//! GPT-2 agent loop: free autoregressive generation -> parse `<bash>` ->
//! sandbox exec -> inject `<result>...</result><resume>` -> re-forward ->
//! repeat until `<answer>` or `max_turns`.
//!
//! The loop is generic over a `step` closure (last-position logits for a token
//! id slice) and an `exec` closure (run a command string -> stdout string) so
//! it is fully unit-testable without real model weights or a real shell. The
//! concrete `run_with_model` wires a real candle `Gpt2` + `sandbox::run_bash`.

use rand::Rng;

use crate::char_tokenizer::CharTokenizer;

/// Control-token literals, byte-identical to agent_special_tokens.py SPECIALS.
#[allow(dead_code)]
pub const SPECIAL_LITERALS: &[&str] = &[
    "<bash>",
    "</bash>",
    "<pause>",
    "<result>",
    "</result>",
    "<resume>",
    "<answer>",
    "</answer>",
];

/// Resolved ids of the 8 control tokens for a given tokenizer.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct SpecialIds {
    pub bash: u32,
    pub bash_close: u32,
    pub pause: u32,
    pub result: u32,
    pub result_close: u32,
    pub resume: u32,
    pub answer: u32,
    pub answer_close: u32,
}

/// Why a generation step stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopKind {
    /// Model emitted `<pause>`: hand control to a tool, then resume.
    Pause,
    /// Model emitted `</answer>` (terminal).
    AnswerClose,
    /// `max_new` reached without a control token.
    MaxNew,
}

/// One observable step of the loop (for printing / asserting).
#[derive(Clone, Debug)]
pub enum AgentEvent {
    Generate { text: String, stop: StopKind },
    Exec { cmd: String, result: String },
    Inject { text: String },
}

/// Result of a full agent run: the concatenated transcript + the event log.
#[derive(Clone, Debug)]
pub struct AgentRun {
    pub transcript: String,
    pub events: Vec<AgentEvent>,
}

/// Resolve the 8 control-token ids from the tokenizer; error if any missing.
pub fn resolve_special_ids(tok: &CharTokenizer) -> anyhow::Result<SpecialIds> {
    let need = |lit: &str| {
        tok.special_id(lit).ok_or_else(|| anyhow::anyhow!("special token {lit:?} missing from vocab"))
    };
    Ok(SpecialIds {
        bash: need("<bash>")?,
        bash_close: need("</bash>")?,
        pause: need("<pause>")?,
        result: need("<result>")?,
        result_close: need("</result>")?,
        resume: need("<resume>")?,
        answer: need("<answer>")?,
        answer_close: need("</answer>")?,
    })
}

/// Classify a generated id as a stop kind, if any.
fn stop_kind(id: u32, s: &SpecialIds) -> Option<StopKind> {
    // `<answer>` (open) is NOT a stop: we generate through it and terminate
    // only on `</answer>`, so the answer *content* is captured.
    if id == s.pause {
        Some(StopKind::Pause)
    } else if id == s.answer_close {
        Some(StopKind::AnswerClose)
    } else {
        None
    }
}

/// Pick the next token id from last-position logits. temp <= 0 => greedy
/// argmax; temp > 0 => temperature sampling.
fn pick_next(logits: &[f32], temp: f32, rng: &mut impl Rng) -> u32 {
    if temp <= 0.0 || logits.is_empty() {
        return argmax(logits);
    }
    let inv = 1.0 / temp;
    let mut m = f32::NEG_INFINITY;
    for &l in logits {
        if l > m {
            m = l;
        }
    }
    let mut probs: Vec<f32> = logits.iter().map(|&l| ((l - m) * inv).exp()).collect();
    let z: f32 = probs.iter().sum();
    if !z.is_finite() || z <= 0.0 {
        return argmax(logits);
    }
    let mut u: f32 = rng.gen::<f32>() * z;
    let mut chosen = 0u32;
    for (i, p) in probs.iter_mut().enumerate() {
        *p /= z;
        u -= *p;
        if u <= 0.0 {
            chosen = i as u32;
            break;
        }
    }
    chosen
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Autoregressively generate, appending to an internal context seeded by
/// `prompt_ids`, until a stop token fires or `max_new` is hit. Returns the
/// generated ids (full, independent of context truncation) and the stop kind.
pub(crate) fn generate_until_stop<S>(
    step: &S,
    prompt_ids: &[u32],
    block_size: usize,
    special: &SpecialIds,
    max_new: usize,
    temp: f32,
    rng: &mut impl Rng,
) -> anyhow::Result<(Vec<u32>, StopKind)>
where
    S: Fn(&[u32]) -> anyhow::Result<Vec<f32>>,
{
    let mut generated: Vec<u32> = Vec::with_capacity(max_new);
    let mut ctx: Vec<u32> = prompt_ids.to_vec();
    for _ in 0..max_new {
        if block_size > 0 && ctx.len() > block_size {
            let start = ctx.len() - block_size;
            ctx = ctx[start..].to_vec();
        }
        let logits = step(&ctx)?;
        let next = pick_next(&logits, temp, rng);
        generated.push(next);
        ctx.push(next);
        if let Some(kind) = stop_kind(next, special) {
            return Ok((generated, kind));
        }
    }
    Ok((generated, StopKind::MaxNew))
}

/// Extract the body of the last `<bash>...</bash>` in `text`. `None` if absent.
pub fn parse_last_bash(text: &str) -> Option<String> {
    let open = text.rfind("<bash>")?;
    let after = open + "<bash>".len();
    let rest = &text[after..];
    let close = rest.find("</bash>")?;
    Some(rest[..close].to_string())
}

/// The agent loop, generic over `step` (logits) and `exec` (command -> stdout).
#[allow(clippy::too_many_arguments)]
pub fn agent_loop<S, E>(
    step: &S,
    tok: &CharTokenizer,
    special: &SpecialIds,
    block_size: usize,
    prompt: &str,
    exec: &E,
    max_turns: usize,
    max_new: usize,
    temp: f32,
    rng: &mut impl Rng,
) -> anyhow::Result<AgentRun>
where
    S: Fn(&[u32]) -> anyhow::Result<Vec<f32>>,
    E: Fn(&str) -> anyhow::Result<String>,
{
    let mut ctx_ids = tok.encode(prompt);
    let mut transcript = String::new();
    let mut events = Vec::new();
    for _ in 0..max_turns {
        let (gen, stop) =
            generate_until_stop(step, &ctx_ids, block_size, special, max_new, temp, rng)?;
        let gen_text = tok.decode(&gen);
        transcript.push_str(&gen_text);
        events.push(AgentEvent::Generate { text: gen_text.clone(), stop });
        match stop {
            StopKind::AnswerClose | StopKind::MaxNew => {
                return Ok(AgentRun { transcript, events });
            }
            StopKind::Pause => {
                let cmd = parse_last_bash(&transcript).unwrap_or_default();
                let result = match exec(&cmd) {
                    Ok(o) => o,
                    Err(e) => format!("error: {e}"),
                };
                events.push(AgentEvent::Exec {
                    cmd: cmd.clone(),
                    result: result.clone(),
                });
                let inject = format!("<result>{result}</result><resume>");
                events.push(AgentEvent::Inject { text: inject.clone() });
                ctx_ids.extend_from_slice(&gen);
                let mut inj_ids = tok.encode(&inject);
                ctx_ids.append(&mut inj_ids);
                transcript.push_str(&inject);
            }
        }
    }
    Ok(AgentRun { transcript, events })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::cell::Cell;

    /// The canonical 65-char tinyshakespeare base, sorted order, so test strings
    /// like "echo R" encode with real ids (matches the real meta.json).
    const BASE_CHARS: &str = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    fn toy_tok() -> CharTokenizer {
        let mut itos: Vec<String> = BASE_CHARS.chars().map(|c| c.to_string()).collect();
        assert_eq!(itos.len(), 65, "base vocab must be 65 chars");
        for lit in SPECIAL_LITERALS {
            itos.push(lit.to_string());
        }
        CharTokenizer::from_itos(itos)
    }

    /// A step closure that emits a pre-scripted sequence of token ids in order,
    /// one per call, regardless of input. Drives the loop deterministically.
    fn scripted_step(script: std::rc::Rc<Vec<u32>>) -> impl Fn(&[u32]) -> anyhow::Result<Vec<f32>> {
        let i = std::rc::Rc::new(Cell::new(0usize));
        move |_ids: &[u32]| {
            let idx = i.get();
            let id = script[idx.min(script.len().saturating_sub(1))];
            i.set(idx + 1);
            let mut logits = vec![-10.0_f32; 81];
            if (id as usize) < logits.len() {
                logits[id as usize] = 10.0;
            }
            Ok(logits)
        }
    }

    #[test]
    fn resolve_ids_ok() {
        let tok = toy_tok();
        let s = resolve_special_ids(&tok).expect("resolve");
        assert_eq!(s.bash, 65);
        assert_eq!(s.pause, 67);
        assert_eq!(s.answer, 71);
        assert_eq!(s.answer_close, 72);
    }

    #[test]
    fn parse_last_bash_basic() {
        assert_eq!(parse_last_bash("x<bash>echo D</bash>y"), Some("echo D".to_string()));
    }

    #[test]
    fn parse_last_bash_picks_last() {
        let t = "<bash>a</bash> mid <bash>b</bash> tail";
        assert_eq!(parse_last_bash(t), Some("b".to_string()));
    }

    #[test]
    fn parse_last_bash_multiline() {
        let t = "<bash>cat > x << 'EOF'\nline1\nline2\nEOF</bash>";
        assert_eq!(
            parse_last_bash(t),
            Some("cat > x << 'EOF'\nline1\nline2\nEOF".to_string())
        );
    }

    #[test]
    fn parse_last_bash_none() {
        assert_eq!(parse_last_bash("no command here"), None);
    }

    #[test]
    fn pick_next_greedy_is_argmax() {
        let mut rng = StdRng::seed_from_u64(2);
        let logits = vec![1.0_f32, 5.0, 2.0, -3.0];
        assert_eq!(pick_next(&logits, 0.0, &mut rng), 1);
    }

    #[test]
    fn generate_stops_at_pause() {
        let tok = toy_tok();
        let s = resolve_special_ids(&tok).unwrap();
        let script: Vec<u32> = tok.encode("<bash>echo R</bash><pause>");
        let step = scripted_step(std::rc::Rc::new(script.clone()));
        let mut rng = StdRng::seed_from_u64(1);
        let (gen, stop) = generate_until_stop(&step, &[], 256, &s, 64, 0.0, &mut rng).unwrap();
        assert_eq!(gen, script);
        assert_eq!(stop, StopKind::Pause);
    }

    #[test]
    fn agent_loop_two_turns_injects_and_terminates() {
        let tok = toy_tok();
        let s = resolve_special_ids(&tok).unwrap();
        let t1: Vec<u32> = tok.encode("<bash>echo R</bash><pause>");
        let t2: Vec<u32> = tok.encode("<answer>done</answer>");
        let script: Vec<u32> = t1.iter().chain(t2.iter()).copied().collect();
        let step = scripted_step(std::rc::Rc::new(script));
        let exec = |cmd: &str| Ok(format!("ran<{cmd}>"));
        let mut rng = StdRng::seed_from_u64(3);
        let run = agent_loop(
            &step, &tok, &s, 256, "PLAY\n", &exec, 8, 64, 0.0, &mut rng,
        )
        .unwrap();
        // Generate(Pause), Exec, Inject, Generate(Answer)
        assert_eq!(run.events.len(), 4, "{:?}", run.events);
        match &run.events[1] {
            AgentEvent::Exec { cmd, result } => {
                assert_eq!(cmd, "echo R");
                assert_eq!(result, "ran<echo R>");
            }
            other => panic!("expected Exec, got {other:?}"),
        }
        if let AgentEvent::Inject { text } = &run.events[2] {
            assert_eq!(text, "<result>ran<echo R></result><resume>");
        } else {
            panic!("expected Inject");
        }
        if let AgentEvent::Generate { stop, .. } = &run.events[3] {
            assert_eq!(*stop, StopKind::AnswerClose);
        } else {
            panic!("expected final Generate(AnswerClose)");
        }
        assert!(run.transcript.contains("<bash>echo R</bash><pause>"));
        assert!(run.transcript.contains("<result>ran<echo R></result><resume>"));
        assert!(run.transcript.ends_with("<answer>done</answer>"));
    }

    #[test]
    fn agent_loop_pause_without_bash_continues() {
        let tok = toy_tok();
        let s = resolve_special_ids(&tok).unwrap();
        let t1: Vec<u32> = tok.encode("<pause>");
        let t2: Vec<u32> = tok.encode("<answer>x</answer>");
        let script: Vec<u32> = t1.iter().chain(t2.iter()).copied().collect();
        let step = scripted_step(std::rc::Rc::new(script));
        let exec = |cmd: &str| Ok(format!("ran<{cmd}>"));
        let mut rng = StdRng::seed_from_u64(4);
        let run = agent_loop(&step, &tok, &s, 256, "Q", &exec, 8, 64, 0.0, &mut rng).unwrap();
        assert!(run.transcript.contains("<answer>x</answer>"));
    }

    #[test]
    fn agent_loop_max_new_terminates() {
        let tok = toy_tok();
        let s = resolve_special_ids(&tok).unwrap();
        // never emits a stop token -> MaxNew ends turn 1.
        let step = scripted_step(std::rc::Rc::new(vec![0u32; 1000]));
        let exec = |_cmd: &str| Ok("ok".to_string());
        let mut rng = StdRng::seed_from_u64(5);
        let run = agent_loop(&step, &tok, &s, 256, "Q", &exec, 8, 10, 0.0, &mut rng).unwrap();
        assert_eq!(run.events.len(), 1);
        if let AgentEvent::Generate { stop, .. } = &run.events[0] {
            assert_eq!(*stop, StopKind::MaxNew);
        } else {
            panic!("expected Generate(MaxNew)");
        }
    }
}
