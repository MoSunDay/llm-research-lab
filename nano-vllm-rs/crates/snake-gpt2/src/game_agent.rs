//! Closed-loop game agent: the GPT-2 agent model actually *plays* Snake.
//!
//! Builds a board prompt byte-identical to the SFT training format, generates a
//! `<bash>echo {U|D|L|R}</bash><pause>` action, parses the UDLR move, steps the
//! `Game`, injects `<result>{dir}</result><resume>` + the new board, and
//! repeats until the snake dies or the turn budget runs out.
//!
//! This bridges the two previously-disjoint code paths: the `agent` text
//! generation loop and the `game` engine. The generic `agent_loop` cannot be
//! reused directly because each turn must inject the *new* board state (the
//! exec closure only returns a result string).

use candle_core::{Device, Tensor};
use rand::Rng;

use crate::agent::{generate_until_stop, parse_last_bash, SpecialIds, StopKind};
use crate::char_tokenizer::CharTokenizer;
use crate::game::{Direction, Game};
use crate::prompt;
use crate::runner::AgentModel;

/// Task preamble, byte-identical to `prepare_agent_snake.py` `gen_game_trace`.
const TASK_PREAMBLE: &str = concat!(
    "Task: choose the snake's next move (U/D/L/R) ",
    "to reach the food without crashing.\nState:\n",
);

// ---------------------------------------------------------------- helpers ---

/// UDLR compass glyph for display (matches the training `DIR=` header).
fn dir_glyph(d: Direction) -> char {
    match d {
        Direction::Up => 'U',
        Direction::Down => 'D',
        Direction::Left => 'L',
        Direction::Right => 'R',
    }
}

/// Parse a single UDLR char into a Direction.
///
/// This is distinct from `prompt::action_char_to_dir` (WASD), which belongs to
/// the constrained-decoding path. The agent model (vocab=73) was trained
/// exclusively on UDLR actions.
pub fn udlr_to_dir(ch: char) -> Option<Direction> {
    match ch.to_ascii_uppercase() {
        'U' => Some(Direction::Up),
        'D' => Some(Direction::Down),
        'L' => Some(Direction::Left),
        'R' => Some(Direction::Right),
        _ => None,
    }
}

/// Extract the UDLR action from a bash command.
///
/// The model is trained on `echo {U|D|L|R}`. We locate `echo` and read the
/// first char of its argument (skipping whitespace/quotes), returning `None`
/// for anything that is not a standalone UDLR char. This avoids false matches
/// like the `l` in `ls` or the `d` in `cd`.
pub fn parse_action(cmd: &str) -> Option<Direction> {
    let lower = cmd.to_ascii_lowercase();
    let pos = lower.find("echo")?;
    let args = &cmd[pos + 4..];
    let first = args
        .trim_start_matches(|c: char| c.is_whitespace() || c == '"' || c == '\'');
    first.chars().next().and_then(udlr_to_dir)
}

/// Render a board prompt segment matching the SFT training `board_prompt`:
/// board rows joined by `\n`, then a `HEAD=… FOOD=… DIR=… SCORE=…` header.
fn board_prompt(game: &Game) -> String {
    let lines = prompt::render_board(game);
    let (hr, hc) = game.head();
    let (fr, fc) = game.food;
    let mut s = lines.join("\n");
    s.push_str(&format!(
        "\nHEAD={},{} FOOD={},{} DIR={} SCORE={}",
        hr, hc, fr, fc, dir_glyph(game.dir), game.score,
    ));
    s
}

// --------------------------------------------------------------- structs ----

/// One observed move of the game-agent loop.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct TurnOutcome {
    /// Raw text the model generated this turn.
    pub generated: String,
    /// The parsed `<bash>…</bash>` body (empty if absent).
    pub cmd: String,
    /// The action applied. `None` when the move was unparsable (fallback).
    pub action: Option<Direction>,
    /// True if the snake ate food this step.
    pub ate: bool,
    /// True if the snake crashed this step.
    pub collided: bool,
    /// True when `action` came from a parse failure fallback (kept direction).
    pub fallback: bool,
}

/// Why the loop terminated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Snake crashed into a wall or its own body.
    Collided,
    /// Model emitted `</answer>`.
    Answer,
    /// Turn budget exhausted (snake still alive).
    MaxTurns,
    /// `max_new` tokens generated without a control token.
    MaxNew,
}

/// Result of a full game-agent run.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct GameAgentRun {
    pub steps: usize,
    pub score: usize,
    pub alive: bool,
    pub turns: Vec<TurnOutcome>,
    pub transcript: String,
    pub stop_reason: StopReason,
}

// --------------------------------------------------------------- the loop ---

/// Run the closed-loop game agent: the model actually plays Snake.
///
/// Each turn: build/extend context -> generate until `<pause>` -> parse the
/// UDLR action -> `game.step` -> inject result + new board -> repeat.
#[allow(clippy::too_many_arguments)]
pub fn run_game_agent<R: Rng>(
    am: &AgentModel,
    special: &SpecialIds,
    rows: usize,
    cols: usize,
    max_turns: usize,
    max_new: usize,
    temp: f32,
    rng: &mut R,
) -> anyhow::Result<GameAgentRun> {
    let tok: &CharTokenizer = &am.tok;
    let block_size = am.block_size;

    let mut game = Game::new(rows, cols, rng);

    // Forward-pass closure: token-id slice -> last-position logits over vocab.
    let step = |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
        let t = ids.len().max(1);
        let data: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let idx = Tensor::from_vec(data, (1, t), &Device::Cpu)?;
        let logits = am.model.forward(&idx)?;
        Ok(logits.get(0)?.get(t - 1)?.to_vec1::<f32>()?)
    };

    // Seed context: preamble + first board + "MOVE:" (matches SFT format).
    let mut ctx_ids = tok.encode(TASK_PREAMBLE);
    let first_board = format!("{}\nMOVE:\n", board_prompt(&game));
    ctx_ids.extend_from_slice(&tok.encode(&first_board));

    let mut transcript = String::from(TASK_PREAMBLE);
    transcript.push_str(&first_board);

    let mut turns: Vec<TurnOutcome> = Vec::new();

    for _ in 0..max_turns {
        let (gen, stop) =
            generate_until_stop(&step, &ctx_ids, block_size, special, max_new, temp, rng)?;
        let gen_text = tok.decode(&gen);
        ctx_ids.extend_from_slice(&gen);
        transcript.push_str(&gen_text);

        match stop {
            StopKind::AnswerClose => {
                turns.push(TurnOutcome {
                    generated: gen_text, cmd: String::new(), action: None,
                    ate: false, collided: false, fallback: false,
                });
                return Ok(finish(game, turns, transcript, StopReason::Answer));
            }
            StopKind::MaxNew => {
                let cmd = parse_last_bash(&transcript).unwrap_or_default();
                turns.push(TurnOutcome {
                    generated: gen_text, cmd, action: None,
                    ate: false, collided: false, fallback: false,
                });
                return Ok(finish(game, turns, transcript, StopReason::MaxNew));
            }
            StopKind::Pause => {
                let cmd = parse_last_bash(&transcript).unwrap_or_default();
                let action = parse_action(&cmd);
                let fallback = action.is_none();
                let dir = action.unwrap_or(game.dir);

                let outcome = game.step(dir, rng);

                turns.push(TurnOutcome {
                    generated: gen_text, cmd, action,
                    ate: outcome.ate, collided: outcome.collided, fallback,
                });

                // Inject result + advance context (matches SFT format).
                let inject = format!("<result>{}</result><resume>", dir_glyph(dir));
                ctx_ids.extend_from_slice(&tok.encode(&inject));
                transcript.push_str(&inject);

                if !game.alive {
                    return Ok(finish(game, turns, transcript, StopReason::Collided));
                }

                // Append the new board for the next turn.
                let next_board = format!("{}\nMOVE:\n", board_prompt(&game));
                ctx_ids.extend_from_slice(&tok.encode(&next_board));
                transcript.push_str(&next_board);
            }
        }
    }

    Ok(finish(game, turns, transcript, StopReason::MaxTurns))
}

fn finish(
    game: Game,
    turns: Vec<TurnOutcome>,
    transcript: String,
    reason: StopReason,
) -> GameAgentRun {
    GameAgentRun {
        steps: turns.len(),
        score: game.score,
        alive: game.alive,
        turns,
        transcript,
        stop_reason: reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn udlr_maps_all_four() {
        assert_eq!(udlr_to_dir('U'), Some(Direction::Up));
        assert_eq!(udlr_to_dir('D'), Some(Direction::Down));
        assert_eq!(udlr_to_dir('L'), Some(Direction::Left));
        assert_eq!(udlr_to_dir('R'), Some(Direction::Right));
        // case-insensitive
        assert_eq!(udlr_to_dir('u'), Some(Direction::Up));
        assert_eq!(udlr_to_dir('r'), Some(Direction::Right));
    }

    #[test]
    fn udlr_rejects_non_actions() {
        assert_eq!(udlr_to_dir('W'), None); // WASD Up — NOT an agent token
        assert_eq!(udlr_to_dir('S'), None); // WASD Down — NOT an agent token
        assert_eq!(udlr_to_dir(' '), None);
        assert_eq!(udlr_to_dir('?'), None);
    }

    #[test]
    fn parse_action_from_echo() {
        assert_eq!(parse_action("echo U"), Some(Direction::Up));
        assert_eq!(parse_action("echo d"), Some(Direction::Down));
        assert_eq!(parse_action("echo L"), Some(Direction::Left));
        assert_eq!(parse_action("echo R"), Some(Direction::Right));
    }

    #[test]
    fn parse_action_garbage_is_none() {
        assert_eq!(parse_action("ls -la"), None);
        assert_eq!(parse_action(""), None);
        assert_eq!(parse_action("echo hello"), None);
        assert_eq!(parse_action("curl http://x"), None);
    }

    #[test]
    fn parse_action_handles_quotes_and_case() {
        assert_eq!(parse_action("echo \"U\""), Some(Direction::Up));
        assert_eq!(parse_action("  echo   r"), Some(Direction::Right));
    }

    #[test]
    fn parse_action_ignores_non_echo_prefix() {
        // "gecho" contains "echo" as substring — acceptable edge case.
        assert_eq!(parse_action("printf U"), None);
    }

    #[test]
    fn dir_glyph_is_udlr() {
        assert_eq!(dir_glyph(Direction::Up), 'U');
        assert_eq!(dir_glyph(Direction::Down), 'D');
        assert_eq!(dir_glyph(Direction::Left), 'L');
        assert_eq!(dir_glyph(Direction::Right), 'R');
    }

    #[test]
    fn board_prompt_matches_training_layout() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let game = Game::new(4, 4, &mut rng);
        let p = board_prompt(&game);
        // Must start with the top border row.
        assert!(p.starts_with("######"));
        // Must contain the header with DIR= and SCORE=.
        assert!(p.contains("DIR="));
        assert!(p.contains("SCORE=0"));
        assert!(p.contains(&format!("HEAD={}", game.head().0)));
        // Must NOT end with a trailing newline (caller appends \nMOVE:\n).
        assert!(!p.ends_with('\n'));
    }

    /// Live end-to-end game-agent run, gated on SNAKE_AGENT_MODEL.
    #[test]
    fn game_agent_live_when_model_present() {
        let path = match std::env::var("SNAKE_AGENT_MODEL") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!("[game_agent] SNAKE_AGENT_MODEL unset; skipping live test");
                return;
            }
        };
        use crate::agent::resolve_special_ids;
        use crate::runner;
        let am = runner::load(std::path::Path::new(&path)).expect("load agent model");
        let special = resolve_special_ids(&am.tok).expect("resolve specials");
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let run = run_game_agent(
            &am, &special, 8, 8, 20, 200, 0.0, &mut rng,
        ).expect("game agent run");
        eprintln!("[game_agent] live: score={} steps={} alive={}",
                  run.score, run.steps, run.alive);
        assert!(run.steps > 0, "must play at least one turn");
        assert!(run.turns.iter().any(|t| t.action.is_some()),
                "at least one turn must produce a parsed action");
    }
}
