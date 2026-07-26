//! Character + special-token tokenizer for the GPT-2 agent.
//!
//! `meta.json` describes the vocabulary: `{ "vocab_size": N, "itos": [...],
//! "stoi": {...}, "special_tokens": {...} }`. Entries 0..64 are the 65
//! tinyshakespeare single characters; entries 65.. are multi-character control
//! tokens (`<bash>`, `</bash>`, ...). Encode does longest-match so the
//! multi-char specials are recognized as single tokens; unknown single
//! characters are skipped (no BPE, fully offline).
//!
//! Pure data + free functions; no interior mutability.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

/// Char + special-token tokenizer.
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    /// id -> token string (single char or special literal).
    itos: Vec<String>,
    /// token string -> id.
    stoi: HashMap<String, u32>,
    /// Multi-char special literals, sorted longest-first for greedy matching.
    specials: Vec<String>,
    #[allow(dead_code)]
    vocab_size: usize,
}

/// Raw shape of `meta.json`.
#[derive(Deserialize)]
struct Meta {
    vocab_size: usize,
    itos: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    stoi: HashMap<String, u32>,
    #[serde(default)]
    special_tokens: HashMap<String, u32>,
}

impl CharTokenizer {
    /// Read `meta.json` from `dir` and build the tokenizer. The source of truth
    /// for ids is the `itos` index; `stoi` is rebuilt from it. Multi-char
    /// `itos` entries become longest-match specials.
    pub fn from_meta_dir(dir: &Path) -> anyhow::Result<Self> {
        let meta_path = dir.join("meta.json");
        let content = std::fs::read_to_string(&meta_path)
            .map_err(|e| anyhow::anyhow!("read meta.json at {}: {e}", meta_path.display()))?;
        let meta: Meta = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("parse meta.json: {e}"))?;

        if meta.itos.len() != meta.vocab_size {
            anyhow::bail!(
                "meta.json inconsistent: vocab_size={} but itos has {} entries",
                meta.vocab_size,
                meta.itos.len()
            );
        }

        // itos index is authoritative; rebuild stoi from it.
        let mut stoi: HashMap<String, u32> = HashMap::with_capacity(meta.itos.len());
        for (id, s) in meta.itos.iter().enumerate() {
            stoi.insert(s.clone(), id as u32);
        }

        // Specials: the union of multi-char itos entries and the declared
        // special_tokens, deduped, sorted by descending char length so the
        // longest match wins during encode.
        let mut specials: Vec<String> = meta
            .itos
            .iter()
            .filter(|s| s.chars().count() > 1)
            .cloned()
            .collect();
        for lit in meta.special_tokens.keys() {
            if !specials.iter().any(|s| s == lit) {
                specials.push(lit.clone());
            }
        }
        specials.sort_by_key(|b| std::cmp::Reverse(b.chars().count()));

        Ok(CharTokenizer {
            itos: meta.itos,
            stoi,
            specials,
            vocab_size: meta.vocab_size,
        })
    }

    /// Build a tokenizer directly from an `itos` list (id == index). Multi-char
    /// entries are inferred as longest-match specials. Useful for tests and for
    /// constructing a tokenizer without a meta.json on disk.
    #[allow(dead_code)]
    pub fn from_itos(itos: Vec<String>) -> Self {
        let stoi: HashMap<String, u32> = itos
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let mut specials: Vec<String> = itos
            .iter()
            .filter(|s| s.chars().count() > 1)
            .cloned()
            .collect();
        specials.sort_by_key(|b| std::cmp::Reverse(b.chars().count()));
        let vocab_size = itos.len();
        CharTokenizer {
            itos,
            stoi,
            specials,
            vocab_size,
        }
    }

    /// Encode a string to token ids with longest-match specials. Unknown single
    /// characters are silently skipped.
    pub fn encode(&self, s: &str) -> Vec<u32> {
        let chars: Vec<char> = s.chars().collect();
        let mut ids = Vec::with_capacity(chars.len());
        let mut i = 0;
        while i < chars.len() {
            let rest: String = chars[i..].iter().collect();
            let mut matched: Option<(&str, usize)> = None;
            for lit in &self.specials {
                if rest.starts_with(lit.as_str()) {
                    let len = lit.chars().count();
                    matched = Some((lit.as_str(), len));
                    break; // specials sorted longest-first
                }
            }
            if let Some((lit, len)) = matched {
                if let Some(&id) = self.stoi.get(lit) {
                    ids.push(id);
                }
                i += len;
                continue;
            }
            // single char
            let one: String = chars[i..i + 1].iter().collect();
            if let Some(&id) = self.stoi.get(one.as_str()) {
                ids.push(id);
            }
            i += 1;
        }
        ids
    }

    /// Decode a slice of ids back to a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(s) = self.itos.get(id as usize) {
                out.push_str(s);
            }
        }
        out
    }

    /// Decode a single id, if in range.
    #[allow(dead_code)]
    pub fn decode_id(&self, id: u32) -> Option<&str> {
        self.itos.get(id as usize).map(|s| s.as_str())
    }

    /// The declared vocabulary size.
    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Resolve the id for a single action character (e.g. 'W').
    pub fn action_token_id(&self, ch: char) -> Option<u32> {
        let s: String = ch.to_string();
        self.stoi.get(s.as_str()).copied()
    }

    /// Resolve the id for a special-token literal (e.g. "<bash>").
    pub fn special_id(&self, literal: &str) -> Option<u32> {
        self.stoi.get(literal).copied()
    }

    /// True if the given id corresponds to a multi-char special token.
    #[allow(dead_code)]
    pub fn is_special(&self, id: u32) -> bool {
        self.itos
            .get(id as usize)
            .map(|s| s.chars().count() > 1)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_tok() -> CharTokenizer {
        // 4 base chars + the 8 specials, matching the agent contract shape.
        let itos: Vec<String> = vec![
            "\n".into(), " ".into(), "W".into(), "D".into(),
            "<bash>".into(), "</bash>".into(), "<pause>".into(),
            "<result>".into(), "</result>".into(), "<resume>".into(),
            "<answer>".into(), "</answer>".into(),
        ];
        let stoi: HashMap<String, u32> = itos
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let specials = vec![
            "</answer>".to_string(), "</result>".to_string(), "</bash>".to_string(),
            "<result>".to_string(), "<answer>".to_string(), "<resume>".to_string(),
            "<bash>".to_string(), "<pause>".to_string(),
        ];
        CharTokenizer { itos, stoi, specials, vocab_size: 12 }
    }

    #[test]
    fn encode_longest_match_specials() {
        let t = agent_tok();
        // "<bash>" is id 4, "</bash>" id 5; must not be split into chars.
        let ids = t.encode("<bash>echo</bash><pause>");
        // "<bash>"(4) 'e'? not in vocab -> skipped ... 'c' skipped 'h' skipped
        // 'o' skipped, "</bash>"(5), "<pause>"(6)
        assert_eq!(ids, vec![4, 5, 6]);
    }

    #[test]
    fn encode_single_chars_and_unknowns() {
        let t = agent_tok();
        let ids = t.encode("W D X\n");
        // W(2) space(1) D(3) space(1) X unknown->skip \n(0)
        assert_eq!(ids, vec![2, 1, 3, 1, 0]);
    }

    #[test]
    fn decode_round_trips() {
        let t = agent_tok();
        let s = "W<bash>D</bash><pause>";
        let ids = t.encode(s);
        assert_eq!(t.decode(&ids), s, "round trip");
    }

    #[test]
    fn decode_id_and_special_lookup() {
        let t = agent_tok();
        assert_eq!(t.decode_id(4), Some("<bash>"));
        assert_eq!(t.decode_id(0), Some("\n"));
        assert_eq!(t.decode_id(99), None);
        assert_eq!(t.special_id("</bash>"), Some(5));
        assert_eq!(t.special_id("<nope>"), None);
        assert!(t.is_special(6));
        assert!(!t.is_special(2));
    }

    #[test]
    fn action_token_still_works() {
        let t = agent_tok();
        assert_eq!(t.action_token_id('W'), Some(2));
        assert_eq!(t.action_token_id('D'), Some(3));
        assert_eq!(t.action_token_id('S'), None);
    }

    #[test]
    fn answer_close_vs_answer_distinct() {
        let t = agent_tok();
        assert_ne!(t.special_id("<answer>"), t.special_id("</answer>"));
        // encode must pick the longest: "</answer>" not "<answer>"+...
        let ids = t.encode("x<answer>y</answer>z");
        // 'x' unknown skip, <answer>(10), 'y' unknown, </answer>(11), 'z' skip
        assert_eq!(ids, vec![10, 11]);
    }
}
