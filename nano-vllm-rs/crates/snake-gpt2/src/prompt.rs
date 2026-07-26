//! Board-to-text prompt builder for the GPT-2 brain.
//!
//! The prompt is a compact ASCII rendering of the board. To keep the
//! strategically-important header (HEAD/FOOD/DIR/SCORE) and the `MOVE:` cue
//! adjacent to the model's read position, the board is rendered FIRST and the
//! header + `MOVE:` footer are appended at the END. Left-truncation (done by
//! the brain when the prompt exceeds `block_size`) therefore only drops top
//! board rows while always preserving the header and the action prompt.
//!
//! All functions are pure and deterministic: same `Game` -> same string.

use crate::game::{Cell, Direction, Game};

/// Action key for a direction (WASD): Up=W, Left=A, Down=S, Right=D.
pub fn dir_to_action_char(d: Direction) -> char {
    match d {
        Direction::Up => 'W',
        Direction::Left => 'A',
        Direction::Down => 'S',
        Direction::Right => 'D',
    }
}

/// Inverse of `dir_to_action_char`. Accepts both cases for robustness.
#[allow(dead_code)]
pub fn action_char_to_dir(ch: char) -> Option<Direction> {
    match ch {
        'W' | 'w' => Some(Direction::Up),
        'A' | 'a' => Some(Direction::Left),
        'S' | 's' => Some(Direction::Down),
        'D' | 'd' => Some(Direction::Right),
        _ => None,
    }
}

/// Compass glyph for the DIR header: U/D/L/R.
fn dir_to_display(d: Direction) -> char {
    match d {
        Direction::Up => 'U',
        Direction::Down => 'D',
        Direction::Left => 'L',
        Direction::Right => 'R',
    }
}

/// Render the board (with a `#` border) as one `String` per line. Body cells
/// use `o` (NOT `S`, which is reserved for the Down action token).
pub fn render_board(game: &Game) -> Vec<String> {
    let cells = game.cells();
    let mut lines = Vec::with_capacity(game.rows + 2);
    let border: String = "#".repeat(game.cols + 2);
    lines.push(border);
    for r in 0..game.rows {
        let mut row = String::with_capacity(game.cols + 2);
        row.push('#');
        for c in 0..game.cols {
            let ch = match cells[r * game.cols + c] {
                Cell::Empty => '.',
                Cell::SnakeHead => 'H',
                Cell::Snake => 'o',
                Cell::Food => 'F',
            };
            row.push(ch);
        }
        row.push('#');
        lines.push(row);
    }
    let border: String = "#".repeat(game.cols + 2);
    lines.push(border);
    lines
}

/// Build the full prompt string. Layout:
/// ```text
/// SNAKE
/// <board>
/// HEAD=r,c FOOD=r,c DIR=X SCORE=n
/// MOVE:
/// ```
/// The `SNAKE` title and top board rows are the first to be dropped by
/// left-truncation; the header and `MOVE:` (at the end) always survive.
pub fn build_prompt(game: &Game, max_tokens: usize) -> String {
    let board_lines = render_board(game);
    let (hr, hc) = game.head();
    let (fr, fc) = game.food;
    let header = format!(
        "HEAD={},{} FOOD={},{} DIR={} SCORE={}",
        hr,
        hc,
        fr,
        fc,
        dir_to_display(game.dir),
        game.score
    );

    let mut parts: Vec<String> = Vec::with_capacity(board_lines.len() + 3);
    parts.push("SNAKE".to_string());
    parts.extend(board_lines);
    parts.push(header);
    parts.push("MOVE:".to_string());
    let full = parts.join("\n");

    // Enforce the char budget locally too: if the caller's max_tokens is
    // smaller than the whole prompt, drop whole lines from the TOP so the
    // header + MOVE: (last two lines) are always retained.
    if max_tokens > 0 && full.chars().count() > max_tokens {
        truncate_top_lines(&full, max_tokens)
    } else {
        full
    }
}

/// Keep the last `max_tokens` chars, but snap the cut point forward to the
/// next line boundary so we never emit a half-board row.
fn truncate_top_lines(full: &str, max_tokens: usize) -> String {
    let chars: Vec<char> = full.chars().collect();
    let n = chars.len();
    let mut start = n.saturating_sub(max_tokens);
    // advance to the character just past the next newline
    while start < n && chars[start] != '\n' {
        start += 1;
    }
    if start < n && chars[start] == '\n' {
        start += 1;
    }
    chars[start..].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn sample_game() -> Game {
        let mut rng = StdRng::seed_from_u64(0);
        Game::new(12, 12, &mut rng)
    }

    #[test]
    fn prompt_contains_required_markers() {
        let g = sample_game();
        let p = build_prompt(&g, 4096);
        assert!(p.starts_with("SNAKE"), "title first");
        assert!(p.contains("HEAD="), "header present");
        assert!(p.contains("DIR="));
        assert!(p.contains("SCORE="));
        assert!(p.contains("MOVE:"), "action cue present");
        // MOVE: is the very last line
        assert!(p.ends_with("MOVE:"));
    }

    #[test]
    fn truncation_keeps_header_and_move() {
        let g = sample_game();
        // tiny budget forces top board rows to be dropped.
        let p = build_prompt(&g, 60);
        assert!(p.contains("MOVE:"));
        assert!(p.contains("DIR="));
        assert!(p.chars().count() <= 60 + 2, "roughly within budget");
    }

    #[test]
    fn action_char_round_trips() {
        for d in Direction::all() {
            let ch = dir_to_action_char(d);
            assert_eq!(action_char_to_dir(ch), Some(d));
        }
        assert_eq!(action_char_to_dir('x'), None);
    }

    #[test]
    fn body_uses_o_not_s() {
        let g = sample_game();
        let p = build_prompt(&g, 4096);
        // The initial snake has body cells; they must render as 'o' so they do
        // not collide with the 'S' (Down) action token.
        assert!(p.contains('o'), "body rendered as 'o'");
    }
}
