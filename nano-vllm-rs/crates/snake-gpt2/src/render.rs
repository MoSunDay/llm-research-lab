//! Terminal renderer.
//!
//! Two paths:
//!  - TTY (interactive): enters the crossterm alternate screen + raw mode via a
//!    `TerminalGuard` whose `Drop` ALWAYS restores the terminal, so a panic or
//!    early `?`-return can never leave the shell in raw mode.
//!  - Non-TTY (piped / sandbox): prints each frame as plain text, separated by
//!    a frame marker, and never touches raw mode.
//!
//! Rendering is a pure function of `Game`; the `Renderer` only owns terminal
//! bookkeeping.

use std::io::{self, IsTerminal, Write};

use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{self, Event, KeyCode},
    execute, queue,
    style::Print,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use crate::game::{Cell, Direction, Game};

/// RAII guard that restores the terminal (leaves alt screen, re-enables echo
/// / canonical mode, shows the cursor) on drop, even on panic.
struct TerminalGuard {
    active: bool,
}

impl TerminalGuard {
    /// Enter raw mode + alternate screen + hide cursor.
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, Hide)?;
        Ok(TerminalGuard { active: true })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.active {
            // Best-effort cleanup; errors here are ignored.
            let _ = execute!(io::stdout(), LeaveAlternateScreen, Show);
            let _ = disable_raw_mode();
        }
    }
}

/// Owns terminal bookkeeping for a run. Dropping it restores the terminal.
pub struct Renderer {
    tty: bool,
    /// Kept alive for its `Drop`; never read directly.
    #[allow(dead_code)]
    guard: Option<TerminalGuard>,
    brain_name: String,
}

impl Renderer {
    /// Construct a renderer for `brain_name`. Detects whether stdout is a TTY
    /// and, if so, takes over the terminal. On any failure to enter raw mode it
    /// silently degrades to the non-TTY path.
    pub fn new(brain_name: &str) -> Self {
        let is_tty = io::stdout().is_terminal();
        let guard = if is_tty {
            match TerminalGuard::enter() {
                Ok(g) => Some(g),
                Err(e) => {
                    eprintln!("[render] could not enter raw mode ({e}); using plain output");
                    None
                }
            }
        } else {
            None
        };
        // Effective tty mode requires the guard to be active.
        let tty = is_tty && guard.is_some();
        Renderer {
            tty,
            guard,
            brain_name: brain_name.to_string(),
        }
    }

    /// Non-blocking check for a 'q' keypress (quit request). Always false in
    /// the non-TTY path (the caller's step cap terminates the run there).
    pub fn should_quit(&self) -> bool {
        if !self.tty {
            return false;
        }
        if let Ok(true) = event::poll(std::time::Duration::from_millis(0)) {
            if let Ok(Event::Key(k)) = event::read() {
                if k.code == KeyCode::Char('q') {
                    return true;
                }
            }
        }
        false
    }

    /// Draw the current game state.
    pub fn draw(&mut self, game: &Game, step: usize, gpt2: bool) {
        let frame = build_frame(game, &self.brain_name, step, gpt2);
        if self.tty {
            let mut out = io::stdout();
            let _ = queue!(out, MoveTo(0, 0), Print(&frame));
            let _ = out.flush();
        } else {
            print!("\n---- frame {step} ----\n{frame}");
            let _ = io::stdout().flush();
        }
    }

    /// Whether we are in interactive TTY mode.
    #[allow(dead_code)]
    pub fn is_tty(&self) -> bool {
        self.tty
    }
}

/// Build the full board + status string for one frame. Pure function.
fn build_frame(game: &Game, brain_name: &str, step: usize, gpt2: bool) -> String {
    let cells = game.cells();
    let border: String = "#".repeat(game.cols + 2);

    let mut s = String::with_capacity((game.cols + 2) * (game.rows + 6));
    s.push_str(&border);
    s.push('\n');
    for r in 0..game.rows {
        s.push('#');
        for c in 0..game.cols {
            let ch = match cells[r * game.cols + c] {
                Cell::Empty => '.',
                Cell::SnakeHead => 'H',
                Cell::Snake => 'o',
                Cell::Food => 'F',
            };
            s.push(ch);
        }
        s.push('#');
        s.push('\n');
    }
    s.push_str(&border);
    s.push('\n');

    let (hr, hc) = game.head();
    let (fr, fc) = game.food;
    let status = format!(
        "score={} steps={} dir={} alive={} brain={} gpt2={} head={},{} food={},{}\nWASD auto | q quits\n",
        game.score,
        step,
        dir_glyph(game.dir),
        game.alive,
        brain_name,
        gpt2,
        hr,
        hc,
        fr,
        fc
    );
    s.push_str(&status);
    s
}

fn dir_glyph(d: Direction) -> char {
    match d {
        Direction::Up => 'U',
        Direction::Down => 'D',
        Direction::Left => 'L',
        Direction::Right => 'R',
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn build_frame_contains_status_and_border() {
        let mut rng = StdRng::seed_from_u64(0);
        let g = Game::new(6, 6, &mut rng);
        let frame = build_frame(&g, "wall-avoider", 3, false);
        assert!(frame.contains("score=0"), "score line present");
        assert!(frame.contains("brain=wall-avoider"));
        assert!(frame.contains("dir=R"), "initial direction is Right");
        // board is 6 wide -> border of 8 '#'
        assert!(frame.contains("########"));
    }

    #[test]
    fn dir_glyph_round_trips_compass() {
        assert_eq!(dir_glyph(Direction::Up), 'U');
        assert_eq!(dir_glyph(Direction::Down), 'D');
        assert_eq!(dir_glyph(Direction::Left), 'L');
        assert_eq!(dir_glyph(Direction::Right), 'R');
    }
}
