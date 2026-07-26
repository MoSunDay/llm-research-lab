//! Brains: strategies that pick the snake's next `Direction`.
//!
//! A `Brain` is a simple trait (object-safe) so the main loop can hold one RNG
//! and drive any brain behind a `Box<dyn Brain>`. The RNG is passed as
//! `&mut dyn rand::RngCore`; rand 0.8's blanket `impl<R: RngCore + ?Sized> Rng
//! for R` means we can still call `rng.gen_range(...)` directly on it.

use std::collections::HashSet;

use rand::{Rng, RngCore};

use crate::game::{Direction, Game};

/// A snake-playing brain. Object-safe: no generics, no `Self` in return.
pub trait Brain {
    /// Choose the direction to move next, given the current game state.
    fn decide(&self, game: &Game, rng: &mut dyn RngCore) -> Direction;
    /// Human-readable label for logging.
    fn name(&self) -> &'static str;
}

/// Picks uniformly at random among the three non-reversing directions.
pub struct RandomBrain;

impl Brain for RandomBrain {
    fn decide(&self, game: &Game, rng: &mut dyn RngCore) -> Direction {
        let current = game.dir;
        let candidates: Vec<Direction> = Direction::all()
            .iter()
            .copied()
            .filter(|d| *d != current.opposite())
            .collect();
        pick(&candidates, rng)
    }

    fn name(&self) -> &'static str {
        "random"
    }
}

/// Greedy heuristic: among the three non-reversing directions, prefer the one
/// whose resulting head cell is safe (in-bounds, not a body cell) AND that
/// minimizes the Manhattan distance to the food. Ties are broken randomly. If
/// every non-reversing move is fatal, fall back to any non-reversing direction.
pub struct WallAvoider;

impl Brain for WallAvoider {
    fn decide(&self, game: &Game, rng: &mut dyn RngCore) -> Direction {
        let current = game.dir;
        let candidates: Vec<Direction> = Direction::all()
            .iter()
            .copied()
            .filter(|d| *d != current.opposite())
            .collect();

        let head = game.head();
        let tail = *game.snake.back().expect("snake is never empty");
        let body: HashSet<(i32, i32)> = game.snake.iter().copied().collect();
        let food = game.food;

        // Score every safe candidate by Manhattan distance to the food.
        let mut safe: Vec<(Direction, i32)> = Vec::new();
        for &d in &candidates {
            let (dr, dc) = d.delta();
            let nh = (head.0 + dr, head.1 + dc);
            if !in_bounds(game, nh) {
                continue;
            }
            // The tail vacates if the snake is not eating this tick.
            let vacates_tail = nh == tail && nh != game.food;
            if body.contains(&nh) && !vacates_tail {
                continue;
            }
            let dist = (nh.0 - food.0).abs() + (nh.1 - food.1).abs();
            safe.push((d, dist));
        }

        if !safe.is_empty() {
            let min_dist = safe.iter().map(|(_, d)| *d).min().unwrap();
            let ties: Vec<Direction> = safe
                .into_iter()
                .filter(|(_, d)| *d == min_dist)
                .map(|(d, _)| d)
                .collect();
            return pick(&ties, rng);
        }

        // No safe move: pick any non-reversing direction (still doomed).
        pick(&candidates, rng)
    }

    fn name(&self) -> &'static str {
        "wall-avoider"
    }
}

fn pick(dirs: &[Direction], rng: &mut dyn RngCore) -> Direction {
    if dirs.len() == 1 {
        return dirs[0];
    }
    dirs[rng.gen_range(0..dirs.len())]
}

fn in_bounds(game: &Game, (r, c): (i32, i32)) -> bool {
    r >= 0 && c >= 0 && (r as usize) < game.rows && (c as usize) < game.cols
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn wall_avoider_never_reverses() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut g = Game::new(12, 12, &mut rng);
        let brain = WallAvoider;
        for _ in 0..120 {
            if !g.alive {
                break;
            }
            let d = brain.decide(&g, &mut rng);
            assert_ne!(
                d,
                g.dir.opposite(),
                "wall-avoider must never pick the reverse direction"
            );
            g.step(d, &mut rng);
        }
    }

    #[test]
    fn random_brain_never_reverses() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut g = Game::new(10, 10, &mut rng);
        let brain = RandomBrain;
        for _ in 0..80 {
            if !g.alive {
                break;
            }
            let d = brain.decide(&g, &mut rng);
            assert_ne!(d, g.dir.opposite());
            g.step(d, &mut rng);
        }
    }

    #[test]
    fn wall_avoider_moves_toward_food_when_clear() {
        // Open 8x8 with food placed directly east of the head and a clear path.
        let mut rng = StdRng::seed_from_u64(3);
        let mut g = Game::new(8, 8, &mut rng);
        let h = g.head();
        g.food = (h.0, h.1 + 3);
        let brain = WallAvoider;
        let d = brain.decide(&g, &mut rng);
        assert_eq!(d, Direction::Right, "should head straight for the food");
    }
}
