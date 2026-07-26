//! Pure Snake game model.
//!
//! Functional design: the game is a plain struct whose state transitions are
//! methods taking `&self`/`&mut self` (state as an explicit argument). No
//! classes, no inheritance. `Game` owns the board geometry and a `VecDeque`
//! of snake cells (index 0 == head).

use std::collections::{HashSet, VecDeque};

use rand::Rng;

/// Compass direction. `delta` returns (d_row, d_col): Up decreases the row
/// index (north), Down increases it (south), Left decreases col (west),
/// Right increases col (east).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    /// The direction that would be a 180-degree reversal of `self`.
    pub fn opposite(self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    /// (row delta, col delta) for one step in this direction.
    pub fn delta(self) -> (i32, i32) {
        match self {
            Direction::Up => (-1, 0),
            Direction::Down => (1, 0),
            Direction::Left => (0, -1),
            Direction::Right => (0, 1),
        }
    }

    /// All four directions in a fixed canonical order.
    pub fn all() -> [Direction; 4] {
        [Direction::Up, Direction::Down, Direction::Left, Direction::Right]
    }
}

/// A single board cell, for rendering / prompting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Empty,
    SnakeHead,
    Snake,
    Food,
}

/// Outcome of a single `step`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepOutcome {
    pub ate: bool,
    pub collided: bool,
}

/// The full game state. `snake[0]` is the head; `snake.back()` is the tail.
#[derive(Clone, Debug)]
pub struct Game {
    pub rows: usize,
    pub cols: usize,
    pub snake: VecDeque<(i32, i32)>,
    pub food: (i32, i32),
    pub dir: Direction,
    pub score: usize,
    pub alive: bool,
}

impl Game {
    /// Create a fresh game: snake of length 3 centered on the board, moving
    /// Right. Food is placed at a uniformly random empty cell.
    pub fn new(rows: usize, cols: usize, rng: &mut impl Rng) -> Game {
        let mid_r = rows as i32 / 2;
        let mid_c = cols as i32 / 2;
        let mut snake = VecDeque::new();
        // head at center; body trails to the west (opposite of Right).
        snake.push_front((mid_r, mid_c));
        snake.push_back((mid_r, mid_c - 1));
        snake.push_back((mid_r, mid_c - 2));
        let food = spawn_food(rows, cols, &snake, rng);
        Game {
            rows,
            cols,
            snake,
            food,
            dir: Direction::Right,
            score: 0,
            alive: true,
        }
    }

    /// Position of the snake head.
    pub fn head(&self) -> (i32, i32) {
        *self.snake.front().expect("snake is never empty")
    }

    /// Advance the game one tick in `dir`. Returns whether food was eaten
    /// and/or the snake collided (and died). A dead game reports
    /// `collided=true` and is left otherwise unchanged.
    ///
    /// Self-collision rule: moving into any body segment is fatal, EXCEPT the
    /// current tail cell when the snake is NOT eating — the tail vacates that
    /// cell as the snake slides forward. Moving into the tail while growing
    /// (eating) is fatal because the tail does not move.
    pub fn step(&mut self, dir: Direction, rng: &mut impl Rng) -> StepOutcome {
        if !self.alive {
            return StepOutcome {
                ate: false,
                collided: true,
            };
        }
        // The direction is updated for display purposes even if the move kills.
        self.dir = dir;

        let (dr, dc) = dir.delta();
        let (hr, hc) = self.head();
        let new_head = (hr + dr, hc + dc);

        let ate = new_head == self.food;
        let in_bounds = in_bounds(self.rows, self.cols, new_head);

        // Build the set of occupied body cells; the tail cell is safe to enter
        // iff the snake slides (does not grow) this tick.
        let tail = *self.snake.back().expect("snake is never empty");
        let allowed_to_enter_tail = !ate && new_head == tail;
        let body: HashSet<(i32, i32)> = self.snake.iter().copied().collect();
        let self_collision = body.contains(&new_head) && !allowed_to_enter_tail;

        let collided = !in_bounds || self_collision;
        if collided {
            self.alive = false;
            return StepOutcome { ate, collided: true };
        }

        if ate {
            self.score += 1;
            self.snake.push_front(new_head); // grow: do not pop the tail
            self.food = spawn_food(self.rows, self.cols, &self.snake, rng);
        } else {
            self.snake.push_front(new_head);
            self.snake.pop_back();
        }

        StepOutcome {
            ate,
            collided: false,
        }
    }

    /// Render the board as a flat `rows*cols` vector (row-major). Head, body
    /// and food are marked; everything else is `Empty`. Pure function of state.
    pub fn cells(&self) -> Vec<Cell> {
        let mut grid = vec![Cell::Empty; self.rows * self.cols];
        let idx = |r: i32, c: i32| -> Option<usize> {
            if r >= 0 && c >= 0 {
                let (ru, cu) = (r as usize, c as usize);
                if ru < self.rows && cu < self.cols {
                    return Some(ru * self.cols + cu);
                }
            }
            None
        };
        for (i, &(r, c)) in self.snake.iter().enumerate() {
            if let Some(i2) = idx(r, c) {
                grid[i2] = if i == 0 { Cell::SnakeHead } else { Cell::Snake };
            }
        }
        // Food never overlaps the snake in a valid state; mark it last.
        if let Some(i2) = idx(self.food.0, self.food.1) {
            if grid[i2] == Cell::Empty {
                grid[i2] = Cell::Food;
            }
        }
        grid
    }

    /// A direction is "valid" if it is not an immediate 180-degree reversal.
    /// Deeper wall/body avoidance is the brain's job; this only prevents the
    /// physically impossible neck-reversal.
    #[allow(dead_code)]
    pub fn is_valid(&self, dir: Direction) -> bool {
        dir != self.dir.opposite()
    }
}

/// Pick a uniformly random empty cell (not on the snake) for new food. If the
/// board is full (snake won), returns the head position as a harmless
/// sentinel — the game is effectively over.
fn spawn_food(rows: usize, cols: usize, snake: &VecDeque<(i32, i32)>, rng: &mut impl Rng) -> (i32, i32) {
    let occupied: HashSet<(i32, i32)> = snake.iter().copied().collect();
    let mut empties: Vec<(i32, i32)> = Vec::new();
    for r in 0..rows as i32 {
        for c in 0..cols as i32 {
            let cell = (r, c);
            if !occupied.contains(&cell) {
                empties.push(cell);
            }
        }
    }
    match empties.is_empty() {
        true => *snake.front().expect("snake is never empty"),
        false => empties[rng.gen_range(0..empties.len())],
    }
}

fn in_bounds(rows: usize, cols: usize, (r, c): (i32, i32)) -> bool {
    r >= 0 && c >= 0 && (r as usize) < rows && (c as usize) < cols
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn default_grid_dimensions() {
        let mut rng = StdRng::seed_from_u64(0);
        let g = Game::new(12, 12, &mut rng);
        assert_eq!(g.rows, 12);
        assert_eq!(g.cols, 12);
        assert_eq!(g.snake.len(), 3);
        assert!(g.alive);
        assert_eq!(g.score, 0);
        assert_eq!(g.dir, Direction::Right);
        assert!(in_bounds(g.rows, g.cols, g.food));
        assert!(in_bounds(g.rows, g.cols, g.head()));
    }

    #[test]
    fn moving_into_wall_kills_snake() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut g = Game::new(4, 4, &mut rng);
        // center col 2 -> move right twice: (2,2)->(2,3) ok, then (2,4) wall.
        assert!(!g.step(Direction::Right, &mut rng).collided);
        assert!(g.alive);
        let out = g.step(Direction::Right, &mut rng);
        assert!(out.collided);
        assert!(!g.alive);
        // a dead game stays dead
        let out2 = g.step(Direction::Right, &mut rng);
        assert!(out2.collided);
    }

    #[test]
    fn eating_food_grows_and_scores_and_respawns() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut g = Game::new(8, 8, &mut rng);
        // Plant food directly east of the head.
        let h = g.head();
        g.food = (h.0, h.1 + 1);
        let len_before = g.snake.len();
        let out = g.step(Direction::Right, &mut rng);
        assert!(out.ate, "should have eaten");
        assert!(!out.collided);
        assert_eq!(g.score, 1);
        assert_eq!(g.snake.len(), len_before + 1, "snake must grow by one");
        // new food must be in-bounds and not on the snake
        assert!(in_bounds(g.rows, g.cols, g.food));
        assert!(!g.snake.iter().any(|&c| c == g.food));
    }

    #[test]
    fn tail_vacate_allows_sliding_into_tail() {
        let mut rng = StdRng::seed_from_u64(0);
        // Construct a game where the head is adjacent to its own tail, so a
        // valid slide moves the head onto the vacating tail cell.
        let mut g = Game::new(5, 5, &mut rng);
        let mut body = VecDeque::new();
        // head (1,2); neck (1,1); (2,1); tail (2,2). head & tail are vertically
        // adjacent. Moving Down lands the head exactly on the current tail.
        body.push_front((1, 2));
        body.push_back((1, 1));
        body.push_back((2, 1));
        body.push_back((2, 2));
        g.snake = body;
        g.dir = Direction::Right; // Down is NOT a reversal of Right
        g.food = (0, 0); // not where we are moving
        let len_before = g.snake.len();
        let out = g.step(Direction::Down, &mut rng);
        assert!(!out.collided, "sliding onto the vacating tail must NOT collide");
        assert!(g.alive);
        assert_eq!(g.snake.len(), len_before, "length is unchanged on a non-eating step");
        assert_eq!(g.head(), (2, 2));
    }

    #[test]
    fn growing_into_tail_is_fatal() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut g = Game::new(5, 5, &mut rng);
        let mut body = VecDeque::new();
        body.push_front((1, 2));
        body.push_back((1, 1));
        body.push_back((2, 1));
        body.push_back((2, 2));
        g.snake = body;
        g.dir = Direction::Right;
        // Food sits exactly on the tail cell -> moving Down "eats" AND lands on
        // the tail. Because the snake grows, the tail does NOT vacate, so this
        // is a self-collision.
        g.food = (2, 2);
        let out = g.step(Direction::Down, &mut rng);
        assert!(out.collided, "growing onto the tail must collide");
        assert!(!g.alive);
    }

    #[test]
    fn reversal_into_neck_is_fatal() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut g = Game::new(6, 6, &mut rng);
        // snake moves Right; reversing to Left moves head onto the neck cell.
        let out = g.step(Direction::Left, &mut rng);
        assert!(out.collided, "180-degree reversal into the neck must be fatal");
        assert!(!g.alive);
    }

    #[test]
    fn food_never_spawns_on_snake() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = Game::new(12, 12, &mut rng);
        for _ in 0..200 {
            if !g.alive {
                break;
            }
            // wander; the heuristic-free random walk keeps the game varied.
            let dirs: Vec<Direction> = Direction::all()
                .iter()
                .copied()
                .filter(|d| g.is_valid(*d))
                .collect();
            let d = dirs[rng.gen_range(0..dirs.len())];
            g.step(d, &mut rng);
            assert!(
                !g.snake.iter().any(|&c| c == g.food),
                "food must never sit on the snake"
            );
        }
    }

    #[test]
    fn cells_marks_head_body_and_food() {
        let mut rng = StdRng::seed_from_u64(0);
        let g = Game::new(6, 6, &mut rng);
        let grid = g.cells();
        let head_count = grid.iter().filter(|&&c| c == Cell::SnakeHead).count();
        let snake_count = grid.iter().filter(|&&c| c == Cell::Snake).count();
        let food_count = grid.iter().filter(|&&c| c == Cell::Food).count();
        assert_eq!(head_count, 1);
        assert_eq!(snake_count, 2, "initial length-3 snake -> 2 body cells");
        assert_eq!(food_count, 1);
    }
}
