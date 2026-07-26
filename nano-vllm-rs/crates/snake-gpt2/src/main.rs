//! `snake-gpt2`: a terminal Snake game whose brain is (optionally) a GPT-2,
//! plus an `--agent "<task>"` mode that runs a bash-tool-use loop driven by a
//! fine-tuned GPT-2 (generate `<bash>` -> exec -> inject `<result>` -> ... ->
//! `<answer>`).
//!
//! Brain selection priority: `--random` > `--heuristic` > `--model <dir>`
//! (with a `WallAvoider` fallback on load failure) > default `WallAvoider`.
//! When `--agent` is given, the Snake game is skipped and the agent loop runs
//! instead, requiring `--model <dir>`.

mod agent;
mod brain;
mod char_tokenizer;
mod game;
mod game_agent;
mod gpt2_brain;
mod prompt;
mod render;
mod runner;
mod sandbox;

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;

use brain::{Brain, RandomBrain, WallAvoider};
use game::Game;
use render::Renderer;

#[derive(Parser)]
#[command(
    name = "snake-gpt2",
    about = "Terminal Snake powered by an optional GPT-2 brain, and a GPT-2 bash agent loop"
)]
struct Cli {
    /// Path to exported model dir (model.safetensors + config.json + meta.json).
    #[arg(long)]
    model: Option<PathBuf>,

    /// Use the random brain (ignores --model).
    #[arg(long)]
    random: bool,

    /// Use the wall-avoider heuristic brain.
    #[arg(long)]
    heuristic: bool,

    /// Grid rows.
    #[arg(long, default_value_t = 12)]
    rows: usize,

    /// Grid cols.
    #[arg(long, default_value_t = 12)]
    cols: usize,

    /// Target frames per second.
    #[arg(long, default_value_t = 8)]
    fps: u64,

    /// Auto-quit after this many game steps (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    steps: usize,

    /// Per-step delay override in milliseconds (0 = use 1000/fps).
    #[arg(long = "delay_ms", default_value_t = 0)]
    delay_ms: u64,

    /// Run the GPT-2 bash agent loop on this task instead of the Snake game.
    /// Requires --model <dir>.
    #[arg(long)]
    agent: Option<String>,

    /// Working directory for agent bash execution (default: a temp dir).
    #[arg(long)]
    agent_workdir: Option<PathBuf>,

    /// Max agent tool-use turns.
    #[arg(long, default_value_t = 6)]
    agent_max_turns: usize,

    /// Max new tokens generated per turn before giving up.
    #[arg(long, default_value_t = 200)]
    agent_max_new: usize,

    /// Sampling temperature for agent generation (0 = greedy).
    #[arg(long, default_value_t = 0.7)]
    temp: f64,

    /// Use the agent model to actually play Snake (closed loop: board ->
    /// generate UDLR action -> step -> new board -> repeat). Requires --model.
    #[arg(long)]
    play: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.play {
        return run_agent_game(&cli);
    }

    if let Some(task) = cli.agent.as_ref() {
        return run_agent(&cli, task);
    }

    run_game(&cli)
}

/// Run the GPT-2 bash agent loop, printing each step to stdout.
fn run_agent(cli: &Cli, task: &str) -> anyhow::Result<()> {
    let dir = cli.model.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--agent requires --model <dir> (an exported agent model)")
    })?;
    let am = runner::load(dir)?;

    let workdir = cli
        .agent_workdir
        .clone()
        .unwrap_or_else(|| {
            let mut p = std::env::temp_dir();
            p.push(format!("snake-agent-{}", std::process::id()));
            p
        });
    std::fs::create_dir_all(&workdir)?;
    eprintln!("agent workdir: {}", workdir.display());

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let run = runner::run_with_model(
        &am,
        task,
        &workdir,
        cli.agent_max_turns,
        cli.agent_max_new,
        cli.temp as f32,
        &mut rng,
    )?;

    for ev in &run.events {
        match ev {
            agent::AgentEvent::Generate { text, stop } => {
                println!("\n--- model generated ({stop:?}) ---\n{text}");
            }
            agent::AgentEvent::Exec { cmd, result } => {
                println!("\n--- tool exec ---\n$ {cmd}\n{result}");
            }
            agent::AgentEvent::Inject { text } => {
                println!("\n--- injected context ---\n{text}");
            }
        }
    }
    println!("\n=== final transcript ===\n{}\n", run.transcript);
    println!("workdir: {}", workdir.display());
    Ok(())
}

/// Run the closed-loop game agent: the model actually plays Snake.
///
/// Each turn the board is fed to the model, it generates a `<bash>echo X</bash>`
/// action, we parse the UDLR move, step the Game, and feed back the new board.
fn run_agent_game(cli: &Cli) -> anyhow::Result<()> {
    let dir = cli.model.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--play requires --model <dir> (an exported agent model)")
    })?;
    let am = runner::load(dir)?;
    let special = agent::resolve_special_ids(&am.tok)?;
    eprintln!(
        "brain: gpt2-agent (play)  rows={} cols={}  temp={}",
        cli.rows, cli.cols, cli.temp
    );

    // Reuse --steps as the turn cap; default to block_size when unlimited.
    let max_turns = if cli.steps > 0 { cli.steps } else { 256 };
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let run = game_agent::run_game_agent(
        &am,
        &special,
        cli.rows,
        cli.cols,
        max_turns,
        cli.agent_max_new,
        cli.temp as f32,
        &mut rng,
    )?;

    for (i, t) in run.turns.iter().enumerate() {
        let act = match t.action {
            Some(d) => format!("{:?}", d),
            None => "(none)".to_string(),
        };
        let fb = if t.fallback { " [fallback]" } else { "" };
        let eaten = if t.ate { " ATE" } else { "" };
        let crash = if t.collided { " CRASHED" } else { "" };
        let cmd_disp = if t.cmd.trim().is_empty() { "(no bash)" } else { t.cmd.trim() };
        println!(
            "turn {:>3}: {:<16} -> {}{}{}{}",
            i, cmd_disp, act, fb, eaten, crash
        );
    }

    println!();
    println!("=== game over ===");
    println!("stop reason: {:?}", run.stop_reason);
    println!("score: {}", run.score);
    println!("steps survived: {}", run.steps);
    println!("alive at exit: {}", run.alive);
    Ok(())
}

/// Run the terminal Snake game with the selected brain.
fn run_game(cli: &Cli) -> anyhow::Result<()> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let (brain, gpt2_used, brain_label) = select_brain(cli);

    let mut game = Game::new(cli.rows, cli.cols, &mut rng);
    let mut renderer = Renderer::new(&brain_label);

    let tick_ms = if cli.delay_ms > 0 {
        cli.delay_ms
    } else {
        1000 / cli.fps.max(1)
    };
    let tick = Duration::from_millis(tick_ms);

    let mut steps_taken: usize = 0;
    loop {
        if !game.alive
            || (cli.steps > 0 && steps_taken >= cli.steps)
            || renderer.should_quit()
        {
            break;
        }
        let dir = brain.decide(&game, &mut rng);
        let _ = game.step(dir, &mut rng);
        renderer.draw(&game, steps_taken, gpt2_used);
        steps_taken += 1;
        std::thread::sleep(tick);
    }

    drop(renderer);
    println!();
    println!("=== game over ===");
    println!("brain: {} (gpt2 path used: {})", brain.name(), gpt2_used);
    println!("score: {}", game.score);
    println!("steps survived: {}", steps_taken);
    println!("alive at exit: {}", game.alive);
    Ok(())
}

/// Choose the brain per the CLI priority.
fn select_brain(cli: &Cli) -> (Box<dyn Brain>, bool, String) {
    if cli.random {
        log_startup("random", false);
        return (Box::new(RandomBrain), false, "random".to_string());
    }
    if cli.heuristic {
        log_startup("wall-avoider", false);
        return (Box::new(WallAvoider), false, "wall-avoider".to_string());
    }
    if let Some(dir) = &cli.model {
        match gpt2_brain::load(dir) {
            Ok(b) => {
                log_startup("gpt2-small (semantics random — validation wiring)", true);
                (Box::new(b), true, "gpt2-small".to_string())
            }
            Err(e) => {
                eprintln!("[warn] failed to load GPT-2 from {}: {e}", dir.display());
                eprintln!("[warn] falling back to wall-avoider");
                log_startup("wall-avoider (fallback)", false);
                (Box::new(WallAvoider), false, "wall-avoider".to_string())
            }
        }
    } else {
        log_startup("wall-avoider (default)", false);
        (Box::new(WallAvoider), false, "wall-avoider".to_string())
    }
}

fn log_startup(label: &str, gpt2: bool) {
    eprintln!("brain: {label} (gpt2 path used: {gpt2})");
}
