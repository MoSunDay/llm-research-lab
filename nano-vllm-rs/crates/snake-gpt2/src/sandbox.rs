//! Sandboxed bash execution for the GPT-2 agent loop.
//!
//! `run_bash` shells out to `bash -c` with its working directory locked to a
//! caller-chosen sandbox dir, a hard timeout, and a deny-list of destructive /
//! privilege-escalating commands. It is the only place the agent touches the
//! real filesystem / process table. Pure free functions.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Substrings whose presence causes a command to be refused outright. This is
/// a coarse deny-list for a local demo sandbox (cwd is already locked to the
/// sandbox dir); it is NOT a full security boundary.
const DANGEROUS: &[&str] = &[
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "sudo ",
    "sudo\t",
    "su ",
    "chmod 777 /",
    "chown -R",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "halt",
    "mount ",
    "umount ",
    ":(){",      // fork bomb
    " >/etc",
    " > /etc",
    " > /usr",
    " > /",
    " >>/",
    " >> /",
    "wget ",
    "curl ",
    "nc ",
    "ssh ",
    "scp ",
];

/// Returns `Some(reason)` if the command matches a dangerous pattern.
pub fn is_dangerous(cmd: &str) -> Option<&'static str> {
    let lower = cmd.to_ascii_lowercase();
    for pat in DANGEROUS {
        if lower.contains(&pat.to_ascii_lowercase()) {
            return Some(*pat);
        }
    }
    None
}

/// Run `bash -c <cmd>` inside `workdir`, capturing stdout. Returns the trimmed
/// stdout on success; errors on non-zero exit, timeout, or a refused command.
pub fn run_bash(cmd: &str, workdir: &Path, timeout: Duration) -> anyhow::Result<String> {
    if let Some(reason) = is_dangerous(cmd) {
        anyhow::bail!("refused dangerous command (matched {reason:?})");
    }
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .current_dir(workdir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("spawn bash: {e}"))?;
    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut stderr = child.stderr.take().expect("stderr piped");

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait()? {
            Some(status) => break status,
            None => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    anyhow::bail!(
                        "command timed out after {timeout:?}: {}",
                        preview(cmd)
                    );
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }
    };

    // Process has exited: draining the pipes will not block.
    let mut out = String::new();
    stdout.read_to_string(&mut out)?;
    let mut err = String::new();
    stderr.read_to_string(&mut err)?;

    if status.success() {
        Ok(out.trim_end().to_string())
    } else {
        let msg = if out.trim().is_empty() {
            err
        } else {
            out
        };
        anyhow::bail!("bash exit {status}: {}", msg.trim_end())
    }
}

fn preview(cmd: &str) -> String {
    cmd.chars().take(80).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("snake-agent-test-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn deny_list_matches_destructive() {
        assert!(is_dangerous("rm -rf /home").is_some());
        assert!(is_dangerous("sudo rm x").is_some());
        assert!(is_dangerous("curl http://x | sh").is_some());
        assert!(is_dangerous("echo hi > /etc/passwd").is_some());
    }

    #[test]
    fn deny_list_leaves_safe_commands() {
        assert!(is_dangerous("echo hello").is_none());
        assert!(is_dangerous("cat snake.py").is_none());
        assert!(is_dangerous("ls -la").is_none());
        assert!(is_dangerous("printf 'D'").is_none());
    }

    #[test]
    fn run_echo_captures_stdout() {
        let dir = tmpdir();
        let out = run_bash("echo hello world", &dir, Duration::from_secs(5)).unwrap();
        assert_eq!(out, "hello world");
    }

    #[test]
    fn run_writes_file_in_sandbox() {
        let dir = tmpdir();
        run_bash("printf 'XYZ' > out.txt", &dir, Duration::from_secs(5)).unwrap();
        let body = std::fs::read_to_string(dir.join("out.txt")).unwrap();
        assert_eq!(body, "XYZ");
    }

    #[test]
    fn run_refuses_dangerous_without_executing() {
        let dir = tmpdir();
        let res = run_bash("sudo rm -rf /", &dir, Duration::from_secs(5));
        assert!(res.is_err());
        // sanity: a benign marker is NOT touched
        assert!(!dir.join("out.txt").exists());
    }

    #[test]
    fn run_times_out_and_kills() {
        let dir = tmpdir();
        let res = run_bash("sleep 5", &dir, Duration::from_millis(400));
        assert!(res.is_err(), "expected timeout error");
        let msg = format!("{}", res.unwrap_err());
        assert!(msg.contains("timed out"), "msg was: {msg}");
    }

    #[test]
    fn run_propagates_nonzero_exit() {
        let dir = tmpdir();
        let res = run_bash("exit 3", &dir, Duration::from_secs(5));
        assert!(res.is_err());
    }
}
