"""
Synthesize SFT data for the GPT-2 agent: two trace families on the 73-token
vocab defined by agent_special_tokens.py.

  1. Snake-playing traces: a heuristic player runs many games; each move
     records  board(state) -> <bash>echo DIR</bash><pause>
              <result>DIR</result><resume>
     and the game ends with <answer>game over score=N</answer>.
  2. Code-writing traces: a complete HTML canvas Snake game is the target,
     emitted via  <bash>cat > snake.py << 'EOF' ... EOF</bash><pause>
                 <result>wrote snake.py</result><resume><answer>done</answer>
     across several natural-language prompt variants.

Outputs into --out_dir:
  train.bin / val.bin            (uint16 token stream)
  train.mask.bin / val.mask.bin  (uint8, 0=ignore / 1=supervise, aligned to
                                  the *target* token = shifted-by-1)
  meta.json                      (vocab_size=73 + special_tokens)

The mask is stored aligned with the token stream (mask[k] supervises token k);
sft_train_gpt2_snake.py shifts it by one when building targets.

Pure functional style: free functions, no classes.
"""
import argparse
import json
import os
import random
from collections import deque

import numpy as np

import agent_special_tokens as ast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}


# ----------------------------------------------------- snake simulator ----

def place_food(rows, cols, occupied, rng):
    empties = [(r, c) for r in range(rows) for c in range(cols)
              if (r, c) not in occupied]
    if not empties:
        return None
    return rng.choice(empties)


def new_game(rows, cols, rng):
    sr, sc = rows // 2, cols // 2
    snake = deque([(sr, sc), (sr, sc - 1), (sr, sc - 2)])
    occupied = set(snake)
    food = place_food(rows, cols, occupied, rng)
    return {
        "rows": rows, "cols": cols, "snake": snake, "occupied": occupied,
        "food": food, "dir": "R", "score": 0, "alive": True,
    }


def heuristic_move(game, rng):
    """Greedy-toward-food + collision-avoiding; never instant-reverse."""
    rows, cols = game["rows"], game["cols"]
    head = game["snake"][0]
    body = set(list(game["snake"])[1:])  # tail may vacate
    food = game["food"]
    rev = OPP[game["dir"]]
    best = None
    for d, (dr, dc) in DIRS.items():
        if d == rev:
            continue
        nr, nc = head[0] + dr, head[1] + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue
        if (nr, nc) in body:
            continue
        dist = abs(nr - food[0]) + abs(nc - food[1]) if food else 0
        key = (dist, rng.random())
        if best is None or key < best[0]:
            best = (key, d)
    if best is None:
        # trapped: pick a non-reverse direction even if fatal
        for d in DIRS:
            if d != rev:
                return d
        return rev
    return best[1]


def step_game(game, d, rng):
    """Apply move d; mutate game in place; return True if still alive."""
    rows, cols = game["rows"], game["cols"]
    dr, dc = DIRS[d]
    game["dir"] = d
    head = game["snake"][0]
    nh = (head[0] + dr, head[1] + dc)
    if nh[0] < 0 or nh[0] >= rows or nh[1] < 0 or nh[1] >= cols:
        game["alive"] = False
        return False
    will_eat = (nh == game["food"])
    # collision check: body excluding tail when not eating (tail vacates)
    body_check = set(list(game["snake"])[1:])
    if not will_eat and game["snake"][-1] in body_check:
        body_check.discard(game["snake"][-1])
    if nh in body_check:
        game["alive"] = False
        return False
    game["snake"].appendleft(nh)
    game["occupied"].add(nh)
    if will_eat:
        game["score"] += 1
        game["food"] = place_food(rows, cols, game["occupied"], rng)
    else:
        tail = game["snake"].pop()
        game["occupied"].discard(tail)
    return True


def render_board(game):
    """ASCII board with # border; H=head o=body F=food .=empty (matches Rust)."""
    rows, cols = game["rows"], game["cols"]
    head = game["snake"][0]
    body = set(list(game["snake"])[1:])
    lines = []
    border = "#" * (cols + 2)
    lines.append(border)
    for r in range(rows):
        row = ["#"]
        for c in range(cols):
            if (r, c) == head:
                row.append("H")
            elif (r, c) == game["food"]:
                row.append("F")
            elif (r, c) in body:
                row.append("o")
            else:
                row.append(".")
        row.append("#")
        lines.append("".join(row))
    lines.append(border)
    return lines


def board_prompt(game):
    hr, hc = game["snake"][0]
    fr, fc = game["food"] if game["food"] else (0, 0)
    lines = render_board(game)
    lines.append(f"HEAD={hr},{hc} FOOD={fr},{fc} DIR={game['dir']} "
                 f"SCORE={game['score']}")
    return "\n".join(lines)


# --------------------------------------------------------- trace build ----

def gen_game_trace(rows, cols, seed, max_steps=60):
    """One full game as (text, mask) segments."""
    rng = random.Random(seed)
    game = new_game(rows, cols, rng)
    segs = [("Task: choose the snake's next move (U/D/L/R) to reach the "
             "food without crashing.\nState:\n", 0)]
    steps = 0
    while game["alive"] and steps < max_steps:
        segs.append((board_prompt(game) + "\nMOVE:\n", 0))
        d = heuristic_move(game, rng)
        segs.append((f"<bash>echo {d}</bash><pause>", 1))
        segs.append((f"<result>{d}</result><resume>", 0))
        step_game(game, d, rng)
        steps += 1
    segs.append((f"<answer>game over score={game['score']}</answer>", 1))
    return segs


SNAKE_HTML = """\
<canvas id="c"></canvas>
<script>
c=document.getElementById("c");x=c.getContext("2d");c.width=c.height=400;
g=20;s=[[10,10],[9,10],[8,10]];f=[5,5];d=[1,0];
function R(){x.fillStyle="#000";x.fillRect(0,0,400,400);x.fillStyle="#f00";x.fillRect(f[0]*g,f[1]*g,g,g);x.fillStyle="#0f0";for(i=0;i<s.length;i++)x.fillRect(s[i][0]*g,s[i][1]*g,g,g)}
function T(){h=[s[0][0]+d[0],s[0][1]+d[1]];if(h[0]<0)return;if(h[1]<0)return;if(h[0]>19)return;if(h[1]>19)return;for(i=0;i<s.length;i++){if(s[i][0]==h[0])if(s[i][1]==h[1])return}s.unshift(h);if(h[0]==f[0]){if(h[1]==f[1]){f=[Math.floor(Math.random()*20),Math.floor(Math.random()*20)]}else{s.pop()}}else{s.pop()}R()}
onkeydown=function(e){k=e.keyCode;if(k==37)d=[-1,0];if(k==39)d=[1,0];if(k==38)d=[0,-1];if(k==40)d=[0,1]};
R();setInterval(T,150);
</script>
"""

CODE_PROMPTS = [
    "Write a Snake game in a single HTML file.",
    "Create a playable snake game using HTML canvas.",
    "Write an HTML file with a snake game.",
    "Implement a snake game in HTML and JavaScript.",
    "Build a browser snake game in one HTML file.",
    "Write snake game code in HTML.",
    "Make a snake game that runs in a browser.",
    "Create an HTML5 canvas snake game.",
]


def gen_code_trace(prompt_text):
    """Emit the snake HTML source as a single bash heredoc command."""
    body = SNAKE_HTML.strip("\n")
    cmd = f"cat > snake.html << 'EOF'\n{body}\nEOF"
    segs = [
        (prompt_text + "\n", 0),
        (f"<bash>{cmd}</bash><pause>", 1),
        ("<result>wrote snake.html</result><resume>", 0),
        ("<answer>done</answer>", 1),
    ]
    return segs


# ------------------------------------------------------------- packing ----

def traces_to_tokens(traces, stoi):
    """Flatten a list of segment-lists into (tokens, mask) lists."""
    all_t, all_m = [], []
    for segs in traces:
        t, m = ast.build_trace(segs, stoi)
        all_t.extend(t)
        all_m.extend(m)
    return all_t, all_m


def split_traces(traces, val_frac, seed):
    rng = random.Random(seed)
    idx = list(range(len(traces)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val = [traces[i] for i in idx[:n_val]]
    train = [traces[i] for i in idx[n_val:]]
    return train, val


def main():
    ap = argparse.ArgumentParser(description="Synthesize agent SFT data.")
    ap.add_argument("--out_dir",
                    default=os.path.join(SCRIPT_DIR, "data-agent"))
    ap.add_argument("--n_games", type=int, default=400)
    ap.add_argument("--n_code", type=int, default=200)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    itos, stoi, vocab_size = ast.build_vocab()

    traces = []
    for i in range(args.n_games):
        rows = random.Random(args.seed + i).choice([6, 7, 8, 9])
        cols = random.Random(args.seed + i + 999).choice([6, 7, 8, 9])
        traces.append(gen_game_trace(rows, cols, args.seed * 7 + i))
    for j in range(args.n_code):
        prompt = CODE_PROMPTS[j % len(CODE_PROMPTS)]
        traces.append(gen_code_trace(prompt))

    train_traces, val_traces = split_traces(traces, args.val_frac, args.seed)
    tr_t, tr_m = traces_to_tokens(train_traces, stoi)
    va_t, va_m = traces_to_tokens(val_traces, stoi)

    np.array(tr_t, dtype=np.uint16).tofile(os.path.join(args.out_dir, "train.bin"))
    np.array(va_t, dtype=np.uint16).tofile(os.path.join(args.out_dir, "val.bin"))
    np.array(tr_m, dtype=np.uint8).tofile(os.path.join(args.out_dir, "train.mask.bin"))
    np.array(va_m, dtype=np.uint8).tofile(os.path.join(args.out_dir, "val.mask.bin"))

    meta = ast.build_meta()
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    sup = int(sum(tr_m))
    print(f"vocab_size={vocab_size}  traces={len(traces)} "
          f"(train {len(train_traces)}/ val {len(val_traces)})")
    print(f"train: {len(tr_t)} tokens ({sup} supervised, "
          f"{100.0*sup/max(1,len(tr_t)):.1f}%)")
    print(f"val:   {len(va_t)} tokens ({int(sum(va_m))} supervised)")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
