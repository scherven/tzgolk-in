#!/usr/bin/env bash
#
# One AlphaZero generation loop: self-play -> train -> export -> benchmark.
#
#   ./run.sh                       # into ./data, defaults below
#   DATA=/Volumes/Big/tz ./run.sh  # onto another drive
#   GENS=40 SIMS=3200 ./run.sh
#
# Resumable: it skips any generation whose checkpoint already exists, so
# Ctrl-C and re-run picks up where it stopped. Every stage is separately
# resumable too -- self-play seals shards as it goes, and `train.py --resume`
# restores the optimiser moments (without them Adam silently restarts and the
# generation is wasted).
set -euo pipefail
cd "$(dirname "$0")"

DATA=${DATA:-data}                 # replay shards, checkpoints, logs
GENS=${GENS:-30}                   # generations to run
SIMS=${SIMS:-3200}                 # search budget per move
GAMES=${GAMES:-4000}               # self-play games per generation
WARM_GAMES=${WARM_GAMES:-200000}   # value warm-start, one-ply agent
STEPS=${STEPS:-3000}               # optimiser steps per generation
SIZE=${SIZE:-small}                # small for ~30 gens, then main (a retrain)
KEEP_GB=${KEEP_GB:-200}            # replay retention; 0 disables
ARENA_GAMES=${ARENA_GAMES:-600}    # games per benchmark

PY=train/.venv/bin/python3
[ -x "$PY" ] || { echo "no venv: python3 -m venv train/.venv && train/.venv/bin/pip install -r train/requirements.txt"; exit 1; }

# Accelerate spawns its own pool and fights the batcher threads for cores.
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "$DATA"/{replay,ckpt,log}
cargo build --release

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- warm start
#
# Value-only records from the one-ply agent. These carry no policy targets --
# their move indices are not regenerable -- but positions paired with the final
# scores of a plausible continuation are exactly what the value head needs, and
# the agent generates them at over a thousand games a second.
if [ ! -f "$DATA/ckpt/gen0001.safetensors" ]; then
  say "warm start: $WARM_GAMES games, one-ply agent"
  ./target/release/selfplay --agent heuristic:32 --games "$WARM_GAMES" \
      --out "$DATA/replay" --gen 0 --keep-gb "$KEEP_GB" \
      2>&1 | tee "$DATA/log/gen0000-selfplay.log"

  say "train gen 1"
  $PY train/train.py --replay "$DATA/replay" --out "$DATA/ckpt" --gen 1 \
      --size "$SIZE" --steps "$STEPS" 2>&1 | tee "$DATA/log/gen0001-train.log"
  $PY train/export.py --ckpt "$DATA/ckpt/gen0001" --out "$DATA/ckpt/gen0001.safetensors"
fi

# ---------------------------------------------------------------- generations
for g in $(seq 2 "$GENS"); do
  prev=$(printf '%s/ckpt/gen%04d.safetensors' "$DATA" $((g - 1)))
  this=$(printf 'gen%04d' "$g")
  [ -f "$DATA/ckpt/$this.safetensors" ] && { echo "skip $this (exists)"; continue; }

  say "$this: self-play, $GAMES games at $SIMS sims"
  # Searching with the previous net is what produces *policy* targets. The
  # warm-start data has none, so until this runs the policy head is untrained.
  ./target/release/selfplay --agent "mcts:$SIMS:$prev" --games "$GAMES" \
      --out "$DATA/replay" --gen "$g" --keep-gb "$KEEP_GB" \
      2>&1 | tee "$DATA/log/$this-selfplay.log"

  say "$this: train"
  # --init warm-starts from the previous net rather than from scratch.
  $PY train/train.py --replay "$DATA/replay" --out "$DATA/ckpt" --gen "$g" \
      --size "$SIZE" --steps "$STEPS" \
      --init "$(printf '%s/ckpt/gen%04d' "$DATA" $((g - 1)))" \
      2>&1 | tee "$DATA/log/$this-train.log"
  $PY train/export.py --ckpt "$DATA/ckpt/$this" --out "$DATA/ckpt/$this.safetensors"

  say "$this: benchmark against gen$((g - 1))"
  # Against the previous generation, not against a fixed baseline: what matters
  # is whether the loop is still climbing. Read the interval, not the point
  # estimate -- an evaluation without one cannot tell 40 from 39.
  ./target/release/arena \
      --candidate "mcts:$SIMS:$DATA/ckpt/$this.safetensors" \
      --baseline  "mcts:$SIMS:$prev" \
      --games "$ARENA_GAMES" --out "$DATA/log/$this-arena.jsonl" \
      2>&1 | tee "$DATA/log/$this-arena.log"
done

say "done. Watch the newest net play:"
newest=$(ls -1 "$DATA"/ckpt/*.safetensors | sort | tail -1)
echo "  ./target/release/tui -- --agent mcts:$SIMS:$newest"
