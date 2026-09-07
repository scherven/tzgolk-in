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
SIMS=${SIMS:-800}                  # search budget per SUB-DECISION, not per turn
GAMES=${GAMES:-4000}               # self-play games per generation
WARM_GAMES=${WARM_GAMES:-200000}   # value warm-start, one-ply agent
STEPS=${STEPS:-3000}               # optimiser steps per generation
SIZE=${SIZE:-small}                # small for ~30 gens, then main (a retrain)
KEEP_GB=${KEEP_GB:-200}            # replay retention; 0 disables
ARENA_GAMES=${ARENA_GAMES:-200}    # games per benchmark
ARENA_SIMS=${ARENA_SIMS:-400}      # benchmark search budget (see below)
ARENA_EVERY=${ARENA_EVERY:-3}      # benchmark every Nth generation; 1 = always
CONCURRENCY=${CONCURRENCY:-128}    # self-play games in flight (default 512)

# SIMS is charged per sub-decision (mcts.rs play_turn searches each link of the
# chain), not per turn. At ~2.28 searched sub-decisions per turn, the old 3200
# was ~7300 sims/turn against LEARNING.md 6.4's stated intent of 800 -- and the
# median searched node has 3 edges. The budget had also made record.rs's
# policy_weight() inert: it normalises by ln(801) and clamps at 1.0, so every
# row weighed exactly 1.0. 800 still spends ~1800 sims/turn.
#
# CONCURRENCY was 512 against 14 cores: load average ~396 at 78% utilisation,
# 8.2 GB resident, and a realised batch of 65 against a cap of 256 -- the
# batchers were starved, not saturated. Sweep 96/128/192 before settling.

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
      --concurrency "$CONCURRENCY" \
      2>&1 | tee "$DATA/log/$this-selfplay.log"

  say "$this: train"
  # --init warm-starts from the previous net rather than from scratch.
  $PY train/train.py --replay "$DATA/replay" --out "$DATA/ckpt" --gen "$g" \
      --size "$SIZE" --steps "$STEPS" \
      --init "$(printf '%s/ckpt/gen%04d' "$DATA" $((g - 1)))" \
      2>&1 | tee "$DATA/log/$this-train.log"
  $PY train/export.py --ckpt "$DATA/ckpt/$this" --out "$DATA/ckpt/$this.safetensors"

  # Against the previous generation, not against a fixed baseline: what matters
  # is whether the loop is still climbing. Read the interval, not the point
  # estimate -- an evaluation without one cannot tell 40 from 39.
  #
  # This block gates nothing: the loop promotes $this unconditionally and the
  # arena only tees to a log. At 600 games with full_share=1.0 it was ~29% of
  # loop compute for a number nothing reads, and over-powered by ~4x --
  # arena-progress.jsonl puts the block-level sd of centred score at 4.36, so
  # 600 games buys +/-0.70 pts where 6.8 only asked to resolve 3. Hence 200
  # games, a lower budget, and every ARENA_EVERY-th generation.
  if [ $((g % ARENA_EVERY)) -eq 0 ]; then
    say "$this: benchmark against gen$((g - 1))"
    ./target/release/arena \
        --candidate "mcts:$ARENA_SIMS:$DATA/ckpt/$this.safetensors" \
        --baseline  "mcts:$ARENA_SIMS:$prev" \
        --games "$ARENA_GAMES" --out "$DATA/log/$this-arena.jsonl" \
        2>&1 | tee "$DATA/log/$this-arena.log"
  else
    echo "skip arena for $this (every ${ARENA_EVERY} gens; ARENA_EVERY=1 to force)"
  fi
done

say "done. Watch the newest net play:"
newest=$(ls -1 "$DATA"/ckpt/*.safetensors | sort | tail -1)
echo "  ./target/release/tui -- --agent mcts:$SIMS:$newest"
