#!/usr/bin/env python3
"""Run one fixed four-agent lineup many times and report every seat's score.

Seats 0 and 1 (R, G) get the research-tilted champion; seats 2 and 3 (B, Y)
get the champion itself.  The heavy lifting is the Rust binary
``target/release/seatmatch``; this script owns the lineup, the bookkeeping and
the statistics.

The games file is the state.  Every completed game is appended and flushed by
the Rust side as it lands, so:

  * Ctrl-C stops the run cleanly and still reports on everything finished.
  * Re-running the same command resumes -- already-played seeds are skipped.
  * Killing the machine loses at most the games in flight.

Nothing is held only in memory.

    tools/seatmatch.py --games 10000
    tools/seatmatch.py --games 10000 --sims 512     # a version that finishes today
    tools/seatmatch.py --report-only                # summarise what is on disk

**On reading the numbers.**  The lineup is fixed, so the four seat averages
are not four measurements of two agents: seat 0 moves first, and the corn
surcharge on higher spaces means seat 0 is not seat 3.  Two copies of one agent
in seats 0 and 1 do not score what two copies score in seats 2 and 3, whoever
they are.  ``--mirror`` plays the reflected lineup as well and subtracts that
away; without it, treat "R+G against B+Y" as suggestive and the four per-seat
numbers as the answer to the question actually asked.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SEATS = ("R", "G", "B", "Y")
REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "target" / "release" / "seatmatch"

# Measured on this machine, `docs/FINDINGS-track-shape.md` §7: 122.6 s user CPU
# per four-game block at 8,192 simulations with the `deeper` preset.  Cost is
# very close to linear in simulations over this range.
CPU_SECONDS_PER_GAME_AT_8192 = 122.6 / 4


def cost_estimate(games: int, sims: int, cores: float) -> tuple[float, float]:
    """(CPU-hours, wall-hours) for `games` games, given usable `cores`."""
    cpu = games * CPU_SECONDS_PER_GAME_AT_8192 * (sims / 8192)
    return cpu / 3600, cpu / max(cores, 0.1) / 3600


# ---------------------------------------------------------------- statistics


@dataclass
class Stat:
    n: int
    mean: float
    sd: float
    ci: float          # half-width of the 95% interval on the mean

    @classmethod
    def of(cls, xs: list[float]) -> "Stat":
        n = len(xs)
        if n == 0:
            return cls(0, float("nan"), float("nan"), float("nan"))
        mean = sum(xs) / n
        if n == 1:
            return cls(1, mean, float("nan"), float("nan"))
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        sd = math.sqrt(var)
        return cls(n, mean, sd, 1.96 * sd / math.sqrt(n))


def read_games(path: Path) -> list[dict]:
    """Every complete line of the games file.  A half-written final line -- the
    process was killed mid-flush -- is dropped rather than crashing the run."""
    if not path.exists():
        return []
    out, bad = [], 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if "scores" in g and len(g["scores"]) == 4:
                out.append(g)
    if bad:
        print(f"  note: skipped {bad} unparsable line(s) in {path.name}", file=sys.stderr)
    return out


# -------------------------------------------------------------------- report


def report(games: list[dict], lineup: list[str], mirror: list[dict] | None) -> None:
    if not games:
        print("no games on file yet")
        return

    n = len(games)
    aborted = sum(1 for g in games if g.get("aborted"))
    print()
    print(f"  {n} games" + (f"   ({aborted} aborted -- a rules bug)" if aborted else ""))
    print()
    print("  seat        agent                                   mean     95% CI"
          "          sd    win rate")
    print("  " + "-" * 96)

    by_seat = [[float(g["scores"][s]) for g in games] for s in range(4)]
    wins = [[float(g["win"][s]) for g in games] for s in range(4)]
    for s in range(4):
        st, w = Stat.of(by_seat[s]), Stat.of(wins[s])
        name = lineup[s]
        short = name if len(name) <= 36 else name[:33] + "..."
        print(f"  {SEATS[s]:<4} {short:<40} {st.mean:8.2f}  "
              f"({st.mean - st.ci:+7.2f},{st.mean + st.ci:+7.2f})  {st.sd:6.2f}"
              f"      {w.mean:.3f}")
    print("  " + "-" * 96)

    # Which seats hold which agent, by name, so the grouping follows the lineup
    # rather than assuming 0/1 against 2/3.
    groups: dict[str, list[int]] = {}
    for s, name in enumerate(lineup):
        groups.setdefault(name, []).append(s)

    if len(groups) == 2:
        (na, sa), (nb, sb) = groups.items()
        # Paired within a game: the same deal, the same opponents, so the
        # per-game difference removes the variance the deal contributes.
        diff = [
            sum(g["scores"][s] for s in sa) / len(sa)
            - sum(g["scores"][s] for s in sb) / len(sb)
            for g in games
        ]
        d = Stat.of(diff)
        print()
        print(f"  paired, per game: seats {''.join(SEATS[s] for s in sa)} minus "
              f"seats {''.join(SEATS[s] for s in sb)}")
        print(f"      {d.mean:+.2f}  95% CI ({d.mean - d.ci:+.2f}, {d.mean + d.ci:+.2f})"
              f"   sd {d.sd:.2f}   n {d.n}")
        verdict = ("clear of zero" if abs(d.mean) > d.ci else
                   "NOT distinguishable from zero")
        print(f"      {verdict}")
        print()
        print("  This difference is seat advantage plus agent difference, and the two")
        print("  are not separated here: seat 0 moves first. Run --mirror to subtract it.")

        if mirror:
            mdiff = [
                sum(g["scores"][s] for s in sb) / len(sb)
                - sum(g["scores"][s] for s in sa) / len(sa)
                for g in mirror
            ]
            m = Stat.of(mdiff)
            # Same agent, the other pair of seats: averaging the two removes the
            # seat term and leaves the agent term.
            both = Stat.of(diff + mdiff)
            print()
            print(f"  mirrored lineup, {m.n} games: {m.mean:+.2f} "
                  f"({m.mean - m.ci:+.2f}, {m.mean + m.ci:+.2f})")
            print(f"  seat-corrected agent effect: {both.mean:+.2f} "
                  f"({both.mean - both.ci:+.2f}, {both.mean + both.ci:+.2f})   "
                  f"n {both.n}")
            print("  Half the seat term cancels in the average of the two lineups.")
    print()


# ---------------------------------------------------------------------- main


def run(cmd: list[str], games_path: Path, want: int, lineup: list[str]) -> int:
    """Run the worker, printing progress.  Ctrl-C asks it to stop and waits."""
    print("  " + " ".join(repr(c) if " " in c else c for c in cmd))
    print()
    proc = subprocess.Popen(cmd, cwd=REPO)

    stopping = False

    def on_sigint(_sig, _frm):
        nonlocal stopping
        if stopping:                      # a second Ctrl-C means business
            proc.kill()
            return
        stopping = True
        print("\n  stopping -- letting games in flight finish and flush; "
              "Ctrl-C again to kill", flush=True)
        proc.send_signal(signal.SIGINT)   # the Rust side installs a handler

    signal.signal(signal.SIGINT, on_sigint)

    started = time.time()
    try:
        while proc.poll() is None:
            time.sleep(5)
            if stopping:
                continue
            done = len(read_games(games_path))
            elapsed = time.time() - started
            if done == 0:
                # Say so rather than showing nothing: a run with more games in
                # flight than games to play finishes them all at once, and
                # silence there is indistinguishable from a hang.
                print(f"\r  0/{want} games   {elapsed / 60:.1f} min elapsed, "
                      f"none finished yet    ", end="", flush=True)
            else:
                rate = done / elapsed
                left = (want - done) / rate
                print(f"\r  {done}/{want} games   {rate:.2f} games/s   "
                      f"eta {left / 3600:5.1f} h    ", end="", flush=True)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    print()
    return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="R and G play the research-tilted champion; B and Y play the champion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--games", type=int, default=10_000)
    ap.add_argument("--sims", type=int, default=8192,
                    help="MCTS simulations per turn for both agents [8192]")
    ap.add_argument("--tilt", default="agri", choices=("agri", "causal"),
                    help="per-track weights for the tilted seats [agri]")
    ap.add_argument("--tiltk", type=float, default=12.0, help="tilt strength [12]")
    ap.add_argument("--seed", type=int, default=3_000_000)
    ap.add_argument("--concurrency", type=int, default=0, help="games in flight [auto]")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "seatmatch",
                    help="directory for the games files")
    ap.add_argument("--mirror", action="store_true",
                    help="also play the reflected lineup, to subtract seat advantage")
    ap.add_argument("--report-only", action="store_true",
                    help="summarise what is already on disk and exit")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing games instead of resuming")
    args = ap.parse_args()

    champ = f"mcts:{args.sims}:heuristic:deeper"
    tilted = f"{champ},tilt={args.tilt},tiltk={args.tiltk:g}"
    lineup = [tilted, tilted, champ, champ]

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"s{args.sims}-{args.tilt}{args.tiltk:g}"
    main_path = args.out / f"{tag}.jsonl"
    mirror_path = args.out / f"{tag}-mirror.jsonl"

    if args.report_only:
        report(read_games(main_path), lineup,
               read_games(mirror_path) if mirror_path.exists() else None)
        return 0

    if not BIN.exists():
        print(f"missing {BIN}\n  cargo build --release --bin seatmatch", file=sys.stderr)
        return 2

    cores = os.cpu_count() or 4
    cpu_h, wall_h = cost_estimate(args.games, args.sims, cores * 0.8)
    done = 0 if args.fresh else len(read_games(main_path))

    print()
    print(f"  seats R,G : {tilted}")
    print(f"  seats B,Y : {champ}")
    print(f"  games     : {args.games}   ({done} already on file)")
    print(f"  cost      : ~{cpu_h:.0f} CPU-hours, ~{wall_h:.1f} h wall on "
          f"{cores} cores if idle")
    if args.mirror:
        print(f"  mirror    : the reflected lineup as well -- double the above")
    print(f"  games file: {main_path}")
    print()
    if wall_h > 4:
        print(f"  That is a long run. It resumes: stop it with Ctrl-C and start the")
        print(f"  same command again, or use --sims 512 for a same-day answer.")
        print()

    def worker(path: Path, seats: list[str], seed: int) -> list[str]:
        cmd = [str(BIN)]
        for i, spec in enumerate(seats):
            cmd += ["--seat", str(i), spec]
        cmd += ["--games", str(args.games), "--seed", str(seed),
                "--out", str(path), "--quiet"]
        if args.concurrency:
            cmd += ["--concurrency", str(args.concurrency)]
        if not args.fresh:
            cmd += ["--resume"]
        return cmd

    rc = run(worker(main_path, lineup, args.seed), main_path, args.games, lineup)
    if rc != 0:
        print(f"  worker exited {rc}", file=sys.stderr)

    if args.mirror:
        mlineup = [champ, champ, tilted, tilted]
        print("\n  mirrored lineup: seats B,Y tilted\n")
        run(worker(mirror_path, mlineup, args.seed + 10_000_000),
            mirror_path, args.games, mlineup)

    report(read_games(main_path), lineup,
           read_games(mirror_path) if args.mirror else None)
    print(f"  games are in {main_path}; re-run the same command to add more.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
