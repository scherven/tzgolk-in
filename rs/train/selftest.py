"""Checks that do not need torch, a GPU, or a trained network.

    python train/selftest.py replay

Three things are worth pinning before a long run:

1. **The policy-target path**, which no real record exercises yet. Until MCTS
   lands every record is ``policy_kind = NONE``, so the code that turns visit
   pairs into a normalised, masked distribution is entirely untested by the data
   on disk. It is fabricated here instead.
2. **Perspective relativity**, the contract that makes one trunk answer for all
   four seats and makes the 4x value augmentation legitimate rather than a lie.
3. **The two copies of the phase table** in ``features.py`` and ``model.py``
   agreeing. They are separate so ``features`` imports without torch; that is a
   reasonable trade only if something checks them.
"""

from __future__ import annotations

import sys

import numpy as np

from features import PHASE_ARITY, batch_arrays, encode_batch, value_targets
from replay import Buffer

FAILED = 0


def check(name: str, ok: bool, detail: str = ""):
    global FAILED
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILED += 1


def test_policy_targets(buf: Buffer):
    """Fabricate tree-backed records and check the distribution that comes out."""
    batch = {k: v.copy() for k, v in buf.sample(64, np.random.default_rng(1)).items()}
    n = len(batch["state"])
    # `Placing` nodes: 7 edges (5 gears + first player + stop).
    batch["policy_kind"][:] = 1
    batch["phase_tag"][:] = 2
    batch["n_edges"][:] = 7
    batch["visits"][:] = 0
    batch["visits"][:, 0] = (3, 600)
    batch["visits"][:, 1] = (6, 200)
    # A visit pair pointing past the head's arity would be a tree bug; the
    # loader clamps rather than crashing, and this pins that it does.
    batch["visits"][:, 2] = (99, 100)
    batch["policy_weight"][:] = 0.5

    _, t = batch_arrays(batch, buf.schema, augment=False)
    key = "policy_" + PHASE_ARITY[2][0]
    check(key + " present", key in t)
    if key not in t:
        return
    sel, dist, weight, mask = t[key]
    check("all rows selected", len(sel) == n, f"{len(sel)} of {n}")
    check("distribution sums to 1", np.allclose(dist.sum(1), 1.0))
    check("visit mass at edge 3", np.allclose(dist[:, 3], 0.6667, atol=1e-3), f"{dist[0, 3]:.4f}")
    check("out-of-range index clamped", np.allclose(dist[:, 6], 0.3333, atol=1e-3))
    check("mask covers exactly n_edges", mask.sum(1).min() == 7 and mask.shape[1] == 7)
    check("policy weight carried through", np.allclose(weight, 0.5))

    # A narrower node must mask the edges it does not have.
    batch["n_edges"][:] = 3
    _, t2 = batch_arrays(batch, buf.schema, augment=False)
    check("narrow node masks the tail", int(t2[key][3].sum(1)[0]) == 3)

    # And a record marked NONE must contribute no policy target at all.
    batch["policy_kind"][:] = 0
    _, t3 = batch_arrays(batch, buf.schema, augment=False)
    check("policy_kind NONE yields no target", not any(k.startswith("policy_") for k in t3))


def test_perspective(buf: Buffer):
    """Encoding the same record from seat p must put p's own holdings in slot 0.

    Checked through the value targets rather than the raw features, because it
    is the targets that would silently teach the net a rotated world.
    """
    batch = buf.sample(32, np.random.default_rng(2))
    seats = [
        value_targets(batch, np.full(len(batch["state"]), p, np.uint8))["z_rel"]
        for p in range(4)
    ]
    raw = batch["z_rel"]
    ok = all(np.allclose(seats[p][:, 0], raw[:, p]) for p in range(4))
    check("z_rel slot 0 is the querying player", ok)

    # And the encoding itself must differ between seats, or the rotation is not
    # actually happening.
    xs = [
        encode_batch(
            batch["state"],
            np.full(len(batch["state"]), p, np.uint8),
            batch["phase_tag"],
            batch["phase_args"],
            buf.schema,
        )
        for p in range(4)
    ]
    differ = sum(1 for p in range(1, 4) if not np.allclose(xs[0], xs[p]))
    check("encoding rotates with the querying seat", differ == 3, f"{differ}/3 differ")


def test_no_hidden_information(buf: Buffer):
    """The undrawn deck must not reach the network.

    ``LEARNING.md`` §1.6: ``Deck::ids`` is the shuffled draw pile, and a network
    fed it learns to read the future. The replay format stores it deliberately
    (the buffer should outlive encoder changes); reading it in the encoder is
    the bug. Shuffle the undrawn tail of every deck and check nothing moves.
    """
    batch = {k: v.copy() for k, v in buf.sample(32, np.random.default_rng(3)).items()}
    args = (batch["mover"], batch["phase_tag"], batch["phase_args"], buf.schema)
    before = encode_batch(batch["state"], *args)

    f = {x["name"]: x for x in buf.schema["state_fields"]}
    rng = np.random.default_rng(4)
    for ids, cursor in (("age1_ids", "age1_next"), ("age2_ids", "age2_next"),
                        ("monument_ids", "monument_next")):
        o, c = f[ids]["offset"], f[ids]["count"]
        nxt = batch["state"][:, f[cursor]["offset"]].astype(int)
        for i in range(len(batch["state"])):
            tail = batch["state"][i, o + nxt[i] : o + c]
            batch["state"][i, o + nxt[i] : o + c] = rng.permutation(tail)

    after = encode_batch(batch["state"], *args)
    check("undrawn deck order is invisible", np.array_equal(before, after))


def test_phase_tables_agree():
    try:
        from model import PHASE_HEADS
    except ImportError:
        print("  SKIP  phase tables agree -- torch not installed")
        return
    check("features and model phase tables agree", PHASE_ARITY == PHASE_HEADS,
          f"{PHASE_ARITY} vs {PHASE_HEADS}")


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "replay"
    buf = Buffer.open(directory)
    print(buf.summary())
    print("\nchecks")
    test_policy_targets(buf)
    test_perspective(buf)
    test_no_hidden_information(buf)
    test_phase_tables_agree()
    print()
    if FAILED:
        print(f"{FAILED} check(s) failed")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
