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

from features import PHASE_ARITY, batch_arrays, edge_slots, encode_batch, value_targets
from replay import Buffer

FAILED = 0


def check(name: str, ok: bool, detail: str = ""):
    global FAILED
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILED += 1


def test_policy_targets(buf: Buffer):
    """Fabricate tree-backed records and check the distribution that comes out.

    The fabrication has to sit on top of *real* records now. Since the loader
    stopped assuming edge index == head slot and started asking
    ``tzolkin::encode::edges`` for the map, a record whose stored ``n_edges``
    does not match what the engine re-derives is reported as unusable rather
    than mapped wrongly -- which is the whole point of the change and also
    means a `phase_tag`/`n_edges` pair invented out of thin air is correctly
    refused. So: pick real ``Placing`` nodes that really have 7 edges, then
    fabricate only the visit counts.
    """
    big = buf.sample(4096, np.random.default_rng(1))
    want = np.where((big["phase_tag"] == 2) & (big["n_edges"] == 7))[0]
    check("real full-width Placing nodes exist", len(want) >= 8, f"{len(want)} found")
    if len(want) < 8:
        return
    want = want[:64]
    batch = {k: v[want].copy() for k, v in big.items()}
    n = len(batch["state"])
    batch["policy_kind"][:] = 1
    batch["visits"][:] = 0
    batch["visits"][:, 0] = (3, 600)
    batch["visits"][:, 1] = (6, 200)
    # A visit pair past the stored edge count would be a tree bug. It used to be
    # *clamped* onto the last slot, which silently credits a move that was never
    # visited; it is dropped now.
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
    # All 7 edges are legal here, so the map is the identity and edge 3 is
    # slot 3. 600 of the 800 counts that survive, the out-of-range 100 dropped.
    check("visit mass at edge 3", np.allclose(dist[:, 3], 0.75, atol=1e-3), f"{dist[0, 3]:.4f}")
    check("out-of-range index dropped", np.allclose(dist[:, 6], 0.25, atol=1e-3),
          f"{dist[0, 6]:.4f}")
    check("mask covers exactly n_edges", mask.sum(1).min() == 7 and mask.shape[1] == 7)
    check("policy weight carried through", np.allclose(weight, 0.5))
    check("no mass outside the mask", float(dist[~mask].sum()) == 0.0)

    # Only the mover's own perspective trains the policy: the other three
    # augmented rows cannot even identify whose move is being predicted.
    _, ta = batch_arrays(batch, buf.schema, augment=True)
    check("augmented rows are 1-in-4", len(ta[key][0]) == n, f"{len(ta[key][0])} of {4 * n}")
    movers_of_sel = np.repeat(batch["mover"], 4)[ta[key][0]]
    picked = np.tile(np.arange(4), n)[ta[key][0]]
    check("and it is the mover's row", bool((movers_of_sel == picked).all()))

    # A narrower node must mask the edges it does not have. Real 3-edge nodes,
    # so the engine agrees with the count.
    narrow = np.where((big["phase_tag"] == 2) & (big["n_edges"] == 3))[0][:32]
    if len(narrow):
        b2 = {k: v[narrow].copy() for k, v in big.items()}
        b2["policy_kind"][:] = 1
        b2["visits"][:] = 0
        b2["visits"][:, 0] = (0, 10)
        _, t2 = batch_arrays(b2, buf.schema, augment=False)
        check("narrow node masks the tail", int(t2[key][3].sum(1)[0]) == 3,
              f"{int(t2[key][3].sum(1)[0])}")


def test_slot_map_is_not_the_identity(buf: Buffer):
    """The bug this file exists to keep out.

    ``features.py`` used to build the fixed-arity targets from ``n_edges``
    alone, which is only right when a head's legal edges are a *prefix* of its
    slots. They are not. Measured on held-out champion records: ``Beg`` 37.7% of
    nodes mismatched, ``Placing`` 32.3%, ``PickWorker`` 100%. If this test ever
    reports 0% for every phase, either the engine changed or the probe broke --
    it must not be read as "the assumption is safe again".
    """
    b = buf.sample(4096, np.random.default_rng(7))
    ok = b["policy_kind"] != 0
    if not ok.any():
        check("slot map probed", False, "no TREE_EDGE records in this buffer")
        return
    idx = np.where(ok)[0]
    ne = b["n_edges"][idx].astype(np.int64)
    kinds, arity, slots = edge_slots(
        b["state"][idx], b["mover"][idx], b["phase_tag"][idx],
        b["phase_args"][idx], ne, int(max(1, ne.max())))
    fixed = kinds == 1
    check("edge_slots reaches the engine", fixed.sum() > 0, f"{int(fixed.sum())} fixed rows")
    if not fixed.any():
        return
    any_bad = False
    for tag, (name, _) in PHASE_ARITY.items():
        rows = np.where(fixed & (b["phase_tag"][idx] == tag))[0]
        if len(rows) == 0:
            continue
        e = np.arange(slots.shape[1])[None, :]
        live = e < ne[rows][:, None]
        bad = ((slots[rows].astype(np.int64) != e) & live).any(1)
        frac = float(bad.mean())
        any_bad |= frac > 0
        print(f"    tag {tag:>2} {name:<10} {len(rows):>5} nodes, {100 * frac:5.1f}% not the identity")
    check("some phase maps edges off the diagonal", any_bad,
          "if this passes as False the targets could be built from n_edges again")

    # And a record marked NONE must contribute no policy target at all.
    none = {k: v[:32].copy() for k, v in b.items()}
    none["policy_kind"][:] = 0
    _, t3 = batch_arrays(none, buf.schema, augment=False)
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
    test_slot_map_is_not_the_identity(buf)
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
