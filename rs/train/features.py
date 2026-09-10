"""Encode replay states into network inputs.

=======================================================================
THIS FILE IS THE ENCODER SEAM. IT IS A PLACEHOLDER.
=======================================================================

``src/encode.rs`` is owned by another agent and is where the real encoder
lives: ``LEARNING.md`` §1 specifies a 3,072-wide flat vector -- 600 board, 1,128
player, 956 global, 388 reserved -- and §5.1 specifies that training batches are
produced by a Rust ``batcher`` binary calling **the same** encoder the search
uses, because "the encoder used to train is bit-identical to the encoder used to
play" is the single correctness property most worth having.

What is below is a much smaller hand-written encoder over the fields the replay
format exposes. Its job is to let ``train.py`` run end to end today, against
records that exist today, so the training loop, the checkpointing, the loss and
the schedule are all debugged before the real encoder arrives. Nothing about the
training script depends on which encoder is in use except the value of ``D_IN``.

To switch over, replace ``encode_batch`` with either:

  * a subprocess call to the Rust batcher, reading pre-encoded ``float32``
    shards, or
  * a ctypes/PyO3 binding to ``tzolkin::encode::encode``.

Both keep this signature::

    encode_batch(raw_states, movers, phase_tags, phase_args, schema) -> [N, D_IN]

and both must honour the two contracts the real encoder honours, which the
placeholder below also honours:

  1. **Perspective relativity.** Everything per-player is rotated by
     ``(owner - p) mod 4``: slot 0 is always the querying player. This is what
     makes one trunk answer for all four seats and what makes the 4x value
     augmentation legitimate rather than a lie (``LEARNING.md`` §1.5, §6.4).
  2. **No hidden information.** ``Deck.ids`` is never read. The cursors and the
     publicly-derivable *unseen* set are (``LEARNING.md`` §1.6). A network fed
     the raw deck learns to read the future and does not transfer to play
     against humans. The replay format stores ``Deck.ids`` because the buffer is
     meant to outlive encoder changes; reading it here would be a bug.
"""

from __future__ import annotations

import os

import numpy as np

from arch import D_CHOICE, POINTER_TAGS, PHASE_HEADS

from replay import State

N_PLAYERS = 4

# The dense absolute score target (`LEARNING.md` §3.2), as
# `clip((score - CENTRE) / SCALE, -2, 2)`.
#
# These have to span the *whole trajectory*, from warm-start play to whatever
# the trained agent reaches, because a value head is only comparable across
# generations if its target means the same thing in each. Centring on each
# generation's own mean would destroy that.
#
# The original 75/50 was calibrated for competent play (scores 50-150) and
# clipped **31% of warm-start targets to exactly -2.0**, leaving the rest with
# a standard deviation of 0.21 -- so the head learned to predict a constant and
# its loss fell to 0.004 while contributing no gradient at all. Measured on
# 20.5M warm-start records: raw scores ran -80..45.
#
# 35/70 spans -105..+175 uncllipped, which covers both ends.
SCORE_CENTRE = 35.0
SCORE_SCALE = 70.0

# ----------------------------------------------------------------------
# the encoder
# ----------------------------------------------------------------------
#
# This used to be a numpy reimplementation of `src/encode.rs`. Two independent
# encoders is a train/inference skew waiting to happen, and an invisible one:
# nothing crashes, the network simply learns against features that differ from
# the ones it is served at play time. So it now calls the real one over the C
# ABI in `src/ffi.rs`.
#
# A precomputed feature file would not do, because of the 4x perspective
# augmentation in `batch_arrays`: every position is presented once per querying
# player, so the encoder has to be callable with any `p`, not just the record's
# mover.

_LIB = None
_D_IN = None
_SLOT = None

_BUILD_HINT = (
    "the Rust encoder is not built. Run:\n"
    "    cargo build --release\n"
    "from the `rs/` directory; it produces target/release/libtzolkin.{dylib,so}."
)


def _lib():
    """Load the cdylib, once."""
    global _LIB, _D_IN, _SLOT
    if _LIB is not None:
        return _LIB

    import ctypes

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    names = ["libtzolkin.dylib", "libtzolkin.so", "tzolkin.dll"]
    cands = [os.path.join(root, "target", p, n) for p in ("release", "debug") for n in names]
    cands += [os.environ[k] for k in ("TZOLKIN_LIB",) if k in os.environ]

    path = next((c for c in cands if os.path.exists(c)), None)
    if path is None:
        raise RuntimeError(_BUILD_HINT)

    lib = ctypes.CDLL(path)
    lib.tzolkin_d_in.restype = ctypes.c_uint32
    lib.tzolkin_state_slot.restype = ctypes.c_uint32
    lib.tzolkin_encode_batch.restype = ctypes.c_int32
    if hasattr(lib, "tzolkin_edge_slots"):
        lib.tzolkin_edge_slots.restype = ctypes.c_int32
    if hasattr(lib, "tzolkin_edge_choices"):
        lib.tzolkin_edge_choices.restype = ctypes.c_int32
    _LIB, _D_IN, _SLOT = lib, int(lib.tzolkin_d_in()), int(lib.tzolkin_state_slot())
    return lib


def _const(which: str) -> int:
    _lib()
    return _D_IN if which == "d_in" else _SLOT


def __getattr__(name):
    """`D_IN` on demand.

    Importing this module must not require a built cdylib -- the target-building
    half is useful without one -- so the width is fetched on first reference
    rather than at import.
    """
    if name == "D_IN":
        return _const("d_in")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def d_in() -> int:
    """Input width, from the Rust encoder."""
    return _const("d_in")


def encode_batch(
    states: np.ndarray,
    movers: np.ndarray,
    phase_tags: np.ndarray,
    phase_args: np.ndarray,
    schema: dict | None = None,
) -> np.ndarray:
    """Encode `n` records to `[n, D_IN]` float32, from each mover's perspective.

    `schema` is accepted and ignored: the layout is now the Rust encoder's
    business, and the schema no longer describes it.
    """
    lib = _lib()
    import ctypes

    n = len(states)
    if n == 0:
        return np.zeros((0, _D_IN), np.float32)

    states = np.ascontiguousarray(states, dtype=np.uint8).reshape(n, -1)
    if states.shape[1] != _SLOT:
        raise ValueError(f"state slot is {states.shape[1]} bytes, encoder wants {_SLOT}")

    movers = np.ascontiguousarray(movers, dtype=np.uint8)
    tags = np.ascontiguousarray(phase_tags, dtype=np.uint8)
    args = np.ascontiguousarray(phase_args, dtype=np.uint8).reshape(n, 5)
    out = np.zeros((n, _D_IN), np.float32)

    u8 = ctypes.POINTER(ctypes.c_uint8)
    rc = lib.tzolkin_encode_batch(
        states.ctypes.data_as(u8),
        movers.ctypes.data_as(u8),
        tags.ctypes.data_as(u8),
        args.ctypes.data_as(u8),
        ctypes.c_uint32(n),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(out.size),
    )
    if rc != 0:
        raise RuntimeError(f"tzolkin_encode_batch failed with {rc}")
    return out


def edge_slots(
    states: np.ndarray,
    movers: np.ndarray,
    phase_tags: np.ndarray,
    phase_args: np.ndarray,
    n_edges: np.ndarray,
    max_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge index -> head slot, from ``tzolkin::encode::edges``.

    Returns ``(kinds, arity, slots)``: ``kinds`` is 0 unusable / 1 fixed-arity /
    2 pointer, ``arity`` the head width, and ``slots[i, e]`` the head slot edge
    ``e`` of row ``i`` maps to.

    **This map is not the identity.** It was assumed to be, and that assumption
    put each visit count on the wrong move for 37.7% of ``Beg`` nodes, 32.3% of
    ``Placing`` and *every* ``PickWorker`` node: a head's legal edges are a
    subset of its slots, not a prefix of them. Nothing downstream can detect a
    scrambled target -- the loss goes down, the head learns to rank the wrong
    move -- so the mapping comes from the same Rust call the network's own prior
    path makes at play time (`src/ffi.rs`), never from a second copy of the rule.
    """
    lib = _lib()
    import ctypes

    n = len(states)
    if n == 0 or max_edges <= 0:
        return (
            np.zeros(0, np.uint8),
            np.zeros(0, np.uint16),
            np.zeros((0, max(max_edges, 1)), np.uint16),
        )
    if not hasattr(lib, "tzolkin_edge_slots"):
        raise RuntimeError(
            "the loaded libtzolkin has no tzolkin_edge_slots; rebuild with "
            "`cargo build --release` so the policy targets can be mapped"
        )

    states = np.ascontiguousarray(states, dtype=np.uint8).reshape(n, -1)
    if states.shape[1] != _SLOT:
        raise ValueError(f"state slot is {states.shape[1]} bytes, encoder wants {_SLOT}")
    movers = np.ascontiguousarray(movers, dtype=np.uint8)
    tags = np.ascontiguousarray(phase_tags, dtype=np.uint8)
    args = np.ascontiguousarray(phase_args, dtype=np.uint8).reshape(n, 5)
    ne = np.ascontiguousarray(n_edges, dtype=np.uint16)

    kinds = np.zeros(n, np.uint8)
    arity = np.zeros(n, np.uint16)
    slots = np.zeros((n, max_edges), np.uint16)

    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    rc = lib.tzolkin_edge_slots(
        states.ctypes.data_as(u8),
        movers.ctypes.data_as(u8),
        tags.ctypes.data_as(u8),
        args.ctypes.data_as(u8),
        ne.ctypes.data_as(u16),
        ctypes.c_uint32(n),
        ctypes.c_uint32(max_edges),
        kinds.ctypes.data_as(u8),
        arity.ctypes.data_as(u16),
        slots.ctypes.data_as(u16),
    )
    if rc != 0:
        raise RuntimeError(f"tzolkin_edge_slots failed with {rc}")
    return kinds, arity, slots


def edge_choices(
    states: np.ndarray,
    movers: np.ndarray,
    phase_tags: np.ndarray,
    phase_args: np.ndarray,
    n_edges: np.ndarray,
    max_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate ``Choice`` features for the pointer phases.

    Returns ``(kinds, cells, feats)`` with ``feats`` of shape
    ``[n, max_edges, D_CHOICE]``, zero-padded past each row's ``n_edges``.
    ``kinds == 2`` marks the rows that actually reconstructed.

    This is what makes ``Take`` trainable. It is 18% of the nodes a champion
    search visits and its widest phase, and it had no policy target at all: the
    pointer head ran at inference on parameters that had never seen a gradient
    and scored worse than uniform.
    """
    lib = _lib()
    import ctypes

    n = len(states)
    d_choice = D_CHOICE
    if n == 0 or max_edges <= 0:
        return (
            np.zeros(0, np.uint8),
            np.zeros(0, np.uint16),
            np.zeros((0, max(max_edges, 1), d_choice), np.float32),
        )
    if not hasattr(lib, "tzolkin_edge_choices"):
        raise RuntimeError(
            "the loaded libtzolkin has no tzolkin_edge_choices; rebuild with "
            "`cargo build --release` to train the pointer head"
        )

    states = np.ascontiguousarray(states, dtype=np.uint8).reshape(n, -1)
    if states.shape[1] != _SLOT:
        raise ValueError(f"state slot is {states.shape[1]} bytes, encoder wants {_SLOT}")
    movers = np.ascontiguousarray(movers, dtype=np.uint8)
    tags = np.ascontiguousarray(phase_tags, dtype=np.uint8)
    args = np.ascontiguousarray(phase_args, dtype=np.uint8).reshape(n, 5)
    ne = np.ascontiguousarray(n_edges, dtype=np.uint16)

    kinds = np.zeros(n, np.uint8)
    cells = np.zeros(n, np.uint16)
    feats = np.zeros((n, max_edges, d_choice), np.float32)

    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    rc = lib.tzolkin_edge_choices(
        states.ctypes.data_as(u8),
        movers.ctypes.data_as(u8),
        tags.ctypes.data_as(u8),
        args.ctypes.data_as(u8),
        ne.ctypes.data_as(u16),
        ctypes.c_uint32(n),
        ctypes.c_uint32(max_edges),
        kinds.ctypes.data_as(u8),
        cells.ctypes.data_as(u16),
        feats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(feats.size),
    )
    if rc != 0:
        raise RuntimeError(f"tzolkin_edge_choices failed with {rc}")
    return kinds, cells, feats


def value_targets(batch: dict, movers: np.ndarray) -> dict:
    """Rotate the stored per-seat targets into the querying player's frame.

    The record stores everything in absolute seat order; the network works in
    perspective order. Doing the rotation here, once, is what lets the same
    record be presented four times with four different queriers.
    """
    n = len(movers)
    rows = np.arange(n)[:, None]
    rot = (movers[:, None] + np.arange(N_PLAYERS)[None, :]) % N_PLAYERS
    scores = batch["final_scores"].astype(np.float32)[rows, rot]
    return {
        "z_rel": batch["z_rel"].astype(np.float32)[rows, rot],
        "z_score": np.clip((scores - SCORE_CENTRE) / SCORE_SCALE, -2.0, 2.0),
        "rank": _rank_targets(scores),
        "win_share": batch["win_share"].astype(np.float32)[rows, rot],
        # The search's own backed-up value at this node, per seat, in the same
        # centred convention as `z_rel` (rowsum 0, measured) and rotated the
        # same way. Present and finite on 100% of champion records and never
        # read until now. It is a *lower-variance* target than the game outcome
        # -- sd 0.294 against z_rel's 0.361 -- and on held-out records it
        # predicts z_rel better than any evaluator this project has (rmse
        # 0.2485, where `eval::heuristic` is 0.2702 and the best net 0.2600).
        "root_value": batch["root_value"].astype(np.float32)[rows, rot],
    }


def _rank_targets(scores: np.ndarray) -> np.ndarray:
    """``rank[i][r] = P(player i finishes in place r)``, ties split evenly.

    Fully vectorised: a Python loop here would run once per record per step and
    is the kind of thing that quietly turns a 100 ms optimiser step into a
    150 ms one.

    A seat's places run from "how many beat it" to "how many beat it, plus how
    many tie with it", and it takes an equal share of each.
    """
    n, p = scores.shape
    better = (scores[:, None, :] > scores[:, :, None]).sum(axis=2)   # [n, p]
    tied = (scores[:, None, :] == scores[:, :, None]).sum(axis=2)    # [n, p], >= 1
    places = np.arange(p)[None, None, :]
    lo = better[..., None]
    hi = (better + tied)[..., None]
    return (((places >= lo) & (places < hi)).astype(np.float32) / tied[..., None]).astype(
        np.float32
    )


# ----------------------------------------------------------------------
# batch assembly
# ----------------------------------------------------------------------

# Fixed-arity policy heads, keyed by `Phase::tag()` in src/phase.rs. Duplicated
# from model.py's PHASE_HEADS because this module must stay importable without
# torch -- `python train/features.py` is a valid smoke test on a machine that
# has numpy and nothing else. The two are checked against each other by
# `selftest.py`.
# The head each phase trains, from `arch.py`. Imported rather than restated:
# this table has to agree with the one the network is built from, and a second
# copy is exactly how it stopped agreeing last time. Tags absent here (4 Take,
# 7 DraftTile) go to the pointer head, which needs per-candidate features the
# records do not carry yet, so those rows train the value heads alone.
PHASE_ARITY = PHASE_HEADS

# `net.rs`'s `TOP_K`: at or under it stage 1 prunes nothing, so the loss
# trained here is exactly the softmax the search reads.
PTR_MAX_EDGES = 64


def batch_arrays(
    batch: dict, schema: dict, augment: bool, value_mix: float = 0.0
) -> tuple[np.ndarray, dict]:
    """Encode one sampled batch into inputs and targets. No torch.

    ``augment`` turns on the 4x perspective augmentation of ``LEARNING.md``
    §6.4: every position is presented once per querying player, and each gives a
    full value target. It is legitimate precisely because the encoder takes the
    querying player independently of whose turn it is, and it trains exactly the
    off-turn query a max^n backup makes. It is this domain's substitute for Go's
    8-fold dihedral augmentation and the only one available.

    ``value_mix`` blends the search's backed-up value into the ``rel`` target:
    ``(1 - mix) * z_rel + mix * root_value``. 0.0 is the game outcome alone,
    which is what every checkpoint before this was trained on. The policy head
    already learns from a search target (the visit distribution) and is the
    strongest thing in the net; the value head learns from the final score and
    is the weakest. This makes that asymmetry a knob rather than an assumption.
    Nothing about the architecture changes -- same tensors, same manifest, same
    `src/net.rs` -- so a checkpoint trained at any mix loads unmodified.
    """
    if not 0.0 <= value_mix <= 1.0:
        raise ValueError(f"value_mix must be in 0..=1, got {value_mix}")
    if augment:
        reps = N_PLAYERS
        rec = {k: np.repeat(v, reps, axis=0) for k, v in batch.items()}
        movers = np.tile(np.arange(reps, dtype=np.uint8), len(batch["state"]))
    else:
        rec = batch
        movers = batch["mover"].astype(np.uint8)

    x = encode_batch(rec["state"], movers, rec["phase_tag"], rec["phase_args"], schema)
    t = value_targets(rec, movers)
    rel = t["z_rel"]
    if value_mix > 0.0:
        rel = (1.0 - value_mix) * rel + value_mix * t["root_value"]
    targets = {"rel": rel, "score": t["z_score"], "rank": t["rank"]}

    # `policy_kind == 0` means the visit indices are not regenerable from
    # `(state, phase)`. Those rows train the value heads alone.
    #
    # **And only the mover's own perspective trains the policy.** The 4x
    # augmentation is legitimate for the value heads because the encoder takes
    # the querying player independently and every seat has a value target. It is
    # not legitimate here: `net.rs` only ever queries the policy from
    # `phase.mover(turn)`'s perspective, and from any other seat the input does
    # not even identify whose move is being predicted -- three of every four
    # augmented rows were asking the head to produce the same distribution from
    # inputs that cannot determine it. Restricting to `movers == mover` costs
    # nothing (those rows still train the value heads) and removes 75% of the
    # policy gradient, all of it noise.
    usable = (rec["policy_kind"] != 0) & (movers == rec["mover"])
    sel_all = np.where(usable)[0]
    if len(sel_all):
        ne_all = rec["n_edges"][sel_all].astype(np.int64)
        max_edges = int(max(1, ne_all.max()))
        kinds, arity_of, slots_all = edge_slots(
            rec["state"][sel_all],
            movers[sel_all],
            rec["phase_tag"][sel_all],
            rec["phase_args"][sel_all],
            ne_all,
            max_edges,
        )
        tags_all = rec["phase_tag"][sel_all]
        for tag, (name, arity) in PHASE_ARITY.items():
            rows = np.where((kinds == 1) & (tags_all == tag))[0]
            if len(rows) == 0:
                continue
            if not (arity_of[rows] == arity).all():
                raise RuntimeError(
                    f"arity disagreement on tag {tag}: arch.py says {arity}, "
                    f"encode::edges says {sorted(set(arity_of[rows].tolist()))}"
                )
            sel = sel_all[rows]
            ne = ne_all[rows]
            sl = slots_all[rows].astype(np.int64)          # [R, max_edges]
            v = rec["visits"][sel]
            eidx = v[:, :, 0].astype(np.int64)             # edge index, not slot
            cnt = v[:, :, 1].astype(np.float32)
            # An unused visit pair is (0, 0); an out-of-range index is a
            # reconstruction failure. Both contribute zero rather than landing
            # somewhere arbitrary.
            good = eidx < ne[:, None]
            cnt = np.where(good, cnt, 0.0)
            eidx = np.where(good, eidx, 0)
            tslot = np.take_along_axis(sl, eidx, axis=1)   # [R, 24] head slots
            dist = np.zeros((len(sel), arity), np.float32)
            np.add.at(dist, (np.arange(len(sel))[:, None], tslot), cnt)
            s = dist.sum(1, keepdims=True)
            dist = np.divide(dist, s, out=np.zeros_like(dist), where=s > 0)
            # Legality mask: exactly the slots the legal edges map to
            # (`LEARNING.md` §6.2 wants it regenerated, not stored -- but
            # regenerated from the mapping, not from the edge count).
            mask = np.zeros((len(sel), arity), bool)
            live = np.arange(max_edges)[None, :] < ne[:, None]
            rr = np.repeat(np.arange(len(sel))[:, None], max_edges, axis=1)
            mask[rr[live], sl[live]] = True
            targets[f"policy_{name}"] = (
                sel.astype(np.int64),
                dist,
                rec["policy_weight"][sel].astype(np.float32),
                mask,
            )

        # ---- the pointer phases (`Take`, `DraftTile`) --------------------
        #
        # Capped at PTR_MAX_EDGES because `net.rs` prunes to `TOP_K = 64` with
        # stage 1 before stage 2 scores anything: at or under the cap, nothing
        # is pruned and the cross-entropy trained here is *exactly* the softmax
        # the search reads. Above it the target would have to model the pruning
        # too. The cap keeps 92.4% of `Take` nodes.
        ptr_rows = np.where(
            np.isin(tags_all, POINTER_TAGS) & (ne_all >= 2) & (ne_all <= PTR_MAX_EDGES)
        )[0]
        if len(ptr_rows):
            sel = sel_all[ptr_rows]
            ne = ne_all[ptr_rows]
            width = int(ne.max())
            kinds2, cells, cand = edge_choices(
                rec["state"][sel],
                movers[sel],
                rec["phase_tag"][sel],
                rec["phase_args"][sel],
                ne,
                width,
            )
            keep = np.where(kinds2 == 2)[0]
            if len(keep):
                sel = sel[keep]
                ne = ne[keep]
                cand = cand[keep]
                v = rec["visits"][sel]
                eidx = v[:, :, 0].astype(np.int64)
                cnt = v[:, :, 1].astype(np.float32)
                good = eidx < ne[:, None]
                cnt = np.where(good, cnt, 0.0)
                eidx = np.where(good, eidx, 0)
                dist = np.zeros((len(sel), width), np.float32)
                np.add.at(dist, (np.arange(len(sel))[:, None], eidx), cnt)
                sm = dist.sum(1, keepdims=True)
                dist = np.divide(dist, sm, out=np.zeros_like(dist), where=sm > 0)
                # The candidate list *is* the legal set, so the mask is only the
                # zero padding out to `width`.
                mask = np.arange(width)[None, :] < ne[:, None]
                targets["policy_ptr"] = (
                    sel.astype(np.int64),
                    dist,
                    rec["policy_weight"][sel].astype(np.float32),
                    mask,
                    cand,
                    cells[keep].astype(np.int64),
                    rec["phase_tag"][sel].astype(np.int64),
                )
    return x, targets


if __name__ == "__main__":
    import sys

    from replay import Buffer

    b = Buffer.open(sys.argv[1] if len(sys.argv) > 1 else "replay")
    batch = b.sample(256, np.random.default_rng(0))
    x, t = batch_arrays(batch, b.schema, augment=True)
    w = d_in()
    print(f"D_IN {w}   x {x.shape} {x.dtype}   finite {np.isfinite(x).all()}")
    print(f"  live columns : {(x != 0).any(axis=0).sum()} / {w}")
    print(f"  range        : {x.min():.3f} .. {x.max():.3f}")
    for k, v in t.items():
        print(f"  {k:<16} {getattr(v, 'shape', type(v).__name__)}")
