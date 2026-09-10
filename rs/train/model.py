"""The network: a pre-norm residual MLP trunk with per-phase heads.

Follows ``LEARNING.md`` §4.1 in shape and §2.2 / §3.1 in what it emits.

**Parameter names and shapes here are load-bearing.** ``Net::load`` in
``src/net.rs`` refuses a file with a missing tensor, an extra tensor or a wrong
shape -- nothing is defaulted -- so this module must reproduce the manifest
exactly. Print it with::

    cargo test --release --test encode -- --ignored --nocapture manifest

``train/selftest.py`` round-trips an export through the Rust loader; run it
after touching anything structural here.

Two earlier caveats in this file are now discharged: the stem is the factored
one (``src/encode.rs`` exists and fixes the 3,072-wide layout), and the ``Take``
pointer head is live, with ``cell_emb`` / ``step_emb`` letting ``DraftTile``
reuse the same machinery.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from arch import (  # noqa: F401  -- re-exported for callers
    Config,
    N_PLAYERS,
    N_PLACE,
    N_WHO,
    N_CELLS,
    N_PHASE_TAGS,
    N_DECOMP,
    PHASE_HEADS,
    POINTER_TAGS,
)

def gelu() -> nn.Module:
    # The tanh approximation, matching src/net.rs. The exact erf form differs by
    # ~1e-3, which would be a systematic train/inference skew.
    return nn.GELU(approximate="tanh")


class Block(nn.Module):
    """Pre-norm residual MLP block: norm1 -> fc1 -> GELU -> norm2 -> fc2 -> skip.

    Both projections are unbiased: a LayerNorm immediately upstream of each one
    already supplies a learned shift, so the bias is redundant.
    """

    def __init__(self, w: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(w)
        self.fc1 = nn.Linear(w, w, bias=False)
        self.norm2 = nn.LayerNorm(w)
        self.fc2 = nn.Linear(w, w, bias=False)
        self.act = gelu()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.fc1(self.norm1(x))
        h = self.fc2(self.norm2(self.act(h)))
        return x + self.drop(h)


class Trunk(nn.Module):
    """Numbered blocks plus a final norm.

    Children are registered as "0".."n-1" and "norm" so the parameter names come
    out as ``trunk.0.fc1.weight`` and ``trunk.norm.weight``, matching the Rust
    reader. An ``nn.ModuleList`` would insert an extra path component.
    """

    def __init__(self, width: int, blocks: int, dropout: float):
        super().__init__()
        for i in range(blocks):
            self.add_module(str(i), Block(width, dropout))
        self.norm = nn.LayerNorm(width)
        self.n = blocks

    def forward(self, x):
        for i in range(self.n):
            x = self._modules[str(i)](x)
        return self.norm(x)


class Stem(nn.Module):
    """Per-group projections, shared over the exchangeable axes.

    One `player` stem is applied to all four seats and one `bslot` stem to all
    six building slots, rather than learning a separate projection per seat.
    Seats and card slots are the axes this game is actually symmetric in; the
    gears are not, which is why there is no convolution anywhere.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.board = nn.Linear(c.w_board, c.board)
        self.player = nn.Linear(c.w_player, c.player)
        self.bslot = nn.Linear(c.w_bslot, c.bslot)
        self.mslot = nn.Linear(c.w_mslot, c.mslot)
        # `global` is a keyword, so it cannot be an attribute.
        self.add_module("global", nn.Linear(c.w_global, c.globals_))
        self.c = c

    def forward(self, x):
        c = self.c
        o = 0
        board = x[:, o : o + c.w_board]; o += c.w_board
        players = x[:, o : o + N_PLAYERS * c.w_player].view(-1, N_PLAYERS, c.w_player)
        o += N_PLAYERS * c.w_player
        gl = x[:, o : o + 254]; o += 254
        bs = x[:, o : o + c.n_bslot * c.w_bslot].view(-1, c.n_bslot, c.w_bslot)
        o += c.n_bslot * c.w_bslot
        ms = x[:, o : o + c.n_mslot * c.w_mslot].view(-1, c.n_mslot, c.w_mslot)
        o += c.n_mslot * c.w_mslot
        reserved = x[:, o:]

        parts = [
            self.board(board),
            self.player(players).flatten(1),
            self.bslot(bs).flatten(1),
            self.mslot(ms).flatten(1),
            self._modules["global"](torch.cat([gl, reserved], dim=1)),
        ]
        return torch.cat(parts, dim=1)


class ValueHead(nn.Module):
    """Four heads off a shared 256-wide projection.

    `decomp` is the score-decomposition auxiliary of LEARNING.md §3.2 -- six
    components per player. It is the cheapest sample-efficiency win available:
    it costs 24 outputs and gives the trunk a much denser gradient than a single
    scalar per game.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.fc = nn.Linear(c.width, c.value_hidden)
        self.rel = nn.Linear(c.value_hidden, N_PLAYERS)
        self.score = nn.Linear(c.value_hidden, N_PLAYERS)
        self.rank = nn.Linear(c.value_hidden, N_PLAYERS * N_PLAYERS)
        self.decomp = nn.Linear(c.value_hidden, N_DECOMP * N_PLAYERS)
        self.act = gelu()

    def forward(self, h):
        v = self.act(self.fc(h))
        return {
            # tanh keeps `rel` in (-1, 1) so c_puct calibrates the way the
            # literature's values assume.
            "rel": torch.tanh(self.rel(v)),
            "score": self.score(v),
            "rank": self.rank(v).view(-1, N_PLAYERS, N_PLAYERS),
            "decomp": self.decomp(v).view(-1, N_DECOMP, N_PLAYERS),
        }


class PolicyHeads(nn.Module):
    """The fixed-arity heads. Variable-arity nodes go to the pointer head."""

    def __init__(self, c: Config):
        super().__init__()
        self.beg = nn.Linear(c.width, 4)
        self.mode = nn.Linear(c.width, 3)
        self.place = nn.Linear(c.width, N_PLACE)
        self.extra_day = nn.Linear(c.width, 2)
        self.who = nn.Linear(c.width, N_WHO)

    def forward(self, h):
        return {
            "beg": self.beg(h),
            "mode": self.mode(h),
            "place": self.place(h),
            "extra_day": self.extra_day(h),
            "who": self.who(h),
        }


class Pointer(nn.Module):
    """Two-stage pointer head over candidates (LEARNING.md §2.3).

    Stage 1 is a cheap dot product that prunes to the top K; stage 2 runs a small
    MLP over the survivors. Both train against the same cross-entropy, so stage 1
    learns to approximate stage 2 rather than being a hand-written heuristic that
    rots when the rules move.

    `cell_emb` and `step_emb` are what let `DraftTile` reuse this head: a
    candidate is scored by what it does to the state, plus an embedding of where
    and which kind of decision it is.
    """

    def __init__(self, c: Config, keep: int = 64):
        super().__init__()
        self.keep = keep
        self.q_fast = nn.Linear(c.width, c.d_choice)
        self.q_deep = nn.Linear(c.width, c.key)
        self.cell_emb = nn.Parameter(torch.zeros(N_WHO, c.key))
        self.step_emb = nn.Parameter(torch.zeros(N_PHASE_TAGS, c.key))
        self.key1 = nn.Linear(c.d_choice, c.key)
        self.key2 = nn.Linear(c.key, c.key)
        self.act = gelu()
        nn.init.normal_(self.cell_emb, std=0.02)
        nn.init.normal_(self.step_emb, std=0.02)

    def forward(self, h, cand, mask, cell=None, tag=None):
        """``h`` [N, W]; ``cand`` [N, C, d_choice]; ``mask`` [N, C] bool.

        ``cell`` and ``tag`` are [N] int64. **Pass them.** `net.rs`'s `finish`
        builds the stage-2 query as ``q_deep(h) + cell_emb[cell] +
        step_emb[tag]``, so leaving them out here trains a different function
        from the one the search reads: the two embeddings stay at their N(0,
        0.02) initialisation and arrive at inference as pure noise on every
        pointer query. They are optional only so an older caller keeps working.
        """
        fast = torch.einsum("nd,ncd->nc", self.q_fast(h), cand)
        fast = fast.masked_fill(~mask, float("-inf"))
        k = self.key2(self.act(self.key1(cand)))
        q = self.q_deep(h)
        if cell is not None:
            q = q + self.cell_emb[cell]
        if tag is not None:
            q = q + self.step_emb[tag]
        deep = torch.einsum("nd,ncd->nc", q, k)
        deep = deep.masked_fill(~mask, float("-inf"))
        return fast, deep


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.stem = Stem(cfg)
        self.fuse = nn.Linear(cfg.fuse_in, cfg.width)
        self.fuse.norm = nn.LayerNorm(cfg.width)
        self.trunk = Trunk(cfg.width, cfg.blocks, cfg.dropout)
        self.value = ValueHead(cfg)
        self.policy = PolicyHeads(cfg)
        self.ptr = Pointer(cfg)

    def forward(self, x, cand=None, cand_mask=None, ptr_rows=None, cell=None, tag=None):
        """``ptr_rows`` selects which rows of the batch are pointer nodes.

        Only ~18% of a batch is `Take`, and the candidate tensor is
        ``[rows, C, 96]``, so gathering first is the difference between a
        4 MB tensor and a 24 MB one of mostly padding.
        """
        h = self.fuse.norm(self.fuse(self.stem(x)))
        h = self.trunk(h)
        out = {"h": h, "policy": self.policy(h)}
        out.update(self.value(h))
        if cand is not None:
            hp = h if ptr_rows is None else h[ptr_rows]
            out["ptr_fast"], out["ptr_deep"] = self.ptr(hp, cand, cand_mask, cell, tag)
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.cfg)


# ----------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------

# LEARNING.md §6.5. Huber throughout rather than MSE: early self-play produces
# scores in the -70..+40 range and MSE lets those outliers dominate the
# gradient.
W_REL = 1.5
W_SCORE = 0.5
W_RANK = 0.3
W_POLICY = {
    "beg": 1.0,
    "mode": 1.0,
    "place": 1.0,
    "who": 1.0,
    "extra_day": 0.5,
}
# The pointer head. `Take` is 18% of a champion search's nodes and its widest
# phase; the same weight as the other big heads. `W_PTR_FAST` is the stage-1
# scorer, trained against the same target so it learns to approximate stage 2
# (LEARNING.md 2.3) -- it only bites above `TOP_K` candidates, which is why it
# is a minority term rather than an equal one.
W_PTR = 1.0
W_PTR_FAST = 0.25
W_DECOMP = 0.3


def loss_fn(out: dict, tgt: dict) -> tuple[torch.Tensor, dict]:
    """Total loss and a dict of components for logging.

    Only the head matching a record's phase contributes a policy term; the value
    terms contribute on every record. A batch with no usable policy targets --
    which is every batch until MCTS lands -- trains the value heads alone, and
    that is exactly the warm-start of §6.7.
    """
    parts = {}

    parts["rel"] = W_REL * F.smooth_l1_loss(out["rel"], tgt["rel"])
    parts["score"] = W_SCORE * F.smooth_l1_loss(out["score"], tgt["score"])
    logp = F.log_softmax(out["rank"], dim=-1)
    parts["rank"] = W_RANK * -(tgt["rank"] * logp).sum(-1).mean()

    for name, w in W_POLICY.items():
        t = tgt.get(f"policy_{name}")
        if t is None:
            continue
        idx, target, weight, mask = t
        if idx.numel() == 0:
            continue
        logits = out["policy"][name][idx]
        # Masked softmax with the mask applied *inside* the loss, not just at
        # inference: illegal logits go to -inf before the softmax, so no
        # gradient reaches them and the net never spends capacity ranking moves
        # it cannot make (LEARNING.md §2.4).
        logits = logits.masked_fill(~mask, float("-inf"))
        lp = F.log_softmax(logits, dim=-1)
        ce = -(target * lp).nan_to_num(0.0).sum(-1)
        parts[f"pi_{name}"] = w * (ce * weight).mean()

    t = tgt.get("policy_ptr")
    if t is not None and "ptr_deep" in out:
        target, weight, mask = t[0], t[1], t[2]
        for key, w in (("ptr_deep", W_PTR), ("ptr_fast", W_PTR_FAST)):
            lp = F.log_softmax(out[key].masked_fill(~mask, float("-inf")), dim=-1)
            ce = -(target * lp).nan_to_num(0.0).sum(-1)
            parts["pi_" + key[4:]] = w * (ce * weight).mean()

    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}
