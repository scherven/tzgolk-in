"""Architecture constants and shapes. **No torch** — deliberately.

`Net::load` in `src/net.rs` refuses a checkpoint with a missing tensor, an extra
tensor or a wrong shape, so these numbers decide whether a trained net can be
loaded at all. Keeping them importable without torch lets `check_manifest.py`
diff them against the Rust manifest anywhere, including in CI with no ML stack.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

N_PLAYERS = 4

# Head arities, fixed by src/encode.rs. Changing one changes the manifest.
N_PLACE = 7
N_WHO = 56
N_CELLS = 55
N_PHASE_TAGS = 8
# `encode::D_CHOICE` -- the per-candidate feature width the pointer head reads.
# `tzolkin_edge_choices` length-checks against it, so a disagreement is a -2
# return rather than a silent reshape.
D_CHOICE = 96
N_DECOMP = 6  # score components per player, LEARNING.md §3.2

# Fixed-arity policy heads, keyed by `Phase::tag()` in src/phase.rs. The arity
# of each is the width of its node type in SEARCH.md §2.7. Exhaustive on
# purpose: `Phase::tag` is an exhaustive match on the Rust side too, so a new
# phase breaks both builds rather than mis-indexing a table.
PHASE_HEADS = {
    0: ("beg", 4),          # none, or step down one of three temples
    1: ("mode", 3),         # place, retrieve, pity
    2: ("place", N_PLACE),  # 5 gears + first player + stop
    3: ("who", N_WHO),      # which worker to pick up next
    # 4 is Take: variable arity, pointer head
    5: ("extra_day", 2),
    6: ("place", N_PLACE),  # pity shares the placement head
    # 7 is DraftTile: pointer head, via step_emb
}
POINTER_TAGS = (4, 7)


@dataclass
class Config:
    """Architecture. Every field here appears in the file's ``__metadata__`` and
    is checked on load, so these *define* the manifest rather than describing it.
    """

    d_in: int = 3072
    width: int = 512
    blocks: int = 6
    board: int = 192
    player: int = 96
    bslot: int = 48
    mslot: int = 32
    globals_: int = 128
    value_hidden: int = 256
    key: int = 128
    d_choice: int = D_CHOICE
    dropout: float = 0.0

    # Input geometry, fixed by src/encode.rs.
    w_board: int = 660
    w_player: int = 282
    w_global: int = 576   # GLOBAL_W 254 + RESERVED_W 322
    w_bslot: int = 73
    w_mslot: int = 45
    n_bslot: int = 6
    n_mslot: int = 6

    @property
    def fuse_in(self) -> int:
        return (
            self.board
            + N_PLAYERS * self.player
            + self.n_bslot * self.bslot
            + self.n_mslot * self.mslot
            + self.globals_
        )

    @staticmethod
    def small() -> "Config":
        """The `small` rung of the ladder in LEARNING.md §4.3.

        Generations 1-30 run on this. Training a 4.2M net on a buffer of a few
        hundred thousand records wastes the first two days; the swap to `main`
        is a from-scratch retrain, not net surgery.
        """
        return Config(
            width=384, blocks=4, board=144, player=72, bslot=36, mslot=24,
            globals_=96, value_hidden=192, key=96,
        )

    @staticmethod
    def main() -> "Config":
        return Config()


