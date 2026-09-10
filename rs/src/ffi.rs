//! C ABI for the trainer.
//!
//! `train/features.py` used to reimplement the encoder in numpy. Two independent
//! encoders is a train/inference skew waiting to happen, and an invisible one:
//! nothing crashes, the network simply learns against features that differ from
//! the ones it is served at play time. This exports the real encoder instead, so
//! there is exactly one.
//!
//! The perspective argument is why a precomputed feature file will not do.
//! `LEARNING.md` §6.4 presents every position once per querying player, so the
//! trainer needs `encode(state, p, phase)` for all four `p` — four times the
//! rows, at 12 KB each. Calling across the ABI is cheaper than storing that.

use crate::encode::{encode, D_IN};
use crate::ids::PlayerId;
use crate::record::{decode_phase, decode_state, STATE_SLOT};

/// Input width, so Python need not hardcode it.
#[no_mangle]
pub extern "C" fn tzolkin_d_in() -> u32 {
    D_IN as u32
}

/// Bytes of state each record carries, so Python can stride correctly.
#[no_mangle]
pub extern "C" fn tzolkin_state_slot() -> u32 {
    STATE_SLOT as u32
}

/// Encode `n` records into `out`, row-major, `n * D_IN` floats.
///
/// * `states` — `n * STATE_SLOT` bytes, the record's state slot verbatim.
/// * `movers` — `n` bytes, the querying player for each row. This is the
///   *perspective*, not necessarily whose turn it is.
/// * `tags` — `n` bytes of `Phase::tag()`.
/// * `args` — `n * 5` bytes of phase arguments.
///
/// Returns 0 on success, or a negative code on a bad argument. Nothing here
/// panics across the ABI: a panic unwinding into Python is undefined behaviour,
/// so every failure is a return code.
///
/// # Safety
/// All four input pointers must be valid for the lengths implied by `n`, and
/// `out` for `n * D_IN` floats.
#[no_mangle]
pub unsafe extern "C" fn tzolkin_encode_batch(
    states: *const u8,
    movers: *const u8,
    tags: *const u8,
    args: *const u8,
    n: u32,
    out: *mut f32,
    out_len: u32,
) -> i32 {
    if states.is_null() || movers.is_null() || tags.is_null() || args.is_null() || out.is_null() {
        return -1;
    }
    let n = n as usize;
    if out_len as usize != n * D_IN {
        return -2;
    }

    let states = std::slice::from_raw_parts(states, n * STATE_SLOT);
    let movers = std::slice::from_raw_parts(movers, n);
    let tags = std::slice::from_raw_parts(tags, n);
    let args = std::slice::from_raw_parts(args, n * 5);
    let out = std::slice::from_raw_parts_mut(out, n * D_IN);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for i in 0..n {
            let g = decode_state(&states[i * STATE_SLOT..(i + 1) * STATE_SLOT]);
            let mut a = [0u8; 5];
            a.copy_from_slice(&args[i * 5..i * 5 + 5]);
            let phase = decode_phase(tags[i], a);
            let p = PlayerId(movers[i] % crate::ids::N_PLAYERS as u8);
            encode(&g, p, phase, &mut out[i * D_IN..(i + 1) * D_IN]);
        }
    }));

    if result.is_err() {
        -3
    } else {
        0
    }
}

/// Edge index -> fixed-arity head slot, for `n` records at once.
///
/// `train/features.py` used to *assume* this map was the identity — that edge
/// `e` trains head slot `e` — and build its policy targets and its legality
/// mask from `n_edges` alone. That is true for `Mode` and `ExtraDay` and false
/// everywhere else: the legal edges of `Beg`, `Placing` and `PickWorker` are a
/// **subset** of the head's slots, not a prefix of them, so the assumption
/// scatters each visit count onto the wrong move. Measured on held-out champion
/// records: `Beg` 37.7% of nodes wrong, `Placing` 32.3%, `PickWorker` **100%**.
///
/// `encode::edges` is the authority on the mapping — it is the same call the
/// network's own prior path makes at play time — so this exports it rather than
/// restating it, for the same reason `tzolkin_encode_batch` exists.
///
/// * `kinds[i]` — 0 `Uniform` (the edge list did not reconstruct; the row is
///   not usable as a policy target), 1 `Fixed`, 2 `Pointer` (`Take` and
///   `DraftTile`, which have no fixed-arity head).
/// * `arity[i]` — the head's width, so Python need not keep a second copy of
///   the table in `arch.py`. 0 unless `kinds[i] == 1`.
/// * `slots[i * max_edges + e]` — the head slot edge `e` maps to, for
///   `e < n_edges[i]`, when `kinds[i] == 1`. Untouched otherwise.
///
/// Returns 0, or a negative code on a bad argument. Never panics across the
/// ABI.
///
/// # Safety
/// Every pointer must be valid for the length implied by `n`, and `slots` for
/// `n * max_edges` `u16`s.
#[no_mangle]
pub unsafe extern "C" fn tzolkin_edge_slots(
    states: *const u8,
    movers: *const u8,
    tags: *const u8,
    args: *const u8,
    n_edges: *const u16,
    n: u32,
    max_edges: u32,
    kinds: *mut u8,
    arity: *mut u16,
    slots: *mut u16,
) -> i32 {
    if states.is_null()
        || movers.is_null()
        || tags.is_null()
        || args.is_null()
        || n_edges.is_null()
        || kinds.is_null()
        || arity.is_null()
        || slots.is_null()
    {
        return -1;
    }
    let n = n as usize;
    let max_edges = max_edges as usize;
    if max_edges == 0 {
        return -2;
    }

    let states = std::slice::from_raw_parts(states, n * STATE_SLOT);
    let movers = std::slice::from_raw_parts(movers, n);
    let tags = std::slice::from_raw_parts(tags, n);
    let args = std::slice::from_raw_parts(args, n * 5);
    let n_edges = std::slice::from_raw_parts(n_edges, n);
    let kinds = std::slice::from_raw_parts_mut(kinds, n);
    let arity = std::slice::from_raw_parts_mut(arity, n);
    let slots = std::slice::from_raw_parts_mut(slots, n * max_edges);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut feats: Vec<f32> = Vec::new();
        for i in 0..n {
            kinds[i] = 0;
            arity[i] = 0;
            let ne = n_edges[i] as usize;
            if ne == 0 || ne > max_edges {
                continue;
            }
            let g = decode_state(&states[i * STATE_SLOT..(i + 1) * STATE_SLOT]);
            let mut a = [0u8; 5];
            a.copy_from_slice(&args[i * 5..i * 5 + 5]);
            let phase = decode_phase(tags[i], a);
            let p = PlayerId(movers[i] % crate::ids::N_PLAYERS as u8);
            feats.clear();
            match crate::encode::edges(&g, p, phase, ne, &mut feats) {
                crate::encode::EdgeSpec::Fixed { head, idx } => {
                    kinds[i] = 1;
                    arity[i] = head.arity() as u16;
                    for (e, &s) in idx.iter().enumerate().take(ne) {
                        slots[i * max_edges + e] = s;
                    }
                }
                crate::encode::EdgeSpec::Pointer { .. } => kinds[i] = 2,
                crate::encode::EdgeSpec::Uniform => kinds[i] = 0,
            }
        }
    }));

    if result.is_err() {
        -3
    } else {
        0
    }
}

/// Per-candidate features for the pointer-head phases, `n` records at once.
///
/// `Take` is 18% of the nodes a champion search visits and the widest phase it
/// has (mean 21.8 edges), and until this existed it was the one phase with no
/// policy target at all: the pointer head ran at inference on parameters that
/// had never seen a gradient, and scored **worse than uniform** on held-out
/// champion records (N28). The obstacle was never the head — `model.py` has had
/// `Pointer` all along — it was that `features.py` cannot build `Choice`
/// features in numpy. `encode::edges` already builds them for exactly this
/// layout, so this hands them across the ABI.
///
/// * `kinds[i]` — 2 when the row is a pointer node whose edge list
///   reconstructed and `n_edges[i] <= max_edges`, 0 otherwise. Only rows marked
///   2 have anything written into `feats`.
/// * `cells[i]` — the board cell a `Take` is resolving, which `net.rs` adds to
///   the pointer query as `cell_emb[cell]`. `N_WHO - 1` (the `STOP` row, which
///   doubles as "no cell") for every other pointer phase, matching
///   `Net::plan`.
/// * `feats[i * max_edges * D_CHOICE ..]` — row `e` is candidate `e`, in the
///   engine's own edge order, zero-padded past `n_edges[i]`.
///
/// Returns 0, or a negative code on a bad argument. Never panics across the ABI.
///
/// # Safety
/// Every pointer must be valid for the length implied by `n`, and `feats` for
/// `n * max_edges * D_CHOICE` floats.
#[no_mangle]
pub unsafe extern "C" fn tzolkin_edge_choices(
    states: *const u8,
    movers: *const u8,
    tags: *const u8,
    args: *const u8,
    n_edges: *const u16,
    n: u32,
    max_edges: u32,
    kinds: *mut u8,
    cells: *mut u16,
    feats: *mut f32,
    feats_len: u32,
) -> i32 {
    use crate::encode::{D_CHOICE, N_WHO};
    if states.is_null()
        || movers.is_null()
        || tags.is_null()
        || args.is_null()
        || n_edges.is_null()
        || kinds.is_null()
        || cells.is_null()
        || feats.is_null()
    {
        return -1;
    }
    let n = n as usize;
    let max_edges = max_edges as usize;
    if max_edges == 0 {
        return -2;
    }
    if feats_len as usize != n * max_edges * D_CHOICE {
        return -2;
    }

    let states = std::slice::from_raw_parts(states, n * STATE_SLOT);
    let movers = std::slice::from_raw_parts(movers, n);
    let tags = std::slice::from_raw_parts(tags, n);
    let args = std::slice::from_raw_parts(args, n * 5);
    let n_edges = std::slice::from_raw_parts(n_edges, n);
    let kinds = std::slice::from_raw_parts_mut(kinds, n);
    let cells = std::slice::from_raw_parts_mut(cells, n);
    let out = std::slice::from_raw_parts_mut(feats, n * max_edges * D_CHOICE);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut scratch: Vec<f32> = Vec::new();
        for i in 0..n {
            kinds[i] = 0;
            cells[i] = (N_WHO - 1) as u16;
            let ne = n_edges[i] as usize;
            if ne < 2 || ne > max_edges {
                continue;
            }
            let g = decode_state(&states[i * STATE_SLOT..(i + 1) * STATE_SLOT]);
            let mut a = [0u8; 5];
            a.copy_from_slice(&args[i * 5..i * 5 + 5]);
            let phase = decode_phase(tags[i], a);
            let p = PlayerId(movers[i] % crate::ids::N_PLAYERS as u8);
            scratch.clear();
            let crate::encode::EdgeSpec::Pointer { off, n: got } =
                crate::encode::edges(&g, p, phase, ne, &mut scratch)
            else {
                continue;
            };
            if got != ne || scratch.len() < off + ne * D_CHOICE {
                continue;
            }
            // Same rule as `Net::plan`: a `Take` keys its query on the cell it
            // is resolving; everything else uses the `STOP` row.
            if let crate::phase::Phase::Take { worker } = phase {
                if let Some((gr, ps)) = g.loc(worker).on_board() {
                    cells[i] = crate::encode::cell(gr, ps) as u16;
                }
            }
            let dst = i * max_edges * D_CHOICE;
            out[dst..dst + ne * D_CHOICE].copy_from_slice(&scratch[off..off + ne * D_CHOICE]);
            kinds[i] = 2;
        }
    }));

    if result.is_err() {
        -3
    } else {
        0
    }
}
