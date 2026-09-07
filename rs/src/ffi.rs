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
