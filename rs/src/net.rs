//! Inference: `(GameState, Phase, PlayerId) -> (priors, value)`.
//!
//! A factored stem into a pre-norm residual MLP, per `docs/LEARNING.md` §4. No
//! convolutions (§1.2) and no attention: the only relational computations worth
//! having are over four players and twelve card slots, and those are cheaper and
//! more reliably learned as the explicit features in `encode.rs` than as a
//! 4-token attention block.
//!
//! Training lives in Python. This file is the other half of that split: ~20
//! matrix multiplies, three elementwise kernels and a bias add, with weights
//! exchanged through safetensors in both directions.
//!
//! **Deviation from §5.1, deliberately.** The doc names Accelerate
//! `cblas_sgemm` as *the* backend. That was right for one machine; the same
//! weights now have to run on a Windows box with a large GPU and possibly in
//! Colab. So every matmul goes through [`Gemm`], which has a portable pure-Rust
//! implementation that is correct everywhere and an Accelerate one behind
//! `cfg(target_os = "macos")`. The whole forward pass needs exactly one
//! primitive — `C = A · Bᵀ` with arbitrary leading dimensions — because weights
//! are stored `[out, in]` and every layer is `Y = X Wᵀ + b`. A third backend
//! implements that one function and nothing else.

use crate::effect::Choice;
use crate::encode::{
    self, EdgeSpec, Head, BOARD_OFF, BOARD_W, BSLOTS_OFF, BSLOT_W, D_CHOICE, D_IN, GLOBAL_OFF,
    GLOBAL_W, MSLOTS_OFF, MSLOT_W, N_WHO, PLAYERS_OFF, PLAYER_W, RESERVED_OFF, RESERVED_W,
};
use crate::ids::*;
use crate::phase::{Evaluation, Evaluator, Phase, Step};
use crate::state::{GameState, N_DISPLAY};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

// ===========================================================================
// GEMM
// ===========================================================================

/// The one linear-algebra primitive the network needs.
///
/// `C(m×n, ld=ldc) = A(m×k, ld=lda) · B(n×k, ld=ldb)ᵀ + beta·C`, all row-major.
///
/// Strides are part of the interface rather than an afterthought: the encoder
/// hands out one `[batch, D_IN]` matrix and each stem reads a *column range* of
/// it, so a leading dimension is what turns "the board block" into a GEMM
/// argument instead of a copy.
pub trait Gemm: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn sgemm_nt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    );

    fn name(&self) -> &'static str;
}

/// Portable fallback. No FFI, no unsafe, runs anywhere Rust does.
///
/// `B` is stored `[n, k]`, so both `A`'s rows and `B`'s rows are contiguous
/// `k`-vectors and the kernel is a dot product rather than a strided gather.
///
/// The shape that matters is the 4x4 register tile with **vector accumulators
/// along `k`**. Sixteen scalar accumulators are sixteen serialised dependency
/// chains, which is what the obvious version does and it runs at 19 GFLOP/s;
/// sixteen four-lane accumulators are sixteen NEON registers with the loads
/// filling the rest, and it runs at ~52. Eight-wide accumulators, or an 8x4
/// tile, spill and give back the gain — so this size is not arbitrary.
///
/// That is still ~20x short of Accelerate's AMX path. It is the floor, not the
/// target: it exists so the same weights run on Windows, in Colab and in CI.
pub struct Portable;

const MR: usize = 4;
const NR: usize = 4;
/// Lanes per accumulator. One NEON register; see the note above.
const KL: usize = 4;

impl Gemm for Portable {
    fn sgemm_nt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        if beta == 0.0 {
            for i in 0..m {
                c[i * ldc..i * ldc + n].fill(0.0);
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for v in &mut c[i * ldc..i * ldc + n] {
                    *v *= beta;
                }
            }
        }
        if k == 0 {
            return;
        }
        let km = k / KL * KL;

        let mut i = 0;
        while i < m {
            let mi = MR.min(m - i);
            let mut j = 0;
            while j < n {
                let nj = NR.min(n - j);
                if mi == MR && nj == NR {
                    let ars: [&[f32]; MR] =
                        std::array::from_fn(|ii| &a[(i + ii) * lda..(i + ii) * lda + km]);
                    let brs: [&[f32]; NR] =
                        std::array::from_fn(|jj| &b[(j + jj) * ldb..(j + jj) * ldb + km]);
                    let mut acc = [[[0f32; KL]; NR]; MR];
                    let mut t = 0;
                    while t < km {
                        let mut x = [[0f32; KL]; MR];
                        let mut y = [[0f32; KL]; NR];
                        for ii in 0..MR {
                            x[ii].copy_from_slice(&ars[ii][t..t + KL]);
                        }
                        for jj in 0..NR {
                            y[jj].copy_from_slice(&brs[jj][t..t + KL]);
                        }
                        for ii in 0..MR {
                            for jj in 0..NR {
                                for l in 0..KL {
                                    acc[ii][jj][l] += x[ii][l] * y[jj][l];
                                }
                            }
                        }
                        t += KL;
                    }
                    for ii in 0..MR {
                        let ar = &a[(i + ii) * lda..(i + ii) * lda + k];
                        for jj in 0..NR {
                            let br = &b[(j + jj) * ldb..(j + jj) * ldb + k];
                            let v = acc[ii][jj];
                            let mut sum = (v[0] + v[1]) + (v[2] + v[3]);
                            for t in km..k {
                                sum += ar[t] * br[t];
                            }
                            c[(i + ii) * ldc + j + jj] += sum;
                        }
                    }
                } else {
                    // Ragged edge. `dot` is already eight-lane, so this is only
                    // a couple of times slower than the tiled path.
                    for ii in 0..mi {
                        let ar = &a[(i + ii) * lda..(i + ii) * lda + k];
                        for jj in 0..nj {
                            let br = &b[(j + jj) * ldb..(j + jj) * ldb + k];
                            c[(i + ii) * ldc + j + jj] += dot(ar, br);
                        }
                    }
                }
                j += NR;
            }
            i += MR;
        }
    }

    fn name(&self) -> &'static str {
        "portable"
    }
}

#[cfg(target_os = "macos")]
mod accelerate {
    //! Apple's BLAS, linked as a framework. No build script and no crate: one
    //! `extern` block against a stable C ABI.

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;
}

/// Accelerate-backed GEMM. ~1.1-2.4 TFLOP/s through AMX with a ~4 µs call
/// latency, measured on an M4 Pro.
///
/// Set `VECLIB_MAXIMUM_THREADS=1` in the self-play driver: otherwise Accelerate
/// spawns its own pool and fights the search threads for the same cores.
#[cfg(target_os = "macos")]
pub struct Accelerate;

#[cfg(target_os = "macos")]
impl Gemm for Accelerate {
    fn sgemm_nt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        if k == 0 {
            Portable.sgemm_nt(m, n, k, a, lda, b, ldb, beta, c, ldc);
            return;
        }
        debug_assert!(a.len() >= (m - 1) * lda + k);
        debug_assert!(b.len() >= (n - 1) * ldb + k);
        debug_assert!(c.len() >= (m - 1) * ldc + n);
        // SAFETY: the three debug_asserts above are the whole contract —
        // cblas_sgemm reads A as m×k with stride lda, B as n×k with stride ldb
        // and writes C as m×n with stride ldc, all in bounds by construction.
        unsafe {
            accelerate::cblas_sgemm(
                accelerate::ROW_MAJOR,
                accelerate::NO_TRANS,
                accelerate::TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
    }

    fn name(&self) -> &'static str {
        "accelerate"
    }
}

// ---------------------------------------------------------------------------
// What a CUDA backend would have to touch
// ---------------------------------------------------------------------------
//
// Not implemented, and this note exists so that whoever does it starts from the
// right place rather than the obvious one.
//
// **A CUDA backend must not be a [`Gemm`].** That is the obvious place — one
// more `sgemm_nt` — and it does not work. The forward pass is ~29 GEMMs plus
// ~26 elementwise ops, so a per-call backend is ~55 kernel launches, and at the
// 5-10 µs per launch that is typical when CPU dispatch rivals the GPU work,
// that is 275-550 µs per forward pass **regardless of batch size**
// (`COMPUTE.md` §2.4). That alone caps a 4070 below what this laptop already
// does. **CUDA Graphs are a precondition here, not an optimisation**: capture
// the whole pass once and replay it with a single `cudaGraphLaunch`. The pass
// is fixed-shape for a fixed batch size, so capture one graph per size from a
// small set (32/64/128/256) and pad the last batch up.
//
// So the seam is [`Net::evaluate_batch`], not `Gemm`. Concretely:
//
// * **Resident on the device:** every `Linear`, `Norm` and embedding table —
//   4.33 M f32, 17.3 MB, or 8.7 MB in FP16. Uploaded once at load, never again.
//   On a 4070 or 4090 that fits entirely in L2 (36 / 72 MB), so after the first
//   pass the weights never leave the chip. That is a real advantage of the Ada
//   cards which has nothing to do with their TFLOPs.
// * **Crossing PCIe per pass:** the `b × D_IN` input. 3,072 floats is 12.3 KB,
//   or 6.1 KB in FP16, and `COMPUTE.md` §2.4's table says FP16 input alone
//   saturates PCIe 3.0 x16 at 2 M evals/s. Send FP16, use pinned memory, and
//   double-buffer on a separate copy stream so the transfer overlaps compute.
// * **Coming back:** `s.h` (`b × width`), `s.rel`, and the two pointer queries.
//
// **Stop at the trunk.** `evaluate_batch` is deliberately split so that
// `trunk` + `value_of` + `queries` are one fixed-shape block and `plan` /
// `select` / `keys` / `finish` are another. The first is graph-capturable as
// written. The second is not: `keys` runs over `Σ min(n_i, TOP_K)` rows, which
// changes with every batch, and padding it to `batch × TOP_K` wastes up to 32x
// the work. It is also only ~1.8 M multiply-adds against the trunk's ~4.2 M,
// and it is already batched. **Leave the pointer head on the CPU**; the split
// costs one device-to-host copy of `s.h`-derived vectors and buys a graph that
// never changes shape.
//
// **Nothing else moves.** `encode.rs`, `phase.rs`, the weight file and
// [`BatchedEvaluator`] are all unchanged — the queue does not care what is
// behind `Net::evaluate_batch`. What is *not* available to this agent: a `cuda`
// feature and a `cudarc`/`cust` dependency in `Cargo.toml`.
//
// Finally, the thing worth knowing before buying anything: `COMPUTE.md` §2.4
// concludes that **core count binds before GPU FLOPs do.** At ~3 µs of CPU per
// leaf, eight cores produce 2.0 M leaves/s; a 3060 delivers ~750 k evals/s and
// is GPU-bound, while a 4070, 3080 and 4090 are all bound by PCIe and by how
// fast the CPU can descend trees, and land in the same 1.5-2.5 M band. The gap
// between a 3060 and a 4090 in *this* workload is about 2.5x, not the 6.5x
// their FP32 specs imply.

/// The fastest backend available here, unless `TZOLKIN_GEMM=portable` says
/// otherwise. The override exists so a test can run both and diff them.
pub fn default_gemm() -> Box<dyn Gemm> {
    #[cfg(target_os = "macos")]
    {
        if std::env::var("TZOLKIN_GEMM").as_deref() != Ok("portable") {
            return Box::new(Accelerate);
        }
    }
    Box::new(Portable)
}

// ===========================================================================
// Weights on disk: safetensors
// ===========================================================================
//
// **This is the seam with the training side.** Everything a Python writer needs
// in order to produce a file `Net::load` accepts is here.
//
// ## Container
//
// Stock safetensors, so `safetensors.torch.save_file` / `save_file` from numpy
// write it and `Bundle::parse` reads it, with no crate on either side beyond
// `serde_json`, which was already a dependency.
//
// ```text
//   offset 0   u64 little-endian   N = header length in bytes
//   offset 8   N bytes             UTF-8 JSON, space-padded (see below)
//   offset 8+N remainder           tensor data, back to back
// ```
//
// * **`8 + N` is a multiple of 8.** The JSON is padded with trailing spaces to
//   make it so, which is what the reference implementation does and what an
//   mmap-based reader needs. `Bundle::parse` does not require it of files it
//   reads; `Bundle::to_bytes` always produces it.
// * The JSON is an object mapping tensor name to
//   `{"dtype", "shape", "data_offsets": [start, end]}`, plus an optional
//   `"__metadata__"` object of string-to-string.
// * `data_offsets` are relative to `8 + N`, half-open, in bytes.
// * Tensors are written **in lexicographic name order**, contiguously, with no
//   gaps and no padding between them. Readers must not rely on that order —
//   `Bundle::parse` does not — but writing it that way keeps files that
//   round-trip byte-identical.
//
// ## dtype and byte order
//
// **`F32` only, little-endian.** `Bundle::parse` rejects any other dtype with
// an error naming the tensor, rather than reinterpreting it: a bf16 checkpoint
// loaded as f32 would produce a net that runs and plays badly. If the training
// side wants to train in bf16, it must cast to f32 on export.
//
// ## Tensor layout
//
// Every weight is `[out, in]` row-major — `nn.Linear.weight` exactly — because
// every layer here is `Y = X Wᵀ + b` and [`Gemm::sgemm_nt`] is the one
// primitive. A `state_dict` therefore round-trips with **no transpose
// anywhere**. Biases are `[out]`. The two embedding tables are `[rows, key]`.
//
// [`Net::tensor_manifest`] is the authoritative list of names and shapes: it is
// what `load` demands and what `bundle` emits, and a `debug_assert` in `bundle`
// pins the two to each other. A file with a missing tensor, an extra tensor, or
// a shape mismatch is refused; nothing is defaulted.
//
// ## `__metadata__`
//
// | key | meaning | checked on load |
// |---|---|---|
// | `format` | always `tzolkin-net` | yes, refused if different |
// | `format_version` | container revision, currently `1` | yes, refused if greater |
// | `rules_version` | `crate::RULES_VERSION` when written | no — recorded for provenance |
// | `d_in`, `d_choice`, `n_who` | encoder geometry | yes, refused if different |
// | `width`, `blocks`, `board`, `player`, `bslot`, `mslot`, `globals`, `value_hidden`, `key` | the `Arch` | no — they *are* the arch, and every shape is then checked against it |
// | `label` | free-text name, shown in `Evaluator::name` | no |
//
// The geometry checks are the whole reason metadata is carried. A checkpoint
// written before a gear-size change has the same tensor shapes and a different
// meaning for every column of the board block; without `d_in` it would load
// happily and read every feature one slot to the left. Missing keys are treated
// as "written by a build that did not record them" and defaulted to
// `Arch::MAIN`, so the shape checks are the backstop.

/// One named f32 tensor, row-major.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A safetensors file in memory.
///
/// safetensors rather than a bespoke header for one reason: PyTorch writes it
/// natively (`safetensors.torch.save_file`) and so does numpy
/// (`safetensors.numpy.save_file`), and `serde_json` is already a dependency,
/// so reading it costs no new crate on either side. The `__metadata__` map
/// carries the arch and the geometry constants, which `Net::load` checks —
/// a checkpoint written before a gear-size change must fail loudly, not load
/// with its first layer reading the wrong columns.
#[derive(Clone, Debug, Default)]
pub struct Bundle {
    pub tensors: BTreeMap<String, Tensor>,
    pub meta: BTreeMap<String, String>,
}

/// `__metadata__["format"]`. A safetensors file that is not one of ours will
/// be missing it, which is allowed; one that names something else is refused.
pub const FORMAT: &str = "tzolkin-net";

/// `__metadata__["format_version"]`. Bump when the *container* changes — a new
/// required metadata key, a dtype, a different offset convention. Not for
/// architecture changes: those are already caught by the shape checks, and not
/// for rule changes, which `rules_version` records.
pub const FORMAT_VERSION: u32 = 1;

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

impl Bundle {
    pub fn read(path: impl AsRef<Path>) -> io::Result<Bundle> {
        Bundle::parse(&std::fs::read(path)?)
    }

    pub fn parse(buf: &[u8]) -> io::Result<Bundle> {
        if buf.len() < 8 {
            return Err(bad("file too short for a safetensors header"));
        }
        let n = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
        let start = 8usize
            .checked_add(n)
            .ok_or_else(|| bad("header length overflows"))?;
        if start > buf.len() {
            return Err(bad("header length runs past the end of the file"));
        }
        let hdr: serde_json::Value =
            serde_json::from_slice(&buf[8..start]).map_err(|e| bad(format!("bad header: {e}")))?;
        let obj = hdr.as_object().ok_or_else(|| bad("header is not an object"))?;

        let mut out = Bundle::default();
        for (k, v) in obj {
            if k == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (mk, mv) in m {
                        if let Some(s) = mv.as_str() {
                            out.meta.insert(mk.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = v.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
            if dtype != "F32" {
                return Err(bad(format!("{k}: dtype {dtype}, expected F32")));
            }
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| bad(format!("{k}: no shape")))?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v
                .get("data_offsets")
                .and_then(|s| s.as_array())
                .ok_or_else(|| bad(format!("{k}: no data_offsets")))?;
            let (b0, b1) = (
                off.first().and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                off.get(1).and_then(|x| x.as_u64()).unwrap_or(0) as usize,
            );
            let (lo, hi) = (start + b0, start + b1);
            if b1 < b0 || hi > buf.len() {
                return Err(bad(format!("{k}: data_offsets out of range")));
            }
            let want: usize = shape.iter().product::<usize>() * 4;
            if hi - lo != want {
                return Err(bad(format!("{k}: {} bytes for shape {shape:?}", hi - lo)));
            }
            let data = buf[lo..hi]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            out.tensors.insert(k.clone(), Tensor { shape, data });
        }
        Ok(out)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut hdr = serde_json::Map::new();
        if !self.meta.is_empty() {
            let mut m = serde_json::Map::new();
            for (k, v) in &self.meta {
                m.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            hdr.insert("__metadata__".into(), serde_json::Value::Object(m));
        }
        let mut off = 0usize;
        let mut body: Vec<u8> = Vec::new();
        for (k, t) in &self.tensors {
            let end = off + t.data.len() * 4;
            hdr.insert(
                k.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": t.shape,
                    "data_offsets": [off, end],
                }),
            );
            for v in &t.data {
                body.extend_from_slice(&v.to_le_bytes());
            }
            off = end;
        }
        let mut json = serde_json::to_vec(&serde_json::Value::Object(hdr)).unwrap();
        // Pad the header with spaces so the data section starts 8-byte aligned.
        // JSON ignores trailing whitespace, the reference implementation does
        // exactly this, and an mmap-based reader on the Python side needs it.
        // `parse` does not require it of files it reads.
        while (8 + json.len()) % 8 != 0 {
            json.push(b' ');
        }
        let mut out = Vec::with_capacity(8 + json.len() + body.len());
        out.extend_from_slice(&(json.len() as u64).to_le_bytes());
        out.extend_from_slice(&json);
        out.extend_from_slice(&body);
        out
    }

    fn take(&mut self, name: &str, shape: &[usize]) -> io::Result<Vec<f32>> {
        let t = self
            .tensors
            .remove(name)
            .ok_or_else(|| bad(format!("missing tensor {name}")))?;
        if t.shape != shape {
            return Err(bad(format!(
                "{name}: shape {:?}, expected {shape:?}",
                t.shape
            )));
        }
        Ok(t.data)
    }
}

// ===========================================================================
// Architecture
// ===========================================================================

/// Layer widths. `MAIN` is §4.2's 4.2 M-parameter net; `SMALL` is the ladder's
/// warm-start rung, kept so generations 1-30 are not spent training a big net
/// on a tiny buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arch {
    pub width: usize,
    pub blocks: usize,
    pub board: usize,
    pub player: usize,
    pub bslot: usize,
    pub mslot: usize,
    pub globals: usize,
    pub value_hidden: usize,
    /// Key/query width of the pointer head's deep stage.
    pub key: usize,
}

impl Arch {
    pub const MAIN: Arch = Arch {
        width: 512,
        blocks: 6,
        board: 192,
        player: 96,
        bslot: 48,
        mslot: 32,
        globals: 128,
        value_hidden: 256,
        key: 128,
    };

    pub const SMALL: Arch = Arch {
        width: 384,
        blocks: 4,
        board: 144,
        player: 72,
        bslot: 36,
        mslot: 24,
        globals: 96,
        value_hidden: 192,
        key: 96,
    };

    /// Width of the concatenated stem output.
    pub const fn fuse(&self) -> usize {
        self.board + N_PLAYERS * self.player + N_DISPLAY * self.bslot + N_DISPLAY * self.mslot
            + self.globals
    }

    fn tag(&self) -> String {
        format!("w{}b{}", self.width, self.blocks)
    }
}

/// The six components `end_game` splits a final score into. Predicting these
/// forces the trunk to represent *why* a position is worth what it is worth
/// rather than only how much; it is the cheapest auxiliary supervision this
/// game offers and it is free on every terminal position.
pub const N_DECOMP: usize = 6;

// ===========================================================================
// Layers
// ===========================================================================

struct Linear {
    /// `[out, inp]`, row-major — PyTorch's `nn.Linear.weight` layout, so a
    /// `state_dict` round-trips without a transpose.
    w: Vec<f32>,
    b: Vec<f32>,
    out: usize,
    inp: usize,
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

struct Block {
    n1: Norm,
    fc1: Vec<f32>,
    n2: Norm,
    fc2: Vec<f32>,
}

/// The four value outputs, in perspective order: index 0 is the querying
/// player, index k the player k seats clockwise.
#[derive(Clone, Copy, Debug)]
pub struct Value {
    /// `tanh((score - mean(score)) / 25)`, centred at inference. Bounded in
    /// (-1, 1), so `c_puct` in 1.5-2.5 is calibrated, and near-linear through
    /// the bulk of the score distribution so the search has a gradient to climb
    /// from round 1. **This is what the search backs up.**
    pub rel: [f32; N_PLAYERS],
    /// `(final_score - 75) / 50`. The dense absolute signal.
    pub score: [f32; N_PLAYERS],
    /// `rank[i][r] = P(player i finishes in place r)`; rows sum to 1.
    /// `rank[i][0]` is P(win), which is what reporting wants.
    pub rank: [[f32; N_PLAYERS]; N_PLAYERS],
    /// Per-player score decomposition: temple, building, monument, corn
    /// conversion, skull VP, starvation.
    pub decomp: [[f32; N_DECOMP]; N_PLAYERS],
}

// ===========================================================================
// The network
// ===========================================================================

pub struct Net {
    pub arch: Arch,
    gemm: Box<dyn Gemm>,
    label: String,
    /// Evaluations served one at a time, through [`Net::evaluate_one`].
    ///
    /// Not a statistic anyone wants; a smoke alarm. See
    /// [`Net::unbatched_calls`].
    solo: AtomicU64,

    s_board: Linear,
    s_player: Linear,
    s_bslot: Linear,
    s_mslot: Linear,
    /// Reads `GLOBAL_W + RESERVED_W` columns. The reserved tail is fed on
    /// purpose: it is dead weight while it is zero, and it is the whole reason
    /// a new scalar feature can be added without discarding the checkpoint.
    s_global: Linear,

    fuse: Linear,
    fuse_norm: Norm,
    blocks: Vec<Block>,
    out_norm: Norm,

    v_fc: Linear,
    v_rel: Linear,
    v_score: Linear,
    v_rank: Linear,
    v_decomp: Linear,

    h_beg: Linear,
    h_mode: Linear,
    h_place: Linear,
    h_extra: Linear,
    h_who: Linear,

    q_fast: Linear,
    q_deep: Linear,
    cell_emb: Vec<f32>,
    step_emb: Vec<f32>,
    key1: Linear,
    key2: Linear,
}

/// One position to evaluate.
pub struct Query<'a> {
    pub state: &'a GameState,
    pub phase: Phase,
    /// Whose turn it is. The decision belongs to `phase.mover(turn)`, which is
    /// also the perspective the encoding and the value vector are rotated to.
    pub turn: PlayerId,
    pub n_edges: usize,
    /// The candidate list the caller already built for a `Take` or `DraftTile`
    /// node. Supplying it skips a `choices_for_worker` re-derivation.
    ///
    /// Superseded by `steps`, which carries the same information for every
    /// phase rather than only the pointer ones.
    pub candidates: Option<&'a [Choice]>,
    /// The edges themselves, in the order the tree enumerated them.
    ///
    /// **Supply this whenever you have it.** `phase::Query` always does, so
    /// anything reaching the network through `Evaluator::evaluate_many` gets it
    /// for free. Without it the encoder re-derives the edge list from the engine
    /// and checks the length, which costs a `choices_for_worker` at every `Take`
    /// node and, worse, cannot detect a list of the right length in the wrong
    /// order. See `encode::edges_for`.
    pub steps: Option<&'a [Step]>,
}

impl<'a> Query<'a> {
    /// A query with only a count. The encoder will re-derive the edge list;
    /// prefer [`Query::with_steps`].
    pub fn new(state: &'a GameState, phase: Phase, turn: PlayerId, n_edges: usize) -> Self {
        Query {
            state,
            phase,
            turn,
            n_edges,
            candidates: None,
            steps: None,
        }
    }

    /// A query carrying the edges the tree enumerated.
    pub fn with_steps(
        state: &'a GameState,
        phase: Phase,
        turn: PlayerId,
        steps: &'a [Step],
    ) -> Self {
        Query {
            state,
            phase,
            turn,
            n_edges: steps.len(),
            candidates: None,
            steps: Some(steps),
        }
    }
}

impl<'a> From<&crate::phase::Query<'a>> for Query<'a> {
    fn from(q: &crate::phase::Query<'a>) -> Self {
        Query::with_steps(q.state, q.phase, q.turn, q.edges)
    }
}

/// Stage-1 keeps this many candidates for the expensive stage-2 scoring.
const TOP_K: usize = 64;
/// Prior mass shared out evenly among everything stage 1 discarded, so a
/// candidate the cheap scorer got wrong is never assigned probability zero.
const TAIL_MASS: f32 = 0.03;

impl Net {
    // ---- construction --------------------------------------------------

    /// Random weights, so the whole pipeline — search, self-play, replay
    /// writing, the arena — can be exercised before anything is trained.
    ///
    /// Hidden layers get Xavier-uniform; every output head gets the same
    /// scaled down by 10. An untrained net therefore emits near-uniform priors
    /// and values near zero, which is the right prior for a search to start
    /// from and keeps a warm start from having to unlearn noise.
    pub fn random(arch: Arch, seed: u64) -> Net {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut lin = |out: usize, inp: usize, gain: f32| {
            let bound = gain * (6.0 / (out + inp) as f32).sqrt();
            Linear {
                w: (0..out * inp).map(|_| rng.gen_range(-bound..bound)).collect(),
                b: vec![0.0; out],
                out,
                inp,
            }
        };
        let head = |out, inp, r: &mut StdRng| {
            let bound = 0.1 * (6.0 / (out + inp) as f32).sqrt();
            Linear {
                w: (0..out * inp).map(|_| r.gen_range(-bound..bound)).collect(),
                b: vec![0.0; out],
                out,
                inp,
            }
        };

        let w = arch.width;
        let s_board = lin(arch.board, BOARD_W, 1.0);
        let s_player = lin(arch.player, PLAYER_W, 1.0);
        let s_bslot = lin(arch.bslot, BSLOT_W, 1.0);
        let s_mslot = lin(arch.mslot, MSLOT_W, 1.0);
        let s_global = lin(arch.globals, GLOBAL_W + RESERVED_W, 1.0);
        let fuse = lin(w, arch.fuse(), 1.0);
        let blocks = (0..arch.blocks)
            .map(|_| Block {
                n1: Norm {
                    g: vec![1.0; w],
                    b: vec![0.0; w],
                },
                fc1: lin(w, w, 1.0).w,
                n2: Norm {
                    g: vec![1.0; w],
                    b: vec![0.0; w],
                },
                // Zero-initialised second projection: every residual block
                // starts as the identity, which is the standard trick for
                // making a deep pre-norm stack trainable from step 1.
                fc2: vec![0.0; w * w],
            })
            .collect();
        let v_fc = lin(arch.value_hidden, w, 1.0);
        let vh = arch.value_hidden;

        let mut net = Net {
            solo: AtomicU64::new(0),
            label: format!("random-{}-{seed}", arch.tag()),
            gemm: default_gemm(),
            s_board,
            s_player,
            s_bslot,
            s_mslot,
            s_global,
            fuse,
            fuse_norm: Norm {
                g: vec![1.0; w],
                b: vec![0.0; w],
            },
            blocks,
            out_norm: Norm {
                g: vec![1.0; w],
                b: vec![0.0; w],
            },
            v_fc,
            v_rel: head(N_PLAYERS, vh, &mut rng),
            v_score: head(N_PLAYERS, vh, &mut rng),
            v_rank: head(N_PLAYERS * N_PLAYERS, vh, &mut rng),
            v_decomp: head(N_PLAYERS * N_DECOMP, vh, &mut rng),
            h_beg: head(4, w, &mut rng),
            h_mode: head(3, w, &mut rng),
            h_place: head(encode::N_PLACE, w, &mut rng),
            h_extra: head(2, w, &mut rng),
            h_who: head(N_WHO, w, &mut rng),
            q_fast: head(D_CHOICE, w, &mut rng),
            q_deep: head(arch.key, w, &mut rng),
            cell_emb: vec![0.0; N_WHO * arch.key],
            step_emb: vec![0.0; Phase::COUNT * arch.key],
            key1: {
                let bound = (6.0 / (arch.key + D_CHOICE) as f32).sqrt();
                Linear {
                    w: (0..arch.key * D_CHOICE)
                        .map(|_| rng.gen_range(-bound..bound))
                        .collect(),
                    b: vec![0.0; arch.key],
                    out: arch.key,
                    inp: D_CHOICE,
                }
            },
            key2: {
                let bound = (6.0 / (2 * arch.key) as f32).sqrt();
                Linear {
                    w: (0..arch.key * arch.key)
                        .map(|_| rng.gen_range(-bound..bound))
                        .collect(),
                    b: vec![0.0; arch.key],
                    out: arch.key,
                    inp: arch.key,
                }
            },
            arch,
        };
        net.label = format!("random-{}-{seed}", arch.tag());
        net
    }

    /// One position, one forward pass.
    ///
    /// The honest cost of a batch of one. [`Unbatched`] is the `Evaluator` that
    /// wraps this; nothing else should call it in a loop.
    pub fn evaluate_one(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation {
        self.solo.fetch_add(1, Ordering::Relaxed);
        let mut out = Vec::with_capacity(1);
        self.evaluate_batch(&[Query::new(state, phase, turn, n_edges)], &mut out);
        out.pop().unwrap_or(Evaluation {
            priors: Vec::new(),
            value: [0.0; N_PLAYERS],
        })
    }

    /// How many evaluations went one at a time — i.e. through [`Unbatched`], at
    /// a fraction of the batched rate.
    ///
    /// Log this at the end of any run that believes it is batched. A non-zero
    /// count on a self-play generation means some thread is holding an
    /// `Unbatched` rather than a [`BatchHandle`], and nothing else in the run
    /// will tell you.
    pub fn unbatched_calls(&self) -> u64 {
        self.solo.load(Ordering::Relaxed)
    }

    /// Swap the matmul backend. Exists so a test can diff Accelerate against
    /// the portable kernel on identical weights.
    pub fn with_gemm(mut self, g: Box<dyn Gemm>) -> Net {
        self.gemm = g;
        self
    }

    pub fn backend(&self) -> &'static str {
        self.gemm.name()
    }

    pub fn n_params(&self) -> usize {
        let mut n = 0;
        for l in [
            &self.s_board,
            &self.s_player,
            &self.s_bslot,
            &self.s_mslot,
            &self.s_global,
            &self.fuse,
            &self.v_fc,
            &self.v_rel,
            &self.v_score,
            &self.v_rank,
            &self.v_decomp,
            &self.h_beg,
            &self.h_mode,
            &self.h_place,
            &self.h_extra,
            &self.h_who,
            &self.q_fast,
            &self.q_deep,
            &self.key1,
            &self.key2,
        ] {
            n += l.w.len() + l.b.len();
        }
        n += 4 * self.arch.width; // fuse_norm + out_norm
        for b in &self.blocks {
            n += b.fc1.len() + b.fc2.len() + 4 * self.arch.width;
        }
        n + self.cell_emb.len() + self.step_emb.len()
    }

    // ---- serialisation -------------------------------------------------

    /// The exact tensor set a checkpoint must contain, with shapes.
    ///
    /// This is the interface with the training side: a Python `state_dict`
    /// whose keys and shapes match this list loads without a transpose, because
    /// every weight is stored `[out, in]` exactly as `nn.Linear` does.
    pub fn tensor_manifest(arch: Arch) -> Vec<(String, Vec<usize>)> {
        let w = arch.width;
        let k = arch.key;
        let vh = arch.value_hidden;
        let mut v: Vec<(String, Vec<usize>)> = vec![
            ("stem.board.weight".into(), vec![arch.board, BOARD_W]),
            ("stem.board.bias".into(), vec![arch.board]),
            ("stem.player.weight".into(), vec![arch.player, PLAYER_W]),
            ("stem.player.bias".into(), vec![arch.player]),
            ("stem.bslot.weight".into(), vec![arch.bslot, BSLOT_W]),
            ("stem.bslot.bias".into(), vec![arch.bslot]),
            ("stem.mslot.weight".into(), vec![arch.mslot, MSLOT_W]),
            ("stem.mslot.bias".into(), vec![arch.mslot]),
            (
                "stem.global.weight".into(),
                vec![arch.globals, GLOBAL_W + RESERVED_W],
            ),
            ("stem.global.bias".into(), vec![arch.globals]),
            ("fuse.weight".into(), vec![w, arch.fuse()]),
            ("fuse.bias".into(), vec![w]),
            ("fuse.norm.weight".into(), vec![w]),
            ("fuse.norm.bias".into(), vec![w]),
        ];
        for i in 0..arch.blocks {
            v.push((format!("trunk.{i}.norm1.weight"), vec![w]));
            v.push((format!("trunk.{i}.norm1.bias"), vec![w]));
            v.push((format!("trunk.{i}.fc1.weight"), vec![w, w]));
            v.push((format!("trunk.{i}.norm2.weight"), vec![w]));
            v.push((format!("trunk.{i}.norm2.bias"), vec![w]));
            v.push((format!("trunk.{i}.fc2.weight"), vec![w, w]));
        }
        v.extend([
            ("trunk.norm.weight".into(), vec![w]),
            ("trunk.norm.bias".into(), vec![w]),
            ("value.fc.weight".into(), vec![vh, w]),
            ("value.fc.bias".into(), vec![vh]),
            ("value.rel.weight".into(), vec![N_PLAYERS, vh]),
            ("value.rel.bias".into(), vec![N_PLAYERS]),
            ("value.score.weight".into(), vec![N_PLAYERS, vh]),
            ("value.score.bias".into(), vec![N_PLAYERS]),
            (
                "value.rank.weight".into(),
                vec![N_PLAYERS * N_PLAYERS, vh],
            ),
            ("value.rank.bias".into(), vec![N_PLAYERS * N_PLAYERS]),
            (
                "value.decomp.weight".into(),
                vec![N_PLAYERS * N_DECOMP, vh],
            ),
            ("value.decomp.bias".into(), vec![N_PLAYERS * N_DECOMP]),
            ("policy.beg.weight".into(), vec![4, w]),
            ("policy.beg.bias".into(), vec![4]),
            ("policy.mode.weight".into(), vec![3, w]),
            ("policy.mode.bias".into(), vec![3]),
            ("policy.place.weight".into(), vec![encode::N_PLACE, w]),
            ("policy.place.bias".into(), vec![encode::N_PLACE]),
            ("policy.extra_day.weight".into(), vec![2, w]),
            ("policy.extra_day.bias".into(), vec![2]),
            ("policy.who.weight".into(), vec![N_WHO, w]),
            ("policy.who.bias".into(), vec![N_WHO]),
            ("ptr.q_fast.weight".into(), vec![D_CHOICE, w]),
            ("ptr.q_fast.bias".into(), vec![D_CHOICE]),
            ("ptr.q_deep.weight".into(), vec![k, w]),
            ("ptr.q_deep.bias".into(), vec![k]),
            ("ptr.cell_emb".into(), vec![N_WHO, k]),
            ("ptr.step_emb".into(), vec![Phase::COUNT, k]),
            ("ptr.key1.weight".into(), vec![k, D_CHOICE]),
            ("ptr.key1.bias".into(), vec![k]),
            ("ptr.key2.weight".into(), vec![k, k]),
            ("ptr.key2.bias".into(), vec![k]),
        ]);
        v
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.bundle().write(path)
    }

    pub fn bundle(&self) -> Bundle {
        let mut b = Bundle::default();
        let mut put = |n: &str, shape: Vec<usize>, data: Vec<f32>| {
            b.tensors.insert(n.into(), Tensor { shape, data });
        };
        let a = self.arch;
        let w = a.width;
        for (l, n, inp) in [
            (&self.s_board, "stem.board", BOARD_W),
            (&self.s_player, "stem.player", PLAYER_W),
            (&self.s_bslot, "stem.bslot", BSLOT_W),
            (&self.s_mslot, "stem.mslot", MSLOT_W),
            (&self.s_global, "stem.global", GLOBAL_W + RESERVED_W),
            (&self.fuse, "fuse", a.fuse()),
            (&self.v_fc, "value.fc", w),
            (&self.v_rel, "value.rel", a.value_hidden),
            (&self.v_score, "value.score", a.value_hidden),
            (&self.v_rank, "value.rank", a.value_hidden),
            (&self.v_decomp, "value.decomp", a.value_hidden),
            (&self.h_beg, "policy.beg", w),
            (&self.h_mode, "policy.mode", w),
            (&self.h_place, "policy.place", w),
            (&self.h_extra, "policy.extra_day", w),
            (&self.h_who, "policy.who", w),
            (&self.q_fast, "ptr.q_fast", w),
            (&self.q_deep, "ptr.q_deep", w),
            (&self.key1, "ptr.key1", D_CHOICE),
            (&self.key2, "ptr.key2", a.key),
        ] {
            put(&format!("{n}.weight"), vec![l.out, inp], l.w.clone());
            put(&format!("{n}.bias"), vec![l.out], l.b.clone());
        }
        put("fuse.norm.weight", vec![w], self.fuse_norm.g.clone());
        put("fuse.norm.bias", vec![w], self.fuse_norm.b.clone());
        for (i, blk) in self.blocks.iter().enumerate() {
            put(&format!("trunk.{i}.norm1.weight"), vec![w], blk.n1.g.clone());
            put(&format!("trunk.{i}.norm1.bias"), vec![w], blk.n1.b.clone());
            put(&format!("trunk.{i}.fc1.weight"), vec![w, w], blk.fc1.clone());
            put(&format!("trunk.{i}.norm2.weight"), vec![w], blk.n2.g.clone());
            put(&format!("trunk.{i}.norm2.bias"), vec![w], blk.n2.b.clone());
            put(&format!("trunk.{i}.fc2.weight"), vec![w, w], blk.fc2.clone());
        }
        put("trunk.norm.weight", vec![w], self.out_norm.g.clone());
        put("trunk.norm.bias", vec![w], self.out_norm.b.clone());
        put("ptr.cell_emb", vec![N_WHO, a.key], self.cell_emb.clone());
        put(
            "ptr.step_emb",
            vec![Phase::COUNT, a.key],
            self.step_emb.clone(),
        );

        b.meta.insert("format".into(), FORMAT.into());
        b.meta
            .insert("format_version".into(), FORMAT_VERSION.to_string());
        b.meta
            .insert("rules_version".into(), crate::RULES_VERSION.to_string());
        b.meta.insert("d_in".into(), D_IN.to_string());
        b.meta.insert("d_choice".into(), D_CHOICE.to_string());
        b.meta.insert("n_who".into(), N_WHO.to_string());
        b.meta.insert("width".into(), a.width.to_string());
        b.meta.insert("blocks".into(), a.blocks.to_string());
        b.meta.insert("board".into(), a.board.to_string());
        b.meta.insert("player".into(), a.player.to_string());
        b.meta.insert("bslot".into(), a.bslot.to_string());
        b.meta.insert("mslot".into(), a.mslot.to_string());
        b.meta.insert("globals".into(), a.globals.to_string());
        b.meta
            .insert("value_hidden".into(), a.value_hidden.to_string());
        b.meta.insert("key".into(), a.key.to_string());
        b.meta.insert("label".into(), self.label.clone());

        // `save` and `load` are written separately, so pin them to one another
        // rather than discovering the mismatch on the first real checkpoint.
        debug_assert!(
            Net::tensor_manifest(a)
                .into_iter()
                .all(|(n, sh)| b.tensors.get(&n).map(|t| t.shape == sh) == Some(true)),
            "bundle() and tensor_manifest() disagree"
        );
        b
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Net> {
        let label = path
            .as_ref()
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "net".into());
        let b = Bundle::read(path)?;
        Net::from_bundle(b, &label)
    }

    pub fn from_bundle(mut b: Bundle, label: &str) -> io::Result<Net> {
        let num = |b: &Bundle, k: &str, d: usize| -> usize {
            b.meta.get(k).and_then(|s| s.parse().ok()).unwrap_or(d)
        };
        // A file from a *later* container revision may put things where this
        // build does not expect them, so refuse it rather than guess. An older
        // one is fine by construction: revisions only add.
        if let Some(f) = b.meta.get("format") {
            if f != FORMAT {
                return Err(bad(format!(
                    "format is {f:?} in the checkpoint but {FORMAT:?} in this build"
                )));
            }
        }
        let v = num(&b, "format_version", 1) as u32;
        if v > FORMAT_VERSION {
            return Err(bad(format!(
                "format_version {v} was written by a newer build; this one reads {FORMAT_VERSION}"
            )));
        }
        // The geometry checks are the point of carrying metadata at all: a
        // checkpoint written before a gear-size change would otherwise load
        // happily and read every column one slot to the left.
        for (k, want) in [("d_in", D_IN), ("d_choice", D_CHOICE), ("n_who", N_WHO)] {
            if let Some(s) = b.meta.get(k) {
                let got: usize = s.parse().unwrap_or(0);
                if got != want {
                    return Err(bad(format!(
                        "{k} is {got} in the checkpoint but {want} in this build; \
                         the encoder geometry changed and the weights do not fit"
                    )));
                }
            }
        }
        let arch = Arch {
            width: num(&b, "width", 512),
            blocks: num(&b, "blocks", 6),
            board: num(&b, "board", 192),
            player: num(&b, "player", 96),
            bslot: num(&b, "bslot", 48),
            mslot: num(&b, "mslot", 32),
            globals: num(&b, "globals", 128),
            value_hidden: num(&b, "value_hidden", 256),
            key: num(&b, "key", 128),
        };

        let w = arch.width;
        let lin = |b: &mut Bundle, n: &str, out: usize, inp: usize| -> io::Result<Linear> {
            Ok(Linear {
                w: b.take(&format!("{n}.weight"), &[out, inp])?,
                b: b.take(&format!("{n}.bias"), &[out])?,
                out,
                inp,
            })
        };
        let s_board = lin(&mut b, "stem.board", arch.board, BOARD_W)?;
        let s_player = lin(&mut b, "stem.player", arch.player, PLAYER_W)?;
        let s_bslot = lin(&mut b, "stem.bslot", arch.bslot, BSLOT_W)?;
        let s_mslot = lin(&mut b, "stem.mslot", arch.mslot, MSLOT_W)?;
        let s_global = lin(&mut b, "stem.global", arch.globals, GLOBAL_W + RESERVED_W)?;
        let fuse = lin(&mut b, "fuse", w, arch.fuse())?;
        let fuse_norm = Norm {
            g: b.take("fuse.norm.weight", &[w])?,
            b: b.take("fuse.norm.bias", &[w])?,
        };
        let mut blocks = Vec::with_capacity(arch.blocks);
        for i in 0..arch.blocks {
            blocks.push(Block {
                n1: Norm {
                    g: b.take(&format!("trunk.{i}.norm1.weight"), &[w])?,
                    b: b.take(&format!("trunk.{i}.norm1.bias"), &[w])?,
                },
                fc1: b.take(&format!("trunk.{i}.fc1.weight"), &[w, w])?,
                n2: Norm {
                    g: b.take(&format!("trunk.{i}.norm2.weight"), &[w])?,
                    b: b.take(&format!("trunk.{i}.norm2.bias"), &[w])?,
                },
                fc2: b.take(&format!("trunk.{i}.fc2.weight"), &[w, w])?,
            });
        }
        let out_norm = Norm {
            g: b.take("trunk.norm.weight", &[w])?,
            b: b.take("trunk.norm.bias", &[w])?,
        };
        let vh = arch.value_hidden;
        let net = Net {
            arch,
            solo: AtomicU64::new(0),
            gemm: default_gemm(),
            label: label.to_string(),
            s_board,
            s_player,
            s_bslot,
            s_mslot,
            s_global,
            fuse,
            fuse_norm,
            blocks,
            out_norm,
            v_fc: lin(&mut b, "value.fc", vh, w)?,
            v_rel: lin(&mut b, "value.rel", N_PLAYERS, vh)?,
            v_score: lin(&mut b, "value.score", N_PLAYERS, vh)?,
            v_rank: lin(&mut b, "value.rank", N_PLAYERS * N_PLAYERS, vh)?,
            v_decomp: lin(&mut b, "value.decomp", N_PLAYERS * N_DECOMP, vh)?,
            h_beg: lin(&mut b, "policy.beg", 4, w)?,
            h_mode: lin(&mut b, "policy.mode", 3, w)?,
            h_place: lin(&mut b, "policy.place", encode::N_PLACE, w)?,
            h_extra: lin(&mut b, "policy.extra_day", 2, w)?,
            h_who: lin(&mut b, "policy.who", N_WHO, w)?,
            q_fast: lin(&mut b, "ptr.q_fast", D_CHOICE, w)?,
            q_deep: lin(&mut b, "ptr.q_deep", arch.key, w)?,
            cell_emb: b.take("ptr.cell_emb", &[N_WHO, arch.key])?,
            step_emb: b.take("ptr.step_emb", &[Phase::COUNT, arch.key])?,
            key1: lin(&mut b, "ptr.key1", arch.key, D_CHOICE)?,
            key2: lin(&mut b, "ptr.key2", arch.key, arch.key)?,
        };
        Ok(net)
    }

    // ---- forward -------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn dense(&self, l: &Linear, m: usize, x: &[f32], ldx: usize, y: &mut [f32], ldy: usize) {
        self.gemm
            .sgemm_nt(m, l.out, l.inp, x, ldx, &l.w, l.inp, 0.0, y, ldy);
        add_bias(y, ldy, m, l.out, &l.b);
    }

    /// Stems, fusion, six residual blocks. Leaves `h` as `[b, width]`.
    fn trunk(&self, b: usize, x: &[f32], s: &mut Scratch) {
        let a = self.arch;
        let f = a.fuse();
        s.fuse.clear();
        s.fuse.resize(b * f, 0.0);

        let mut col = 0usize;
        self.gemm.sgemm_nt(
            b,
            a.board,
            BOARD_W,
            &x[BOARD_OFF..],
            D_IN,
            &self.s_board.w,
            BOARD_W,
            0.0,
            &mut s.fuse[col..],
            f,
        );
        add_bias(&mut s.fuse[col..], f, b, a.board, &self.s_board.b);
        col += a.board;

        // The player stem is applied four times with tied weights: "how to read
        // a player's holdings" is then learned from 4x the data. Sharing over
        // the axis that *is* exchangeable, and not over board positions, is
        // this domain's real analogue of a convolution.
        for seat in 0..N_PLAYERS {
            self.gemm.sgemm_nt(
                b,
                a.player,
                PLAYER_W,
                &x[PLAYERS_OFF + seat * PLAYER_W..],
                D_IN,
                &self.s_player.w,
                PLAYER_W,
                0.0,
                &mut s.fuse[col..],
                f,
            );
            add_bias(&mut s.fuse[col..], f, b, a.player, &self.s_player.b);
            col += a.player;
        }
        for slot in 0..N_DISPLAY {
            self.gemm.sgemm_nt(
                b,
                a.bslot,
                BSLOT_W,
                &x[BSLOTS_OFF + slot * BSLOT_W..],
                D_IN,
                &self.s_bslot.w,
                BSLOT_W,
                0.0,
                &mut s.fuse[col..],
                f,
            );
            add_bias(&mut s.fuse[col..], f, b, a.bslot, &self.s_bslot.b);
            col += a.bslot;
        }
        for slot in 0..N_DISPLAY {
            self.gemm.sgemm_nt(
                b,
                a.mslot,
                MSLOT_W,
                &x[MSLOTS_OFF + slot * MSLOT_W..],
                D_IN,
                &self.s_mslot.w,
                MSLOT_W,
                0.0,
                &mut s.fuse[col..],
                f,
            );
            add_bias(&mut s.fuse[col..], f, b, a.mslot, &self.s_mslot.b);
            col += a.mslot;
        }
        // Global and reserved are one weight matrix over two disjoint column
        // ranges of the input, accumulated with beta=1 rather than copied.
        let gin = GLOBAL_W + RESERVED_W;
        self.gemm.sgemm_nt(
            b,
            a.globals,
            GLOBAL_W,
            &x[GLOBAL_OFF..],
            D_IN,
            &self.s_global.w,
            gin,
            0.0,
            &mut s.fuse[col..],
            f,
        );
        self.gemm.sgemm_nt(
            b,
            a.globals,
            RESERVED_W,
            &x[RESERVED_OFF..],
            D_IN,
            &self.s_global.w[GLOBAL_W..],
            gin,
            1.0,
            &mut s.fuse[col..],
            f,
        );
        add_bias(&mut s.fuse[col..], f, b, a.globals, &self.s_global.b);
        debug_assert_eq!(col + a.globals, f);

        let w = a.width;
        s.h.clear();
        s.h.resize(b * w, 0.0);
        s.t1.clear();
        s.t1.resize(b * w, 0.0);
        s.t2.clear();
        s.t2.resize(b * w, 0.0);

        self.dense(&self.fuse, b, &s.fuse, f, &mut s.h, w);
        layer_norm(&mut s.h, b, w, &self.fuse_norm);
        gelu(&mut s.h);

        for blk in &self.blocks {
            s.t1.copy_from_slice(&s.h);
            layer_norm(&mut s.t1, b, w, &blk.n1);
            self.gemm
                .sgemm_nt(b, w, w, &s.t1, w, &blk.fc1, w, 0.0, &mut s.t2, w);
            gelu(&mut s.t2);
            layer_norm(&mut s.t2, b, w, &blk.n2);
            self.gemm
                .sgemm_nt(b, w, w, &s.t2, w, &blk.fc2, w, 0.0, &mut s.t1, w);
            for i in 0..b * w {
                s.h[i] += s.t1[i];
            }
        }
        layer_norm(&mut s.h, b, w, &self.out_norm);
    }

    fn value_of(&self, b: usize, s: &mut Scratch) {
        let a = self.arch;
        let vh = a.value_hidden;
        s.vh.clear();
        s.vh.resize(b * vh, 0.0);
        self.dense(&self.v_fc, b, &s.h, a.width, &mut s.vh, vh);
        gelu(&mut s.vh);
        for (l, dst, n) in [
            (&self.v_rel, &mut s.rel, N_PLAYERS),
            (&self.v_score, &mut s.score, N_PLAYERS),
            (&self.v_rank, &mut s.rank, N_PLAYERS * N_PLAYERS),
            (&self.v_decomp, &mut s.decomp, N_PLAYERS * N_DECOMP),
        ] {
            dst.clear();
            dst.resize(b * n, 0.0);
            self.gemm
                .sgemm_nt(b, n, vh, &s.vh, vh, &l.w, vh, 0.0, dst, n);
            add_bias(dst, n, b, n, &l.b);
        }
    }

    // ---- the public entry points ---------------------------------------

    /// One trunk pass over `qs.len()` positions, results in the same order.
    ///
    /// **Everything here is batch-shaped, including the pointer head.** The two
    /// stages of the candidate scorer run as one GEMM each over every surviving
    /// candidate in the batch, not two GEMMs per node. That matters more than it
    /// sounds like it should: with every node a `Take`, the per-node form issues
    /// `2 * batch` small GEMM calls against a ~1.3 µs call floor, which at batch
    /// 128 is 340 µs of pure call overhead — a sixth of the whole pass — for
    /// ~1.8 M multiply-adds that the batched form gets for almost nothing.
    ///
    /// A search that calls this one leaf at a time runs at roughly a tenth of
    /// the batched rate. That is what [`BatchedEvaluator`] exists to prevent;
    /// [`Unbatched`] is how you say you meant it.
    pub fn evaluate_batch(&self, qs: &[Query<'_>], out: &mut Vec<Evaluation>) {
        out.clear();
        if qs.is_empty() {
            return;
        }
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            let s = &mut *s;
            let b = qs.len();
            s.x.clear();
            s.x.resize(b * D_IN, 0.0);
            s.movers.clear();
            for (i, q) in qs.iter().enumerate() {
                let mover = q.phase.mover(q.turn);
                s.movers.push(mover);
                encode::encode(q.state, mover, q.phase, &mut s.x[i * D_IN..(i + 1) * D_IN]);
            }

            // `x` is moved out for the duration so the trunk can hold an
            // immutable view of it while writing the other scratch buffers.
            let x = std::mem::take(&mut s.x);
            self.trunk(b, &x, s);
            self.value_of(b, s);
            s.x = x;

            // Both pointer queries are `W h`, so they are two GEMMs over the
            // batch rather than 224 dot products per node. Cleared rather than
            // left stale when no node needs them, so the length check in `plan`
            // is exact instead of accidentally satisfied by a previous batch.
            s.qf.clear();
            s.qd.clear();
            if qs
                .iter()
                .any(|q| matches!(q.phase, Phase::Take { .. } | Phase::DraftTile { .. }))
            {
                self.queries(b, s);
            }

            self.plan(qs, s);
            self.keys(s);

            for (i, q) in qs.iter().enumerate() {
                let value = self.rotate_value(&s.rel[i * N_PLAYERS..], s.movers[i]);
                let plan = std::mem::take(&mut s.plans[i]);
                let priors = self.finish(q.n_edges, plan, s);
                out.push(Evaluation { priors, value });
            }
        });
    }

    /// The full value structure for one position, including the auxiliary
    /// heads. Used by reporting and by the arena, not by the search.
    pub fn value_full(&self, g: &GameState, p: PlayerId, phase: Phase) -> Value {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            let s = &mut *s;
            s.x.clear();
            s.x.resize(D_IN, 0.0);
            encode::encode(g, p, phase, &mut s.x);
            let x = std::mem::take(&mut s.x);
            self.trunk(1, &x, s);
            self.value_of(1, s);
            s.x = x;

            let mut rel: [f32; N_PLAYERS] = std::array::from_fn(|i| tanh_fast(s.rel[i]));
            let mean = rel.iter().sum::<f32>() / N_PLAYERS as f32;
            for v in &mut rel {
                *v -= mean;
            }
            let mut rank = [[0f32; N_PLAYERS]; N_PLAYERS];
            for i in 0..N_PLAYERS {
                let row = &s.rank[i * N_PLAYERS..(i + 1) * N_PLAYERS];
                let mut r = [0f32; N_PLAYERS];
                r.copy_from_slice(row);
                softmax(&mut r);
                rank[i] = r;
            }
            Value {
                rel,
                score: std::array::from_fn(|i| s.score[i]),
                rank,
                decomp: std::array::from_fn(|i| {
                    std::array::from_fn(|c| s.decomp[i * N_DECOMP + c])
                }),
            }
        })
    }

    /// `rel` comes out in perspective order; the search wants seat order.
    ///
    /// The mean is subtracted here rather than constrained during training:
    /// it is one subtraction and it enforces the zero-sum invariant a max^n
    /// backup relies on without distorting the optimisation.
    fn rotate_value(&self, rel: &[f32], mover: PlayerId) -> [f32; N_PLAYERS] {
        let mut v: [f32; N_PLAYERS] = std::array::from_fn(|k| tanh_fast(rel[k]));
        let mean = v.iter().sum::<f32>() / N_PLAYERS as f32;
        for x in &mut v {
            *x -= mean;
        }
        std::array::from_fn(|seat| v[encode::seat_off(mover, PlayerId(seat as u8))])
    }

    /// The two pointer query projections, batched: `q_fast` is the stage-1
    /// query and `q_deep` the stage-2 one, and both are `W h`.
    fn queries(&self, b: usize, s: &mut Scratch) {
        let k = self.arch.key;
        s.qf.resize(b * D_CHOICE, 0.0);
        s.qd.resize(b * k, 0.0);
        self.dense(&self.q_fast, b, &s.h, self.arch.width, &mut s.qf, D_CHOICE);
        self.gemm.sgemm_nt(
            b,
            k,
            self.arch.width,
            &s.h,
            self.arch.width,
            &self.q_deep.w,
            self.arch.width,
            0.0,
            &mut s.qd,
            k,
        );
        add_bias(&mut s.qd, k, b, k, &self.q_deep.b);
    }

    /// Pass A: decide what each node needs, and append every pointer node's
    /// candidate rows to one batch-wide matrix.
    fn plan(&self, qs: &[Query<'_>], s: &mut Scratch) {
        let w = self.arch.width;
        let k = self.arch.key;
        s.feats.clear();
        s.keep.clear();
        s.kin.clear();
        s.plans.clear();
        s.plans.reserve(qs.len());

        for (i, q) in qs.iter().enumerate() {
            let n = q.n_edges;
            if n == 0 {
                s.plans.push(Plan::Done(Vec::new()));
                continue;
            }
            if n == 1 {
                s.plans.push(Plan::Done(vec![1.0]));
                continue;
            }
            let mover = s.movers[i];
            let ptr_phase = matches!(q.phase, Phase::Take { .. } | Phase::DraftTile { .. });
            let spec = match (q.steps, q.candidates) {
                // The edges themselves: no re-derivation, and no way for the
                // prior to line up against a differently ordered list.
                (Some(st), _) => encode::edges_for(q.state, mover, q.phase, st, &mut s.feats),
                // The caller already built the candidate list; reuse it rather
                // than paying `choices_for_worker` a second time.
                (None, Some(cs)) if ptr_phase => {
                    let off = s.feats.len();
                    s.feats.resize(off + n * D_CHOICE, 0.0);
                    for j in 0..n {
                        if let Some(c) = cs.get(j) {
                            let r = off + j * D_CHOICE;
                            encode::encode_choice(q.state, mover, c, &mut s.feats[r..r + D_CHOICE]);
                        }
                    }
                    EdgeSpec::Pointer { off, n }
                }
                _ => encode::edges(q.state, mover, q.phase, n, &mut s.feats),
            };

            let plan = match spec {
                EdgeSpec::Uniform => Plan::Done(vec![1.0 / n as f32; n]),
                EdgeSpec::Fixed { head, idx } => {
                    let l = match head {
                        Head::Beg => &self.h_beg,
                        Head::Mode => &self.h_mode,
                        Head::Place => &self.h_place,
                        Head::Who => &self.h_who,
                        Head::ExtraDay => &self.h_extra,
                    };
                    debug_assert_eq!(idx.len(), n);
                    let h = &s.h[i * w..(i + 1) * w];
                    // Masked softmax: only the slots an edge points at are ever
                    // read, so an illegal move contributes no probability and,
                    // at training time, no gradient. Masked dots beat a batched
                    // GEMM here — `Head::Who` has `N_WHO` rows (one per board
                    // cell plus STOP) and a node names at most 7 of them.
                    let mut logits: Vec<f32> = idx
                        .iter()
                        .map(|&j| {
                            let j = j as usize;
                            l.b[j] + dot(h, &l.w[j * l.inp..(j + 1) * l.inp])
                        })
                        .collect();
                    softmax(&mut logits);
                    Plan::Done(logits)
                }
                EdgeSpec::Pointer { off, n } => {
                    // The batched query projections are computed only when the
                    // batch contains a pointer node; a phase that reaches here
                    // without them is a wiring bug, and a uniform prior is a far
                    // better failure than an out-of-bounds index inside a search
                    // thread.
                    debug_assert!(s.qf.len() >= (i + 1) * D_CHOICE && s.qd.len() >= (i + 1) * k);
                    if s.qf.len() < (i + 1) * D_CHOICE || s.qd.len() < (i + 1) * k {
                        Plan::Done(vec![1.0 / n as f32; n])
                    } else {
                        // `Take` keys its query on the cell being resolved; every
                        // other pointer node uses the last slot, which is
                        // `STOP`'s row and doubles as "no cell".
                        let cell = match q.phase {
                            Phase::Take { worker } => q
                                .state
                                .loc(worker)
                                .on_board()
                                .map(|(gr, ps)| encode::cell(gr, ps))
                                .unwrap_or(N_WHO - 1),
                            _ => N_WHO - 1,
                        };
                        self.select(i, off, n, cell, q.phase, s)
                    }
                }
            };
            s.plans.push(plan);
        }
    }

    /// Pass B: stage 1, top-`TOP_K` selection, and the copy of the survivors
    /// into the batch-wide key input.
    ///
    /// Stage 1 is 96 multiply-adds per candidate against the raw delta features,
    /// which is free even at 500 candidates. Stage 2 runs the key MLP only on
    /// the top 64. Both are trained with the same cross-entropy, so stage 1
    /// learns to be a cheap approximation of stage 2 rather than a hand-written
    /// heuristic that rots when the rules move.
    fn select(
        &self,
        row: usize,
        off: usize,
        n: usize,
        cell: usize,
        phase: Phase,
        s: &mut Scratch,
    ) -> Plan {
        s.fast.clear();
        {
            let qf = &s.qf[row * D_CHOICE..(row + 1) * D_CHOICE];
            for j in 0..n {
                let r = off + j * D_CHOICE;
                s.fast.push(dot(qf, &s.feats[r..r + D_CHOICE]));
            }
        }

        let k0 = s.keep.len();
        s.keep.extend(0..n as u32);
        if n > TOP_K {
            {
                let fast = &s.fast;
                s.keep[k0..].sort_unstable_by(|&a, &b| fast[b as usize].total_cmp(&fast[a as usize]));
            }
            s.keep.truncate(k0 + TOP_K);
            // Back into candidate order, so the scatter in `finish` and the row
            // order in the key matrix agree.
            s.keep[k0..].sort_unstable();
        }

        let kin = s.kin.len() / D_CHOICE;
        for j in k0..s.keep.len() {
            let r = off + s.keep[j] as usize * D_CHOICE;
            s.kin.extend_from_slice(&s.feats[r..r + D_CHOICE]);
        }

        Plan::Ptr {
            row,
            n,
            keep: (k0, s.keep.len()),
            kin,
            cell,
            phase,
        }
    }

    /// Pass C: the key MLP, once for every surviving candidate in the batch.
    ///
    /// This is the whole point of the three-pass shape. Per node it is two
    /// GEMMs of at most 64 rows; batched it is two GEMMs of up to `64 * batch`.
    fn keys(&self, s: &mut Scratch) {
        let k = self.arch.key;
        let m = s.kin.len() / D_CHOICE;
        s.k1.clear();
        s.k2.clear();
        if m == 0 {
            return;
        }
        s.k1.resize(m * k, 0.0);
        s.k2.resize(m * k, 0.0);
        self.dense(&self.key1, m, &s.kin, D_CHOICE, &mut s.k1, k);
        gelu(&mut s.k1);
        self.dense(&self.key2, m, &s.k1, k, &mut s.k2, k);
    }

    /// Pass D: the per-node query dot, softmax and scatter.
    fn finish(&self, n_edges: usize, plan: Plan, s: &mut Scratch) -> Vec<f32> {
        let (row, n, keep, kin, cell, phase) = match plan {
            Plan::Done(v) => return v,
            Plan::Ptr {
                row,
                n,
                keep,
                kin,
                cell,
                phase,
            } => (row, n, keep, kin, cell, phase),
        };
        debug_assert_eq!(n, n_edges);
        let k = self.arch.key;

        // The cell a `Take` is resolving, and which phase asked, are additive
        // embeddings on the query rather than a concatenated context. That is
        // what makes the `DraftTile` head the same machinery plus one embedding
        // row instead of a second pointer head: §3b's "~1 k parameters".
        s.qt.clear();
        let step = phase.tag() as usize * k;
        for j in 0..k {
            s.qt
                .push(s.qd[row * k + j] + self.cell_emb[cell * k + j] + self.step_emb[step + j]);
        }

        let m = keep.1 - keep.0;
        s.deep.clear();
        for r in 0..m {
            let o = (kin + r) * k;
            s.deep.push(dot(&s.qt, &s.k2[o..o + k]));
        }
        softmax(&mut s.deep);

        let mut out = vec![0.0f32; n];
        if m == n {
            for (r, &j) in s.keep[keep.0..keep.1].iter().enumerate() {
                out[j as usize] = s.deep[r];
            }
        } else {
            let tail = TAIL_MASS / (n - m) as f32;
            out.fill(tail);
            for (r, &j) in s.keep[keep.0..keep.1].iter().enumerate() {
                out[j as usize] = s.deep[r] * (1.0 - TAIL_MASS);
            }
        }
        out
    }
}

/// What still has to happen to turn one node's trunk output into priors.
///
/// Splitting the pointer head into "decide, select, one GEMM, finish" is what
/// makes it batchable. Everything cheap enough not to care — uniform priors, a
/// masked softmax over a fixed-arity head — is finished in pass A and arrives
/// here already `Done`.
enum Plan {
    /// Nothing left to do; these are the priors.
    Done(Vec<f32>),
    /// A pointer node waiting on the batch's stage-2 keys.
    Ptr {
        /// This node's row in the trunk output, for the stage-2 query.
        row: usize,
        /// Edge count, which for a pointer node is the candidate count.
        n: usize,
        /// Half-open range into `Scratch::keep`: the candidates that survived
        /// stage 1, in increasing candidate order.
        keep: (usize, usize),
        /// First row this node occupies in the batch-wide key matrix.
        kin: usize,
        cell: usize,
        phase: Phase,
    },
}

impl Default for Plan {
    fn default() -> Self {
        Plan::Done(Vec::new())
    }
}

// ===========================================================================
// Reaching the network through `phase::Evaluator`
// ===========================================================================
//
// Three ways in, and the difference between them is worth about 9x.
//
// * [`Unbatched`] — the network with nothing in front of it. Batches whatever
//   one caller hands to `evaluate_many`, and cannot assemble anything wider.
// * [`BatchedEvaluator`] + [`BatchHandle`] — a queue and one or two batcher
//   threads. N search threads block inside `evaluate`; the batcher drains them
//   into one `evaluate_batch`. This is `docs/COMPUTE.md` §2.6's design and
//   KataGo's; it changes nothing in `mcts.rs`, needs no futures, and needs no
//   virtual loss, because the batch is assembled *across games* rather than
//   within one tree (§2.5), so each tree's search is bit-for-bit what it would
//   have been alone.
// * `Net` itself, which implements `Evaluator` directly. Safe now that
//   `phase::Evaluator` names `evaluate_many` as the method to implement.
//
// The one thing none of them can do is manufacture concurrency. A single
// `Mcts` descends one path at a time, so `Mcts::new(net, cfg)` runs at batch 1
// however good the impl is; the concurrency has to come from the *game* count.
// `COMPUTE.md` §2.6 note 1: size the game pool by memory, not by cores, and
// keep the batcher off the worker pool (note 2) or it deadlocks.

/// A `Net` with no queue in front of it.
///
/// It batches whatever one caller hands it in a single `evaluate_many`, and
/// nothing more. **The batch it cannot assemble is the one that matters**: a
/// single MCTS descent produces one leaf at a time whatever the trait
/// signature, and `docs/COMPUTE.md` §0 measured the same forward pass at 5,491
/// evaluations/s at batch 1 against 111,469/s at batch 256, on one core, with
/// no GPU. On this file as it now stands the gap is 8-9x.
///
/// So use this where there genuinely is only one position in flight: unit
/// tests, `LEARNING.md` §6.8's anchor panel, and a human at a board. For
/// anything that runs more than a few thousand evaluations, put a
/// [`BatchedEvaluator`] in front and hand each search thread a
/// [`BatchHandle`] — that is `COMPUTE.md` §2.6's design, it needs no change in
/// `mcts.rs`, and because the batch is assembled across *games* rather than
/// within one tree it costs no search quality (§2.5).
///
/// Generic over the pointer, so `Unbatched(net)` and `Unbatched(arc)` both work.
pub struct Unbatched<N = Net>(pub N);

impl<N: Borrow<Net>> Unbatched<N> {
    pub fn net(&self) -> &Net {
        self.0.borrow()
    }

    pub fn into_inner(self) -> N {
        self.0
    }
}

impl<N: Borrow<Net> + Send + Sync> Evaluator for Unbatched<N> {
    fn evaluate_many(&self, queries: &[crate::phase::Query<'_>]) -> Vec<Evaluation> {
        self.0.borrow().evaluate_many(queries)
    }

    fn evaluate(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation {
        self.0.borrow().evaluate_one(state, phase, turn, n_edges)
    }

    fn name(&self) -> String {
        let n = self.0.borrow();
        format!("net[{}/{}]:unqueued", n.label, n.gemm.name())
    }
}

/// The network itself, as an `Evaluator`.
///
/// Safe to hold directly now that `phase::Evaluator` names `evaluate_many` as
/// the method to implement: a caller that hands over a batch gets a batch, and
/// one that hands over a single node gets the honest cost of a single node,
/// counted by [`Net::unbatched_calls`].
///
/// What this still cannot do is *create* concurrency. A single `Mcts` descends
/// one path at a time, so `Mcts::new(net, cfg)` runs at batch 1 no matter how
/// good this impl is. [`BatchedEvaluator`] is what turns many such searches
/// into full batches.
impl Evaluator for Net {
    fn evaluate_many(&self, queries: &[crate::phase::Query<'_>]) -> Vec<Evaluation> {
        let qs: Vec<Query> = queries.iter().map(Query::from).collect();
        let mut out = Vec::with_capacity(qs.len());
        self.evaluate_batch(&qs, &mut out);
        out
    }

    fn evaluate(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation {
        self.evaluate_one(state, phase, turn, n_edges)
    }

    fn name(&self) -> String {
        format!("net[{}/{}]", self.label, self.gemm.name())
    }
}

/// How the batcher trades latency for batch size.
#[derive(Clone, Copy, Debug)]
pub struct BatchConfig {
    /// Largest batch handed to one `evaluate_batch` call.
    ///
    /// `COMPUTE.md` §1.4: the GEMM itself saturates at **32**, and everything
    /// above that buys amortised *fixed* cost — the queue handoff, the
    /// elementwise tiling, and on a GPU the kernel launch. 128 is the measured
    /// whole-machine optimum (§2.2) and is insurance against a half-full batch,
    /// not a GEMM-efficiency argument.
    pub max_batch: usize,
    /// How long a batcher lingers for a partial batch before giving up and
    /// running what it has. Zero means never wait.
    pub max_wait: Duration,
    /// Batcher threads. `COMPUTE.md` §1.4 measures GEMM throughput on this
    /// machine as fully claimed by **two** threads, so more than two is only
    /// useful where the elementwise half dominates.
    pub threads: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            max_batch: 128,
            max_wait: Duration::from_micros(200),
            threads: 1,
        }
    }
}

impl BatchConfig {
    /// Sized for `games` concurrent self-play games.
    ///
    /// The batch comes from the number of games in flight, not the core count:
    /// each game thread has exactly one descent outstanding, so the ceiling on
    /// the batch *is* the game count. `COMPUTE.md` §2.6's note 1 — size the
    /// game pool by memory, not by cores, and do not run it on a rayon pool
    /// whose width is `available_parallelism()`.
    pub fn for_games(games: usize) -> Self {
        BatchConfig {
            max_batch: games.clamp(1, 256),
            ..BatchConfig::default()
        }
    }
}

/// What the batcher actually saw.
///
/// `COMPUTE.md` §2.6: *"if the mean batch size is not close to the configured
/// maximum, none of §5's numbers are happening, and it is the only symptom you
/// will get."* So this is not optional instrumentation — it is the only way to
/// tell a working pipeline from one that is quietly running at batch 3.
#[derive(Clone, Copy, Debug, Default)]
pub struct BatchStats {
    pub batches: u64,
    pub evals: u64,
    /// `size[j]` counts batches of size in `[2^j, 2^(j+1))`, so `size[0]` is
    /// batch 1 — the number to watch.
    pub size: [u64; 12],
    /// `wait[j]` counts requests that waited in `[2^j, 2^(j+1))` µs.
    pub wait: [u64; 12],
    pub wait_ns: u64,
    pub max_seen: usize,
    /// Requests served inline because the pool had already shut down.
    pub bypassed: u64,
}

fn bucket(v: u64) -> usize {
    (64 - v.max(1).leading_zeros() as usize - 1).min(11)
}

impl BatchStats {
    pub fn mean_batch(&self) -> f64 {
        if self.batches == 0 {
            0.0
        } else {
            self.evals as f64 / self.batches as f64
        }
    }

    pub fn mean_wait_us(&self) -> f64 {
        if self.evals == 0 {
            0.0
        } else {
            self.wait_ns as f64 / self.evals as f64 / 1000.0
        }
    }

    /// One line per histogram, for a log at the end of a generation.
    pub fn report(&self) -> String {
        let hist = |h: &[u64; 12]| -> String {
            h.iter()
                .enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(j, &c)| format!("{}:{}", 1u64 << j, c))
                .collect::<Vec<_>>()
                .join(" ")
        };
        format!(
            "batches {} evals {} mean {:.1} (max {}) wait {:.0} us bypassed {}\n  size  {}\n  wait  {}",
            self.batches,
            self.evals,
            self.mean_batch(),
            self.max_seen,
            self.mean_wait_us(),
            self.bypassed,
            hist(&self.size),
            hist(&self.wait),
        )
    }
}

/// One queued request.
///
/// Owned, deliberately. `GameState` is `Copy` and 312 bytes; copying it costs
/// 0.005 µs (`COMPUTE.md` §1.2) against the ~9 µs the evaluation costs, and it
/// buys the queue a payload with no borrows in it and this file zero `unsafe`.
/// The edge list is cloned for the same reason, and pays for itself: without it
/// the encoder re-derives the edges at every `Take` node, which costs more and
/// cannot detect a right-length list in the wrong order.
struct Req {
    state: GameState,
    phase: Phase,
    turn: PlayerId,
    n_edges: usize,
    /// Empty when the caller only had a count.
    steps: Vec<Step>,
}

impl Req {
    fn query(&self) -> Query<'_> {
        if self.steps.is_empty() {
            Query::new(&self.state, self.phase, self.turn, self.n_edges)
        } else {
            Query::with_steps(&self.state, self.phase, self.turn, &self.steps)
        }
    }
}

struct Slot {
    /// Taken by the batcher when it claims the request.
    req: Option<Req>,
    out: Option<Evaluation>,
    filled: bool,
    /// A batcher has taken this into a batch and is committed to filling it.
    claimed: bool,
    queued: Instant,
}

#[derive(Default)]
struct Queue {
    slots: Vec<Option<Slot>>,
    free: Vec<usize>,
    ready: VecDeque<usize>,
    stop: bool,
    stats: BatchStats,
}

struct Shared {
    net: Arc<Net>,
    q: Mutex<Queue>,
    /// Batchers park here for work.
    work: Condvar,
    /// Requesters park here for their result. One `notify_all` per batch, not
    /// one signal per request: `COMPUTE.md` §2.6 note 4.
    done: Condvar,
    cfg: BatchConfig,
}

/// A queue and its batcher threads, in front of one [`Net`].
///
/// Own one of these per process and give every search thread a
/// [`handle`](BatchedEvaluator::handle). Dropping it drains the queue and joins
/// the batchers.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use tzolkin::net::{Net, Arch, BatchedEvaluator, BatchConfig};
/// let net = Arc::new(Net::random(Arch::MAIN, 0));
/// let pool = BatchedEvaluator::new(net, BatchConfig::for_games(256));
/// // one per game thread; each blocks inside `Evaluator::evaluate`
/// let ev = pool.handle();
/// ```
///
/// Two things will otherwise cost a day, both from `COMPUTE.md` §2.6:
///
/// 1. **The batchers are `std::thread`s, deliberately.** If the game workers
///    are rayon tasks and the batcher were queued behind them in the same pool,
///    every worker parking on a result would deadlock the pool.
/// 2. **Set `VECLIB_MAXIMUM_THREADS=1`** or Accelerate spawns its own pool and
///    fights the game threads for the same cores.
pub struct BatchedEvaluator {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// A cheap, cloneable way in. Give one to each game thread's `Mcts`.
///
/// Handles outlive the pool safely: once the pool is dropped the queue is
/// closed, and a handle then evaluates inline rather than parking forever. That
/// is slow, and `BatchStats::bypassed` counts it, so it shows up as a number
/// rather than as a mysterious slowdown.
#[derive(Clone)]
pub struct BatchHandle(Arc<Shared>);

impl BatchedEvaluator {
    pub fn new(net: Arc<Net>, cfg: BatchConfig) -> BatchedEvaluator {
        let cfg = BatchConfig {
            max_batch: cfg.max_batch.max(1),
            threads: cfg.threads.max(1),
            ..cfg
        };
        let shared = Arc::new(Shared {
            net,
            q: Mutex::new(Queue::default()),
            work: Condvar::new(),
            done: Condvar::new(),
            cfg,
        });
        let workers = (0..cfg.threads)
            .map(|i| {
                let sh = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("tzolkin-batcher-{i}"))
                    .spawn(move || batcher(sh))
                    .expect("spawn batcher")
            })
            .collect();
        BatchedEvaluator { shared, workers }
    }

    pub fn handle(&self) -> BatchHandle {
        BatchHandle(Arc::clone(&self.shared))
    }

    pub fn stats(&self) -> BatchStats {
        self.shared.q.lock().unwrap().stats
    }

    pub fn net(&self) -> &Arc<Net> {
        &self.shared.net
    }

    /// Close the queue, let the batchers finish what is already in it, and join
    /// them. Idempotent; `Drop` calls it.
    pub fn shutdown(&mut self) {
        {
            let mut g = self.shared.q.lock().unwrap();
            g.stop = true;
        }
        self.shared.work.notify_all();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

impl Drop for BatchedEvaluator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl BatchHandle {
    pub fn stats(&self) -> BatchStats {
        self.0.q.lock().unwrap().stats
    }
}

/// Longest a parked requester sleeps before re-checking its own slot.
///
/// The broadcast normally arrives in microseconds; this exists only so that a
/// lost wakeup, or a batcher that died before claiming the request, degrades
/// into 20 ms of latency instead of a hung self-play run.
const PARK_RECHECK: Duration = Duration::from_millis(20);

impl BatchHandle {
    /// Queue `reqs` and block until every one of them has an answer.
    ///
    /// Submitting the whole set before parking is the point: a caller with
    /// several nodes in hand contributes all of them to one batch instead of
    /// serialising on the first. Cross-*game* concurrency is still where most
    /// of the batch comes from — `COMPUTE.md` §2.5 — but this costs nothing and
    /// means `evaluate_many` is never worse than `evaluate` in a loop.
    fn submit(&self, reqs: Vec<Req>) -> Vec<Evaluation> {
        let sh = &*self.0;
        let n = reqs.len();
        let mut out: Vec<Option<Evaluation>> = (0..n).map(|_| None).collect();
        let mut mine: Vec<usize> = Vec::with_capacity(n);
        let now = Instant::now();

        let mut g = sh.q.lock().unwrap();
        if g.stop {
            g.stats.bypassed += n as u64;
            drop(g);
            return reqs.into_iter().map(|r| one(&sh.net, r)).collect();
        }
        for req in reqs {
            let i = match g.free.pop() {
                Some(i) => i,
                None => {
                    g.slots.push(None);
                    g.slots.len() - 1
                }
            };
            g.slots[i] = Some(Slot {
                req: Some(req),
                out: None,
                filled: false,
                claimed: false,
                queued: now,
            });
            g.ready.push_back(i);
            mine.push(i);
        }
        if n == 1 {
            sh.work.notify_one();
        } else {
            sh.work.notify_all();
        }

        let mut left = n;
        while left > 0 {
            let mut progressed = false;
            for (k, &i) in mine.iter().enumerate() {
                if out[k].is_some() {
                    continue;
                }
                let s = g.slots[i].as_ref().expect("slot vanished");
                if s.filled {
                    let slot = g.slots[i].take().expect("slot vanished");
                    g.free.push(i);
                    out[k] = Some(slot.out.unwrap_or_else(empty_eval));
                    left -= 1;
                    progressed = true;
                } else if g.stop && !s.claimed {
                    // Nobody is coming: the pool shut down before this was
                    // claimed. Take the request back and run it here.
                    let mut slot = g.slots[i].take().expect("slot vanished");
                    g.free.push(i);
                    g.stats.bypassed += 1;
                    let req = slot.req.take().expect("unclaimed slot with no request");
                    drop(g);
                    out[k] = Some(one(&sh.net, req));
                    left -= 1;
                    progressed = true;
                    g = sh.q.lock().unwrap();
                }
            }
            if left > 0 && !progressed {
                g = sh.done.wait_timeout(g, PARK_RECHECK).unwrap().0;
            }
        }
        drop(g);
        out.into_iter().map(|e| e.expect("unfilled slot")).collect()
    }
}

impl Evaluator for BatchHandle {
    fn evaluate_many(&self, queries: &[crate::phase::Query<'_>]) -> Vec<Evaluation> {
        if queries.is_empty() {
            return Vec::new();
        }
        self.submit(
            queries
                .iter()
                .map(|q| Req {
                    state: *q.state,
                    phase: q.phase,
                    turn: q.turn,
                    n_edges: q.edges.len(),
                    steps: q.edges.to_vec(),
                })
                .collect(),
        )
    }

    fn evaluate(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation {
        self.submit(vec![Req {
            state: *state,
            phase,
            turn,
            n_edges,
            steps: Vec::new(),
        }])
        .pop()
        .unwrap_or_else(empty_eval)
    }

    fn name(&self) -> String {
        let n = &self.0.net;
        format!(
            "net[{}/{}]:batched({})",
            n.label,
            n.gemm.name(),
            self.0.cfg.max_batch
        )
    }
}

/// The inline fallback. Named rather than inlined so the places that reach for
/// it are obvious.
fn one(net: &Net, r: Req) -> Evaluation {
    let mut out = Vec::with_capacity(1);
    net.evaluate_batch(&[r.query()], &mut out);
    out.pop().unwrap_or(Evaluation {
        priors: Vec::new(),
        value: [0.0; N_PLAYERS],
    })
}

fn empty_eval() -> Evaluation {
    Evaluation {
        priors: Vec::new(),
        value: [0.0; N_PLAYERS],
    }
}

fn batcher(sh: Arc<Shared>) {
    let cap = sh.cfg.max_batch;
    let mut taken: Vec<usize> = Vec::with_capacity(cap);
    let mut reqs: Vec<Req> = Vec::with_capacity(cap);
    let mut out: Vec<Evaluation> = Vec::with_capacity(cap);

    loop {
        taken.clear();
        reqs.clear();
        {
            let mut g = sh.q.lock().unwrap();
            while g.ready.is_empty() && !g.stop {
                g = sh.work.wait(g).unwrap();
            }
            if g.ready.is_empty() {
                break; // stopped and drained
            }
            // Linger for a fuller batch, but never past the deadline and never
            // once shutdown has started.
            if g.ready.len() < cap && !g.stop && !sh.cfg.max_wait.is_zero() {
                let deadline = Instant::now() + sh.cfg.max_wait;
                while g.ready.len() < cap && !g.stop {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        break;
                    };
                    let (ng, t) = sh.work.wait_timeout(g, left).unwrap();
                    g = ng;
                    if t.timed_out() {
                        break;
                    }
                }
            }

            let now = Instant::now();
            let n = g.ready.len().min(cap);
            let mut wait_ns = 0u64;
            for _ in 0..n {
                let i = g.ready.pop_front().expect("ready shrank");
                let w = {
                    let s = g.slots[i].as_mut().expect("slot vanished");
                    s.claimed = true;
                    reqs.push(s.req.take().expect("queued slot with no request"));
                    now.saturating_duration_since(s.queued).as_nanos() as u64
                };
                wait_ns += w;
                g.stats.wait[bucket(w / 1000)] += 1;
                taken.push(i);
            }
            g.stats.batches += 1;
            g.stats.evals += n as u64;
            g.stats.wait_ns += wait_ns;
            g.stats.size[bucket(n as u64)] += 1;
            g.stats.max_seen = g.stats.max_seen.max(n);
        }

        // The forward pass runs with the lock released, so game threads keep
        // filling the next batch while this one computes.
        let qs: Vec<Query> = reqs.iter().map(Req::query).collect();
        sh.net.evaluate_batch(&qs, &mut out);
        drop(qs);

        {
            let mut g = sh.q.lock().unwrap();
            for (j, &i) in taken.iter().enumerate() {
                let s = g.slots[i].as_mut().expect("slot vanished");
                s.out = out.get_mut(j).map(|e| {
                    std::mem::replace(
                        e,
                        Evaluation {
                            priors: Vec::new(),
                            value: [0.0; N_PLAYERS],
                        },
                    )
                });
                s.filled = true;
            }
        }
        sh.done.notify_all();
    }

    // Whatever the reason for leaving, nobody may be left parked.
    {
        let mut g = sh.q.lock().unwrap();
        g.stop = true;
    }
    sh.done.notify_all();
}

// ===========================================================================
// Kernels and scratch
// ===========================================================================

#[derive(Default)]
struct Scratch {
    x: Vec<f32>,
    fuse: Vec<f32>,
    h: Vec<f32>,
    t1: Vec<f32>,
    t2: Vec<f32>,
    vh: Vec<f32>,
    rel: Vec<f32>,
    score: Vec<f32>,
    rank: Vec<f32>,
    decomp: Vec<f32>,
    /// `phase.mover(turn)` per row: the perspective the encoding and the value
    /// vector are rotated to.
    movers: Vec<PlayerId>,
    /// Per-row plan, filled by pass A and consumed by pass D.
    plans: Vec<Plan>,
    qf: Vec<f32>,
    qd: Vec<f32>,
    /// Every candidate of every pointer node in the batch, `D_CHOICE` wide.
    /// `EdgeSpec::Pointer` hands back an offset into this rather than a `Vec`.
    feats: Vec<f32>,
    /// Stage-1 scores for the node currently being selected.
    fast: Vec<f32>,
    /// Concatenated top-`TOP_K` candidate indices, one run per pointer node.
    keep: Vec<u32>,
    /// The survivors' feature rows, gathered into one matrix for `keys`.
    kin: Vec<f32>,
    k1: Vec<f32>,
    k2: Vec<f32>,
    /// The stage-2 query for the node being finished: `q_deep` plus the cell
    /// and phase embeddings.
    qt: Vec<f32>,
    /// Stage-2 logits for the node being finished.
    deep: Vec<f32>,
}

// One set of buffers per inference thread. The doc's batching contract is one
// or two dedicated inference threads, so this is two allocations for the run
// rather than twenty per forward pass -- which at batch 1 is most of the cost.
thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Eight independent accumulators, then one tree reduction.
///
/// Float addition is not associative, so a naive `iter().sum()` is a single
/// dependency chain and the compiler is not allowed to vectorise it: 512
/// elements become 512 serialised 4-cycle adds. Splitting the chain by hand is
/// what turns the LayerNorms from the most expensive thing in the forward pass
/// into a rounding error. The tree order is fixed, so results are still bitwise
/// reproducible run to run.
const LANES: usize = 8;

#[inline]
fn hsum(a: [f32; LANES]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

#[inline]
fn sum_of(x: &[f32]) -> f32 {
    let mut a = [0f32; LANES];
    let ch = x.chunks_exact(LANES);
    let rem = ch.remainder();
    for c in ch {
        for l in 0..LANES {
            a[l] += c[l];
        }
    }
    let mut s = hsum(a);
    for &v in rem {
        s += v;
    }
    s
}

/// Sum of squared deviations from `mean`, eight lanes.
///
/// A second pass rather than `E[x²] - E[x]²`: the row was just read so it is in
/// L1, and the one-pass form loses precision exactly when the residual stream
/// develops a large mean, which is the case a trained net can reach and an
/// untrained one cannot. This also makes the numerics identical to
/// `nn.LayerNorm`, which matters because the two sides share weights.
#[inline]
fn dev_sq(x: &[f32], mean: f32) -> f32 {
    let mut q = [0f32; LANES];
    let ch = x.chunks_exact(LANES);
    let rem = ch.remainder();
    for c in ch {
        for l in 0..LANES {
            let d = c[l] - mean;
            q[l] += d * d;
        }
    }
    let mut t = hsum(q);
    for &v in rem {
        t += (v - mean) * (v - mean);
    }
    t
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; LANES];
    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..LANES {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = hsum(acc);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

fn add_bias(y: &mut [f32], ldy: usize, rows: usize, n: usize, b: &[f32]) {
    for r in 0..rows {
        let row = &mut y[r * ldy..r * ldy + n];
        for (v, &bb) in row.iter_mut().zip(b) {
            *v += bb;
        }
    }
}

/// PyTorch's `nn.LayerNorm` with the default `eps`: biased variance, affine.
fn layer_norm(x: &mut [f32], rows: usize, n: usize, w: &Norm) {
    const EPS: f32 = 1e-5;
    let inv_n = 1.0 / n as f32;
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        let mean = sum_of(row) * inv_n;
        let var = dev_sq(row, mean) * inv_n;
        let inv = 1.0 / (var + EPS).sqrt();
        for ((v, &g), &b) in row.iter_mut().zip(&w.g).zip(&w.b) {
            *v = (*v - mean) * inv * g + b;
        }
    }
}

/// The [7/6] Pade approximant of `tanh`, clamped.
///
/// `f32::tanh` is a libm call: it does not inline, does not vectorise, and at
/// ~1.7 ns each it was the second most expensive thing in the forward pass
/// after the LayerNorms. This is five times faster and agrees with `tanh` to
/// better than 1e-4 everywhere, with the worst case at |x| ~ 5 where `gelu(x)`
/// is already within 1e-4 of `x`. That is far inside f32 accumulation noise
/// over a 512-wide dot product.
#[inline]
fn tanh_fast(x: f32) -> f32 {
    let x2 = x * x;
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    (num / den).clamp(-1.0, 1.0)
}

/// The tanh form of GELU, which is `nn.GELU(approximate='tanh')` on the
/// training side. The exact erf form differs by ~1e-3, which would show up as a
/// systematic train/inference skew, so the two sides have to name the same
/// function; they do.
fn gelu(x: &mut [f32]) {
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    for v in x.iter_mut() {
        let t = *v;
        *v = 0.5 * t * (1.0 + tanh_fast(C * (t + 0.044715 * t * t * t)));
    }
}

fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        let u = 1.0 / x.len() as f32;
        x.fill(u);
        return;
    }
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    } else {
        let u = 1.0 / x.len() as f32;
        x.fill(u);
    }
}
