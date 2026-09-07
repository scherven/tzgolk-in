"""Export a trained checkpoint into whatever ``src/net.rs`` wants to read.

=======================================================================
THIS FILE IS THE WEIGHT-FORMAT SEAM. RECONCILE IT WITH src/net.rs.
=======================================================================

``src/net.rs`` is owned by another agent and did not exist when this was
written, so the exact tensor names and layout it expects are not yet known.
What *is* agreed, from ``LEARNING.md`` §5.1, is the container: **safetensors,
both directions** -- PyTorch writes it natively and the ``safetensors`` crate
reads it with no C dependency.

This script writes two things:

* ``<out>.safetensors`` (or ``<out>.tzw``, see below) -- the weights.
* ``<out>.json`` -- a manifest listing every tensor's name, shape and dtype,
  plus the architecture config. **Read this first when wiring up the loader.**
  If ``src/net.rs`` wants different names, change ``rename`` below; it is the
  only place names are decided.

Fallback container
------------------

If ``safetensors`` is not installed, this writes ``.tzw``: a 4-byte
little-endian JSON header length, that many bytes of UTF-8 JSON
(``{name: {dtype, shape, offset, nbytes}}``), then the tensors back to back as
little-endian ``f32``, each 64-byte aligned. That is about thirty lines to read
in Rust with no crate at all. Prefer safetensors; the fallback exists so that a
missing ``pip install`` never blocks a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch import N_WHO  # noqa: E402

# Must track `RULES_VERSION` in src/lib.rs. Written into every checkpoint so a
# net trained on one ruleset is identifiable after the fact.
RULES_VERSION = 1


def rename(k: str) -> str:
    """PyTorch parameter name -> the name ``src/net.rs`` looks for.

    The identity. `Net::load` refuses a file with a missing tensor, so the two
    sides must already agree; `train/check_manifest.py --check` is what proves
    they do.
    """
    return k


def metadata(cfg: dict, label: str) -> dict[str, str]:
    """The `__metadata__` block `Net::from_bundle` reads.

    Without it every dimension falls back to the `main` defaults, and a `small`
    checkpoint is refused with a shape mismatch that looks like a bug in the
    model rather than a missing header. `width`, `blocks` and the stem widths do
    not merely *describe* the file — they are what the reader builds the
    architecture from before checking a single shape.
    """
    keys = (
        "width", "blocks", "board", "player", "bslot", "mslot",
        "value_hidden", "key", "d_in", "d_choice",
    )
    meta = {k: str(cfg[k]) for k in keys if k in cfg}
    # `globals_` avoids the Python keyword; the reader calls it `globals`.
    if "globals_" in cfg:
        meta["globals"] = str(cfg["globals_"])
    meta.update(
        {
            "format": "tzolkin-net",
            "format_version": "1",
            "rules_version": str(RULES_VERSION),
            "n_who": str(N_WHO),
            "label": label,
        }
    )
    return meta


def write_tzw(path: str, tensors: dict[str, torch.Tensor]) -> None:
    """The dependency-free fallback container. See the module docstring."""
    meta, blobs, off = {}, [], 0
    for name, t in tensors.items():
        a = t.detach().contiguous().to(torch.float32).cpu().numpy()
        b = a.tobytes()
        pad = (-len(b)) % 64
        meta[name] = {
            "dtype": "f32",
            "shape": list(a.shape),
            "offset": off,
            "nbytes": len(b),
        }
        blobs.append(b + b"\0" * pad)
        off += len(b) + pad
    head = json.dumps(meta, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint base, e.g. ckpt/gen0001")
    ap.add_argument("--out", default=None, help="output path [<ckpt>.safetensors]")
    ap.add_argument("--force-tzw", action="store_true", help="use the fallback container")
    args = ap.parse_args()

    ck = torch.load(f"{args.ckpt}.pt", map_location="cpu", weights_only=False)
    tensors = {rename(k): v for k, v in ck["model"].items()}
    cfg = ck.get("config", {})

    out = args.out
    use_st = not args.force_tzw
    if use_st:
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("[warn] safetensors not installed; writing the .tzw fallback.")
            print("       pip install safetensors  -- it is small and pure Python + rust wheel.")
            use_st = False

    if out is None:
        out = f"{args.ckpt}.safetensors" if use_st else f"{args.ckpt}.tzw"

    if use_st:
        from safetensors.torch import save_file

        save_file(
            {k: v.contiguous() for k, v in tensors.items()},
            out,
            metadata=metadata(cfg, os.path.basename(args.ckpt)),
        )
    else:
        write_tzw(out, tensors)

    manifest = {
        "container": "safetensors" if use_st else "tzw",
        "config": cfg,
        "note": (
            "Tensor names come from train/export.py:rename(). If src/net.rs "
            "expects different names, change that function -- it is the only "
            "place names are decided."
        ),
        "tensors": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
            for k, v in tensors.items()
        },
    }
    with open(f"{os.path.splitext(out)[0]}.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n = sum(v.numel() for v in tensors.values())
    print(f"wrote {out}  ({len(tensors)} tensors, {n:,} parameters)")
    print(f"      {os.path.splitext(out)[0]}.json  (the manifest src/net.rs should read)")


if __name__ == "__main__":
    main()
