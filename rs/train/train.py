"""Train one generation from the replay buffer.

    python train/train.py --replay replay --out ckpt --gen 1 --steps 3000

CPU by default, and deliberately so. A 4.2M-parameter MLP at batch 1024 costs
~26 GFLOP per optimiser step, which is under 100 ms on this machine's CPU.
There is nothing to buy with a GPU here, and ``LEARNING.md`` §5.2 documents a
standing pattern of small nets that train fine on CPU and **silently fail to
learn on MPS with no error**. If you enable MPS anyway, diff its loss curve
against CPU for the first few hundred steps before trusting it.

Interruption
------------

Ctrl-C at any point saves the model, the optimiser moments and the step counter
and exits cleanly; ``--resume`` picks up from there. The optimiser moments are
not optional: a resume without them silently restarts Adam and loses a
generation's progress (``LEARNING.md`` §6.9). Checkpoints are also written every
``--checkpoint-every`` steps, so a crash costs at most that many.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import batch_arrays, d_in  # noqa: E402
from model import Config, Net, loss_fn  # noqa: E402
from replay import Buffer  # noqa: E402

STOP = False


def _on_sigint(sig, frame):
    """First Ctrl-C winds up; a second one is the usual hard kill."""
    global STOP
    if STOP:
        raise KeyboardInterrupt
    STOP = True
    print("\n[interrupt] finishing this step, then checkpointing...", flush=True)


# ----------------------------------------------------------------------
# learning rate
# ----------------------------------------------------------------------


def lr_for(generation: int) -> float:
    """LEARNING.md §6.6: constant with drops, not cosine.

    Cosine assumes a known horizon and the run will be stopped whenever it is
    stopped.
    """
    if generation < 60:
        return 2e-3
    if generation < 150:
        return 1e-3
    return 3e-4


def warmup(step: int, n: int = 200) -> float:
    """200-step linear warmup after every LR change and after the size swap."""
    return min(1.0, (step + 1) / n)


# ----------------------------------------------------------------------
# batching
# ----------------------------------------------------------------------


def make_batch(buf: Buffer, n: int, rng, augment: bool):
    """One training batch as tensors.

    All the actual work is `features.batch_arrays`, which is torch-free so that
    the encoding and the target construction can be smoke-tested on a machine
    with numpy and nothing else:

        python train/features.py replay
    """
    x, t = batch_arrays(buf.sample(n, rng), buf.schema, augment)
    tgt: dict = {}
    ptr = None
    for k, v in t.items():
        if k == "policy_ptr":
            # `sel` goes to the forward pass (which rows of the batch are
            # pointer nodes) rather than into the loss, so the pointer head runs
            # on ~18% of the batch instead of on a mostly-padded full one.
            sel, dist, weight, mask, cand, cell, tag = v
            ptr = (
                torch.from_numpy(sel),
                torch.from_numpy(cand),
                torch.from_numpy(mask),
                torch.from_numpy(cell),
                torch.from_numpy(tag),
            )
            tgt[k] = (
                torch.from_numpy(dist),
                torch.from_numpy(weight),
                torch.from_numpy(mask),
            )
        elif isinstance(v, tuple):
            sel, dist, weight, mask = v
            tgt[k] = (
                torch.from_numpy(sel),
                torch.from_numpy(dist),
                torch.from_numpy(weight),
                torch.from_numpy(mask),
            )
        else:
            tgt[k] = torch.from_numpy(v)
    return torch.from_numpy(x), tgt, ptr


# ----------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------


def save(path_base: str, net: Net, opt, step: int, gen: int, meta: dict):
    """Write model, optimiser moments and progress, temp-then-rename.

    Renaming over the target is what stops a Ctrl-C during the write from
    leaving a half a checkpoint that ``--resume`` would then load.
    """
    for name, obj in (
        (f"{path_base}.pt", {"model": net.state_dict(), "config": net.config_dict()}),
        (f"{path_base}.opt.pt", {"opt": opt.state_dict(), "step": step, "gen": gen}),
    ):
        tmp = name + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, name)
    tmp = f"{path_base}.json.tmp"
    with open(tmp, "w") as f:
        json.dump({"step": step, "generation": gen, **meta}, f, indent=2)
    os.replace(tmp, f"{path_base}.json")


def load_into(path_base: str, net: Net, opt) -> tuple[int, bool]:
    if not os.path.exists(f"{path_base}.pt"):
        return 0, False
    ck = torch.load(f"{path_base}.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    step = 0
    if os.path.exists(f"{path_base}.opt.pt"):
        o = torch.load(f"{path_base}.opt.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(o["opt"])
        step = o.get("step", 0)
    return step, True


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", default="replay", help="shard directory")
    ap.add_argument("--out", default="ckpt", help="checkpoint directory")
    ap.add_argument("--gen", type=int, default=0, help="generation number")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    # COMPUTE.md section 7: LEARNING.md section 6.6 sized this at 1.5 M records,
    # which is 768 MB and was very conservative even then. At the throughput the
    # batched self-play driver reaches, one generation overflows 1.5 M several
    # times over, so the loop would be discarding most of what it generates.
    # 8 M x 512 B is 4.1 GB, memory-mapped rather than resident.
    ap.add_argument("--window", type=int, default=8_000_000, help="max records in the buffer")
    ap.add_argument("--size", choices=["small", "main"], default="small")
    # A contiguous tail of whole games the optimiser never draws from, so
    # `valprobe --from` measures generalisation rather than memorisation.
    ap.add_argument("--holdout", type=float, default=0.02,
                    help="fraction of the newest records reserved for evaluation")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=None, help="override the §6.6 schedule")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--no-augment", action="store_true", help="disable 4x perspective augmentation")
    ap.add_argument("--init", default=None, help="warm-start from this checkpoint base")
    ap.add_argument("--resume", action="store_true", help="continue this generation")
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)
    torch.manual_seed(1234 + args.gen)
    rng = np.random.default_rng(1234 + args.gen)
    os.makedirs(args.out, exist_ok=True)

    if args.device != "cpu":
        print(
            f"[warn] --device {args.device}: LEARNING.md §5.2 documents small nets that "
            "train fine on CPU and silently fail to learn on MPS. Diff the loss curve "
            "against CPU before trusting this.",
            file=sys.stderr,
        )
    dev = torch.device(args.device)

    buf = Buffer.open(args.replay, max_records=args.window, holdout=args.holdout)
    print(buf.summary())
    print()

    cfg = Config.small() if args.size == "small" else Config.main()
    if cfg.d_in != d_in():
        raise SystemExit(f"encoder is {d_in()} wide, model expects {cfg.d_in}")
    net = Net(cfg).to(dev)
    print(f"net {args.size}: d_in {cfg.d_in}, width {cfg.width}, "
          f"{cfg.blocks} blocks, {net.n_params():,} parameters")

    # AdamW, not SGD+Nesterov: the latter needs LR tuning this project has no
    # time for. Decay excludes LayerNorm parameters and biases (§6.6).
    decay, no_decay = [], []
    for n_, p in net.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    base_lr = args.lr if args.lr is not None else lr_for(args.gen)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=base_lr,
    )

    base = os.path.join(args.out, f"gen{args.gen:04d}")
    start = 0
    if args.resume:
        start, ok = load_into(base, net, opt)
        print(f"resumed at step {start}" if ok else "nothing to resume from")
    elif args.init:
        ck = torch.load(f"{args.init}.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        print(f"initialised from {args.init}.pt")

    print(f"lr {base_lr:g}, batch {args.batch}"
          f"{' x4 perspective' if not args.no_augment else ''}, "
          f"{args.steps} steps, device {args.device}\n")

    t0 = time.time()
    running: dict[str, float] = {}
    step = start
    try:
        for step in range(start, args.steps):
            if STOP:
                break
            for g in opt.param_groups:
                g["lr"] = base_lr * warmup(step - start)

            x, tgt, ptr = make_batch(buf, args.batch, rng, augment=not args.no_augment)
            x = x.to(dev)
            tgt = {
                k: (v.to(dev) if torch.is_tensor(v) else tuple(t.to(dev) for t in v))
                for k, v in tgt.items()
            }

            if ptr is None:
                out = net(x)
            else:
                rows, cand, cmask, cell, ptag = (t.to(dev) for t in ptr)
                out = net(x, cand=cand, cand_mask=cmask, ptr_rows=rows, cell=cell, tag=ptag)
            total, parts = loss_fn(out, tgt)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()

            for k, v in parts.items():
                running[k] = 0.98 * running.get(k, v) + 0.02 * v
            if (step + 1) % args.log_every == 0:
                el = time.time() - t0
                comp = "  ".join(f"{k} {v:.4f}" for k, v in sorted(running.items()))
                print(
                    f"  step {step + 1:>5}/{args.steps}  loss {float(total.detach()):.4f}  "
                    f"{comp}  {(step + 1 - start) / max(el, 1e-9):.1f} steps/s",
                    flush=True,
                )
            if (step + 1) % args.checkpoint_every == 0:
                save(base, net, opt, step + 1, args.gen, {"loss": running})
    except KeyboardInterrupt:
        print("\n[interrupt] hard stop; the last periodic checkpoint stands.")
        raise SystemExit(130)

    save(base, net, opt, step + 1, args.gen, {"loss": running, "records": buf.n})
    print(f"\nsaved {base}.pt (+ .opt.pt, .json) at step {step + 1}")
    if STOP:
        print("interrupted; resume with --resume")
        raise SystemExit(130)

    print("\nnext:")
    print(f"  python train/export.py --ckpt {base} --out {base}.safetensors")
    print(f"  cargo run --release --bin arena -- --candidate {base}.safetensors \\")
    print(f"      --baseline heuristic:32 --games 600")


if __name__ == "__main__":
    main()
