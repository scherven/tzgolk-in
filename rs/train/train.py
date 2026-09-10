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


def make_batch(buf: Buffer, n: int, rng, augment: bool, held: bool = False,
               value_mix: float = 0.0):
    """One batch as tensors, from the training head or the held-out tail.

    All the actual work is `features.batch_arrays`, which is torch-free so that
    the encoding and the target construction can be smoke-tested on a machine
    with numpy and nothing else:

        python train/features.py replay
    """
    draw = buf.sample_holdout if held else buf.sample
    x, t = batch_arrays(draw(n, rng), buf.schema, augment, value_mix=value_mix)
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
# held-out loss
# ----------------------------------------------------------------------


def forward(net, x, ptr, dev):
    """One forward pass, routing the pointer rows if the batch has any."""
    if ptr is None:
        return net(x)
    rows, cand, cmask, cell, ptag = (t.to(dev) for t in ptr)
    return net(x, cand=cand, cand_mask=cmask, ptr_rows=rows, cell=cell, tag=ptag)


def to_dev(tgt, dev):
    return {
        k: (v.to(dev) if torch.is_tensor(v) else tuple(t.to(dev) for t in v))
        for k, v in tgt.items()
    }


def evaluate(net, buf: Buffer, dev, batch: int, batches: int, gen: int, augment: bool,
             value_mix: float = 0.0):
    """Mean loss on the held-out tail, by component.

    Two things make this comparable across steps rather than a fresh sample of
    noise each time. The rng is **re-seeded identically on every call**, so it
    is the same records every time and a change in the number is a change in
    the net; and `net.eval()` is entered even though `Config.dropout` is 0.0
    today, because a future non-zero dropout would otherwise turn this into a
    quietly different quantity.

    Returned separately from the training loss on purpose: the training loss
    cannot distinguish a net that is learning from a net that is memorising,
    and "is more data worth generating?" is exactly that question.
    """
    rng = np.random.default_rng(99_991 + gen)
    was_training = net.training
    net.eval()
    acc: dict[str, float] = {}
    n = 0
    try:
        with torch.no_grad():
            for _ in range(batches):
                x, tgt, ptr = make_batch(buf, batch, rng, augment=augment, held=True,
                                         value_mix=value_mix)
                out = forward(net, x.to(dev), ptr, dev)
                total, parts = loss_fn(out, to_dev(tgt, dev))
                parts["total"] = float(total.detach())
                for k, v in parts.items():
                    acc[k] = acc.get(k, 0.0) + v
                n += 1
    finally:
        if was_training:
            net.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


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
    # NOT the newest records, which is what this used to claim and what N53
    # cost: `Buffer.open` indexes the newest shard first, so the reserved tail
    # is the *oldest* games. `buf.summary()` prints the VALPROBE_FROM that
    # walks it.
    ap.add_argument("--holdout", type=float, default=0.02,
                    help="fraction of the buffer reserved for evaluation "
                         "(a contiguous tail of whole games; see buf.summary() "
                         "for which shard it lands in)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=None, help="override the §6.6 schedule")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--no-augment", action="store_true", help="disable 4x perspective augmentation")
    # `root_value` -- the search's own backed-up value -- is in every record and
    # was never read. 0.0 reproduces every checkpoint before this one.
    ap.add_argument("--value-mix", type=float, default=0.0,
                    help="blend root_value into the `rel` target: "
                         "(1-mix)*z_rel + mix*root_value")
    ap.add_argument("--init", default=None, help="warm-start from this checkpoint base")
    ap.add_argument("--resume", action="store_true", help="continue this generation")
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=50)
    # The holdout existed before this and nothing read it: the loop reported a
    # running *training* loss and nothing else, which is the one quantity that
    # cannot tell learning from memorising.
    ap.add_argument("--eval-every", type=int, default=500,
                    help="held-out loss every N steps; 0 disables")
    ap.add_argument("--eval-batches", type=int, default=8,
                    help="batches averaged per held-out evaluation")
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

    do_eval = args.eval_every > 0 and buf.n_train < buf.n
    if args.eval_every > 0 and not do_eval:
        print("[warn] --eval-every set but the buffer has no holdout; "
              "no held-out loss will be reported", file=sys.stderr)

    t0 = time.time()
    running: dict[str, float] = {}
    held: list[dict] = []
    step = start
    try:
        for step in range(start, args.steps):
            if STOP:
                break
            for g in opt.param_groups:
                g["lr"] = base_lr * warmup(step - start)

            x, tgt, ptr = make_batch(buf, args.batch, rng, augment=not args.no_augment,
                                     value_mix=args.value_mix)
            out = forward(net, x.to(dev), ptr, dev)
            total, parts = loss_fn(out, to_dev(tgt, dev))
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
            if do_eval and (step + 1) % args.eval_every == 0:
                h = evaluate(net, buf, dev, args.batch, args.eval_batches,
                             args.gen, augment=not args.no_augment,
                             value_mix=args.value_mix)
                held.append({"step": step + 1, **h})
                with open(base + ".held.jsonl", "a") as f:
                    f.write(json.dumps(held[-1]) + "\n")
                comp = "  ".join(f"{k} {v:.4f}" for k, v in sorted(h.items())
                                 if k != "total")
                print(f"  HELD {step + 1:>5}/{args.steps}  loss {h['total']:.4f}  "
                      f"{comp}", flush=True)
            if (step + 1) % args.checkpoint_every == 0:
                save(base, net, opt, step + 1, args.gen,
                     {"loss": running, "held": held[-1] if held else None})
    except KeyboardInterrupt:
        print("\n[interrupt] hard stop; the last periodic checkpoint stands.")
        raise SystemExit(130)

    # ...unless the loop's last step was already an evaluation step, which it is
    # whenever --steps is a multiple of --eval-every.
    if do_eval and not (held and held[-1]["step"] == step + 1):
        h = evaluate(net, buf, dev, args.batch, args.eval_batches, args.gen,
                     augment=not args.no_augment, value_mix=args.value_mix)
        held.append({"step": step + 1, **h})
        with open(base + ".held.jsonl", "a") as f:
            f.write(json.dumps(held[-1]) + "\n")
        comp = "  ".join(f"{k} {v:.4f}" for k, v in sorted(h.items()) if k != "total")
        print(f"  HELD {step + 1:>5}/{args.steps}  loss {h['total']:.4f}  {comp}",
              flush=True)
    save(base, net, opt, step + 1, args.gen,
         {"loss": running, "held": held[-1] if held else None, "records": buf.n,
          "train_records": buf.n_train, "shards": len(buf.shards),
          "value_mix": args.value_mix})
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
