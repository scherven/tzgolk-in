"""Cross-check the PyTorch parameter names and shapes against src/net.rs.

`Net::load` refuses a file with a missing tensor, an extra tensor or a wrong
shape, so a mismatch here is a checkpoint that cannot be loaded at all. This
derives the expected manifest from `Config` alone -- deliberately a second
statement of the structure, so a typo in `model.py` shows up as a diff rather
than as a runtime refusal hours into a training run.

No torch required, which is the point: it runs anywhere.

    python3 train/check_manifest.py            # print
    python3 train/check_manifest.py --check    # diff against the Rust manifest

The authoritative list comes from:
    cargo test --release --test encode -- --ignored --nocapture manifest
"""

import subprocess
import sys

from arch import Config, N_PLAYERS, N_PLACE, N_WHO, N_DECOMP, N_PHASE_TAGS


def linear(name, out, inp):
    return [(f"{name}.weight", [out, inp]), (f"{name}.bias", [out])]


def norm(name, w):
    return [(f"{name}.weight", [w]), (f"{name}.bias", [w])]


def manifest(c: Config):
    t = []
    t += linear("stem.board", c.board, c.w_board)
    t += linear("stem.player", c.player, c.w_player)
    t += linear("stem.bslot", c.bslot, c.w_bslot)
    t += linear("stem.mslot", c.mslot, c.w_mslot)
    t += linear("stem.global", c.globals_, c.w_global)
    t += linear("fuse", c.width, c.fuse_in)
    t += norm("fuse.norm", c.width)
    for i in range(c.blocks):
        t += norm(f"trunk.{i}.norm1", c.width)
        t += [(f"trunk.{i}.fc1.weight", [c.width, c.width])]   # unbiased
        t += norm(f"trunk.{i}.norm2", c.width)
        t += [(f"trunk.{i}.fc2.weight", [c.width, c.width])]   # unbiased
    t += norm("trunk.norm", c.width)
    t += linear("value.fc", c.value_hidden, c.width)
    t += linear("value.rel", N_PLAYERS, c.value_hidden)
    t += linear("value.score", N_PLAYERS, c.value_hidden)
    t += linear("value.rank", N_PLAYERS * N_PLAYERS, c.value_hidden)
    t += linear("value.decomp", N_DECOMP * N_PLAYERS, c.value_hidden)
    t += linear("policy.beg", 4, c.width)
    t += linear("policy.mode", 3, c.width)
    t += linear("policy.place", N_PLACE, c.width)
    t += linear("policy.extra_day", 2, c.width)
    t += linear("policy.who", N_WHO, c.width)
    t += linear("ptr.q_fast", c.d_choice, c.width)
    t += linear("ptr.q_deep", c.key, c.width)
    t += [("ptr.cell_emb", [N_WHO, c.key]), ("ptr.step_emb", [N_PHASE_TAGS, c.key])]
    t += linear("ptr.key1", c.key, c.d_choice)
    t += linear("ptr.key2", c.key, c.key)
    return t


def count(t):
    n = 0
    for _, shape in t:
        p = 1
        for d in shape:
            p *= d
        n += p
    return n


def rust_manifest():
    out = subprocess.run(
        ["cargo", "test", "--release", "--test", "encode", "--",
         "--ignored", "--nocapture", "manifest"],
        cwd="..", capture_output=True, text=True,
    ).stdout
    arches, cur = {}, None
    for line in out.splitlines():
        if line.startswith("# "):
            cur = line.split(":")[0][2:].strip()
            arches[cur] = []
        elif cur and line.startswith(("stem", "fuse", "trunk", "value", "policy", "ptr")):
            name, rest = line.split(maxsplit=1)
            arches[cur].append((name, [int(x) for x in rest.strip("[] ").split(", ")]))
    return arches


if __name__ == "__main__":
    ours = {"MAIN": manifest(Config.main()), "SMALL": manifest(Config.small())}
    for k, v in ours.items():
        print(f"# {k}: {len(v)} tensors, {count(v)} parameters")

    if "--check" not in sys.argv:
        for name, shape in ours["MAIN"]:
            print(f"{name:32s} {shape}")
        sys.exit(0)

    theirs = rust_manifest()
    bad = 0
    for arch, want in ours.items():
        got = theirs.get(arch)
        if not got:
            print(f"{arch}: no Rust manifest found"); bad += 1; continue
        w, g = dict(want), dict(got)
        for n in sorted(set(w) | set(g)):
            if n not in g:
                print(f"{arch}: {n} only in python {w[n]}"); bad += 1
            elif n not in w:
                print(f"{arch}: {n} only in rust {g[n]}"); bad += 1
            elif w[n] != g[n]:
                print(f"{arch}: {n} python {w[n]} != rust {g[n]}"); bad += 1
        if len(want) != len(got):
            print(f"{arch}: {len(want)} tensors vs rust {len(got)}"); bad += 1
    print("MISMATCHES:", bad)
    sys.exit(1 if bad else 0)
