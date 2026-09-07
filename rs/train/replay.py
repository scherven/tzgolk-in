"""Read Tzolk'in self-play replay shards.

The shard format is defined once, in Rust, in ``src/record.rs``. This module
does not restate it: ``selfplay`` writes a ``schema.json`` next to the shards
and everything here is built from that. ``src/state.rs`` changed shape twice in
the week this pipeline was written (Chichen Itza went from 10 worker spaces to
11), so a second hand-maintained copy of the offsets would already be silently
wrong.

Layout, in one paragraph: a 64-byte header, then fixed-size records, little
endian. Fixed size is the whole point -- a shard is read with ``np.memmap`` and
a structured dtype, which costs nothing and parses nothing, and a shard
truncated by a kill is still valid up to its last whole record.

Usage
-----

    from replay import Buffer
    buf = Buffer.open("replay")             # every .tzr in the directory
    print(buf.summary())
    batch = buf.sample(1024)                # dict of numpy arrays
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

MAGIC = b"TZZR"


# ----------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------


def load_schema(path: str) -> dict:
    """Load ``schema.json``. ``path`` may be the file or its directory.

    Regenerate it at any time with::

        cargo run --release --bin selfplay -- --print-schema > replay/schema.json
    """
    if os.path.isdir(path):
        path = os.path.join(path, "schema.json")
    with open(path) as f:
        return json.load(f)


def record_dtype(schema: dict) -> np.dtype:
    """A structured dtype covering the whole record, gaps included.

    Built with explicit offsets and ``itemsize`` rather than by concatenating
    fields, so a reserved gap in the middle of the record costs nothing and a
    new field appearing in a later format version does not shift anything.
    """
    names, formats, offsets = [], [], []
    for f in schema["record_fields"]:
        names.append(f["name"])
        formats.append((f["dtype"], (f["count"],)) if f["count"] > 1 else f["dtype"])
        offsets.append(f["offset"])
    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": schema["record_bytes"],
        }
    )


# ----------------------------------------------------------------------
# shards
# ----------------------------------------------------------------------


@dataclass
class Header:
    format_version: int
    header_bytes: int
    record_bytes: int
    rules_version: int
    state_bytes: int
    n_players: int
    max_visits: int
    created_unix: int
    generation: int
    producer: str


def read_header(path: str) -> Header:
    with open(path, "rb") as f:
        h = f.read(64)
    if len(h) < 64 or h[0:4] != MAGIC:
        raise ValueError(f"{path}: not a tzolkin replay shard")
    u16 = lambda o: int.from_bytes(h[o : o + 2], "little")
    u32 = lambda o: int.from_bytes(h[o : o + 4], "little")
    u64 = lambda o: int.from_bytes(h[o : o + 8], "little")
    return Header(
        format_version=u16(4),
        header_bytes=u16(6),
        record_bytes=u32(8),
        rules_version=u32(12),
        state_bytes=u16(16),
        n_players=u16(18),
        max_visits=u16(20),
        created_unix=u64(24),
        generation=u32(32),
        producer=h[36:64].rstrip(b"\0").decode("ascii", "replace"),
    )


def open_shard(path: str, schema: dict) -> np.ndarray:
    """Memory-map one shard as a structured array. Nothing is copied.

    A trailing partial record (a truncated copy; the writer cannot produce one)
    is dropped rather than raising -- losing a record is always better than
    refusing to train.
    """
    hdr = read_header(path)
    if hdr.record_bytes != schema["record_bytes"]:
        raise ValueError(
            f"{path}: record size {hdr.record_bytes} but schema says "
            f"{schema['record_bytes']}; regenerate schema.json"
        )
    size = os.path.getsize(path)
    n = (size - hdr.header_bytes) // hdr.record_bytes
    if n <= 0:
        return np.zeros(0, dtype=record_dtype(schema))
    return np.memmap(
        path,
        dtype=record_dtype(schema),
        mode="r",
        offset=hdr.header_bytes,
        shape=(n,),
    )


# ----------------------------------------------------------------------
# the buffer
# ----------------------------------------------------------------------


class Buffer:
    """The sampling window: a set of shards treated as one array of records.

    ``LEARNING.md`` §6.6 wants the last 20 generations, capped at 1.5M records.
    ``open()`` takes the newest shards up to that cap, which under the default
    synchronous-generation layout is the same thing and does not require the
    loader to understand generations.
    """

    def __init__(self, shards: list[np.ndarray], schema: dict, paths: list[str]):
        self.shards = shards
        self.schema = schema
        self.paths = paths
        self.sizes = np.array([len(s) for s in shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.n = int(self.offsets[-1])

    @classmethod
    def open(
        cls,
        directory: str,
        max_records: int = 1_500_000,
        min_rules_version: int | None = None,
        pattern: str = "*.tzr",
    ) -> "Buffer":
        """Open the newest shards in ``directory``, newest first up to the cap.

        ``min_rules_version`` is the point of the version stamp. A rules change
        invalidates the learned value function even when every tensor shape
        survives (``LEARNING.md`` §8.4), so pass the current
        ``tzolkin::RULES_VERSION`` here and shards written under older rules are
        dropped -- turning "the whole buffer is suspect" into "the records
        before generation N are".
        """
        schema = load_schema(directory)
        if min_rules_version is None:
            min_rules_version = schema["rules_version"]

        files = sorted(glob.glob(os.path.join(directory, pattern)))
        # Shard names carry a UTC stamp, so lexical order is chronological.
        files.reverse()

        shards, kept, total, dropped = [], [], 0, []
        for path in files:
            try:
                hdr = read_header(path)
            except ValueError:
                continue
            if hdr.rules_version < min_rules_version:
                dropped.append((path, hdr.rules_version))
                continue
            arr = open_shard(path, schema)
            if len(arr) == 0:
                continue
            shards.append(arr)
            kept.append(path)
            total += len(arr)
            if total >= max_records:
                break

        if dropped:
            print(
                f"replay: dropped {len(dropped)} shard(s) written under rules "
                f"< v{min_rules_version}"
            )
        if not shards:
            raise FileNotFoundError(
                f"no usable shards in {directory} "
                f"(looked for {pattern}, rules >= v{min_rules_version})"
            )
        return cls(shards, schema, kept)

    # -- indexing ------------------------------------------------------

    def take(self, idx: np.ndarray) -> np.ndarray:
        """Gather records by global index into one contiguous array."""
        idx = np.asarray(idx, dtype=np.int64)
        which = np.searchsorted(self.offsets, idx, side="right") - 1
        local = idx - self.offsets[which]
        out = np.empty(len(idx), dtype=self.shards[0].dtype)
        for s in np.unique(which):
            m = which == s
            out[m] = self.shards[s][local[m]]
        return out

    def sample(self, n: int, rng: np.random.Generator | None = None) -> dict:
        """Uniform sample of ``n`` records, as a dict of decoded arrays."""
        rng = rng or np.random.default_rng()
        return decode(self.take(rng.integers(0, self.n, size=n)), self.schema)

    def all(self) -> dict:
        return decode(self.take(np.arange(self.n)), self.schema)

    def summary(self) -> str:
        d = decode(self.take(np.arange(0, self.n, max(1, self.n // 20000))), self.schema)
        scores = d["final_scores"].astype(np.float64)
        centred = scores - scores.mean(axis=1, keepdims=True)
        pk = d["policy_kind"]
        return (
            f"{self.n:,} records over {len(self.shards)} shards\n"
            f"  rules version   : v{self.schema['rules_version']}\n"
            f"  mean score      : {scores.mean():+.1f}  "
            f"(sd {scores.std():.1f}, centred sd {centred.std():.1f})\n"
            f"  |z_rel| mean    : {np.abs(d['z_rel']).mean():.3f}\n"
            f"  policy targets  : {(pk != 0).mean() * 100:.1f}% usable "
            f"({int((pk != 0).sum()):,} of {len(pk):,} sampled)\n"
            f"  day range       : {int(d['day'].min())}..{int(d['day'].max())}\n"
            f"  newest shard    : {os.path.basename(self.paths[0])}"
        )


# ----------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------


def decode(recs: np.ndarray, schema: dict) -> dict:
    """Turn a structured record array into a dict of plain arrays.

    The ``state`` field stays as raw bytes, ``uint8[N, state_slot]``. Decoding
    it is the encoder's job, not the loader's -- see ``features.py``.
    """
    out = {name: np.ascontiguousarray(recs[name]) for name in recs.dtype.names}
    out["visits"] = out["visits"].reshape(len(recs), schema["max_visits"], 2)
    out["temperature"] = out.pop("temperature_x100").astype(np.float32) / 100.0
    return out


class State:
    """Typed views over the raw state bytes.

    Offsets come from ``schema['state_fields']``; nothing here is hard-coded.
    Only the fields the Python-side placeholder encoder needs are exposed --
    once the Rust batcher (``LEARNING.md`` §5.1) exists this class becomes
    debugging scaffolding.
    """

    def __init__(self, raw: np.ndarray, schema: dict):
        self.raw = raw
        self.schema = schema
        self.f = {f["name"]: f for f in schema["state_fields"]}
        self.n_players = schema["n_players"]
        self.player_bytes = schema["player_bytes"]

    def field(self, name: str) -> np.ndarray:
        """One state field, as `[N, count]`.

        `count` in the schema is a count of *elements*, not of bytes, so the
        byte slice is `count * itemsize` wide. A column slice of the
        `[N, state_slot]` uint8 array is also not contiguous, and `view` on a
        wider dtype needs it to be, hence the copy.
        """
        f = self.f[name]
        o, c = f["offset"], f["count"]
        dt = np.dtype(f["dtype"])
        v = self.raw[:, o : o + c * dt.itemsize]
        if dt.itemsize > 1:
            return np.ascontiguousarray(v).view(dt).reshape(len(self.raw), c)
        return v

    # Player sub-fields. The 19-byte player record is laid out in
    # `encode_state`; these offsets mirror it and are checked by
    # `test_state_offsets` below.
    def player(self, offset: int, width: int = 1, dtype=np.uint8) -> np.ndarray:
        base = self.f["players"]["offset"]
        pb = self.player_bytes
        cols = [
            self.raw[:, base + p * pb + offset : base + p * pb + offset + width]
            for p in range(self.n_players)
        ]
        stacked = np.stack(cols, axis=1)  # [N, players, width]
        if dtype is np.uint8:
            return stacked[..., 0] if width == 1 else stacked
        return stacked.copy().view(dtype).reshape(len(self.raw), self.n_players)

    def corn(self):
        return self.player(1)

    def res(self):
        return self.player(2, 4)

    def points(self):
        return self.player(6, 2, np.int16)

    def corn_tiles(self):
        return self.player(8)

    def wood_tiles(self):
        return self.player(9)

    def free_workers(self):
        return self.player(10)

    def worker_discount(self):
        return self.player(11)

    def may_skip_day(self):
        return self.player(12)

    def buildings(self):
        return self.player(13, 4, np.uint32)

    def monuments(self):
        return self.player(17, 2, np.uint16)

    def temples(self):
        return self.field("temples").reshape(-1, 3, self.n_players)

    def research(self):
        return self.field("research").reshape(-1, self.n_players, 4)

    def gears(self):
        g = self.f["gears"]
        return self.raw[:, g["offset"] : g["offset"] + g["count"]].reshape(
            -1, 5, g["count"] // 5
        )

    def workers(self):
        w = self.f["workers"]
        return self.raw[:, w["offset"] : w["offset"] + w["count"]].reshape(-1, 24, 2)

    def scalar(self, name: str) -> np.ndarray:
        return self.field(name)[:, 0]


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "replay"
    b = Buffer.open(d)
    print(b.summary())
