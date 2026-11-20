#!/usr/bin/env python3
"""Convert TencentGameMate/chinese-wav2vec2-base (pytorch_model.bin) to the
flat f32 blob + JSON manifest consumed by liveact-model's wav2vec2 loader.

Uses only the Python standard library (+ numpy if available for speed) by
implementing a custom Unpickler with `persistent_load` for torch storages.

Usage:
    python3 convert_wav2vec2.py [models_dir]

Produces <models_dir>/wav2vec2-weights.bin and <models_dir>/wav2vec2-manifest.json
"""
import io
import json
import os
import pickle
import struct
import sys
import urllib.request
import zipfile

REPO = "TencentGameMate/chinese-wav2vec2-base"
URL = f"https://huggingface.co/{REPO}/resolve/main/pytorch_model.bin"

# keys we keep (Wav2Vec2Model feature extractor + encoder)
KEEP_PREFIXES = (
    "feature_extractor.conv_layers.",
    "feature_projection.layer_norm.",
    "feature_projection.projection.",
    "encoder.pos_conv_embed.conv.",
    "encoder.layer_norm.",
    "encoder.layers.",
)


class Storage:
    def __init__(self, dtype, data):
        self.dtype = dtype  # 'FloatStorage' | 'HalfStorage' | 'BFloat16Storage'
        self.data = data    # raw bytes


class Unpickler(pickle.Unpickler):
    def __init__(self, f, zf):
        super().__init__(f)
        self.zf = zf
        self.storage_by_key = {}

    def find_class(self, module, name):
        import collections
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "collections" and name == "defaultdict":
            return collections.defaultdict
        if name == "__container__":
            return lambda: None

        # named placeholder class: pickle passes it to persistent_load as the
        # storage type; its __name__ carries the dtype (e.g. FloatStorage)
        return type(name, (), {"__module__": module})

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, storage_type, key, location, numel = pid
        dtype = getattr(storage_type, "__name__", str(storage_type))
        entry = f"archive/data/{key}"
        try:
            raw = self.zf.read(entry)
        except KeyError:
            try:
                raw = self.zf.read(f"data/{key}")
            except KeyError:
                raw = self.zf.read(f"archive/data/{len(self.storage_by_key)}")
        self.storage_by_key[key] = (dtype, raw)
        return Storage(dtype, raw)


def rebuild_tensor(storage, offset, size, stride, *args):
    import array

    dtype = storage.dtype
    raw = storage.data
    if dtype == "FloatStorage":
        item = 4
        code = "f"
    elif dtype == "HalfStorage":
        item = 2
        code = "e"
    elif dtype == "BFloat16Storage":
        item = 2
        code = None  # handled below
    else:
        raise RuntimeError(f"unsupported storage dtype {dtype}")

    def read_f32(i):
        # i = element index
        if dtype == "FloatStorage":
            return struct.unpack_from("<f", raw, i * 4)[0]
        if dtype == "HalfStorage":
            v = struct.unpack_from("<e", raw, i * 2)[0]
            return float(v)
        # bf16 -> f32: shift into upper 16 bits
        (h,) = struct.unpack_from("<H", raw, i * 2)
        return struct.unpack("<f", struct.pack("<I", h << 16))[0]

    import array as _array

    numel = 1
    for s in size:
        numel *= s
    out = _array.array("f", bytes(4 * numel))
    # walk with strides (C-order iteration)
    idx = [0] * len(size)
    # check contiguity fast path
    contiguous = True
    acc = 1
    for d in reversed(range(len(size))):
        if stride[d] != acc:
            contiguous = False
        acc *= size[d]
    if contiguous or len(size) == 1:
        base = offset
        for i in range(numel):
            out[i] = read_f32(base + i)
    else:
        for i in range(numel):
            # compute flat storage offset
            off = offset
            for d in range(len(size)):
                off += idx[d] * stride[d]
            out[i] = read_f32(off)
            # increment multi-dim index
            for d in reversed(range(len(size))):
                idx[d] += 1
                if idx[d] < size[d]:
                    break
                idx[d] = 0
    return (dtype, list(size), out)


def load_bin(path):
    tensors = {}
    with zipfile.ZipFile(path) as zf:
        with zf.open("archive/data.pkl") as f:
            up = Unpickler(io.BufferedReader(f), zf)
            state = up.load()
    # state: dict from the pickled object (may be the raw state_dict or a
    # wrapper); walk it for (dtype, size, list) triples
    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(obj, tuple) and len(obj) == 3 and obj[0] in (
            "FloatStorage",
            "HalfStorage",
            "BFloat16Storage",
        ) and isinstance(obj[1], list) and isinstance(obj[2], list):
            yield_key = prefix.rstrip(".")
            tensors[yield_key] = obj
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, prefix)

    # direct dict case: keys -> tensor tuples
    found_direct = isinstance(state, dict) and any(
        isinstance(v, tuple) and len(v) == 3 and v[0] in ("FloatStorage", "HalfStorage", "BFloat16Storage")
        for v in state.values()
    )
    if not found_direct:
        walk(state)
    else:
        for k, v in state.items():
            if isinstance(v, tuple) and len(v) == 3 and v[0] in (
                "FloatStorage",
                "HalfStorage",
                "BFloat16Storage",
            ):
                tensors[k] = v
    return tensors


def main():
    models_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    os.makedirs(models_dir, exist_ok=True)
    bin_path = os.path.join(models_dir, "pytorch_model.bin")
    out_bin = os.path.join(models_dir, "wav2vec2-weights.bin")
    out_json = os.path.join(models_dir, "wav2vec2-manifest.json")

    if not os.path.exists(bin_path):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, bin_path + ".part")
        os.replace(bin_path + ".part", bin_path)

    tensors = load_bin(bin_path)
    print(f"parsed {len(tensors)} tensors")

    manifest = {}
    offset = 0
    kept = 0
    with open(out_bin, "wb") as out:
        for orig in sorted(tensors):
            key = orig.removeprefix("wav2vec2.")
            if not key.startswith(KEEP_PREFIXES):
                continue
            dtype, size, values = tensors[orig]
            out.write(values.tobytes())  # array('f'), little-endian on x86
            manifest[key] = {"shape": size, "offset": offset, "dtype": "f32"}
            offset += len(values) * 4
            kept += 1

    with open(out_json, "w") as f:
        json.dump(manifest, f)
    print(f"kept {kept} tensors, wrote {offset/1e6:.1f} MB to {out_bin}")

    # sanity: report a few key shapes
    for probe in [
        "feature_extractor.conv_layers.0.conv.weight",
        "encoder.pos_conv_embed.conv.weight_v",
        "encoder.pos_conv_embed.conv.weight_g",
        "encoder.layers.0.attention.q_proj.weight",
        "encoder.layers.11.final_layer_norm.weight",
    ]:
        if probe in manifest:
            print(" ", probe, manifest[probe]["shape"])
        else:
            print(" ", probe, "MISSING (relative-position checkpoint variant?)")
    # any pos_bias keys => old relative-attention variant
    pb = [k for k in tensors if "pos_bias" in k]
    print("pos_bias keys:", pb[:4] if pb else "none (standard attention)")


if __name__ == "__main__":
    main()
