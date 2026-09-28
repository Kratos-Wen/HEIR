"""Losslessly gather a committed FSDP checkpoint on CPU, without an optimizer."""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from safetensors.torch import save_file

from .tokenizer import digest


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-run", dest="run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_num_threads(1)
    pointer = json.loads((args.run / "latest.json").read_text())
    complete = json.loads((args.run / "training_complete.json").read_text())
    manifest = json.loads((args.run / pointer["directory"] / "complete.json").read_text())
    fingerprint = digest(args.run / "protocol.json")
    if (world != pointer["world_size"] or complete["cursor"] != pointer["cursor"]
            or complete["final_checkpoint"] != pointer["directory"]
            or any(pointer.get(k) != v for k, v in manifest.items())
            or any(v["protocol_sha256"] != fingerprint for v in (pointer, complete, manifest))):
        raise ValueError("Require all ranks of the committed fixed-final checkpoint")
    if sorted(v["rank"] for v in pointer["shards"]) != list(range(world)):
        raise ValueError("Checkpoint rank coverage mismatch")
    shard = pointer["shards"][rank]
    path = args.run / pointer["directory"] / shard["file"]
    if shard["rank"] != rank or digest(path) != shard["sha256"]:
        raise ValueError("Checkpoint shard hash mismatch")
    protocol = {"checkpoint": str(path.parent.resolve()), "checkpoint_manifest_sha256": digest(path.parent / "complete.json"),
                "training_protocol_sha256": fingerprint, "format": "lossless FP32 model-only safetensors",
                "source_sha256": digest(Path(__file__)), "world_size_at_training": world}
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        pp = args.output / "protocol.json"
        if pp.exists() and json.loads(pp.read_text()) != protocol:
            raise ValueError("Refuse a different conversion in the same directory")
        atomic_json(pp, protocol)
    dist.barrier()
    # mmap leaves optimizer storages unmaterialized; no optimizer is reconstructed.
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if state["cursor"] != pointer["cursor"] or state["rank"] != rank or state["protocol_sha256"] != fingerprint:
        raise ValueError("Shard contents differ from the committed manifest")
    model = state["model"]
    del state
    keys = sorted(model)
    layouts = [None] * world
    dist.all_gather_object(layouts, keys)
    if any(k != keys for k in layouts):
        raise ValueError("Different model key sets across ranks")
    index = {}
    for i, name in enumerate(keys):
        value = model.pop(name)
        if isinstance(value, ShardedTensor):
            local = [{"offsets": s.metadata.shard_offsets, "sizes": s.metadata.shard_sizes,
                      "sha256": tensor_hash(s.tensor)} for s in value.local_shards()]
            full = torch.empty(tuple(value.size()), dtype=value.dtype) if rank == 0 else None
            value.gather(dst=0, out=full, dtype=value.dtype)
        else:
            local = [{"offsets": [0] * value.ndim, "sizes": list(value.shape), "sha256": tensor_hash(value)}]
            full = value if rank == 0 else None
        records = [None] * world
        dist.all_gather_object(records, local)
        if rank == 0:
            for rank_records in records:
                for record in rank_records:
                    slices = tuple(slice(o, o + n) for o, n in zip(record["offsets"], record["sizes"]))
                    if tensor_hash(full[slices]) != record["sha256"]:
                        raise ValueError(f"Gathered tensor differs from trained shard: {name}")
            file = args.output / f"tensor_{i:04d}.safetensors"
            tmp = file.with_suffix(".partial")
            save_file({name: full.contiguous()}, str(tmp))
            tmp.replace(file)
            index[name] = {"file": file.name, "sha256": digest(file), "shape": list(full.shape),
                           "dtype": str(full.dtype), "numel": full.numel()}
            if i % 25 == 0 or i == len(keys) - 1:
                print(f"Verified/gathered {i + 1}/{len(keys)}: {name}", flush=True)
        del value, full
        dist.barrier()
    if rank == 0:
        atomic_json(args.output / "complete.json", {"protocol": protocol, "tensors": index,
            "parameters_and_buffers": sum(v["numel"] for v in index.values()),
            "all_source_shards_sha256_verified": True, "all_gathered_slices_bitwise_verified": True})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
