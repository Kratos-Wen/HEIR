"""Resumable, label-free greedy inference for the fixed supervised checkpoint."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .detection import ANNOTATIONS, BASE, PROMPT, DetectionLanguageModel
from .export_weights import atomic_json
from .prediction import action_roles, parse_prediction
from .tokenizer import ROOT, digest, encode, letterbox, load_vqgan


def load_model(weights, device):
    manifest = json.loads((weights / "complete.json").read_text())
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    config.vocab_size = 128256 + 8192
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        llm = AutoModelForCausalLM.from_config(config)
        model = DetectionLanguageModel(llm, 128256)
    expected = set(model.state_dict())
    if expected != set(manifest["tensors"]):
        raise ValueError(f"Converted model key mismatch: {expected ^ set(manifest['tensors'])}")
    for name, info in manifest["tensors"].items():
        path = weights / info["file"]
        if digest(path) != info["sha256"]:
            raise ValueError("Converted tensor integrity failure")
        values = load_file(str(path))
        if set(values) != {name} or list(values[name].shape) != info["shape"]:
            raise ValueError("Converted tensor name/shape mismatch")
        if str(values[name].dtype) != info["dtype"]:
            raise ValueError("Converted tensor dtype mismatch")
        result = model.load_state_dict(values, strict=False, assign=True)
        if result.unexpected_keys:
            raise ValueError("Unexpected inference parameter")
    materialize_rope(model)
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Inference model still has unmaterialized tensors")
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)


def materialize_rope(model):
    # Reuse the installed Llama implementation, including legacy nonpersistent
    # cosine/sine buffers, instead of leaving meta tensors in the loaded model.
    for module in model.modules():
        if hasattr(module, "inv_freq") and module.inv_freq.is_meta:
            fresh = type(module)(module.dim, module.max_position_embeddings,
                                 module.base, device="cpu", scaling_factor=module.scaling_factor)
            for name, buffer in fresh.named_buffers(recurse=False):
                module.register_buffer(name, buffer, persistent=False)


def prefix_embeddings(model, ids):
    modality = (ids >= model.visual_start).long()
    embeddings = model.llm.get_input_embeddings()(ids) + model.modality(modality)
    return model.prefix_adapter(embeddings, modality, torch.ones_like(ids, dtype=torch.bool),
                                ids.new_full((len(ids),), ids.shape[1]), "detection")


@torch.inference_mode()
def greedy_generate(model, prefix, eos_id, max_new_tokens):
    if max_new_tokens < 1:
        raise ValueError("Positive generation budget required")
    attention = torch.ones_like(prefix)
    result = model.llm.model(inputs_embeds=prefix_embeddings(model, prefix),
        attention_mask=attention, use_cache=True, return_dict=True)
    output, logprobs = [[] for _ in prefix], [[] for _ in prefix]
    finished = [False for _ in prefix]
    for step in range(max_new_tokens):
        logits = model.llm.lm_head(result.last_hidden_state[:, -1]).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite generation logits")
        tokens = logits.argmax(-1)
        scores = logits.log_softmax(-1).gather(1, tokens[:, None])[:, 0]
        for i, (token, score) in enumerate(zip(tokens.tolist(), scores.tolist())):
            if not finished[i]:
                output[i].append(token)
                logprobs[i].append(score)
                finished[i] = token == eos_id
        if all(finished) or step == max_new_tokens - 1:
            break
        # Generated tokens receive the same modality embeddings as teacher forcing.
        embeddings = (model.llm.get_input_embeddings()(tokens[:, None])
                      + model.modality((tokens[:, None] >= model.visual_start).long()))
        attention = torch.cat((attention, attention.new_ones((len(prefix), 1))), dim=1)
        result = model.llm.model(inputs_embeds=embeddings, attention_mask=attention,
            past_key_values=result.past_key_values, use_cache=True, return_dict=True)
    return output, logprobs, finished


@torch.inference_mode()
def check_cached_forward(model, ids, suffix):
    prefix = prefix_embeddings(model, ids)
    pre = model.llm.model(inputs_embeds=prefix, use_cache=True, return_dict=True)
    next_emb = model.llm.get_input_embeddings()(suffix) + model.modality((suffix >= model.visual_start).long())
    cached = model.llm.model(inputs_embeds=next_emb, past_key_values=pre.past_key_values,
                             use_cache=True, return_dict=True).last_hidden_state
    full_ids = torch.cat((ids, suffix), dim=1)
    full = model.hidden(full_ids, torch.ones_like(full_ids), ids.new_full((len(ids),), ids.shape[1]))
    reference = full[:, -suffix.shape[1]:]
    error = (cached.float() - reference.float()).abs()
    return {"max_abs_hidden_difference": error.max().item(), "mean_abs_hidden_difference": error.mean().item(),
            "relative_l2_hidden_difference": (error.norm() / reference.float().norm().clamp_min(1e-12)).item(),
            "next_token_argmax_equal": bool(torch.equal(model.llm.lm_head(cached).argmax(-1),
                                                        model.llm.lm_head(reference).argmax(-1)))}


def image_path(image_id):
    root = ROOT.parent / "data/v-coco/images"
    paths = [root / split / f"COCO_{split}_{image_id:012d}.jpg" for split in ("train2014", "val2014")]
    existing = [p for p in paths if p.exists()]
    if len(existing) != 1:
        raise ValueError(f"Expected one source image for {image_id}")
    return existing[0]


def render_prediction(tokenizer, tokens, logprobs, finished, geometry, image_id, actions):
    if not finished:
        return {"events": [], "errors": [{"scope": "image", "reason": "Generation reached context limit before EOS"}],
                "text": None, "confidence_mode": None}
    ids, scores = tokens[:-1], logprobs[:-1]
    if any(t >= len(tokenizer) for t in ids):
        return {"events": [], "errors": [{"scope": "image", "reason": "Generated visual token in JSON answer"}],
                "text": None, "confidence_mode": None}
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    retokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    exact = retokenized["input_ids"] == ids
    events, errors = parse_prediction(text, geometry, image_id, actions, scores,
        retokenized["offset_mapping"] if exact else None)
    return {"events": events, "errors": errors, "text": text,
            "confidence_mode": "event_token_geometric_mean" if exact else "sequence_token_geometric_mean_retokenization_mismatch"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("qualify", "test"), required=True)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("Invalid batch size")
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                          (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    if len(tokenizer) != 128256:
        raise ValueError("Wrong base tokenizer")
    actions = action_roles(json.loads(ANNOTATIONS.read_text())["channels"])
    split_name = "vcoco_test.ids" if args.mode == "test" else "vcoco_trainval.ids"
    ids_path = ROOT.parent / "data/v-coco/data/splits" / split_name
    ids = [int(v) for v in ids_path.read_text().split()]
    if len(ids) != len(set(ids)) or len(ids) != (4946 if args.mode == "test" else 5400):
        raise ValueError("Wrong split IDs")
    if args.mode == "qualify":
        ids = ids[:4]
    sources = (Path(__file__), Path(__file__).with_name("prediction.py"), Path(__file__).with_name("detection.py"),
               Path(__file__).with_name("attention.py"), Path(__file__).with_name("tokenizer.py"))
    shared = {"weights_sha256": digest(args.weights / "complete.json"),
              "source_sha256": {str(f): digest(f) for f in sources},
              "prompt": PROMPT, "decoding": "greedy full expanded vocabulary; KV cache; no beam, sampling, grammar or GT",
              "context_limit": 4096, "precision": "BF16 inference, FP32 logsoftmax; exact FP32 checkpoint retained",
              "confidence": "event-token geometric mean; sequence-token mean only on retokenization mismatch",
              "coordinates": "normalized padded continuous xyxy inverted then converted to official inclusive xyxy",
              "missing_role": "explicit generated null only", "invalid_output": "saved; empty or rejected event, never dropped image",
              "annotation_channels_sha256": digest(ANNOTATIONS), "test_labels_used_for_generation": False,
              "batch_size": args.batch_size, "evidence_type": "INDEPENDENT_SHORTENED_SUPERVISED_ADAPTATION_NOT_FULL_UNIHOI"}
    if args.mode == "test":
        if args.qualification is None:
            raise ValueError("Real train-image qualification required before test")
        qualification = json.loads(args.qualification.read_text())
        if qualification["shared_protocol"] != shared or not qualification["passed"]:
            raise ValueError("Qualification differs from fixed inference protocol")
    protocol = {**shared, "mode": args.mode, "image_ids": ids, "split_sha256": digest(ids_path)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "images").mkdir(exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Different inference protocol; refusing to mix predictions")
    # Independent workers write identical protocol; outputs are disjoint image IDs.
    atomic_json(protocol_path, protocol)
    fingerprint = digest(protocol_path)
    mine = ids[rank::world]
    pending = []
    for image_id in mine:
        path = args.output / "images" / f"{image_id:012d}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if (old["protocol_sha256"] != fingerprint or old["image_id"] != image_id
                    or old["image_sha256"] != digest(image_path(image_id))):
                raise ValueError("Invalid saved image result")
        else:
            pending.append(image_id)
    print(f"rank={rank} {len(mine)-len(pending)}/{len(mine)} already complete; loading fixed checkpoint", flush=True)
    model = load_model(args.weights, device)
    vq, _ = load_vqgan(device)
    prompt = tokenizer.encode(PROMPT, add_special_tokens=False)
    separator = tokenizer.encode("\nAnswer: ", add_special_tokens=False)
    cached_check = None
    completed = len(mine) - len(pending)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        images, geometry, hashes = [], [], []
        for image_id in batch:
            path = image_path(image_id)
            hashes.append(digest(path))
            with Image.open(path) as image:
                tensor, geom = letterbox(image)
            images.append(tensor)
            geometry.append(geom)
        tick = time.monotonic()
        with torch.inference_mode():
            codes = encode(vq, torch.stack(images).to(device))
            prefixes = [[tokenizer.bos_token_id] + prompt + (row + len(tokenizer)).tolist() + separator for row in codes]
            prefix = torch.tensor(prefixes, dtype=torch.long, device=device)
            if args.mode == "qualify" and cached_check is None:
                suffix = torch.tensor([tokenizer.encode("[]", add_special_tokens=False)], device=device)
                cached_check = check_cached_forward(model, prefix[:1], suffix)
                if not cached_check["next_token_argmax_equal"] or cached_check["relative_l2_hidden_difference"] > .01:
                    raise ValueError(f"Cached/full inference mismatch: {cached_check}")
            tokens, scores, finished = greedy_generate(model, prefix, tokenizer.eos_token_id, 4096 - prefix.shape[1])
        for i, image_id in enumerate(batch):
            result = render_prediction(tokenizer, tokens[i], scores[i], finished[i], geometry[i], image_id, actions)
            result.update({"image_id": image_id, "image_sha256": hashes[i], "protocol_sha256": fingerprint,
                           "geometry": geometry[i], "generated_ids": tokens[i], "token_logprobs": scores[i],
                           "eos_reached": finished[i], "vq_codes": codes[i].tolist()})
            atomic_json(args.output / "images" / f"{image_id:012d}.json", result)
        completed += len(batch)
        progress = {"rank": rank, "completed": completed, "assigned": len(mine), "mode": args.mode,
                    "batch_seconds": time.monotonic() - tick, "peak_mib": torch.cuda.max_memory_allocated() / 2**20}
        atomic_json(args.output / f"progress_rank{rank}.json", progress)
        print(json.dumps(progress), flush=True)
    if args.mode == "qualify":
        if world != 1:
            raise ValueError("Run qualification in one process")
        if cached_check is None:
            raise ValueError("Use a new qualification directory to execute the cache check")
        atomic_json(args.output / "qualification.json", {"passed": True, "shared_protocol": shared,
            "cached_forward": cached_check, "train_images": ids, "test_labels_read": False,
            "prediction_quality_not_a_gate": True, "peak_mib": torch.cuda.max_memory_allocated() / 2**20})
    atomic_json(args.output / f"complete_rank{rank}.json", {"protocol_sha256": fingerprint, "image_ids": mine})


if __name__ == "__main__":
    main()
