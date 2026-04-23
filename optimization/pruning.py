"""
Structured pruning for OpenVLA (attention head & FFN pruning).
Usage:
    python optimization/pruning.py --config configs/default.yaml --sparsity 0.25
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image


def compute_head_importance_taylor(model, processor, calibration_samples: list,
                                    cfg: dict) -> dict:
    """Compute attention head importance using first-order Taylor expansion."""
    model.eval()
    head_importance = {}

    for name, module in model.named_modules():
        if hasattr(module, "num_heads") and "self_attn" in name:
            head_importance[name] = torch.zeros(module.num_heads, device=cfg["device"])

    model.zero_grad()
    for sample in calibration_samples:
        image = Image.open(sample["image"]).convert("RGB")
        prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut:"
        inputs = processor(prompt, image).to(cfg["device"])
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is not None:
            loss.backward()

    for name, module in model.named_modules():
        if name in head_importance and hasattr(module, "q_proj"):
            weight = module.q_proj.weight
            if weight.grad is not None:
                head_dim = weight.shape[0] // module.num_heads
                for h in range(module.num_heads):
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    importance = (weight[start:end] * weight.grad[start:end]).abs().sum()
                    head_importance[name][h] += importance

    return head_importance


def compute_head_importance_entropy(model, processor, calibration_samples: list,
                                     cfg: dict) -> dict:
    """Compute head importance based on attention entropy (lower entropy = more decisive)."""
    head_importance = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(-1).mean(0).mean(-1)
                    if name not in head_importance:
                        head_importance[name] = []
                    head_importance[name].append(entropy.detach().cpu())
        return hook_fn

    for name, module in model.named_modules():
        if "self_attn" in name and hasattr(module, "num_heads"):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for sample in calibration_samples:
            image = Image.open(sample["image"]).convert("RGB")
            prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut:"
            inputs = processor(prompt, image).to(cfg["device"])
            model(**inputs, output_attentions=True)

    for h in hooks:
        h.remove()

    # Lower entropy → more important (invert for ranking)
    result = {}
    for name, entropies in head_importance.items():
        avg_entropy = torch.stack(entropies).mean(0)
        result[name] = -avg_entropy  # negate so higher = more important
    return result


def prune_heads(model, head_importance: dict, target_sparsity: float):
    """Remove the least important attention heads up to target sparsity."""
    all_heads = []
    for layer_name, scores in head_importance.items():
        for h_idx, score in enumerate(scores):
            all_heads.append((layer_name, h_idx, score.item()))

    all_heads.sort(key=lambda x: x[2])
    n_prune = int(len(all_heads) * target_sparsity)
    heads_to_prune = all_heads[:n_prune]

    prune_map = {}
    for layer_name, h_idx, score in heads_to_prune:
        if layer_name not in prune_map:
            prune_map[layer_name] = []
        prune_map[layer_name].append(h_idx)

    print(f"Pruning {n_prune}/{len(all_heads)} heads ({target_sparsity:.0%})")
    for layer_name, heads in prune_map.items():
        print(f"  {layer_name}: removing heads {heads}")

    # Apply pruning via HuggingFace API if available
    if hasattr(model, "prune_heads"):
        model.prune_heads(prune_map)
    else:
        print("Warning: model.prune_heads() not available, using manual zeroing")
        for name, module in model.named_modules():
            if name in prune_map and hasattr(module, "q_proj"):
                head_dim = module.q_proj.weight.shape[0] // module.num_heads
                for h_idx in prune_map[name]:
                    start = h_idx * head_dim
                    end = (h_idx + 1) * head_dim
                    module.q_proj.weight.data[start:end] = 0
                    module.k_proj.weight.data[start:end] = 0
                    module.v_proj.weight.data[start:end] = 0
                    module.o_proj.weight.data[:, start:end] = 0

    return model, prune_map


def main():
    parser = argparse.ArgumentParser(description="Structured Pruning for OpenVLA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str, default=None, choices=["taylor", "entropy"])
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--calibration-data", type=str, help="Path to calibration manifest.json")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--recovery-finetune", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    method = args.method or cfg["pruning"]["method"]
    sparsity = args.sparsity or cfg["pruning"]["target_sparsity"]

    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name"], cache_dir=cfg["model"]["cache_dir"], trust_remote_code=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"]["cache_dir"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        trust_remote_code=True,
    ).to(cfg["device"])

    calibration_samples = []
    if args.calibration_data:
        with open(args.calibration_data) as f:
            calibration_samples = json.load(f)[:50]

    print(f"Computing head importance with method='{method}'...")
    if method == "taylor":
        importance = compute_head_importance_taylor(model, processor, calibration_samples, cfg)
    else:
        importance = compute_head_importance_entropy(model, processor, calibration_samples, cfg)

    print(f"Pruning with target sparsity={sparsity:.0%}...")
    model, prune_map = prune_heads(model, importance, sparsity)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)
        with open(os.path.join(args.save, "prune_config.json"), "w") as f:
            json.dump({"method": method, "sparsity": sparsity, "pruned_heads": {
                k: v for k, v in prune_map.items()
            }}, f, indent=2)
        print(f"Pruned model saved to {args.save}")

    print("Pruning complete.")


if __name__ == "__main__":
    main()
