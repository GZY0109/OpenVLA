"""
OpenVLA single-step inference entry point.
Usage:
    python inference/run_inference.py --config configs/default.yaml --test
    python inference/run_inference.py --image path/to/img.png --instruction "pick up the red cup"
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from PIL import Image


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"]["cache_dir"],
        trust_remote_code=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"]["cache_dir"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"],
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(cfg["device"])
    model.eval()
    return model, processor


def predict_action(model, processor, image: Image.Image, instruction: str, cfg: dict) -> list:
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image).to(cfg["device"], dtype=torch.bfloat16)

    with torch.no_grad():
        action_ids = model.predict_action_token_ids(inputs, max_new_tokens=cfg["action"]["dim"])

    action = processor.decode_action(
        action_ids,
        n_action_bins=cfg["action"]["n_bins"],
        unnorm_key=cfg["action"]["unnorm_key"],
    )
    return action.tolist()


def benchmark_latency(model, processor, cfg: dict, n_iters: int = 20):
    dummy_image = Image.new("RGB", tuple(cfg["libero"]["image_size"]), color=(128, 128, 128))
    instruction = "pick up the red cup"

    # warmup
    for _ in range(3):
        predict_action(model, processor, dummy_image, instruction, cfg)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        predict_action(model, processor, dummy_image, instruction, cfg)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    print(f"Latency over {n_iters} iters: avg={avg:.1f}ms, p50={p50:.1f}ms")
    return avg


def main():
    parser = argparse.ArgumentParser(description="OpenVLA Inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--instruction", type=str, help="Language instruction")
    parser.add_argument("--test", action="store_true", help="Run latency benchmark")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device

    print(f"Loading model: {cfg['model']['name']}")
    model, processor = load_model(cfg)
    print("Model loaded.")

    if args.test:
        benchmark_latency(model, processor, cfg)
        return

    if args.image and args.instruction:
        image = Image.open(args.image).convert("RGB")
        action = predict_action(model, processor, image, args.instruction, cfg)
        print(f"Predicted action (7-DoF): {action}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
