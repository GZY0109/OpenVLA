"""
INT8 quantization for OpenVLA inference acceleration.
Usage:
    python optimization/quantize_int8.py --config configs/default.yaml --method dynamic
    python optimization/quantize_int8.py --config configs/default.yaml --method static --calibration-data data/sft/manifest.json
"""

import argparse
import json
import os
import time

import torch
import yaml
from PIL import Image


def dynamic_quantize(model, sensitive_layers: list):
    from torch.quantization import quantize_dynamic

    modules_to_quantize = {torch.nn.Linear}
    quantized = quantize_dynamic(model, modules_to_quantize, dtype=torch.qint8)

    # Restore sensitive layers to FP16 (mixed precision)
    # NOTE: torch quantize_dynamic doesn't support per-layer exclusion natively;
    # for production, use bitsandbytes or GPTQ
    return quantized


def static_quantize_with_calibration(model, processor, calibration_data_path: str,
                                      cfg: dict, n_samples: int = 100):
    """Static quantization with calibration data for better accuracy."""
    from torch.quantization import prepare, convert, QConfig, default_observer, default_weight_observer

    qconfig = QConfig(
        activation=default_observer,
        weight=default_weight_observer,
    )
    model.qconfig = qconfig

    prepared = prepare(model, inplace=False)

    # Run calibration
    with open(calibration_data_path) as f:
        samples = json.load(f)[:n_samples]

    print(f"Running calibration with {len(samples)} samples...")
    with torch.no_grad():
        for i, sample in enumerate(samples):
            image = Image.open(sample["image"]).convert("RGB")
            prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut:"
            inputs = processor(prompt, image).to(cfg["device"])
            prepared(**inputs)
            if (i + 1) % 20 == 0:
                print(f"  Calibrated {i+1}/{len(samples)}")

    quantized = convert(prepared, inplace=False)
    return quantized


def bitsandbytes_int8(model_name: str, cache_dir: str, device: str):
    """Load model with bitsandbytes 8-bit quantization (recommended for LLMs)."""
    from transformers import AutoModelForVision2Seq, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["action_head", "lm_head"],
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def benchmark_quantized(model, processor, cfg: dict, n_iters: int = 20):
    dummy_image = Image.new("RGB", tuple(cfg["libero"]["image_size"]), color=(128, 128, 128))
    instruction = "pick up the red cup"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # warmup
    for _ in range(3):
        inputs = processor(prompt, dummy_image).to(cfg["device"])
        with torch.no_grad():
            model(**inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        inputs = processor(prompt, dummy_image).to(cfg["device"])
        with torch.no_grad():
            model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"Quantized latency: avg={avg:.1f}ms, p50={p50:.1f}ms, peak_mem={mem_gb:.2f}GB")
    return {"avg_ms": avg, "p50_ms": p50, "peak_mem_gb": mem_gb}


def main():
    parser = argparse.ArgumentParser(description="INT8 Quantization for OpenVLA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str, default="bitsandbytes",
                        choices=["dynamic", "static", "bitsandbytes"])
    parser.add_argument("--calibration-data", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--save", type=str, default=None, help="Save quantized model to path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name"], cache_dir=cfg["model"]["cache_dir"], trust_remote_code=True,
    )

    if args.method == "bitsandbytes":
        print("Loading model with bitsandbytes INT8...")
        model = bitsandbytes_int8(cfg["model"]["name"], cfg["model"]["cache_dir"], cfg["device"])
    else:
        print("Loading FP16 model...")
        model = AutoModelForVision2Seq.from_pretrained(
            cfg["model"]["name"],
            cache_dir=cfg["model"]["cache_dir"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(cfg["device"])

        if args.method == "dynamic":
            print("Applying dynamic INT8 quantization...")
            model = dynamic_quantize(model, cfg["quantization"]["sensitive_layers"])
        elif args.method == "static":
            if not args.calibration_data:
                print("Error: --calibration-data required for static quantization")
                return
            print("Applying static INT8 quantization with calibration...")
            model = static_quantize_with_calibration(
                model, processor, args.calibration_data, cfg,
            )

    if args.benchmark:
        benchmark_quantized(model, processor, cfg)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)
        print(f"Quantized model saved to {args.save}")

    print("Quantization complete.")


if __name__ == "__main__":
    main()
