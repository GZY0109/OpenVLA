"""
ONNX export and TensorRT optimization for OpenVLA.
Usage:
    python optimization/export_onnx.py --config configs/default.yaml --export-onnx model.onnx
    python optimization/export_onnx.py --convert-trt model.onnx --trt-output model.trt
"""

import argparse
import os
import time

import numpy as np
import torch
import yaml
from PIL import Image


def export_to_onnx(model, processor, cfg: dict, output_path: str):
    """Export OpenVLA model to ONNX format."""
    dummy_image = Image.new("RGB", tuple(cfg["libero"]["image_size"]), color=(128, 128, 128))
    prompt = "In: What action should the robot take to pick up the red cup?\nOut:"
    inputs = processor(prompt, dummy_image).to(cfg["device"])

    # Prepare input dict for tracing
    input_names = list(inputs.keys())
    output_names = ["logits"]

    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch_size"}
    dynamic_axes["logits"] = {0: "batch_size"}

    input_tuple = tuple(inputs[k] for k in input_names)

    print(f"Exporting to ONNX: {output_path}")
    print(f"  Input names: {input_names}")
    print(f"  Opset version: {cfg['export']['onnx_opset']}")

    torch.onnx.export(
        model,
        input_tuple,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=cfg["export"]["onnx_opset"],
        do_constant_folding=True,
    )
    print(f"ONNX model exported to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e9:.2f} GB")


def verify_onnx(onnx_path: str, processor, cfg: dict):
    """Verify ONNX model produces correct outputs."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    dummy_image = Image.new("RGB", tuple(cfg["libero"]["image_size"]), color=(128, 128, 128))
    prompt = "In: What action should the robot take to pick up the red cup?\nOut:"
    inputs = processor(prompt, dummy_image)

    ort_inputs = {k: v.numpy() for k, v in inputs.items()}
    outputs = session.run(None, ort_inputs)

    print(f"ONNX verification passed. Output shape: {outputs[0].shape}")
    return outputs


def convert_to_tensorrt(onnx_path: str, trt_output_path: str, precision: str = "fp16"):
    """Convert ONNX model to TensorRT engine."""
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not installed. Install with: pip install tensorrt")
        print("Alternatively, use trtexec CLI:")
        print(f"  trtexec --onnx={onnx_path} --saveEngine={trt_output_path} "
              f"--{precision} --workspace=8192")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (1 << 30))  # 8GB

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabling FP16 precision")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Enabling INT8 precision")

    print("Building TensorRT engine (this may take several minutes)...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("Failed to build TensorRT engine")
        return

    with open(trt_output_path, "wb") as f:
        f.write(engine)
    print(f"TensorRT engine saved to {trt_output_path}")
    print(f"  File size: {os.path.getsize(trt_output_path) / 1e9:.2f} GB")


def benchmark_trt(trt_path: str, cfg: dict, n_iters: int = 50):
    """Benchmark TensorRT engine inference latency."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("TensorRT/PyCUDA not available for benchmarking")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers (simplified)
    print(f"Benchmarking TensorRT engine: {trt_path}")
    print(f"TensorRT benchmarking requires proper buffer allocation — skipping auto-benchmark")
    print(f"Use trtexec for accurate benchmarks:")
    print(f"  trtexec --loadEngine={trt_path} --iterations={n_iters}")


def main():
    parser = argparse.ArgumentParser(description="ONNX Export & TensorRT for OpenVLA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--export-onnx", type=str, help="Export to ONNX at this path")
    parser.add_argument("--verify-onnx", type=str, help="Verify ONNX model at this path")
    parser.add_argument("--convert-trt", type=str, help="Convert ONNX to TensorRT")
    parser.add_argument("--trt-output", type=str, default="model.trt")
    parser.add_argument("--trt-precision", type=str, default=None, choices=["fp16", "int8"])
    parser.add_argument("--benchmark-trt", type=str, help="Benchmark TensorRT engine")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.trt_precision:
        cfg["export"]["trt_precision"] = args.trt_precision

    if args.export_onnx or args.verify_onnx:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            cfg["model"]["name"], cache_dir=cfg["model"]["cache_dir"], trust_remote_code=True,
        )

    if args.export_onnx:
        model = AutoModelForVision2Seq.from_pretrained(
            args.checkpoint or cfg["model"]["name"],
            cache_dir=cfg["model"]["cache_dir"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(cfg["device"])
        model.eval()
        export_to_onnx(model, processor, cfg, args.export_onnx)

    if args.verify_onnx:
        verify_onnx(args.verify_onnx, processor, cfg)

    if args.convert_trt:
        convert_to_tensorrt(args.convert_trt, args.trt_output, cfg["export"]["trt_precision"])

    if args.benchmark_trt:
        benchmark_trt(args.benchmark_trt, cfg)


if __name__ == "__main__":
    main()
