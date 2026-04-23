"""
LIBERO benchmark evaluation for OpenVLA.
Usage:
    python eval/libero_eval.py --task object --episodes 20
    python eval/libero_eval.py --task goal --episodes 20 --checkpoint outputs/lora_epoch_3
    python eval/libero_eval.py --task goal --episodes 20 --use-yolo
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image


TASK_SUITE_MAP = {
    "object": "libero_object",
    "goal": "libero_goal",
    "spatial": "libero_spatial",
    "long": "libero_long",
}


def create_libero_env(task_suite: str, task_idx: int = 0):
    """Create a LIBERO environment for evaluation."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark(task_suite)()
    task = bench.get_task(task_idx)
    task_name = task.name
    task_bddl = task.problem

    env_args = {
        "bddl_file_name": task_bddl,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    instruction = task.language
    return env, instruction, task_name


def run_episode(model, processor, env, instruction: str, cfg: dict,
                yolo_preprocessor=None, max_steps: int = 300) -> dict:
    obs = env.reset()
    total_reward = 0.0
    success = False
    actions_taken = []

    for step in range(max_steps):
        image = Image.fromarray(obs["agentview_image"])

        current_instruction = instruction
        if yolo_preprocessor:
            from inference.yolo_preprocess import inject_bbox_into_prompt, bbox_to_spatial_prior
            enhanced, detections, target = yolo_preprocessor.process(
                image, instruction, target_name=None,
            )
            current_instruction = enhanced

        prompt = f"In: What action should the robot take to {current_instruction}?\nOut:"
        inputs = processor(prompt, image).to(cfg["device"], dtype=torch.bfloat16)

        with torch.no_grad():
            action = model.predict_action_token_ids(inputs, max_new_tokens=cfg["action"]["dim"])
            action = processor.decode_action(
                action, n_action_bins=cfg["action"]["n_bins"],
                unnorm_key=cfg["action"]["unnorm_key"],
            )

        obs, reward, done, info = env.step(action.cpu().numpy().flatten())
        total_reward += reward
        actions_taken.append(action.cpu().numpy().tolist())

        if done or info.get("success", False):
            success = info.get("success", False)
            break

    return {
        "success": success,
        "total_reward": total_reward,
        "steps": step + 1,
        "actions": actions_taken,
    }


def evaluate_task_suite(model, processor, cfg: dict, task_suite: str,
                         n_episodes: int, yolo_preprocessor=None) -> dict:
    results = defaultdict(list)
    suite_name = TASK_SUITE_MAP.get(task_suite, task_suite)

    print(f"\n{'='*60}")
    print(f"Evaluating: {suite_name} | Episodes: {n_episodes}")
    print(f"{'='*60}")

    from libero.libero import benchmark
    bench = benchmark.get_benchmark(suite_name)()
    n_tasks = bench.n_tasks

    for task_idx in range(n_tasks):
        env, instruction, task_name = create_libero_env(suite_name, task_idx)
        print(f"\nTask {task_idx}: {task_name}")
        print(f"  Instruction: {instruction}")

        task_successes = 0
        task_steps = []

        for ep in range(n_episodes):
            t0 = time.perf_counter()
            result = run_episode(
                model, processor, env, instruction, cfg,
                yolo_preprocessor=yolo_preprocessor,
                max_steps=cfg["libero"]["max_steps"],
            )
            elapsed = time.perf_counter() - t0

            status = "SUCCESS" if result["success"] else "FAIL"
            print(f"  Episode {ep+1}/{n_episodes}: {status} "
                  f"({result['steps']} steps, {elapsed:.1f}s)")

            task_successes += int(result["success"])
            task_steps.append(result["steps"])
            results["all_results"].append({
                "task": task_name,
                "episode": ep,
                **result,
            })

        task_sr = task_successes / n_episodes
        avg_steps = np.mean(task_steps)
        print(f"  Task SR: {task_sr:.0%} ({task_successes}/{n_episodes}), "
              f"Avg steps: {avg_steps:.0f}")
        results["per_task"].append({
            "task": task_name,
            "success_rate": task_sr,
            "avg_steps": avg_steps,
        })
        env.close()

    overall_sr = np.mean([t["success_rate"] for t in results["per_task"]])
    results["overall"] = {
        "suite": suite_name,
        "success_rate": overall_sr,
        "n_tasks": n_tasks,
        "n_episodes": n_episodes,
    }
    print(f"\n{'='*60}")
    print(f"Overall {suite_name} success rate: {overall_sr:.1%}")
    print(f"{'='*60}")
    return dict(results)


def main():
    parser = argparse.ArgumentParser(description="LIBERO Evaluation for OpenVLA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--task", type=str, required=True,
                        choices=["object", "goal", "spatial", "long"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA/DPO checkpoint path")
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLOv8 preprocessing")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-results", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.device:
        cfg["device"] = args.device

    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name"], cache_dir=cfg["model"]["cache_dir"], trust_remote_code=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"]["cache_dir"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"],
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(cfg["device"])

    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()
        print(f"Loaded checkpoint: {args.checkpoint}")

    model.eval()

    yolo_preprocessor = None
    if args.use_yolo:
        from inference.yolo_preprocess import YOLOPreprocessor
        yolo_preprocessor = YOLOPreprocessor(
            model_path=cfg["yolo"]["model"],
            device=cfg["device"],
            confidence_threshold=cfg["yolo"]["confidence_threshold"],
        )
        print("YOLOv8 preprocessing enabled")

    results = evaluate_task_suite(
        model, processor, cfg,
        task_suite=args.task,
        n_episodes=args.episodes,
        yolo_preprocessor=yolo_preprocessor,
    )

    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results) or ".", exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.save_results}")


if __name__ == "__main__":
    main()
