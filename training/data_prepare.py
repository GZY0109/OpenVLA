"""
LIBERO data preparation: extract instruction-image-action triplets for SFT/DPO.
Usage:
    python training/data_prepare.py --task libero_object --output data/sft
    python training/data_prepare.py --task libero_goal --mode dpo --output data/dpo
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def load_libero_hdf5(hdf5_path: str) -> dict:
    data = {"observations": [], "actions": [], "dones": []}
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        for demo_key in demos:
            demo = f["data"][demo_key]
            obs = demo["obs"]["agentview_rgb"][:]
            acts = demo["actions"][:]
            dones = demo["dones"][:]
            data["observations"].append(obs)
            data["actions"].append(acts)
            data["dones"].append(dones)
    return data


def extract_sft_triplets(hdf5_path: str, instruction: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    data = load_libero_hdf5(hdf5_path)

    triplets = []
    for ep_idx, (obs, acts) in enumerate(zip(data["observations"], data["actions"])):
        ep_dir = os.path.join(output_dir, f"episode_{ep_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        for step_idx in range(len(acts)):
            img = Image.fromarray(obs[step_idx])
            img_path = os.path.join(ep_dir, f"step_{step_idx:04d}.png")
            img.save(img_path)

            triplets.append({
                "image": img_path,
                "instruction": instruction,
                "action": acts[step_idx].tolist(),
            })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(triplets, f, indent=2)
    print(f"Extracted {len(triplets)} SFT triplets → {manifest_path}")
    return triplets


def extract_dpo_pairs(hdf5_paths: dict, instruction: str, output_dir: str):
    """
    hdf5_paths: {"success": [path, ...], "failure": [path, ...]}
    """
    os.makedirs(output_dir, exist_ok=True)

    success_data = []
    for p in hdf5_paths["success"]:
        d = load_libero_hdf5(p)
        for obs, acts in zip(d["observations"], d["actions"]):
            success_data.append({"obs": obs, "acts": acts})

    failure_data = []
    for p in hdf5_paths["failure"]:
        d = load_libero_hdf5(p)
        for obs, acts in zip(d["observations"], d["actions"]):
            failure_data.append({"obs": obs, "acts": acts})

    pairs = []
    n_pairs = min(len(success_data), len(failure_data))
    for i in range(n_pairs):
        img = Image.fromarray(success_data[i]["obs"][0])
        img_path = os.path.join(output_dir, f"pair_{i:04d}.png")
        img.save(img_path)

        pairs.append({
            "image": img_path,
            "instruction": instruction,
            "chosen_actions": success_data[i]["acts"].tolist(),
            "rejected_actions": failure_data[i]["acts"].tolist(),
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Extracted {len(pairs)} DPO pairs → {manifest_path}")
    return pairs


def augment_instructions(instruction: str) -> list:
    templates = [
        instruction,
        instruction.replace("pick up", "grasp").replace("place", "put"),
        instruction.lower(),
        "please " + instruction.lower(),
    ]
    return list(set(templates))


def main():
    parser = argparse.ArgumentParser(description="LIBERO Data Preparation")
    parser.add_argument("--hdf5", type=str, help="Path to LIBERO HDF5 file")
    parser.add_argument("--task", type=str, default="libero_object",
                        choices=["libero_object", "libero_goal", "libero_spatial", "libero_long"])
    parser.add_argument("--instruction", type=str, default="pick up the object")
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "dpo"])
    parser.add_argument("--output", type=str, default="data/sft")
    parser.add_argument("--success-dir", type=str, help="Dir of success rollout HDF5s (DPO mode)")
    parser.add_argument("--failure-dir", type=str, help="Dir of failure rollout HDF5s (DPO mode)")
    args = parser.parse_args()

    if args.mode == "sft":
        if not args.hdf5:
            print("Error: --hdf5 required for SFT mode")
            return
        extract_sft_triplets(args.hdf5, args.instruction, args.output)
    elif args.mode == "dpo":
        if not args.success_dir or not args.failure_dir:
            print("Error: --success-dir and --failure-dir required for DPO mode")
            return
        success_files = sorted(Path(args.success_dir).glob("*.hdf5"))
        failure_files = sorted(Path(args.failure_dir).glob("*.hdf5"))
        hdf5_paths = {
            "success": [str(f) for f in success_files],
            "failure": [str(f) for f in failure_files],
        }
        extract_dpo_pairs(hdf5_paths, args.instruction, args.output)


if __name__ == "__main__":
    main()
