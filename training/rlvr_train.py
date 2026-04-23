"""
RLVR (RL from Verifiable Rewards) training for OpenVLA with LIBERO.
Supports GRPO, PPO, and REINFORCE.

Usage:
    python training/rlvr_train.py --config configs/default.yaml --algorithm grpo
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def compute_reward(trajectory: dict, task_spec: dict, cfg: dict) -> float:
    weights = cfg["rlvr"]["reward_weights"]

    task_success = float(trajectory.get("success", False))
    progress = trajectory.get("progress", 0.0)
    safety_penalty = trajectory.get("safety_penalty", 0.0)

    reward = (task_success * weights["task_success"]
              + progress * weights["progress"]
              + safety_penalty * weights["safety_penalty"])
    return reward


def rollout_episode(model, processor, env, instruction: str, cfg: dict) -> dict:
    obs = env.reset()
    trajectory = {"observations": [], "actions": [], "log_probs": [], "rewards": []}
    done = False
    step = 0
    max_steps = cfg["libero"]["max_steps"]

    while not done and step < max_steps:
        from PIL import Image
        image = Image.fromarray(obs["agentview_rgb"])
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, image).to(cfg["device"])

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -cfg["action"]["dim"]:, :]
            probs = F.softmax(logits, dim=-1)
            action_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).squeeze(-1)
            log_prob = F.log_softmax(logits, dim=-1)
            selected_log_prob = log_prob.view(-1, log_prob.size(-1)).gather(1, action_ids.unsqueeze(-1))

        action = processor.decode_action(
            action_ids, n_action_bins=cfg["action"]["n_bins"],
            unnorm_key=cfg["action"]["unnorm_key"],
        )

        obs, reward, done, info = env.step(action.cpu().numpy())

        trajectory["observations"].append(obs)
        trajectory["actions"].append(action_ids)
        trajectory["log_probs"].append(selected_log_prob.sum())
        trajectory["rewards"].append(reward)
        step += 1

    trajectory["success"] = info.get("task_success", False)
    trajectory["progress"] = info.get("progress", 0.0)
    trajectory["safety_penalty"] = 0.0
    trajectory["total_reward"] = compute_reward(trajectory, {}, cfg)
    return trajectory


def grpo_update(model, ref_model, trajectories: list, cfg: dict):
    """GRPO: Group Relative Policy Optimization."""
    optimizer = getattr(grpo_update, "_optimizer", None)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["rlvr"]["learning_rate"])
        grpo_update._optimizer = optimizer

    rewards = torch.tensor([t["total_reward"] for t in trajectories])
    # Group relative advantage
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss = 0.0
    for traj, adv in zip(trajectories, advantages):
        log_probs = torch.stack(traj["log_probs"])
        policy_loss = -(log_probs * adv).mean()

        # KL penalty against reference model
        kl_penalty = cfg["rlvr"]["kl_coeff"] * log_probs.mean()

        loss = policy_loss + kl_penalty
        total_loss += loss

    total_loss /= len(trajectories)
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
    optimizer.step()

    return total_loss.item(), rewards.mean().item()


def ppo_update(model, ref_model, trajectories: list, cfg: dict):
    """PPO with clipped surrogate objective."""
    optimizer = getattr(ppo_update, "_optimizer", None)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["rlvr"]["learning_rate"])
        ppo_update._optimizer = optimizer

    clip_eps = 0.2
    rewards = torch.tensor([t["total_reward"] for t in trajectories])
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss = 0.0
    for traj, adv in zip(trajectories, advantages):
        old_log_probs = torch.stack(traj["log_probs"]).detach()
        new_log_probs = torch.stack(traj["log_probs"])
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        loss = -torch.min(ratio * adv, clipped * adv).mean()
        total_loss += loss

    total_loss /= len(trajectories)
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
    optimizer.step()

    return total_loss.item(), rewards.mean().item()


def train_rlvr(model, ref_model, processor, env, instruction: str, cfg: dict):
    algorithm = cfg["rlvr"]["algorithm"]
    update_fn = {"grpo": grpo_update, "ppo": ppo_update}[algorithm]
    group_size = cfg["rlvr"]["group_size"]
    max_epochs = cfg["rlvr"]["max_epochs"]

    reward_history = deque(maxlen=100)
    best_reward = float("-inf")

    for epoch in range(max_epochs):
        trajectories = []
        for _ in range(group_size):
            traj = rollout_episode(model, processor, env, instruction, cfg)
            trajectories.append(traj)
            reward_history.append(traj["total_reward"])

        loss, avg_reward = update_fn(model, ref_model, trajectories, cfg)
        success_rate = sum(1 for t in trajectories if t["success"]) / len(trajectories)

        print(f"Epoch {epoch+1}/{max_epochs} | "
              f"Loss: {loss:.4f} | "
              f"Reward: {avg_reward:.2f} | "
              f"Success: {success_rate:.0%} | "
              f"Running avg: {np.mean(reward_history):.2f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            save_dir = os.path.join(cfg["output_dir"], "rlvr_best")
            model.save_pretrained(save_dir)
            print(f"  New best! Saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="RLVR Training for OpenVLA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--algorithm", type=str, default=None, choices=["grpo", "ppo"])
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA/DPO checkpoint to start from")
    parser.add_argument("--task", type=str, default="libero_object")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.algorithm:
        cfg["rlvr"]["algorithm"] = args.algorithm
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
        trust_remote_code=True,
    ).to(cfg["device"])

    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")

    ref_model = AutoModelForVision2Seq.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"]["cache_dir"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        trust_remote_code=True,
    ).to(cfg["device"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # LIBERO env setup (requires libero package)
    print(f"Setting up LIBERO environment: {args.task}")
    # env = create_libero_env(args.task)  # TODO: integrate with libero API
    # instruction = get_task_instruction(args.task)
    # train_rlvr(model, ref_model, processor, env, instruction, cfg)

    print("RLVR training setup complete. Uncomment env creation to run.")


if __name__ == "__main__":
    main()
