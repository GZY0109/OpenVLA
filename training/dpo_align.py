"""
DPO (Direct Preference Optimization) alignment for OpenVLA.
Usage:
    python training/dpo_align.py --config configs/default.yaml --data data/dpo/manifest.json
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DPODataset(Dataset):
    def __init__(self, manifest_path: str, processor):
        with open(manifest_path) as f:
            self.pairs = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair["image"]).convert("RGB")
        instruction = pair["instruction"]
        chosen = torch.tensor(pair["chosen_actions"], dtype=torch.float32)
        rejected = torch.tensor(pair["rejected_actions"], dtype=torch.float32)

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image)
        return {**inputs, "chosen_actions": chosen, "rejected_actions": rejected}


def compute_log_probs(model, inputs, actions):
    outputs = model(**inputs)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log probs for action tokens
    action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return action_log_probs.sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss, chosen_rewards.mean(), rejected_rewards.mean()


def train_dpo(policy_model, ref_model, train_loader, cfg: dict):
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg["dpo"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    beta = cfg["dpo"]["beta"]
    device = cfg["device"]
    ref_model.eval()

    for epoch in range(cfg["dpo"]["max_epochs"]):
        epoch_loss = 0.0
        epoch_chosen_reward = 0.0
        epoch_rejected_reward = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            chosen_inputs = {k: v for k, v in batch.items()
                           if k not in ("chosen_actions", "rejected_actions")}

            with torch.no_grad():
                ref_chosen_logps = compute_log_probs(ref_model, chosen_inputs, batch["chosen_actions"])
                ref_rejected_logps = compute_log_probs(ref_model, chosen_inputs, batch["rejected_actions"])

            policy_chosen_logps = compute_log_probs(policy_model, chosen_inputs, batch["chosen_actions"])
            policy_rejected_logps = compute_log_probs(policy_model, chosen_inputs, batch["rejected_actions"])

            loss, chosen_reward, rejected_reward = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps, beta,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg["training"]["max_grad_norm"])
            optimizer.step()

            epoch_loss += loss.item()
            epoch_chosen_reward += chosen_reward.item()
            epoch_rejected_reward += rejected_reward.item()

            if (step + 1) % 20 == 0:
                n = step + 1
                print(f"Epoch {epoch+1}, Step {n}, "
                      f"Loss: {epoch_loss/n:.4f}, "
                      f"Chosen R: {epoch_chosen_reward/n:.4f}, "
                      f"Rejected R: {epoch_rejected_reward/n:.4f}")

        n = len(train_loader)
        print(f"Epoch {epoch+1} done. Loss: {epoch_loss/n:.4f}, "
              f"Reward margin: {(epoch_chosen_reward - epoch_rejected_reward)/n:.4f}")

        save_dir = os.path.join(cfg["output_dir"], f"dpo_epoch_{epoch+1}")
        policy_model.save_pretrained(save_dir)
        print(f"Saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="DPO Alignment for OpenVLA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to DPO manifest.json")
    parser.add_argument("--policy-checkpoint", type=str, default=None,
                        help="Path to SFT LoRA checkpoint to start from")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.beta:
        cfg["dpo"]["beta"] = args.beta
    if args.device:
        cfg["device"] = args.device

    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name"], cache_dir=cfg["model"]["cache_dir"], trust_remote_code=True,
    )

    print("Loading reference model (frozen)...")
    ref_model = AutoModelForVision2Seq.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"]["cache_dir"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        trust_remote_code=True,
    ).to(cfg["device"])
    if args.policy_checkpoint:
        ref_model = PeftModel.from_pretrained(ref_model, args.policy_checkpoint)
        ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("Loading policy model...")
    policy_model = AutoModelForVision2Seq.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"]["cache_dir"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        trust_remote_code=True,
    ).to(cfg["device"])
    if args.policy_checkpoint:
        policy_model = PeftModel.from_pretrained(policy_model, args.policy_checkpoint)

    from training.lora_finetune import setup_lora
    if not args.policy_checkpoint:
        policy_model = setup_lora(policy_model, cfg)

    dataset = DPODataset(args.data, processor)
    train_loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=4)

    print(f"Starting DPO training: β={cfg['dpo']['beta']}, epochs={cfg['dpo']['max_epochs']}")
    train_dpo(policy_model, ref_model, train_loader, cfg)
    print("DPO training complete.")


if __name__ == "__main__":
    main()
