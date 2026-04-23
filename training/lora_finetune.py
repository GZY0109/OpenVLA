"""
LoRA fine-tuning for OpenVLA on LIBERO tasks.
Usage:
    python training/lora_finetune.py --config configs/default.yaml --data data/sft/manifest.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class VLAFineTuneDataset(Dataset):
    def __init__(self, manifest_path: str, processor):
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        instruction = sample["instruction"]
        action = torch.tensor(sample["action"], dtype=torch.float32)

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image)
        return {**inputs, "labels": action}


def setup_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_and_processor(cfg: dict):
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
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return model, processor


def train(model, train_loader, cfg: dict):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["training"]["max_epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    device = cfg["device"]
    grad_accum = cfg["training"]["gradient_accumulation_steps"]

    for epoch in range(cfg["training"]["max_epochs"]):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"]["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum

            if (step + 1) % 50 == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"Epoch {epoch+1}, Step {step+1}/{len(train_loader)}, Loss: {avg_loss:.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")

        save_dir = os.path.join(cfg["output_dir"], f"lora_epoch_{epoch+1}")
        model.save_pretrained(save_dir)
        print(f"LoRA weights saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for OpenVLA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to SFT manifest.json")
    parser.add_argument("--rank", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.rank:
        cfg["lora"]["rank"] = args.rank
        cfg["lora"]["alpha"] = args.rank * 2
    if args.epochs:
        cfg["training"]["max_epochs"] = args.epochs
    if args.device:
        cfg["device"] = args.device

    print("Loading model and processor...")
    model, processor = load_model_and_processor(cfg)

    print("Applying LoRA...")
    model = setup_lora(model, cfg)
    model.to(cfg["device"])

    print(f"Loading dataset from {args.data}...")
    dataset = VLAFineTuneDataset(args.data, processor)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Starting training: {cfg['training']['max_epochs']} epochs, "
          f"LoRA rank={cfg['lora']['rank']}")
    train(model, train_loader, cfg)
    print("Training complete.")


if __name__ == "__main__":
    main()
