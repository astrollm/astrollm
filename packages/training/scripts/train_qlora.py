"""
AstroLLM QLoRA Training Script

Usage:
    uv run python packages/training/scripts/train_qlora.py --config configs/qwen3-8b-qlora-astro-sft-v001.yaml

Supports:
    - QLoRA and LoRA fine-tuning
    - Checkpoint resumption (spot instance safe)
    - W&B experiment tracking
    - Flash Attention 2
    - Gradient checkpointing
"""

import os
import yaml
import torch
import typer
import wandb
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

# These imports will work once dependencies are installed
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     TrainingArguments,
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from trl import SFTTrainer
# from datasets import load_dataset

console = Console()
app = typer.Typer()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict):
    """Configure BitsAndBytes quantization for QLoRA."""
    quant_config = config.get("quantization", {})
    if not quant_config.get("load_in_4bit", False):
        return None

    # from transformers import BitsAndBytesConfig
    # return BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
    #     bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
    #     bnb_4bit_compute_dtype=getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16")),
    # )
    console.print("[yellow]Quantization config prepared (uncomment imports to use)[/yellow]")
    return quant_config


def setup_lora(config: dict):
    """Configure LoRA adapter."""
    lora_config = config.get("lora", {})
    # from peft import LoraConfig
    # return LoraConfig(
    #     r=lora_config.get("r", 64),
    #     lora_alpha=lora_config.get("lora_alpha", 128),
    #     lora_dropout=lora_config.get("lora_dropout", 0.05),
    #     bias=lora_config.get("bias", "none"),
    #     target_modules=lora_config.get("target_modules", []),
    #     task_type=lora_config.get("task_type", "CAUSAL_LM"),
    # )
    console.print("[yellow]LoRA config prepared (uncomment imports to use)[/yellow]")
    return lora_config


def print_config_summary(config: dict):
    """Print a formatted summary of the training configuration."""
    table = Table(title="AstroLLM Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    exp = config.get("experiment", {})
    table.add_row("Experiment", exp.get("name", "unnamed"))
    table.add_row("Hypothesis", exp.get("hypothesis", "none")[:80])

    model = config.get("model", {})
    table.add_row("Base Model", model.get("base_model", "unknown"))

    lora = config.get("lora", {})
    table.add_row("LoRA Rank", str(lora.get("r", "N/A")))
    table.add_row("LoRA Alpha", str(lora.get("lora_alpha", "N/A")))

    training = config.get("training", {})
    table.add_row("Epochs", str(training.get("num_train_epochs", "N/A")))
    table.add_row("Batch Size", str(training.get("per_device_train_batch_size", "N/A")))
    table.add_row("Grad Accum", str(training.get("gradient_accumulation_steps", "N/A")))
    eff_batch = (
        training.get("per_device_train_batch_size", 1)
        * training.get("gradient_accumulation_steps", 1)
    )
    table.add_row("Effective Batch", str(eff_batch))
    table.add_row("Learning Rate", str(training.get("learning_rate", "N/A")))
    table.add_row("Max Seq Length", str(training.get("max_seq_length", "N/A")))

    console.print(table)


@app.command()
def train(
    config: str = typer.Option(..., help="Path to training config YAML"),
    resume: str = typer.Option(None, help="Path to checkpoint to resume from"),
    dry_run: bool = typer.Option(False, help="Print config and exit without training"),
):
    """Run QLoRA/LoRA fine-tuning for AstroLLM."""

    console.print("\n[bold blue]AstroLLM Training[/bold blue]")
    console.print(f"Config: {config}\n")

    # Load config
    cfg = load_config(config)
    print_config_summary(cfg)

    if dry_run:
        console.print("\n[yellow]Dry run — exiting without training.[/yellow]")
        return

    # Override resume path if provided via CLI
    if resume:
        cfg.setdefault("checkpointing", {})["resume_from_checkpoint"] = resume
        console.print(f"\n[yellow]Resuming from: {resume}[/yellow]")

    # Check for GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"\n[green]GPU: {gpu_name} ({gpu_mem:.1f} GB)[/green]")
    else:
        console.print("\n[red]No GPU detected! Training will be extremely slow.[/red]")
        console.print("[red]Use a cloud GPU (RunPod, Lambda Labs) for training.[/red]")
        return

    # ─── TRAINING PIPELINE ───
    # Uncomment and implement when ready to train:
    #
    # 1. Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    # tokenizer.pad_token = tokenizer.eos_token
    #
    # 2. Load model with quantization
    # bnb_config = setup_quantization(cfg)
    # model = AutoModelForCausalLM.from_pretrained(
    #     cfg["model"]["base_model"],
    #     quantization_config=bnb_config,
    #     device_map="auto",
    #     attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
    #     torch_dtype=getattr(torch, cfg["model"].get("torch_dtype", "bfloat16")),
    # )
    # model = prepare_model_for_kbit_training(model)
    #
    # 3. Apply LoRA
    # lora_config = setup_lora(cfg)
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    #
    # 4. Load dataset
    # dataset = load_dataset("json", data_files={
    #     "train": cfg["data"]["train_file"],
    #     "eval": cfg["data"]["eval_file"],
    # })
    #
    # 5. Configure training
    # training_args = TrainingArguments(
    #     output_dir=cfg["checkpointing"]["output_dir"],
    #     **cfg["training"],
    #     report_to="wandb",
    # )
    #
    # 6. Initialize W&B
    # wandb.init(
    #     project=cfg["wandb"]["project"],
    #     name=f"{cfg['experiment']['name']}-{datetime.now().strftime('%Y%m%d-%H%M')}",
    #     tags=cfg["wandb"].get("tags", []),
    #     config=cfg,
    # )
    #
    # 7. Train
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["eval"],
    #     args=training_args,
    #     max_seq_length=cfg["training"]["max_seq_length"],
    # )
    # trainer.train(resume_from_checkpoint=cfg["checkpointing"].get("resume_from_checkpoint"))
    #
    # 8. Save
    # trainer.save_model()
    # console.print(f"\n[green]Model saved to {cfg['checkpointing']['output_dir']}[/green]")

    console.print("\n[yellow]Training pipeline is scaffolded. Uncomment imports and pipeline code to run.[/yellow]")
    console.print("[yellow]See docs/V1_FINAL_PLAN.md for step-by-step instructions.[/yellow]")


if __name__ == "__main__":
    app()
