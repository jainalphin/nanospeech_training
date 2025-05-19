from functools import partial
from torch.optim import AdamW
from datasets import load_dataset
import argparse
import os
import wandb

from nanospeech.nanospeech_torch import (
    Nanospeech,
    DiT,
    list_str_to_vocab_tensor,
    SAMPLES_PER_SECOND
)
from nanospeech.trainer_torch import NanospeechTrainer


def parse_args():
    """
    Parse command-line arguments for training.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train NanoSpeech Model")

    # Add command-line arguments
    parser.add_argument("--total_steps", type=int, default=80,
                        help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size for training")
    parser.add_argument("--max_duration_sec", type=float, default=10.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--save_step", type=int, default=80,
                        help="Save checkpoint every N steps")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--dataset", type=str, default="AlphJain/tts_10_sec_dataset",
                        help="HuggingFace dataset to use")
    parser.add_argument("--vocab_file", type=str, default="vocab.txt",
                        help="Path to vocabulary file")
    parser.add_argument("--wandb_project", type=str, default="nanospeech",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (default: auto-generated)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")

    # Model architecture arguments
    parser.add_argument("--dim", type=int, default=512,
                        help="Model embedding dimension")
    parser.add_argument("--depth", type=int, default=18,
                        help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--ff_mult", type=int, default=2,
                        help="Feed forward multiplier")
    parser.add_argument("--conv_layers", type=int, default=4,
                        help="Number of convolutional layers")

    return parser.parse_args()


def setup_wandb(args):
    """
    Set up Weights & Biases logging if enabled.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    wandb.run or None
        The initialized W&B run object, or None if W&B is disabled.
    """
    if args.no_wandb:
        return None

    # Try to get API key from environment variable
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if not wandb_api_key:
        print("Warning: WANDB_API_KEY environment variable not found.")
        print("Either set this environment variable or run with --no_wandb to disable W&B logging.")
        print("To set the environment variable:")
        print("  - Linux/macOS: export WANDB_API_KEY=your_api_key")
        print("  - Windows: set WANDB_API_KEY=your_api_key")
        print("Continuing without W&B logging...")
        return None

    # Initialize wandb
    try:
        wandb.login(key=wandb_api_key)
        print(f"W&B logging enabled.")
        return True
    except Exception as e:
        print(f"Error initializing W&B: {e}")
        print("Continuing without W&B logging...")
        return None


def train():
    """
    Main training function for Nanospeech model.
    """
    # Parse command-line arguments
    args = parse_args()

    # Set up Weights & Biases
    wandb_run = setup_wandb(args)

    # Configure accelerate kwargs based on wandb availability
    accelerate_kwargs = {
        "mixed_precision": "bf16",
    }

    if wandb_run is not None:
        accelerate_kwargs["log_with"] = "wandb"

    # Set up constants
    SAMPLE_RATE = 24_000
    HOP_LENGTH = 256
    SAMPLES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH

    # Load vocabulary
    try:
        with open(args.vocab_file, "r") as f:
            vocab = {v: i for i, v in enumerate(f.read().splitlines())}
        tokenizer = partial(list_str_to_vocab_tensor, vocab=vocab)
        text_num_embeds = len(vocab)
        print(f"Loaded vocabulary with {text_num_embeds} tokens")
    except FileNotFoundError:
        print(f"Vocabulary file {args.vocab_file} not found. Please check the path.")
        return

    # Set up the model
    model = Nanospeech(
        transformer=DiT(
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            text_dim=args.dim,
            ff_mult=args.ff_mult,
            conv_layers=args.conv_layers,
            text_num_embeds=text_num_embeds,
        ),
        tokenizer=tokenizer,
    )

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Configure optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Set up the trainer
    trainer = NanospeechTrainer(
        model,
        optimizer,
        num_warmup_steps=args.warmup_steps,
        sample_rate=SAMPLE_RATE,
        accelerate_kwargs=accelerate_kwargs,
    )

    # Load dataset
    try:
        dataset = load_dataset(
            args.dataset,
            split="train",            
        )
        print(f"Dataset '{args.dataset}' loaded successfully")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Configure batch size and duration limits
    batch_size = args.batch_size
    max_duration_sec = args.max_duration_sec
    max_duration = int(max_duration_sec * SAMPLES_PER_SECOND)
    max_batch_frames = int(batch_size * max_duration)

    print(f"\n===== Training Configuration =====")
    print(f"Total Steps: {args.total_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"Max Duration: {max_duration_sec} seconds")
    print(f"Learning Rate: {args.lr}")
    print(f"Warmup Steps: {args.warmup_steps}")
    print(f"Save Checkpoint Every: {args.save_step} steps")
    print(f"W&B Logging: {'Enabled' if wandb_run else 'Disabled'}")
    print(f"================================\n")

    # Train the model
    trainer.train(
        dataset,
        args.total_steps,
        batch_size=batch_size,
        max_batch_frames=max_batch_frames,
        max_duration=max_duration,
        num_workers=args.num_workers,
        save_step=args.save_step,
    )

    # Close wandb run if active
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    train()
