#!/usr/bin/env python3
"""
Simple inference script for OLMo models.

Usage:
    python inference/simple_inference.py \
        --checkpoint-dir /path/to/checkpoint \
        --prompt "Once upon a time"
    
    # Or for batch inference:
    python inference/simple_inference.py \
        --checkpoint-dir /path/to/checkpoint \
        --prompts "Prompt 1" "Prompt 2" "Prompt 3"
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch

# Add parent directory to path to import olmo
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.config import TrainConfig
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer


def load_model_and_tokenizer(checkpoint_dir: str, device: str = "cuda"):
    """
    Load model and tokenizer from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., step0-unsharded)
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {checkpoint_dir}...")
    
    # Load model from checkpoint
    model = OLMo.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device=device,
        checkpoint_type=None  # Will auto-detect
    )
    model.eval()
    
    # Fix for STU models: ensure all tensors (including non-parameters) are on the correct device
    # The phi tensor in STU is not registered as a buffer, so it needs manual device placement
    if device == "cuda":
        for module in model.modules():
            if hasattr(module, 'phi') and isinstance(module.phi, torch.Tensor):
                if module.phi.device.type != 'cuda':
                    module.phi = module.phi.cuda()
    
    # Load tokenizer from config
    config_path = Path(checkpoint_dir) / "config.yaml"
    train_config = TrainConfig.load(config_path, validate_paths=False)
    tokenizer = Tokenizer.from_train_config(train_config)
    
    print(f"Model loaded successfully on {device}")
    print(f"Model config: {model.config.n_layers} layers, {model.config.d_model} hidden size")
    
    return model, tokenizer


def generate_text(
    model: OLMo,
    tokenizer: Tokenizer,
    prompts: List[str],
    max_steps: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda"
) -> List[str]:
    """
    Generate text completions for given prompts.
    
    Args:
        model: Loaded OLMo model
        tokenizer: Loaded tokenizer
        prompts: List of text prompts
        max_steps: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no change, < 1.0 = more focused)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        device: Device model is on
    
    Returns:
        List of generated text completions
    """
    from olmo.beam_search import TopPSampler, TopKSampler, MultinomialSampler
    
    completions = []
    
    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        
        # Generate
        with torch.inference_mode():
            # Use greedy decoding (beam_size=1) with optional sampling
            sampler = None
            if top_p < 1.0:
                sampler = TopPSampler(p=top_p, temperature=temperature)
            elif top_k > 0 and top_k < model.config.vocab_size:
                sampler = TopKSampler(k=top_k, temperature=temperature)
            elif temperature != 1.0:
                sampler = MultinomialSampler(temperature=temperature)
            
            output = model.generate(
                input_tensor,
                max_steps=max_steps,
                beam_size=1,  # Greedy decoding
                sampler=sampler,
            )
        
        # Decode the generated tokens
        # token_ids shape: (batch_size, beam_size, max_steps)
        # We use beam_size=1, so take [0, 0] to get the first beam of first batch
        generated_ids = output.token_ids[0, 0].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        print(f"Generated: {generated_text}")
        completions.append(generated_text)
    
    return completions


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on OLMo models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt
  python inference/simple_inference.py \\
      --checkpoint-dir /scratch/gpfs/EHAZAN/tharuntk/OLMo-data/checkpoints/OLMo-150M-Ablations/OLMo-STU-All-150M/step0-unsharded \\
      --prompt "The capital of France is"
  
  # Multiple prompts
  python inference/simple_inference.py \\
      --checkpoint-dir /scratch/gpfs/EHAZAN/tharuntk/OLMo-data/checkpoints/OLMo-150M-Ablations/OLMo-STU-All-150M/step0-unsharded \\
      --prompts "Once upon a time" "In a galaxy far far away" "The meaning of life is"
  
  # Use latest checkpoint
  python inference/simple_inference.py \\
      --checkpoint-dir /scratch/gpfs/EHAZAN/tharuntk/OLMo-data/checkpoints/OLMo-150M-Ablations/OLMo-STU-All-150M/latest-unsharded \\
      --prompt "Hello, world"
  
  # Control generation parameters
  python inference/simple_inference.py \\
      --checkpoint-dir /path/to/checkpoint \\
      --prompt "Once upon a time" \\
      --max-tokens 100 \\
      --top-p 0.9 \\
      --device cpu
        """
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., step0-unsharded or latest-unsharded)"
    )
    
    # Prompt arguments (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt text"
    )
    prompt_group.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Multiple prompts"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter (default: 0.95)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda if available, else cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint directory exists
    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory does not exist: {args.checkpoint_dir}")
        sys.exit(1)
    
    # Check if config.yaml exists
    if not (checkpoint_path / "config.yaml").exists():
        print(f"Error: No config.yaml found in {args.checkpoint_dir}")
        sys.exit(1)
    
    # Check if model.pt exists (for unsharded checkpoints)
    if not (checkpoint_path / "model.pt").exists():
        print(f"Warning: No model.pt found in {args.checkpoint_dir}")
        print("This might be a sharded checkpoint. The script will attempt to load it.")
    
    # Prepare prompts
    prompts = [args.prompt] if args.prompt else args.prompts
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
    
    # Generate text
    print(f"\nGenerating completions for {len(prompts)} prompt(s)...")
    completions = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_steps=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device
    )
    
    print(f"\n{'='*80}")
    print("Generation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

