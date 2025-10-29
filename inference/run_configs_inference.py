#!/usr/bin/env python3
"""
Run inference on all models from a config directory.

This script:
1. Reads all YAML config files in a specified directory
2. Extracts the save_folder path from each config
3. Finds the latest-unsharded checkpoint for each model
4. Runs inference on all models with the same prompts
5. Saves results to a JSON file

Usage:
    python inference/run_configs_inference.py \
        --config-dir configs/stu-ablations \
        --prompts "Once upon a time" "The capital of France is" \
        --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

# Add parent directory to path to import olmo
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.config import TrainConfig
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer
from olmo.beam_search import TopPSampler


def find_config_files(config_dir: Path) -> List[Path]:
    """Find all YAML config files in a directory."""
    config_files = []
    for pattern in ["*.yaml", "*.yml"]:
        config_files.extend(config_dir.glob(pattern))
    return sorted(config_files)


def extract_save_folder(config_path: Path) -> Optional[str]:
    """Extract save_folder path from a config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            save_folder = config.get('save_folder')
            if save_folder:
                # Handle variable substitution like ${run_name}
                run_name = config.get('run_name')
                if run_name and '${run_name}' in save_folder:
                    save_folder = save_folder.replace('${run_name}', run_name)
                return save_folder
    except Exception as e:
        print(f"Error reading {config_path}: {e}")
    return None


def find_latest_checkpoint(save_folder: str) -> Optional[Path]:
    """Find the latest-unsharded checkpoint in a save folder."""
    save_path = Path(save_folder)
    
    # Check if save folder exists
    if not save_path.exists():
        print(f"Save folder does not exist: {save_folder}")
        return None
    
    # Try latest-unsharded symlink first
    latest_link = save_path / "latest-unsharded"
    if latest_link.exists():
        return latest_link
    
    # Otherwise, find the highest numbered step*-unsharded directory
    unsharded_dirs = list(save_path.glob("step*-unsharded"))
    if not unsharded_dirs:
        print(f"No unsharded checkpoints found in {save_folder}")
        return None
    
    # Sort by step number
    def get_step_num(path: Path) -> int:
        try:
            return int(path.name.replace("step", "").replace("-unsharded", ""))
        except:
            return -1
    
    unsharded_dirs.sort(key=get_step_num, reverse=True)
    return unsharded_dirs[0]


def load_model_and_tokenizer(checkpoint_dir: Path, device: str = "cuda"):
    """Load model and tokenizer from checkpoint directory."""
    print(f"  Loading model from {checkpoint_dir}...")
    
    model = OLMo.from_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        device=device,
        checkpoint_type=None
    )
    model.eval()
    
    # Fix for STU models: ensure all tensors are on the correct device
    if device == "cuda":
        for module in model.modules():
            if hasattr(module, 'phi') and isinstance(module.phi, torch.Tensor):
                if module.phi.device.type != 'cuda':
                    module.phi = module.phi.cuda()
    
    config_path = checkpoint_dir / "config.yaml"
    train_config = TrainConfig.load(config_path, validate_paths=False)
    tokenizer = Tokenizer.from_train_config(train_config)
    
    return model, tokenizer


def generate_for_model(
    checkpoint_dir: Path,
    prompts: List[str],
    max_steps: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
    device: str = "cuda"
) -> List[Dict[str, str]]:
    """Generate completions for a model."""
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, device)
    
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        
        with torch.inference_mode():
            sampler = TopPSampler(p=top_p, temperature=temperature) if top_p < 1.0 else None
            output = model.generate(
                input_tensor,
                max_steps=max_steps,
                beam_size=1,
                sampler=sampler,
            )
        
        generated_ids = output.token_ids[0, 0].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        results.append({
            "prompt": prompt,
            "completion": generated_text
        })
    
    # Clean up
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on all models from a config directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python inference/run_configs_inference.py \\
      --config-dir configs/stu-ablations \\
      --prompts "Once upon a time" "The capital of France is" "Machine learning is" \\
      --output stu_ablations_results.json \\
      --max-tokens 50
        """
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Directory containing config YAML files"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="List of prompts to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results.json",
        help="Output JSON file (default: inference_results.json)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        default="latest",
        choices=["latest", "step0"],
        help="Which checkpoint to use: 'latest' for latest-unsharded, 'step0' for step0-unsharded (default: latest)"
    )
    
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        print(f"Error: Config directory does not exist: {args.config_dir}")
        sys.exit(1)
    
    # Find all config files
    config_files = find_config_files(config_dir)
    if not config_files:
        print(f"Error: No config files found in {args.config_dir}")
        sys.exit(1)
    
    print(f"Found {len(config_files)} config file(s) in {args.config_dir}")
    print()
    
    # Build list of models to process
    models_to_process = []
    for config_file in config_files:
        print(f"Processing config: {config_file.name}")
        
        # Extract save folder
        save_folder = extract_save_folder(config_file)
        if not save_folder:
            print(f"  ⚠️  Could not extract save_folder from config")
            continue
        
        print(f"  Save folder: {save_folder}")
        
        # Find checkpoint
        if args.checkpoint_type == "step0":
            checkpoint_dir = Path(save_folder) / "step0-unsharded"
            if not checkpoint_dir.exists():
                print(f"  ⚠️  step0-unsharded not found")
                continue
        else:
            checkpoint_dir = find_latest_checkpoint(save_folder)
            if not checkpoint_dir:
                continue
        
        print(f"  Checkpoint: {checkpoint_dir}")
        
        # Check if checkpoint has required files
        if not (checkpoint_dir / "config.yaml").exists():
            print(f"  ⚠️  No config.yaml found in checkpoint")
            continue
        
        if not (checkpoint_dir / "model.pt").exists():
            print(f"  ⚠️  No model.pt found in checkpoint (might be sharded)")
            continue
        
        models_to_process.append({
            "config_file": config_file,
            "model_name": config_file.stem,  # Filename without extension
            "checkpoint_dir": checkpoint_dir
        })
        print(f"  ✓ Added to processing queue")
        print()
    
    if not models_to_process:
        print("Error: No valid models found to process")
        sys.exit(1)
    
    print(f"{'='*80}")
    print(f"Will process {len(models_to_process)} model(s)")
    print(f"{'='*80}\n")
    
    # Run inference on all models
    all_results = {}
    
    for i, model_info in enumerate(models_to_process, 1):
        model_name = model_info["model_name"]
        checkpoint_dir = model_info["checkpoint_dir"]
        
        print(f"[{i}/{len(models_to_process)}] Processing: {model_name}")
        print(f"Checkpoint: {checkpoint_dir}")
        print()
        
        try:
            results = generate_for_model(
                checkpoint_dir=checkpoint_dir,
                prompts=args.prompts,
                max_steps=args.max_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
                device=args.device
            )
            
            all_results[model_name] = {
                "checkpoint_dir": str(checkpoint_dir),
                "config_file": str(model_info["config_file"]),
                "results": results
            }
            
            # Print results
            for result in results:
                print(f"  Prompt: {result['prompt']}")
                print(f"  Output: {result['completion'][:100]}...")
                print()
            
            print(f"✓ Completed {model_name}")
            
        except Exception as e:
            print(f"✗ Error processing {model_name}: {e}")
            all_results[model_name] = {
                "checkpoint_dir": str(checkpoint_dir),
                "config_file": str(model_info["config_file"]),
                "error": str(e)
            }
        
        print(f"{'='*80}\n")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            "config_dir": str(config_dir),
            "prompts": args.prompts,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "device": args.device,
            "checkpoint_type": args.checkpoint_type,
            "models": all_results
        }, f, indent=2)
    
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"Successfully processed: {len([r for r in all_results.values() if 'error' not in r])}/{len(models_to_process)} models")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

