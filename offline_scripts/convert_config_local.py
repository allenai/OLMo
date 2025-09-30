#!/usr/bin/env python3
"""
Script to convert OLMo config from HTTP URLs to local file paths.
"""

import argparse
from pathlib import Path

def convert_config_to_local(input_config: Path, output_config: Path, data_dir: Path):
    """Convert HTTP URLs in config to local file paths."""
    
    with open(input_config, 'r') as f:
        content = f.read()
    
    # Replace HTTP URLs with local paths
    content = content.replace(
        "http://olmo-data.org/",
        f"{data_dir.absolute()}/"
    )
    
    # Update save folder to local path
    content = content.replace(
        "save_folder: http://olmo-data.org/checkpoints/OLMo-small/${run_name}",
        f"save_folder: {data_dir.absolute()}/checkpoints/OLMo-small/${{run_name}}"
    )
    
    with open(output_config, 'w') as f:
        f.write(content)
    
    print(f"Converted config saved to: {output_config}")

def main():
    parser = argparse.ArgumentParser(description="Convert OLMo config to use local data paths")
    parser.add_argument("--input-config", "-i", type=str, 
                       default="configs/official-0425/OLMo2-1B-stage1-10B-tokens.yaml",
                       help="Input config file")
    parser.add_argument("--output-config", "-o", type=str,
                       default="OLMo2-1B-stage1-10B-tokens-local.yaml",
                       help="Output config file")
    parser.add_argument("--data-dir", "-d", type=str, default="./olmo_data",
                       help="Local data directory")
    
    args = parser.parse_args()
    
    input_config = Path(args.input_config)
    output_config = Path(args.output_config)
    data_dir = Path(args.data_dir)
    
    if not input_config.exists():
        print(f"Input config file not found: {input_config}")
        return
    
    convert_config_to_local(input_config, output_config, data_dir)

if __name__ == "__main__":
    main()
