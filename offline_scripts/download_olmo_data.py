#!/usr/bin/env python3
"""
Script to download OLMo 1B training data locally for offline training.
This downloads all the data files from olmo-data.org to a local directory.
"""

import os
import requests
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

def download_file(url: str, local_path: Path, max_retries: int = 5) -> bool:
    """Download a single file with retry logic."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=local_path.name, leave=False) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to download {url} after {max_retries} attempts")
                return False
    
    return False

def get_data_urls(config_file=None):
    """Extract all data URLs from the OLMo 1B config."""
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = Path(__file__).parent / "configs" / "official-0425" / "OLMo2-1B-stage1-10B-tokens.yaml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return []
    
    urls = []
    in_paths_section = False
    
    with open(config_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            
            # Check if we're entering the paths section
            if stripped == "paths:":
                in_paths_section = True
                continue
            
            # If in paths section, look for URLs
            if in_paths_section:
                # Skip comments and empty lines
                if not stripped or stripped.startswith("#"):
                    continue
                # Found a URL
                if stripped.startswith("- http://olmo-data.org"):
                    url = stripped[2:].strip()  # Remove "- " prefix
                    urls.append(url)
                # End of paths section (reached another YAML key at same indentation level)
                elif stripped and not stripped.startswith("-") and ":" in stripped:
                    break
    
    return urls

def main():
    parser = argparse.ArgumentParser(description="Download OLMo 1B training data locally")
    parser.add_argument("--config", "-c", type=str, default=None,
                       help="Path to config file (default: configs/official-0425/OLMo2-1B-stage1-10B-tokens.yaml)")
    parser.add_argument("--output-dir", "-o", type=str, default="./olmo_data", 
                       help="Directory to save downloaded data")
    parser.add_argument("--max-workers", "-w", type=int, default=8,
                       help="Maximum number of concurrent downloads")
    parser.add_argument("--max-files", "-n", type=int, default=None,
                       help="Maximum number of files to download (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be downloaded without actually downloading")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting data URLs from config...")
    urls = get_data_urls(args.config)
    
    if not urls:
        print("No data URLs found in config file!")
        return
    
    # Limit number of files if requested
    if args.max_files:
        urls = urls[:args.max_files]
        print(f"Limiting download to first {len(urls)} files (--max-files={args.max_files})")
    
    print(f"Found {len(urls)} data files to download")
    
    if args.dry_run:
        print("\nDry run - would download:")
        for url in urls[:10]:  # Show first 10
            print(f"  {url}")
        if len(urls) > 10:
            print(f"  ... and {len(urls) - 10} more files")
        return
    
    print(f"Downloading to: {output_dir}")
    print(f"Using {args.max_workers} concurrent workers")
    
    # Create local file paths
    download_tasks = []
    for url in urls:
        # Convert URL to local path
        # http://olmo-data.org/preprocessed/... -> ./olmo_data/preprocessed/...
        relative_path = url.replace("http://olmo-data.org/", "")
        local_path = output_dir / relative_path
        download_tasks.append((url, local_path))
    
    # Download files
    successful_downloads = 0
    failed_downloads = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all download tasks
        future_to_task = {
            executor.submit(download_file, url, local_path): (url, local_path)
            for url, local_path in download_tasks
        }
        
        # Process completed downloads
        with tqdm(total=len(download_tasks), desc="Downloading files") as pbar:
            for future in as_completed(future_to_task):
                url, local_path = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                    failed_downloads += 1
                
                pbar.update(1)
    
    print(f"\nDownload complete!")
    print(f"Successful: {successful_downloads}")
    print(f"Failed: {failed_downloads}")
    
    if successful_downloads > 0:
        print(f"\nData downloaded to: {output_dir}")
        print("\nTo use local data, modify your config file to replace:")
        print("  http://olmo-data.org/")
        print("with:")
        print(f"  {output_dir.absolute()}/")

if __name__ == "__main__":
    main()
