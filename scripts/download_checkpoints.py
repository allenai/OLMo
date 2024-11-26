import argparse
import csv
import os
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm


def convert_to_r2_url(http_url):
    """Convert HTTP URL to R2 URL format."""
    if http_url.startswith('https://olmo-checkpoints.org/'):
        return http_url.replace('https://olmo-checkpoints.org/', 'r2://olmo-checkpoints/')
    return http_url

def convert_to_public_url(r2_url):
    """Convert R2 URL to public HTTP URL format."""
    if r2_url.startswith('r2://olmo-checkpoints/'):
        return r2_url.replace('r2://olmo-checkpoints/', 'https://olmo-checkpoints.org/')
    return r2_url

def download_file(url, save_path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def try_get_directory_listing(url):
    common_files = [
        "config.yaml",
        "model.pt",
        "optim.pt",
        "train.pt",
        "model.safetensors",
        "optim.safetensors",
    ]
    found_files = []
    for pattern in common_files:
        try:
            test_url = urljoin(url.rstrip('/') + '/', pattern)
            response = requests.head(test_url)
            # response.raise_for_status()
            if response.status_code == 200:
                found_files.append(pattern)
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error for {pattern}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Connection error for {pattern}: {e}")
    return found_files

def download_checkpoint(url, save_dir):
   """Download all files from a checkpoint directory."""
   r2_url = convert_to_r2_url(url)
   public_url = convert_to_public_url(r2_url)
   base_path = Path(save_dir)
   base_path.mkdir(parents=True, exist_ok=True)
   print(f"Saving to: {base_path}")
   available_files = try_get_directory_listing(public_url)
   
   if not available_files:
       raise ValueError("No matching files found in directory")
   
   failed_files = []
   for file in available_files:
       file_url = urljoin(public_url.rstrip('/') + '/', file)
       file_path = base_path / file
       try:
           print(f"\nDownloading: {file}")
           download_file(file_url, file_path)
       except requests.exceptions.Timeout:
           print(f"Timeout error for {file}, retrying once...")
           try:
               download_file(file_url, file_path)
           except requests.exceptions.RequestException as e:
               failed_files.append(file)
               print(f"Failed to download {file}: {e}")
       except requests.exceptions.RequestException as e:
           failed_files.append(file)
           print(f"Failed to download {file}: {e}")
   if failed_files:
       print(f"\nWARNING: Failed to download these files: {failed_files}")

def main():
    parser = argparse.ArgumentParser(description='Download OLMo checkpoints')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    download_parser = subparsers.add_parser('download', 
                                          help='Download checkpoints from CSV file')
    download_parser.add_argument('csv_file', type=str, 
                               help='Path to the CSV file containing checkpoint URLs')
    download_parser.add_argument('--step', type=str, required=True,
                               help='Specific step number to download')
    download_parser.add_argument('--save-dir', type=str, default='./checkpoints',
                               help='Base directory to save downloaded checkpoints')
    list_parser = subparsers.add_parser('list',
                                       help='List available checkpoint steps')
    list_parser.add_argument('csv_file', type=str,
                            help='Path to the CSV file containing checkpoint URLs')
    args = parser.parse_args()
    
    print(f"Reading CSV file: {args.csv_file}")
    
    with open(args.csv_file, 'r') as f:
        reader = csv.DictReader(f)
        urls = [(row['Step'], row['Checkpoint Directory']) for row in reader]
    
    if args.command == 'list':
        print("Available steps:")
        for step, _ in urls:
            print(f"Step {step}")
        return
    
    if args.step:
        urls = [(step, url) for step, url in urls if step == args.step]
        if not urls:
            print(f"Error: Step {args.step} not found in the CSV file.")
            print("Use list argument to see available step numbers.")
            return
    
    print(f"Saving checkpoints to: {args.save_dir}")
    for step, url in urls:
        r2_url = convert_to_r2_url(url)
        public_url = convert_to_public_url(r2_url)
        print(f"\nStep {step}:")
        print(f"Public URL: {public_url}")
        save_path = os.path.join(args.save_dir, f"step{step}")
        download_checkpoint(url, save_path)
    

if __name__ == "__main__":
    main()