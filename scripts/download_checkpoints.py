import csv
import os
import requests
from tqdm import tqdm
import argparse
from pathlib import Path
from urllib.parse import urljoin

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
        test_url = urljoin(url.rstrip('/') + '/', pattern)
        try:
            response = requests.head(test_url)
            if response.status_code == 200:
                found_files.append(pattern)
        except requests.exceptions.RequestException:
            continue
            
    return found_files

def download_checkpoint(url, save_dir):
    """Download all files from a checkpoint directory."""
    r2_url = convert_to_r2_url(url)
    public_url = convert_to_public_url(r2_url)
    
    base_path = Path(save_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nR2 URL: {r2_url}")
    print(f"Public URL: {public_url}")
    print(f"Saving to: {base_path}")
    
    print("Checking for available files...")
    available_files = try_get_directory_listing(public_url)
    
    if not available_files:
        print("No files found using common patterns. The directory might be empty or use different file patterns.")
        return
    
    for file in available_files:
        file_url = urljoin(public_url.rstrip('/') + '/', file)
        file_path = base_path / file
        
        try:
            print(f"\nDownloading: {file}")
            download_file(file_url, file_path)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Download OLMo checkpoints from CSV')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing checkpoint URLs')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Base directory to save downloaded checkpoints')
    parser.add_argument('--step', type=str, help='Specific step number to download (optional)')
    parser.add_argument('--list-steps', action='store_true', help='List available step numbers and exit')
    
    args = parser.parse_args()
    
    print(f"Reading CSV file: {args.csv_file}")
    
    with open(args.csv_file, 'r') as f:
        reader = csv.DictReader(f)
        urls = [(row['Step'], row['Checkpoint Directory']) for row in reader]
    
    if args.list_steps:
        print("\nAvailable steps:")
        for step, _ in urls:
            print(f"Step {step}")
        return
    
    if args.step:
        urls = [(step, url) for step, url in urls if step == args.step]
        if not urls:
            print(f"Error: Step {args.step} not found in the CSV file.")
            print("Use --list-steps to see available step numbers.")
            return
    
    print(f"Saving checkpoints to: {args.save_dir}")
    print("\nURL conversions:")
    for step, url in urls:
        r2_url = convert_to_r2_url(url)
        public_url = convert_to_public_url(r2_url)
        print(f"\nStep {step}:")
        print(f"Original URL: {url}")
        print(f"R2 URL: {r2_url}")
        print(f"Public URL: {public_url}")
    
    proceed = input("\nDo you want to proceed with the download? (y/n): ")
    if proceed.lower() != 'y':
        print("Download cancelled.")
        return
    
    for step, url in urls:
        save_path = os.path.join(args.save_dir, f"step{step}")
        try:
            download_checkpoint(url, save_path)
        except Exception as e:
            print(f"Error during download of step {step}: {e}")

if __name__ == "__main__":
    main()