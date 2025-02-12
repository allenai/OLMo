import os
import json
import gzip
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

DEBUG = False  # Debug flag

def process_chunk(chunk, start_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"chunk_{start_idx}.jsonl.gz")
    
    if DEBUG:
        print(f"\nProcessing chunk {start_idx}")
        print(f"First 2 items in chunk:")
        for i in range(min(2, len(chunk))):
            print(f"Item {i}:")
            print(f"Problem: {chunk[i]['problem'][:100]}...")
            print(f"Solution: {chunk[i]['generated_solution'][:100]}...")
            print("---")
    
    with gzip.open(output_file, 'wt') as f:
        for i, item in enumerate(tqdm(chunk, desc=f"Processing chunk {start_idx}", disable=not DEBUG)):
            try:
                doc = {
                    "id": f"omath2_{start_idx + i}",
                    "text": f"{item['problem']}\n\n{item['generated_solution']}\n\nThe final answer is {item['expected_answer']}",
                    "source": "OpenMathInstruct-2",
                    "added": datetime.now().isoformat(),
                    "metadata": {
                        "problem": item['problem'],
                        "generated_solution": item['generated_solution'],
                        "expected_answer": item['expected_answer'],
                        "problem_source": item['problem_source']
                    }
                }
                f.write(json.dumps(doc) + '\n')
            except Exception as e:
                if DEBUG:
                    print(f"Error processing item {i}: {e}")
                    print(f"Item content: {item}")
                continue

def main():
    # Load the dataset
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    
    # Debug dataset structure
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset length: {len(dataset)}")
    print(f"First item type: {type(dataset[0])}")
    print(f"Sample keys: {dataset[0].keys()}")
    
    # Process in chunks of 100k
    chunk_size = 100000
    output_dir = "documents"
    
    # Use select to slice the dataset properly
    for i in range(0, len(dataset), chunk_size):
        end_idx = min(i + chunk_size, len(dataset))
        chunk = dataset.select(range(i, end_idx))
        process_chunk(chunk, i // 100_000, output_dir)

if __name__ == "__main__":
    main()