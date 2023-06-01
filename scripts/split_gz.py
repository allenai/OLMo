"""
Split gzipped files into smaller gzipped files.

Author: @soldni
"""

import concurrent.futures
import gzip
import os
from contextlib import ExitStack

import click

MAX_SIZE_4_GB = 4 * 1024 * 1024 * 1024


@click.command()
@click.option("--input_dir", required=True, help="Path to input directory containing gzip files")
@click.option("--input_ext", default=".gz", help="Extension of the input files")
@click.option("--output_dir", required=True, help="Path to output directory for the split files")
@click.option("--output_ext", default=".gz", help="Extension of the output files")
@click.option("--size_limit", default=MAX_SIZE_4_GB, help="Size limit for each split file in bytes")
def main(input_dir: str, input_ext: str, output_dir: str, output_ext: str, size_limit: int):
    os.makedirs(output_dir, exist_ok=True)

    def split_gzip_file(input_file, output_base, size_limit=size_limit, output_ext=output_ext):
        print(f"Splitting {input_file} into {output_base} with size limit {size_limit:,}")
        with ExitStack() as stack, gzip.open(input_file, "rt") as f:
            count = 0
            path = f"{output_base}_{count:04d}{output_ext}"
            output = stack.enter_context(gzip.open(path, "wt"))
            current_size = 0
            for line in f:
                line_size = len(line)
                if current_size + line_size > size_limit:
                    stack.pop_all().close()
                    count += 1
                    print(f"Wrote {path}")
                    path = f"{output_base}_{count:04d}{output_ext}"
                    output = stack.enter_context(gzip.open(path, "wt"))
                    current_size = 0
                output.write(str(line))
                current_size += line_size
            print(f"Wrote {path}")
            stack.pop_all().close()
            print(f"Finished splitting {input_file} into {count + 1:,} files")

    def process_file(file_name, input_ext=input_ext, input_dir=input_dir, output_dir=output_dir):
        input_file = os.path.join(input_dir, file_name)
        if file_name.endswith(input_ext) and os.path.isfile(input_file):
            base_name = file_name.rstrip(input_ext)
            output_base = os.path.join(output_dir, base_name)
            split_gzip_file(input_file, output_base)

    files_to_process = [
        file_name
        for file_name in os.listdir(input_dir)
        if file_name.endswith(input_ext) and os.path.isfile(os.path.join(input_dir, file_name))
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_file, files_to_process)


if __name__ == "__main__":
    main()
