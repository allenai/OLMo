import gzip
from contextlib import ExitStack
from functools import reduce
from multiprocessing import Pool, current_process
from pathlib import Path

import springs as sp
from tqdm import tqdm


def get_current_process_number() -> int:
    if not (pid := current_process()._identity):
        return 0
    return reduce(lambda x, y: x * y, pid, 1)


def concat_and_compress(
    source_path: Path,
    dest_path: Path,
    max_bytes: int = 1_000_000,
):
    """Open all files in source_path, concatenate up to max_rows, and compress
    the result in files at dest_path."""

    # list all files in source_path
    all_files = [p for p in source_path.glob("**/*") if p.is_file()]

    # create dest_path if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # open files in source path one by one, concatenate them up to max_rows,
    # and compress the result in files at dest_path
    num_output = 0
    num_bytes = 0
    fout = None

    with ExitStack() as stack:
        pbar = stack.enter_context(
            tqdm(
                desc=dest_path.name,
                position=max(get_current_process_number() - 1, 0),
            )
        )
        for f in all_files:
            fin = stack.enter_context(open(f, "r"))

            for line in fin:
                if fout is None:
                    fout = stack.enter_context(gzip.open(dest_path / f"{num_output:03}.gz", "wb"))

                enc = line.encode("utf-8")
                fout.write(enc)
                num_bytes += len(enc)
                pbar.update(1)

                if num_bytes >= max_bytes:
                    fout.close()
                    num_output += 1
                    num_bytes = 0
                    fout = None


@sp.dataclass
class DownloadConfig:
    local_src: str = sp.MISSING
    local_dst: str = sp.MISSING
    max_bytes: int = 300_000_000
    parallel: int = 1
    debug: bool = False


@sp.cli(DownloadConfig)
def main(config: DownloadConfig):
    base_src_path = Path(config.local_src).expanduser().resolve()
    all_base_path = [base_src_path / f"{d}" for d in base_src_path.iterdir() if d.is_dir()]
    base_dst_path = Path(config.local_dst).expanduser().resolve()
    all_dst_path = [
        base_dst_path / f"lang={d.name.replace('wiki_', '')}" for d in base_src_path.iterdir() if d.is_dir()
    ]

    if config.debug:
        for src, dst in zip(all_base_path, all_dst_path):
            concat_and_compress(src, dst, max_bytes=config.max_bytes)
        return

    with Pool(config.parallel) as pool:
        for src, dst in zip(all_base_path, all_dst_path):
            pool.apply_async(concat_and_compress, (src, dst, config.max_bytes))
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
