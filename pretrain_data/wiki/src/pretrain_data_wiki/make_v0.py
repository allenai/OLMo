import datetime
import gzip
import json
import string
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial, reduce
from io import BytesIO
from multiprocessing import current_process, get_context
from queue import Queue
from time import sleep
from typing import List, Optional, TextIO, Tuple

import springs as sp
from smashed.utils.io_utils import (
    MultiPath,
    open_file_for_read,
    open_file_for_write,
    recursively_list_files,
)
from tqdm import tqdm
from uniseg.wordbreak import words as uniseg_get_words

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def get_current_process_number() -> int:
    if not (pid := current_process()._identity):
        return 0
    return reduce(lambda x, y: x * y, pid, 1)


def get_words(text: str) -> List[str]:
    return [word for word in uniseg_get_words(text) if not all(char in string.whitespace for char in word)]


def worker_process(
    source_path: str,
    result_queue: "Queue[dict]",
):
    """Open all files in source_path, concatenate up to max_rows, and compress
    the result in files at dest_path."""

    with open_file_for_read(source_path, "rt") as f:
        for _, line in enumerate(f, start=1):
            data = json.loads(line)

            id_ = data.pop("id")
            title = data.pop("title", "").strip()
            body = data.pop("text", "").strip()

            if not id_ or not title or not body:
                continue

            text = f"{title}\n\n{body}".strip()

            json_data = {
                "id": id_,
                "source": "wikipedia",
                "version": "v0",
                "text": text,
                "created": TIMESTAMP,
                "added": TIMESTAMP,
                "metadata": {**data, "length": len(get_words(text))},
            }

            result_queue.put(json_data)


def make_new_file_and_stream(
    stack: ExitStack,
    dst: MultiPath,
    current_file: Optional[BytesIO] = None,
    current_stream: Optional[TextIO] = None,
) -> Tuple[BytesIO, TextIO]:
    current_stream.close() if current_stream else None
    current_file.close() if current_file else None
    file_out = stack.enter_context(open_file_for_write(dst, "wb", temp_dir=None))
    stream_out = stack.enter_context(gzip.open(file_out, "wt"))

    return file_out, stream_out  # type: ignore


def writer_process(
    source_path: MultiPath,
    dest_path: MultiPath,
    max_bytes: int = 1_000_000,
    num_workers_per_lang: int = 1,
    debug: bool = False,
):
    # list all files in source_path
    all_files = list(recursively_list_files(source_path))
    num_workers_per_lang = min(num_workers_per_lang, len(all_files))

    # open files in source path one by one, concatenate them up to max_rows,
    # and compress the result in files at dest_path
    num_output = 0
    valid_cnt = 0
    stream_out = None
    file_out = None
    short_desc = dest_path.as_str.split("/")[-1]
    proc_number = get_current_process_number()

    # add some delay for when each process starts
    sleep(proc_number * 1.0)

    with ExitStack() as stack:
        pbar_desc = f"Parsing {short_desc} - {num_workers_per_lang} worker(s)"
        pbar_pos = max(proc_number - 1, 0)
        pbar = stack.enter_context(tqdm(desc=pbar_desc, position=pbar_pos, unit=" docs", unit_scale=True))

        queue: Queue[dict]
        pool: Optional[ProcessPoolExecutor] = None
        futures: List[Future] = []

        if debug:
            queue = Queue()
            for file_path in all_files:
                # worker processes each path and adds to queue
                worker_process(source_path=file_path, result_queue=queue)
        else:
            queue = stack.enter_context(get_context("spawn").Manager()).Queue()
            pool = stack.enter_context(ProcessPoolExecutor(num_workers_per_lang))
            worker_fn = partial(worker_process, result_queue=queue)
            futures = [pool.submit(worker_fn, fn) for fn in all_files]

        def must_make_new_file(current_file: Optional[BytesIO], current_stream: Optional[TextIO]) -> bool:
            if current_file is None or current_stream is None:
                return True
            return current_file.tell() > max_bytes

        while True:
            if queue.empty() and futures is None:
                # single process debug mode
                break
            if queue.empty() and all(f.done() for f in futures):
                # multi-process mode, and they're all done
                break
            if queue.empty():
                # multi-process mode, but some are still working
                # wait a bit and try again
                sleep(0.1)
                continue

            data = queue.get()

            if must_make_new_file(current_file=file_out, current_stream=stream_out):
                fp = dest_path / f"{num_output:05d}.gz"
                file_out, stream_out = make_new_file_and_stream(
                    stack=stack, dst=fp, current_file=file_out, current_stream=stream_out
                )
                num_output += 1

            stream_out.write(json.dumps(data) + "\n")  # type: ignore
            valid_cnt += 1
            pbar.update(1)

        pbar.desc = f"[Done parsing {short_desc}: {valid_cnt:,} docs, {num_output:,} files]"


@sp.dataclass
class DownloadConfig:
    src: str = sp.MISSING
    section: str = "wikipedia"
    dst: str = "s3://ai2-llm/pretraining-data/sources/{section}/v0/documents"
    max_bytes: int = 100_000_000
    num_parallel_langs: int = 1
    num_workers_per_lang: int = 1
    debug: bool = False


@sp.cli(DownloadConfig)
def main(config: DownloadConfig):
    base_src_path = MultiPath.parse(config.src)
    all_lang_dirs = set(
        (MultiPath.parse(d) - base_src_path).as_str.lstrip("/").rsplit("/")[0]
        for d in recursively_list_files(base_src_path)
    )
    all_src_lang_paths = [base_src_path / lang.replace("wiki_", "lang=") for lang in all_lang_dirs]

    base_dst_path = MultiPath.parse(config.dst.format(section=config.section))
    all_dst_lang_paths = [base_dst_path / lang for lang in all_lang_dirs]

    if config.debug:
        for src, dst in zip(all_src_lang_paths, all_dst_lang_paths):
            writer_process(source_path=src, dest_path=dst, max_bytes=config.max_bytes, debug=config.debug)
        return

    with ProcessPoolExecutor(max_workers=config.num_parallel_langs) as pool:
        fn = partial(writer_process, max_bytes=config.max_bytes, num_workers_per_lang=config.num_workers_per_lang)
        for src, dst in zip(all_src_lang_paths, all_dst_lang_paths):
            pool.submit(fn, src, dst)


if __name__ == "__main__":
    main()
