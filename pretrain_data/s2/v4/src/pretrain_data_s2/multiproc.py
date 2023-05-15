from contextlib import ExitStack
from functools import partial
from multiprocessing import Manager, Pool, cpu_count, set_start_method
from queue import Queue
from threading import Thread
from time import sleep
from typing import Callable, NamedTuple, Optional, Union

import springs as sp
from smashed.utils import io_utils
from tqdm import tqdm


@sp.dataclass
class ProcessTextConfig:
    src: str = sp.field(default=sp.MISSING, help="Path to S3 prefix containing parqet files")
    dst: str = sp.field(default=sp.MISSING, help="Path to S3 prefix to write parqet files")
    debug: int = sp.field(default=0, help="Debug mode. Set to >0 to enable")
    parallel: int = sp.field(default=cpu_count(), help="Number of processes to use")


class PbarUnit(NamedTuple):
    files: int
    docs: int
    tokens: int

    @classmethod
    def new(cls, files: int = 0, docs: int = 0, tokens: int = 0) -> "PbarUnit":
        return cls(files, docs, tokens)


def threaded_progressbar(q: "Queue[Union[PbarUnit, None]]", timeout: float, total_files: Optional[int] = None):
    with ExitStack() as stack:
        files_pbar = stack.enter_context(tqdm(desc=" Files", unit="files", position=0, total=total_files))
        docs_pbar = stack.enter_context(tqdm(desc="  Docs", unit=" docs", position=1, unit_scale=True))
        tokens_pbar = stack.enter_context(tqdm(desc="Tokens", unit=" tokens", position=2, unit_scale=True))
        while True:
            item = q.get()
            if item is None:
                break

            files_pbar.update(item.files)
            docs_pbar.update(item.docs)
            tokens_pbar.update(item.tokens)
            sleep(timeout)


def make_runner(fn: Callable):
    _runner = partial(runner, fn=fn)
    _decorated = sp.cli(ProcessTextConfig)(_runner)
    return _decorated


def runner(cfg: ProcessTextConfig, fn: Callable):
    src = io_utils.MultiPath.parse(cfg.src)
    dst = io_utils.MultiPath.parse(cfg.dst)

    src_paths = [io_utils.MultiPath.parse(p) for p in io_utils.recursively_list_files(src)]
    dst_paths = [dst / (diff) if len(diff := (single_src - src)) > 0 else dst for single_src in src_paths]

    if cfg.debug:
        with tqdm(total=len(src_paths)) as pbar:
            for single_src, single_dst in zip(src_paths, dst_paths):
                fn((single_src, single_dst), debug=cfg.debug)
                pbar.update(1)

    else:
        set_start_method("spawn")

        with Pool(processes=cfg.parallel) as pool:
            pbar_queue: Queue = (manager := Manager()).Queue()
            pbar_thread = Thread(
                target=threaded_progressbar,
                args=(pbar_queue, 0.1, len(src_paths)),
                daemon=True,
            )
            pbar_thread.start()

            for _ in pool.imap_unordered(
                partial(fn, pbar_queue=pbar_queue, debug=cfg.debug), tuple(zip(src_paths, dst_paths))
            ):
                ...

            pool.close()
            pool.join()

            pbar_queue.put(None)
            pbar_thread.join()
            manager.shutdown()
