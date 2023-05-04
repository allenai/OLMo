import multiprocessing
import random
import time
from contextlib import ExitStack
from datetime import datetime
from functools import partial
from queue import Queue
from threading import Thread
from typing import Dict, List, Tuple, Union

import tqdm
from smashed.utils.io_utils import (
    MultiPath,
    open_file_for_write,
    recursively_list_files,
)

METADATA_SUFFIX = ".done.txt"


class BaseParallelProcessor:
    def __init__(
        self,
        source_prefix: str,
        destination_prefix: str,
        metadata_prefix: str,
        num_processes: int = 1,
        debug: bool = False,
        seed: int = 0,
        pbar_timeout: float = 0.01,
        ignore_existing: bool = False,
    ):
        self.source_prefix = MultiPath.parse(source_prefix)
        self.destination_prefix = MultiPath.parse(destination_prefix)
        self.metadata_prefix = MultiPath.parse(metadata_prefix)
        self.num_processes = num_processes
        self.debug = debug
        self.seed = seed
        self.pbar_timeout = pbar_timeout
        self.ignore_existing = ignore_existing

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: "Queue[Union[None, Tuple[int, ...]]]",
    ):
        raise NotImplementedError()

    @classmethod
    def _process_single_and_save_status(
        cls,
        source_path: str,
        destination_path: str,
        metadata_path: str,
        queue: "Queue[Union[None, Tuple[int, ...]]]",
    ):
        cls.process_single(source_path, destination_path, queue)
        with open_file_for_write(metadata_path) as f:
            f.write(datetime.now().isoformat())

    @classmethod
    def increment_progressbar(
        cls, queue: "Queue[Union[None, Tuple[int, ...]]]", /, **kwargs: int
    ) -> Dict[str, int]:
        assert len(kwargs) > 0, "You must define increment_progressbar and provide at least one kwarg argument"
        queue.put(tuple(kwargs.get(k, 0) for k in kwargs))
        return kwargs

    @classmethod
    def _run_threaded_progressbar(
        cls,
        queue: "Queue[Union[None, Tuple[int, ...]]]",
        timeout: float,
    ):
        sample_queue_output = cls.increment_progressbar(queue)

        with ExitStack() as stack:
            pbars = [
                stack.enter_context(tqdm.tqdm(desc=str(k), unit=str(k)[:1], position=i, unit_scale=True))
                for i, k in enumerate(sample_queue_output)
            ]

            while True:
                item = queue.get()
                if item is None:
                    break

                for pbar, value in zip(pbars, item):
                    pbar.update(value)

                time.sleep(timeout)

    def _debug_run_all(
        self,
        all_source_paths: List[MultiPath],
        all_destination_paths: List[MultiPath],
        all_metadata_paths: List[MultiPath],
    ):
        it = zip(all_source_paths, all_destination_paths, all_metadata_paths)
        pbar_queue: "Queue[Union[None, Tuple[int, ...]]]" = Queue()
        thread = Thread(target=self._run_threaded_progressbar, args=(pbar_queue, self.pbar_timeout), daemon=True)
        thread.start()

        for source_prefix, destination_prefix, metadata_prefix in it:
            self._process_single_and_save_status(
                source_path=source_prefix.as_str,
                destination_path=destination_prefix.as_str,
                metadata_path=metadata_prefix.as_str,
                queue=pbar_queue,
            )

        pbar_queue.put(None)
        thread.join()

    def _multiprocessing_run_all(
        self,
        all_source_paths: List[MultiPath],
        all_destination_paths: List[MultiPath],
        all_metadata_paths: List[MultiPath],
    ):
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            assert multiprocessing.get_start_method() == "spawn", "Multiprocessing start method must be spawn"

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            pbar_queue: "Queue[Union[None, Tuple[int, ...]]]" = (manager := multiprocessing.Manager()).Queue()
            thread = Thread(
                target=self._run_threaded_progressbar, args=(pbar_queue, self.pbar_timeout), daemon=True
            )
            thread.start()

            process_single_fn = partial(self.process_single, queue=pbar_queue)
            results = []

            for s, d, m in zip(all_source_paths, all_destination_paths, all_metadata_paths):
                process_single_fn = partial(
                    self._process_single_and_save_status,
                    queue=pbar_queue,
                    source_path=s.as_str,
                    destination_path=d.as_str,
                    metadata_path=m.as_str,
                )
                result = pool.apply_async(process_single_fn)
                results.append(result)

            for result in results:
                result.get()

            pool.close()
            pool.join()

            pbar_queue.put(None)
            thread.join()
            manager.shutdown()

    def _get_all_paths(self) -> Tuple[List[MultiPath], List[MultiPath], List[MultiPath]]:
        all_source_paths, all_destination_paths, all_metadata_paths = [], [], []

        existing_metadata_names = set(
            (MultiPath.parse(path) - self.metadata_prefix).as_str.rstrip(METADATA_SUFFIX)
            for path in recursively_list_files(self.metadata_prefix)
        )
        for path in recursively_list_files(self.source_prefix):
            source_path = MultiPath.parse(path)
            if not self.ignore_existing and (source_path - self.source_prefix).as_str in existing_metadata_names:
                continue

            all_source_paths.append(source_path)
            all_destination_paths.append(self.destination_prefix / (source_path - self.source_prefix))

            metadata_path = MultiPath.parse(source_path.as_str + METADATA_SUFFIX)
            all_metadata_paths.append(self.metadata_prefix / (metadata_path - self.source_prefix))

        return all_source_paths, all_destination_paths, all_metadata_paths

    def __call__(self):
        random.seed(self.seed)
        all_source_paths, all_destination_paths, all_metadata_paths = self._get_all_paths()

        fn = self._debug_run_all if self.debug else self._multiprocessing_run_all

        fn(
            all_source_paths=all_source_paths,
            all_destination_paths=all_destination_paths,
            all_metadata_paths=all_metadata_paths,
        )
