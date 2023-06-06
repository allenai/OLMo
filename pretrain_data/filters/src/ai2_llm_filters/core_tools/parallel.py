import inspect
import multiprocessing
import pickle
import random
import time
from contextlib import ExitStack
from datetime import datetime
from functools import partial
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import tqdm
from smashed.utils.io_utils import (
    MultiPath,
    open_file_for_write,
    recursively_list_files,
)

from .data_types import Ai2LlmFilterError, Ai2LlmRetryableFailure

METADATA_SUFFIX = ".done.txt"


class BaseParallelProcessor:
    """A base parallel processor that supports applying the same process_single method to a list of files.

    This class is meant to be subclassed. The subclass must implement:
        - `process_single` method, which takes a source path file to transform, and a destination path where
           to save the transformed file.
        - `increment_progressbar` method, which defines which units to keep track of in the progress bar.

    See documentation of both methods for more details on how to implement them correctly.
    """

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
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
    ):
        """Initialize the parallel processor.

        Args:
            source_prefix (str): The location where source files are stored. This can be a local directory or a
                prefix to an S3 location.
            destination_prefix (str): The location where to save the transformed files. This can be a local
                directory or a prefix to an S3 location. Local directories will be created if they do not exist.
                The directory structure from the source prefix will be replicated in the destination prefix;
                file names will also be the same.
            metadata_prefix (str): The prefix of the metadata files to save. This can be a local path or an
                S3 path. Metadata output will be created for each file after it is processed. Filenames are
                checked to verify if a file has been processed and can be skipped unless `ignore_existing` is
                set to true.
            num_processes (int, optional): The number of processes to use. Defaults to 1.
            debug (bool, optional): Whether to run in debug mode; if true, no multiprocessing will be used.
                Defaults to False.
            seed (int, optional): The random seed to use when shuffling input files. Defaults to 0.
            pbar_timeout (float, optional): How often to update progress bars in seconds.
                Defaults to 0.01 seconds.
            ignore_existing (bool, optional): Whether to ignore files that have been already processed and
                re-run the processor on all files from scratch. Defaults to False.
            include_paths (Optional[List[str]], optional): A list of paths to include. If provided, only files
                that match one of the paths will be processed. Defaults to None.
            exclude_paths (Optional[List[str]], optional): A list of paths to exclude. If provided, files that
                match one of the paths will be skipped. Defaults to None.
        """

        self.source_prefix = MultiPath.parse(source_prefix)
        self.destination_prefix = MultiPath.parse(destination_prefix)
        self.metadata_prefix = MultiPath.parse(metadata_prefix)
        self.num_processes = num_processes
        self.debug = debug
        self.seed = seed
        self.pbar_timeout = pbar_timeout
        self.ignore_existing = ignore_existing

        self.include_paths = set(include_paths) if include_paths is not None else None
        self.exclude_paths = set(exclude_paths) if exclude_paths is not None else None

        # checking that the increment_progressbar method is subclassed
        # correctly
        sig = inspect.signature(self.increment_progressbar)
        if "queue" not in sig.parameters or sig.parameters["queue"].kind != inspect.Parameter.POSITIONAL_ONLY:
            raise AttributeError(
                "increment_progressbar must have a positional-only argument named 'queue'; "
                "Check that you have subclassed BaseParallelProcessor correctly!"
            )
        if "kwargs" in sig.parameters and sig.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD:
            raise AttributeError(
                "increment_progressbar must not have a **kwargs argument; "
                "Check that you have subclassed BaseParallelProcessor correctly!"
            )
        if any(p.name != "queue" and p.default != 0 for p in sig.parameters.values()):
            raise AttributeError(
                "increment_progressbar must have a default value of 0 for all arguments except 'queue'; "
                "Check that you have subclassed BaseParallelProcessor correctly!"
            )

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: "Queue[Union[None, Tuple[int, ...]]]",
        **kwargs: Any,
    ):
        """Process a single file.

        This method must be implemented by the subclass. It takes a source path file to transform, and a
        destination path where to save the transformed file. It also takes a queue to increment the progress
        bars. The queue should be passed to the `increment_progressbar` method.

        Args:
            source_path (str): The path to the source file to transform. Can be an S3 path or a local path.
            destination_path (str): The path to the destination file to save. Can be an S3 path or a local path.
            queue (Queue[Union[None, Tuple[int, ...]]]): The queue to increment the progress bars.
        """
        raise NotImplementedError()

    @classmethod
    def _process_single_and_save_status(
        cls,
        source_path: str,
        destination_path: str,
        metadata_path: str,
        queue: "Queue[Union[None, Tuple[int, ...]]]",
        serialized_kwargs: bytes,
    ):
        """A wrapper around process single that saves a metadata file if processing is successful."""

        kwargs = pickle.loads(serialized_kwargs)
        tries_remaining = kwargs.get("retry_on_read_error", 0) + 1
        while True:
            try:
                cls.process_single(
                    source_path=source_path, destination_path=destination_path, queue=queue, **kwargs
                )
                break
            except Ai2LlmRetryableFailure as e:
                tries_remaining -= 1
                if tries_remaining == 0:
                    raise Ai2LlmFilterError from e
        with open_file_for_write(metadata_path) as f:
            f.write(datetime.now().isoformat())

    @classmethod
    def increment_progressbar(
        cls, queue: "Queue[Union[None, Tuple[int, ...]]]", /, **kwargs: int
    ) -> Dict[str, int]:
        """Increment the progress bar by putting a tuple in the queue.

        When subclassing, we recommend defining which units to keep track of in the progress bar by
        defining keyword arguments. Then you can call the base class via `super()` and pass the keyword.
        Example:

        ```python
        class MyProcessor(BaseParallelProcessor):
            def increment_progressbar(self, queue, /, files = 0, documents = 0):   # we use two progress bars
                return super().increment_progressbar(queue, files=files, documents=documents)
        ```
        """
        queue.put(tuple(kwargs.get(k, 0) for k in kwargs))
        return kwargs

    @classmethod
    def _run_threaded_progressbar(
        cls,
        queue: "Queue[Union[None, Tuple[int, ...]]]",
        timeout: float,
    ):
        """Run a progress bar in a separate thread.

        Args:
            queue (Queue[Union[None, Tuple[int, ...]]]): The queue to increment the progress bars.
            timeout (float): How often to update the progress bars in seconds.
        """

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
        **process_single_kwargs: Any,
    ):
        """Run files one by one on the main process

        Args:
            all_source_paths (List[MultiPath]): The list of source paths to process.
            all_destination_paths (List[MultiPath]): The list of destination paths to save.
            all_metadata_paths (List[MultiPath]): The locations where to save metadata.
        """

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
                serialized_kwargs=pickle.dumps(process_single_kwargs),
            )

        pbar_queue.put(None)
        thread.join()

    def _multiprocessing_run_all(
        self,
        all_source_paths: List[MultiPath],
        all_destination_paths: List[MultiPath],
        all_metadata_paths: List[MultiPath],
        **process_single_kwargs: Any,
    ):
        """Run files in parallel using multiprocessing.

        Args:
            all_source_paths (List[MultiPath]): The list of source paths to process.
            all_destination_paths (List[MultiPath]): The list of destination paths to save.
            all_metadata_paths (List[MultiPath]): The locations where to save metadata.
        """
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
                    serialized_kwargs=pickle.dumps(process_single_kwargs),
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
        """Get all paths to process using prefixes provided"""
        all_source_paths, all_destination_paths, all_metadata_paths = [], [], []

        def _valid_path(path: str) -> bool:
            return (self.include_paths is None or path in self.include_paths) and (
                self.exclude_paths is None or path not in self.exclude_paths
            )

        existing_metadata_names = set(
            (MultiPath.parse(path) - self.metadata_prefix).as_str.rstrip(METADATA_SUFFIX)
            for path in recursively_list_files(self.metadata_prefix)
            if _valid_path(path)
        )
        paths = list(recursively_list_files(self.source_prefix))
        random.shuffle(paths)

        for path in paths:
            source_path = MultiPath.parse(path)
            if not self.ignore_existing and (source_path - self.source_prefix).as_str in existing_metadata_names:
                continue

            all_source_paths.append(source_path)
            all_destination_paths.append(self.destination_prefix / (source_path - self.source_prefix))

            metadata_path = MultiPath.parse(source_path.as_str + METADATA_SUFFIX)
            all_metadata_paths.append(self.metadata_prefix / (metadata_path - self.source_prefix))

        return all_source_paths, all_destination_paths, all_metadata_paths

    def __call__(self, **process_single_kwargs: Any):
        """Run the processor."""
        random.seed(self.seed)
        all_source_paths, all_destination_paths, all_metadata_paths = self._get_all_paths()

        fn = self._debug_run_all if self.debug else self._multiprocessing_run_all

        fn(
            all_source_paths=all_source_paths,
            all_destination_paths=all_destination_paths,
            all_metadata_paths=all_metadata_paths,
            **process_single_kwargs,
        )
