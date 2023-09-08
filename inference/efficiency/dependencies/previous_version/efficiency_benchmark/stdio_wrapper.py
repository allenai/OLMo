import json
import os
import subprocess
from abc import ABC
from subprocess import SubprocessError
from typing import Any, Dict, Iterator, List, Sequence

import more_itertools
import tqdm


class StdioWrapper(ABC):
    """
    A model that wraps a binary that reads from stdin and writes to stdout.
    """

    def __init__(self, *, cmd: List[str]):
        """
        binary_cmd: the command to start the inference binary
        """
        self._cmd = cmd

    def _exhaust_and_yield_stdout(self, block_until_read_num_batches: int = None):
        """
        Read everything from the stdout pipe.
        This function uses stdout.readline() to read one prediction at a time.
        stdout.readline() is either blocking or non-blocking (in this case returns "" if nothing is available),
        and the behavior is determined by calling os.set_blocking(self._process.stdout.fileno(), False/True).
        To avoid complicated async/threaded code, we instead toggle the blocking behavior as needed.
        During non-blocking operation we empty the pipe, but don't wait for additional predictions.
        During blocking, we block reads until a certain number of predicitons is read (used to ensure we receive predictions for all instances).

        block_until_read_num_instances: if None then non-blocking. Otherwise, block until this many predictions are read.
        """
        self._set_blocking(block_until_read_num_batches)
        if block_until_read_num_batches is None:
            block_until_read_num_batches = 1000000000

        num_batches_yielded = 0
        while num_batches_yielded < block_until_read_num_batches:
            # output is bytes, decode to str
            # Also necessary to remove the \n from the end of the label.
            try:
                output_batch = self._read_batch()
            except ValueError:
                break
            try:
                output_batch = json.loads(output_batch)
            except:
                # Irrelavent output in stdout
                continue
            yield output_batch
            num_batches_yielded += 1

    def _set_blocking(self, block_until_read_num_batches: int = None):
        blocking = block_until_read_num_batches is not None
        os.set_blocking(self._process.stdout.fileno(), blocking)

    def _write_batch(self, batch: Sequence[Dict[str, Any]]) -> None:
        try:
            self._process.stdin.write(f"{json.dumps(batch)}\n".encode("utf-8"))
            self._process.stdin.flush()
        except:
            self.stop()
            raise SubprocessError

    def _read_batch(self) -> str:
        line = self._process.stdout.readline().decode("utf-8").strip()
        if line == "":
            raise ValueError
        elif line == "Efficiency benchmark exception: SubprocessError":
            self.stop()
            print("Below is the traceback of the subprocess:")
            print("=========================")
            while line != "":
                print(line)
                line = self._process.stdout.readline().decode("utf-8").strip()
            raise SubprocessError
        return line

    def predict(  # type: ignore
        self, *, input_batches: List[List[Dict[str, Any]]], max_batch_size: int
    ) -> Iterator[str]:
        for input_batch in tqdm.tqdm(input_batches, desc="Making predictions", miniters=10):
            # Make sure the batch size does not exceed a user defined maximum.
            # Split into smaller batches if necessary.
            splitted_batches = list(more_itertools.chunked(input_batch, max_batch_size))
            num_splitted_batches = len(splitted_batches)
            num_batches_yielded, num_outputs_yielded = 0, 0
            for batch in splitted_batches:
                self._write_batch(batch)
                # Feed all splitted batches without blocking.
                output_batches = self._exhaust_and_yield_stdout(None)
                for output_batch in output_batches:
                    num_batches_yielded += 1
                    for output in output_batch:
                        yield output
                        num_outputs_yielded += 1

            # Now read from stdout until we have hit the required number.
            num_batches_to_read = num_splitted_batches - num_batches_yielded
            if num_batches_to_read > 0:
                for output_batch in self._exhaust_and_yield_stdout(num_batches_to_read):
                    for output in output_batch:
                        yield output
                        num_outputs_yielded += 1
            assert num_outputs_yielded == len(input_batch), "Number of outputs does not match number of inputs."

    def provide_offline_configs(self, offline_data_path: str, offline_output_file: str, limit: int = -1) -> bool:
        configs = {
            "offline_data_path": offline_data_path,
            "offline_output_path": offline_output_file,
            "limit": limit,
        }
        os.set_blocking(self._process.stdout.fileno(), True)
        self._process.stdin.write(f"{json.dumps(configs)}\n".encode("utf-8"))
        self._process.stdin.flush()

        while True:
            line = self._process.stdout.readline()
            if line.decode("utf-8").strip() == "Model and data loaded. Start the timer.":
                break

    def block_for_prediction(self) -> bool:
        os.set_blocking(self._process.stdout.fileno(), True)

        while True:
            line = self._process.stdout.readline()
            if line.decode("utf-8").strip() == "Offiline prediction done. Stop the timer.":
                break

    def block_for_outputs(self) -> bool:
        os.set_blocking(self._process.stdout.fileno(), True)

        while True:
            line = self._process.stdout.readline()
            if line.decode("utf-8").strip() == "Offiline outputs written. Exit.":
                break

    def start(self):
        self._process = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def dummy_predict(self, dummy_inputs: List[Dict[str, Any]], max_batch_size: int) -> List[str]:
        dummy_outputs = self.predict(input_batches=[dummy_inputs], max_batch_size=max_batch_size)
        return list(dummy_outputs)

    def stop(self):
        try:
            self._process.kill()
        except:
            pass
