import io
import logging
import os
import re
import socket
import sys
import time
import warnings
from datetime import datetime
from enum import Enum
from itertools import cycle, islice
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, Optional, Tuple, Union

import boto3
import botocore.exceptions as boto_exceptions
import datasets
import rich
from botocore.config import Config
from cached_path.schemes import SchemeClient, add_scheme_client
from rich.console import Console, ConsoleRenderable
from rich.highlighter import NullHighlighter
from rich.progress import Progress
from rich.text import Text
from rich.traceback import Traceback

from .aliases import PathOrStr
from .exceptions import (
    OLMoCliError,
    OLMoEnvironmentError,
    OLMoError,
    OLMoNetworkError,
    OLMoThreadError,
)
from .torch_util import (
    barrier,
    get_fs_local_rank,
    get_global_rank,
    get_local_rank,
    get_node_rank,
    is_distributed,
)

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


_log_extra_fields: Dict[str, Any] = {}
log = logging.getLogger(__name__)


class LogFilterType(StrEnum):
    rank0_only = "rank0_only"
    local_rank0_only = "local_rank0_only"
    all_ranks = "all_ranks"


def log_extra_field(field_name: str, field_value: Any) -> None:
    global _log_extra_fields
    if field_value is None:
        if field_name in _log_extra_fields:
            del _log_extra_fields[field_name]
    else:
        _log_extra_fields[field_name] = field_value


def setup_logging(log_filter_type: LogFilterType = LogFilterType.rank0_only) -> None:
    """
    :param rank0_only: INFO and below messages will only be emitted on the rank0 process.
    """
    log_extra_field("hostname", socket.gethostname())
    if is_distributed():
        log_extra_field("node_rank", get_node_rank())
        log_extra_field("local_rank", get_local_rank())
        log_extra_field("global_rank", get_global_rank())
    else:
        log_extra_field("node_rank", 0)
        log_extra_field("local_rank", 0)
        log_extra_field("global_rank", 0)

    old_log_record_factory = logging.getLogRecordFactory()

    def log_record_factory(*args, **kwargs) -> logging.LogRecord:
        record = old_log_record_factory(*args, **kwargs)
        for field_name, field_value in _log_extra_fields.items():
            setattr(record, field_name, field_value)
        return record

    logging.setLogRecordFactory(log_record_factory)

    handler: logging.Handler
    if (
        os.environ.get("OLMo_NONINTERACTIVE", False)
        or os.environ.get("DEBIAN_FRONTEND", None) == "noninteractive"
        or not sys.stdout.isatty()
    ):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s\t%(hostname)s:%(local_rank)s\t%(name)s:%(lineno)s\t%(levelname)s\t%(message)s"
        )
        formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        formatter.default_msec_format = "%s.%03d"
        handler.setFormatter(formatter)
    else:
        handler = RichHandler()

    def rank0_filter(record: logging.LogRecord) -> int:
        if record.levelno > logging.INFO:
            return 1
        if getattr(record, "global_rank", 0) == 0:
            return 1
        else:
            return 0

    def local_rank0_filter(record: logging.LogRecord) -> int:
        if record.levelno > logging.INFO:
            return 1
        if getattr(record, "local_rank", 0) == 0:
            return 1
        else:
            return 0

    if log_filter_type == LogFilterType.rank0_only:
        filter = rank0_filter
    elif log_filter_type == LogFilterType.local_rank0_only:
        filter = local_rank0_filter  # type: ignore
    elif log_filter_type == LogFilterType.all_ranks:
        filter = None
    else:
        raise ValueError(log_filter_type)

    if filter is not None:
        handler.addFilter(filter)  # type: ignore
    logging.basicConfig(handlers=[handler], level=logging.INFO)

    logging.captureWarnings(True)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def excepthook(exctype, value, traceback):
    """
    Used to patch `sys.excepthook` in order to log exceptions.
    """
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, traceback)
    elif issubclass(exctype, OLMoCliError):
        rich.get_console().print(f"[yellow]{value}[/]", highlight=False)
    elif issubclass(exctype, OLMoError):
        rich.get_console().print(Text(f"{exctype.__name__}:", style="red"), value, highlight=False)
    else:
        log.critical("Uncaught %s: %s", exctype.__name__, value, exc_info=(exctype, value, traceback))


def install_excepthook():
    sys.excepthook = excepthook


def filter_warnings():
    # Filter internal deprecation warnings from torch
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="torch.distributed.*_base is a private function and will be deprecated.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="TypedStorage is deprecated.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="Please use DTensor instead.*",
    )
    # Torchvision warnings. We don't actually use torchvision.
    warnings.filterwarnings(
        action="ignore",
        message="failed to load.*",
        module="torchvision.io.image",
    )


def set_env_variables():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_cli_environment(log_filter_type: Optional[LogFilterType] = None):
    if log_filter_type is None:
        log_filter_type = LogFilterType(os.environ.get("LOG_FILTER_TYPE", "rank0_only"))
    rich.reconfigure(width=max(rich.get_console().width, 180), soft_wrap=True)
    setup_logging(log_filter_type=log_filter_type)
    install_excepthook()
    filter_warnings()
    set_env_variables()


def clean_opt(arg: str) -> str:
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    return f"{name}={val}"


class RichHandler(logging.Handler):
    """
    A simplified version of rich.logging.RichHandler from
    https://github.com/Textualize/rich/blob/master/rich/logging.py
    """

    def __init__(
        self,
        *,
        level: Union[int, str] = logging.NOTSET,
        console: Optional[Console] = None,
        markup: bool = False,
    ) -> None:
        super().__init__(level=level)
        self.console = console or rich.get_console()
        self.highlighter = NullHighlighter()
        self.markup = markup

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if hasattr(record.msg, "__rich__") or hasattr(record.msg, "__rich_console__"):
                self.console.print(record.msg)
            else:
                msg: Any = record.msg
                if isinstance(record.msg, str):
                    msg = self.render_message(record=record, message=record.getMessage())
                renderables = [
                    self.get_time_text(record),
                    self.get_level_text(record),
                    self.get_location_text(record),
                    msg,
                ]
                if record.exc_info is not None:
                    tb = Traceback.from_exception(*record.exc_info)  # type: ignore
                    renderables.append(tb)
                self.console.print(*renderables)
        except Exception:
            self.handleError(record)

    def render_message(self, *, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        use_markup = getattr(record, "markup", self.markup)
        message_text = Text.from_markup(message) if use_markup else Text(message)

        highlighter = getattr(record, "highlighter", self.highlighter)
        if highlighter:
            message_text = highlighter(message_text)

        return message_text

    def get_time_text(self, record: logging.LogRecord) -> Text:
        log_time = datetime.fromtimestamp(record.created)
        time_str = log_time.strftime("[%Y-%m-%d %X]")
        return Text(time_str, style="log.time", end=" ")

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname
        level_text = Text.styled(level_name.ljust(8), f"logging.level.{level_name.lower()}")
        level_text.style = "log.level"
        level_text.end = " "
        return level_text

    def get_location_text(self, record: logging.LogRecord) -> Text:
        name_and_line = f"{record.name}:{record.lineno}" if record.name != "root" else "root"
        text = f"[{name_and_line}, rank={record.local_rank}]"  # type: ignore
        return Text(text, style="log.path")


def wait_for(condition: Callable[[], bool], description: str, timeout: float = 10.0):
    """Wait for the condition function to return True."""
    start_time = time.monotonic()
    while not condition():
        time.sleep(0.5)
        if time.monotonic() - start_time > timeout:
            raise TimeoutError(f"{description} timed out")


def is_url(path: PathOrStr) -> bool:
    return re.match(r"[a-z0-9]+://.*", str(path)) is not None


def dir_is_empty(dir: PathOrStr) -> bool:
    dir = Path(dir)
    if not dir.is_dir():
        return True
    try:
        next(dir.glob("*"))
        return False
    except StopIteration:
        return True


def get_progress_bar() -> Progress:
    from cached_path import get_download_progress

    return get_download_progress()


def resource_path(
    folder: PathOrStr, fname: str, local_cache: Optional[PathOrStr] = None, progress: Optional[Progress] = None
) -> Path:
    if local_cache is not None and (local_path := Path(local_cache) / fname).is_file():
        log.info(f"Found local cache of {fname} at {local_path}")
        return local_path
    else:
        from cached_path import cached_path

        return cached_path(f"{str(folder).rstrip('/')}/{fname}", progress=progress)


def file_size(path: PathOrStr) -> int:
    """
    Get the size of a local or remote file in bytes.
    """
    if is_url(path):
        from urllib.parse import urlparse

        parsed = urlparse(str(path))
        if parsed.scheme == "gs":
            return _gcs_file_size(parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_file_size(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme in ("http", "https"):
            return _http_file_size(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme == "file":
            return file_size(str(path).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"file size not implemented for '{parsed.scheme}' files")
    else:
        return os.stat(path).st_size


def upload(source: PathOrStr, target: str, save_overwrite: bool = False):
    """Upload source file to a target location on GCS or S3."""
    from urllib.parse import urlparse

    source = Path(source)
    assert source.is_file()
    parsed = urlparse(target)
    if parsed.scheme == "gs":
        _gcs_upload(source, parsed.netloc, parsed.path.strip("/"), save_overwrite=save_overwrite)
    elif parsed.scheme in ("s3", "r2", "weka"):
        _s3_upload(source, parsed.scheme, parsed.netloc, parsed.path.strip("/"), save_overwrite=save_overwrite)
    else:
        raise NotImplementedError(f"Upload not implemented for '{parsed.scheme}' scheme")


def get_bytes_range(source: PathOrStr, bytes_start: int, num_bytes: int) -> bytes:
    if is_url(source):
        from urllib.parse import urlparse

        parsed = urlparse(str(source))
        if parsed.scheme == "gs":
            return _gcs_get_bytes_range(parsed.netloc, parsed.path.strip("/"), bytes_start, num_bytes)
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_get_bytes_range(
                parsed.scheme, parsed.netloc, parsed.path.strip("/"), bytes_start, num_bytes
            )
        elif parsed.scheme in ("http", "https"):
            return _http_get_bytes_range(
                parsed.scheme, parsed.netloc, parsed.path.strip("/"), bytes_start, num_bytes
            )
        elif parsed.scheme == "file":
            return get_bytes_range(str(source).replace("file://", "", 1), bytes_start, num_bytes)
        else:
            raise NotImplementedError(f"get bytes range not implemented for '{parsed.scheme}' files")
    else:
        with open(source, "rb") as f:
            f.seek(bytes_start)
            return f.read(num_bytes)


def find_latest_checkpoint(dir: PathOrStr) -> Optional[PathOrStr]:
    if is_url(dir):
        from urllib.parse import urlparse

        parsed = urlparse(str(dir))
        if parsed.scheme == "gs":
            raise NotImplementedError
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_find_latest_checkpoint(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme == "file":
            return find_latest_checkpoint(str(dir).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"find_latest_checkpoint not implemented for '{parsed.scheme}' files")
    else:
        latest_step = 0
        latest_checkpoint: Optional[Path] = None
        for path in Path(dir).glob("step*"):
            if path.is_dir():
                try:
                    step = int(path.name.replace("step", "").replace("-unsharded", ""))
                except ValueError:
                    continue
                # We prioritize sharded checkpoints over unsharded checkpoints.
                if step > latest_step or (step == latest_step and not path.name.endswith("-unsharded")):
                    latest_step = step
                    latest_checkpoint = path
        return latest_checkpoint


def _gcs_upload(source: Path, bucket_name: str, key: str, save_overwrite: bool = False):
    from google.cloud import storage as gcs

    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    if not save_overwrite and blob.exists():
        raise FileExistsError(f"gs://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it.")
    blob.upload_from_filename(source)


def _gcs_file_size(bucket_name: str, key: str) -> int:
    from google.api_core.exceptions import NotFound
    from google.cloud import storage as gcs

    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload()
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket_name}/{key}")
    assert blob.size is not None
    return blob.size


def _gcs_get_bytes_range(bucket_name: str, key: str, bytes_start: int, num_bytes: int) -> bytes:
    from google.api_core.exceptions import NotFound
    from google.cloud import storage as gcs

    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload()
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket_name}/{key}")
    return blob.download_as_bytes(start=bytes_start, end=bytes_start + num_bytes - 1)


def _get_s3_profile_name(scheme: str) -> Optional[str]:
    if scheme == "s3":
        # For backwards compatibility, we assume S3 uses the default profile if S3_PROFILE is not set.
        return os.environ.get("S3_PROFILE")
    if scheme == "r2":
        profile_name = os.environ.get("R2_PROFILE")
        if profile_name is None:
            raise OLMoEnvironmentError(
                "R2 profile name is not set. Did you forget to set the 'R2_PROFILE' env var?"
            )

        return profile_name
    if scheme == "weka":
        profile_name = os.environ.get("WEKA_PROFILE")
        if profile_name is None:
            raise OLMoEnvironmentError(
                "Weka profile name is not set. Did you forget to set the 'WEKA_PROFILE' env var?"
            )

        return profile_name

    raise NotImplementedError(f"Cannot get profile name for scheme {scheme}")


def _get_s3_endpoint_url(scheme: str) -> Optional[str]:
    if scheme == "s3":
        return None
    if scheme == "r2":
        r2_endpoint_url = os.environ.get("R2_ENDPOINT_URL")
        if r2_endpoint_url is None:
            raise OLMoEnvironmentError(
                "R2 endpoint url is not set. Did you forget to set the 'R2_ENDPOINT_URL' env var?"
            )

        return r2_endpoint_url
    if scheme == "weka":
        weka_endpoint_url = os.environ.get("WEKA_ENDPOINT_URL")
        if weka_endpoint_url is None:
            raise OLMoEnvironmentError(
                "Weka endpoint url is not set. Did you forget to set the 'WEKA_ENDPOINT_URL' env var?"
            )

        return weka_endpoint_url

    raise NotImplementedError(f"Cannot get endpoint url for scheme {scheme}")


@cache
def _get_s3_client(scheme: str):
    session = boto3.Session(profile_name=_get_s3_profile_name(scheme))
    return session.client(
        "s3",
        endpoint_url=_get_s3_endpoint_url(scheme),
        config=Config(retries={"max_attempts": 10, "mode": "standard"}),
        use_ssl=not int(os.environ.get("OLMO_NO_SSL", "0")),
    )


def _wait_before_retry(attempt: int):
    time.sleep(min(0.5 * 2**attempt, 3.0))


def _s3_upload(
    source: Path, scheme: str, bucket_name: str, key: str, save_overwrite: bool = False, max_attempts: int = 3
):
    err: Optional[Exception] = None
    if not save_overwrite:
        for attempt in range(1, max_attempts + 1):
            try:
                _get_s3_client(scheme).head_object(Bucket=bucket_name, Key=key)
                raise FileExistsError(
                    f"s3://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it."
                )
            except boto_exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    err = None
                    break
                err = e

            if attempt < max_attempts:
                log.warning("%s failed attempt %d with retriable error: %s", _s3_upload.__name__, attempt, err)
                _wait_before_retry(attempt)

        if err is not None:
            raise OLMoNetworkError(f"Failed to check object existence during {scheme} upload") from err

    try:
        _get_s3_client(scheme).upload_file(source, bucket_name, key)
    except boto_exceptions.ClientError as e:
        raise OLMoNetworkError(f"Failed to upload to {scheme}") from e


def _s3_file_size(scheme: str, bucket_name: str, key: str, max_attempts: int = 3) -> int:
    err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _get_s3_client(scheme).head_object(Bucket=bucket_name, Key=key)["ContentLength"]
        except boto_exceptions.ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                raise FileNotFoundError(f"s3://{bucket_name}/{key}") from e
            err = e

        if attempt < max_attempts:
            log.warning("%s failed attempt %d with retriable error: %s", _s3_file_size.__name__, attempt, err)
            _wait_before_retry(attempt)

    raise OLMoNetworkError(f"Failed to get {scheme} file size") from err


def _s3_get_bytes_range(
    scheme: str, bucket_name: str, key: str, bytes_start: int, num_bytes: int, max_attempts: int = 3
) -> bytes:
    err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return (
                _get_s3_client(scheme)
                .get_object(
                    Bucket=bucket_name, Key=key, Range=f"bytes={bytes_start}-{bytes_start + num_bytes - 1}"
                )["Body"]
                .read()
            )
        except boto_exceptions.ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                raise FileNotFoundError(f"{scheme}://{bucket_name}/{key}") from e
            err = e
        except (boto_exceptions.HTTPClientError, boto_exceptions.ConnectionError) as e:
            # ResponseStreamingError (subclass of HTTPClientError) can happen as
            # a result of a failed read from the stream (http.client.IncompleteRead).
            # Retrying can help in this case.
            err = e

        if attempt < max_attempts:
            log.warning(
                "%s failed attempt %d with retriable error: %s", _s3_get_bytes_range.__name__, attempt, err
            )
            _wait_before_retry(attempt)

    # When torch's DataLoader intercepts exceptions, it may try to re-raise them
    # by recalling their constructor with a single message arg. Torch has some
    # logic to deal with the absence of a single-parameter constructor, but it
    # doesn't gracefully handle other possible failures in calling such a constructor
    # This can cause an irrelevant exception (e.g. KeyError: 'error'), resulting
    # in us losing the true exception info. To avoid this, we change the exception
    # to a type that has a single-parameter constructor.
    raise OLMoNetworkError(f"Failed to get bytes range from {scheme}") from err


def _s3_find_latest_checkpoint(scheme: str, bucket_name: str, prefix: str) -> Optional[str]:
    if not prefix.endswith("/"):
        prefix = f"{prefix}/"
    response = _get_s3_client(scheme).list_objects(Bucket=bucket_name, Prefix=prefix, Delimiter="/")
    assert not response["IsTruncated"]  # need to handle this if it happens
    latest_step = 0
    latest_checkpoint: Optional[str] = None
    for item in response["CommonPrefixes"]:
        prefix = item["Prefix"].strip("/")
        checkpoint_name = os.path.split(prefix)[-1]
        if not checkpoint_name.startswith("step"):
            continue
        try:
            step = int(checkpoint_name.replace("step", "").replace("-unsharded", ""))
        except ValueError:
            continue
        # Make sure the checkpoint dir contains a config, otherwise the checkpoint is incomplete
        # (upload might have have failed part way through).
        try:
            _s3_file_size(scheme, bucket_name, f"{prefix}/config.yaml")
        except FileNotFoundError:
            continue
        # We prioritize sharded checkpoints over unsharded ones.
        if step > latest_step or (step == latest_step and not checkpoint_name.endswith("-unsharded")):
            latest_step = step
            latest_checkpoint = f"{scheme}://{bucket_name}/{prefix}"
    return latest_checkpoint


def _http_file_size(scheme: str, host_name: str, path: str) -> int:
    import requests

    response = requests.head(f"{scheme}://{host_name}/{path}", allow_redirects=True)
    return int(response.headers.get("content-length"))


def _http_get_bytes_range(scheme: str, host_name: str, path: str, bytes_start: int, num_bytes: int) -> bytes:
    import requests

    response = requests.get(
        f"{scheme}://{host_name}/{path}", headers={"Range": f"bytes={bytes_start}-{bytes_start+num_bytes-1}"}
    )
    result = response.content
    assert (
        len(result) == num_bytes
    ), f"expected {num_bytes} bytes, got {len(result)}"  # Some web servers silently ignore range requests and send everything
    return result


def _load_hf_dataset_from_disk(hf_path: str, name: Optional[str], split: str, datasets_dir: str):
    dataset_path = os.path.join(datasets_dir, hf_path, name or "none", split)
    return datasets.load_from_disk(dataset_path)


def _save_hf_dataset_to_disk(
    dataset: datasets.DatasetDict | datasets.Dataset,
    hf_path: str,
    name: Optional[str],
    split: str,
    datasets_dir: str,
):
    dataset_path = os.path.join(datasets_dir, hf_path, name or "none", split)
    return dataset.save_to_disk(dataset_path)


def load_hf_dataset(path: str, name: Optional[str], split: str, datasets_cache_dir: Optional[str] = None):
    dataset = None

    # First try to load dataset on only FS rank 0, to avoid unnecessary network load.
    # This will hopefully cache the dataset for use in other FS ranks.
    if get_fs_local_rank() == 0:
        # Try get dataset from disk.
        if datasets_cache_dir is not None:
            try:
                dataset = _load_hf_dataset_from_disk(path, name, split, datasets_cache_dir)
            except FileNotFoundError:
                log.info(
                    "Path %s name %s split %s not present in local dir %s, loading from online",
                    path,
                    name,
                    split,
                    datasets_cache_dir,
                )

        # Get dataset from online if not available on disk
        if dataset is None:
            dataset = datasets.load_dataset(
                path=path,
                name=name,
                split=split,
                trust_remote_code=True,
            )
            assert isinstance(dataset, (datasets.DatasetDict, datasets.Dataset))
            if datasets_cache_dir is not None:
                _save_hf_dataset_to_disk(dataset, path, name, split, datasets_cache_dir)
    barrier()

    # Dataset is loaded in FS rank 0
    if dataset is not None:
        return dataset

    # Load dataset on non-zero FS ranks
    if datasets_cache_dir is not None:
        return _load_hf_dataset_from_disk(path, name, split, datasets_cache_dir)
    return datasets.load_dataset(
        path=path,
        name=name,
        split=split,
        trust_remote_code=True,
    )


def default_thread_count() -> int:
    return int(os.environ.get("OLMO_NUM_THREADS") or min(32, (os.cpu_count() or 1) + 4))


def pass_through_fn(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def threaded_generator(g, maxsize: int = 16, thread_name: Optional[str] = None):
    q: Queue = Queue(maxsize=maxsize)

    sentinel = object()

    def fill_queue():
        try:
            for value in g:
                q.put(value)
        except Exception as e:
            q.put(e)
        finally:
            q.put(sentinel)

    thread_name = thread_name or repr(g)
    thread = Thread(name=thread_name, target=fill_queue, daemon=True)
    thread.start()

    for x in iter(q.get, sentinel):
        if isinstance(x, Exception):
            raise OLMoThreadError(f"generator thread {thread_name} failed") from x
        else:
            yield x


def roundrobin(*iterables):
    """
    Call the given iterables in a round-robin fashion. For example:
    ``roundrobin('ABC', 'D', 'EF') --> A D E B F C``
    """
    # Adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def add_cached_path_clients():
    add_scheme_client(WekaClient)


class WekaClient(SchemeClient):
    recoverable_errors = SchemeClient.recoverable_errors + (
        boto_exceptions.HTTPClientError,
        boto_exceptions.ConnectionError,
    )

    scheme = "weka"

    def __init__(self, resource: str) -> None:
        SchemeClient.__init__(self, resource)
        self.bucket_name, self.path = WekaClient._split_cloud_path(resource, "weka")
        self.s3 = _get_s3_client("weka")
        self.object_info = None

    @staticmethod
    def _split_cloud_path(url: str, provider: str) -> Tuple[str, str]:
        """Split a full s3 path into the bucket name and path."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if not parsed.netloc or not parsed.path:
            raise ValueError("bad {} path {}".format(provider, url))
        bucket_name = parsed.netloc
        provider_path = parsed.path
        # Remove '/' at beginning of path.
        if provider_path.startswith("/"):
            provider_path = provider_path[1:]
        return bucket_name, provider_path

    def _ensure_object_info(self):
        if self.object_info is None:
            try:
                self.object_info = self.s3.head_object(Bucket=self.bucket_name, Key=self.path)
            except boto_exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    raise FileNotFoundError(f"weka://{self.bucket_name}/{self.path}") from e
                raise e

    def get_etag(self) -> Optional[str]:
        self._ensure_object_info()
        assert self.object_info is not None
        return self.object_info.get("ETag")

    def get_size(self) -> Optional[int]:
        self._ensure_object_info()
        assert self.object_info is not None
        return self.object_info.get("ContentLength")

    def get_resource(self, temp_file: io.BufferedWriter) -> None:
        self.s3.download_fileobj(Fileobj=temp_file, Bucket=self.bucket_name, Key=self.path)

    def get_bytes_range(self, index: int, length: int) -> bytes:
        response = self.s3.get_object(
            Bucket=self.bucket_name, Key=self.path, Range=f"bytes={index}-{index+length-1}"
        )
        return response["Body"].read()
