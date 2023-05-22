import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from multiprocessing import current_process
from typing import Union

import gutenbergpy.textget
import requests
import springs as sp
import tqdm
from rdflib import Graph, plugin  # noqa: F401
from rdflib.serializer import Serializer  # noqa: F401
from smashed.utils.io_utils import open_file_for_write


def get_current_process_number() -> int:
    if not (pid := current_process()._identity):
        return 0
    return reduce(lambda x, y: x * y, pid, 1)


@sp.dataclass
class DownloadConfig:
    dst: str = "s3://ai2-llm/pretraining-data/sources/gutenberg/raw/json"
    wait: float = 0.1
    max_retry: int = 10
    latest: int = 70640
    num_processes: int = 1


REQUEST_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
RDF_URL = "https://www.gutenberg.org/ebooks/{id}.rdf"


def download_rdf(_id: int) -> dict:
    url = RDF_URL.format(id=_id)
    g = Graph()
    g.parse(url)
    return json.loads(g.serialize(format="json-ld"))


def raw_request(_id: int, max_retry: int = 10, wait: float = 1) -> str:
    req = None
    url = REQUEST_URL.format(id=_id)
    retry = 0
    while retry < max_retry:
        req = requests.get(url)
        if req.status_code == 200:
            break
        time.sleep(wait)
        retry += 1

    if req is None or req.status_code != 200:
        raise RuntimeError(f"[WARNING] Failed to download {url}")
    return req.text


def download_range(start: int, end: int, prefix: str, wait: float = 0.1, max_retry: int = 10):
    pid = get_current_process_number()
    logger = sp.configure_logging(f"gutenberg_{pid}")

    for _id in tqdm.tqdm(
        range(start, end + 1),
        desc=f"Downloading ({start}-{end})...",
        total=(end - start + 1),
        unit="b",
        position=(pid - 1) if pid > 0 else None,
    ):
        # url = config.src.format(id=_id)
        group = _id // 100
        dst = prefix.rstrip("/") + f"/{group}/{_id}.json"

        raw: Union[str, None] = None
        msg = pipeline = None
        try:
            raw = gutenbergpy.textget.get_text_by_id(_id).decode("utf-8")
            pipeline = "gutenbergpy"
        except Exception as e:
            msg = f"Failed to download {_id} via gutenbergpy ({e})"

        if raw is None:
            try:
                raw = raw_request(_id, max_retry=max_retry, wait=wait)
                pipeline = "requests"
            except Exception as e:
                msg = f"Failed to download {_id} via requests ({e})"

        if raw is None:
            logger.warning(msg)
            continue

        try:
            rdf = download_rdf(_id)
        except Exception:
            rdf = {}
            logger.warning(f"Failed to download {_id} via rdf")

        data = {
            "id": _id,
            "raw": raw,
            "pipeline": pipeline,
            "rdf": rdf,
        }

        with open_file_for_write(dst, "w") as f:
            f.write(json.dumps(data, indent=2))

        time.sleep(wait)


@sp.cli(DownloadConfig)
def download(cfg: DownloadConfig):
    if cfg.num_processes > 1:
        with ProcessPoolExecutor(max_workers=cfg.num_processes) as executor:
            futures = []
            for i in range(cfg.num_processes):
                start = cfg.latest // cfg.num_processes * i + 1
                end = cfg.latest // cfg.num_processes * (i + 1)
                futures.append(
                    executor.submit(
                        download_range,
                        start,
                        end,
                        prefix=cfg.dst,
                        wait=cfg.wait,
                        max_retry=cfg.max_retry,
                    )
                )
            for future in futures:
                future.result()
    else:
        download_range(1, cfg.latest, prefix=cfg.dst, wait=cfg.wait, max_retry=cfg.max_retry)


if __name__ == "__main__":
    download()
