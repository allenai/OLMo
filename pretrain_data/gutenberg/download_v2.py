import time
from typing import Union
import tqdm
import requests
import gutenbergpy.textget
from multiprocessing import current_process
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
import springs as sp
from smashed.utils.io_utils import (
    open_file_for_write
)

def get_current_process_number() -> int:
    if not (pid := current_process()._identity):
        return 0
    return reduce(lambda x, y: x * y, pid, 1)

@sp.dataclass
class DownloadConfig:
    dst: str = 's3://ai2-llm/pretraining-data/sources/gutenberg/raw/pg'
    wait: float = 0.1
    max_retry: int = 10
    latest: int = 70640
    num_processes: int = 1


REQUEST_URL = 'https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt'


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

    if req is None:
        raise RuntimeError(f'[WARNING] Failed to download {url}')
    return req.text


def download_range(start: int, end: int, prefix: str, wait: float = 0.1, max_retry: int = 10):
    pid = get_current_process_number()
    logger = sp.configure_logging(f'gutenberg_{pid}.log')
    for _id in tqdm.tqdm(
            range(start, end + 1),
            desc=f'Downloading ({pid})...',
            total=(end - start + 1),
            unit='b',
            position=(pid - 1) if pid > 0 else None,
        ):
        # url = config.src.format(id=_id)
        group = _id // 1000
        dst = prefix.rstrip('/') + f'/{group}/{_id}.txt'

        text: Union[str, None] = None
        msg = None
        try:
            text_bytes = gutenbergpy.textget.get_text_by_id(_id)
            text = text_bytes.decode('utf-8')
        except Exception as e:
            msg = f'Failed to download {_id} via gutenbergpy ({e})'

        try:
            text = raw_request(_id, max_retry=max_retry, wait=wait)
        except Exception as e:
            msg = f'Failed to download {_id} via requests ({e})'

        if text is None:
            logger.warning(msg)
            continue

        with open_file_for_write(dst, 'w') as f:
            f.write(text)

        time.sleep(wait)


@sp.cli(DownloadConfig)
def download(cfg: DownloadConfig):
    if cfg.num_processes > 1:
        with ProcessPoolExecutor(max_workers=cfg.num_processes) as executor:
            futures = []
            for i in range(cfg.num_processes):
                start = cfg.latest // cfg.num_processes * i + 1
                end = cfg.latest // cfg.num_processes * (i + 1)
                futures.append(executor.submit(
                    download_range,
                    start,
                    end,
                    prefix=cfg.dst,
                    wait=cfg.wait,
                    max_retry=cfg.max_retry,
                ))
            for future in futures:
                future.result()
    else:
        download_range(1, cfg.latest, prefix=cfg.dst, wait=cfg.wait, max_retry=cfg.max_retry)


if __name__ == '__main__':
    download()
