from functools import reduce
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import List, Optional, Union

import requests
from tqdm import tqdm
import springs as sp


def get_current_process_number() -> int:
    if not (pid := current_process()._identity):
        return 0
    return reduce(lambda x, y: x * y, pid, 1)


def download_single(
    url: str,
    fname: Union[Path, str],
    chunk_size: int = 1024 ** 2,
):
    """Download a file from a URL and show a progress bar. Adapted from
    https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51"""

    fname = Path(fname)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    tqdm_position = max(get_current_process_number() - 1, 0)

    with open(fname, 'wb') as file, tqdm(
        desc=fname.name,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        position=tqdm_position
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


@sp.dataclass
class DownloadConfig:
    langs: List[str] = sp.flist(
        "en", "de", "fr", "nl", "ru", "es", "it", "arz", "pl", "ja", "zh",
        "vi", "war", "uk", "ar", "pt", "fa", "ca", "sr", "id", "ko", "no",
        "ce", "fi", "hu", "cs", "tr"
    )
    url: str = (
        "https://dumps.wikimedia.org/{lang_code}wiki/20230320/"
        "{lang_code}wiki-20230320-pages-articles-multistream.xml.bz2"
    )
    local_dst: str = sp.MISSING
    local_name: str = "wiki_{lang_code}.xml.bz2"
    remote_dst: Optional[str] = None
    parallel: int = 1
    debug: bool = False


def prepare(config: DownloadConfig, lang: str):
    url = config.url.format(lang_code=lang)
    (local_dst := Path(config.local_dst)).mkdir(parents=True, exist_ok=True)
    fname = local_dst / config.local_name.format(lang_code=lang)

    return url, fname


@sp.cli(DownloadConfig)
def main(config: DownloadConfig):
    if config.debug:
        for lang in config.langs:
            url, fname = prepare(config, lang)
            download_single(url, fname)
        return

    with Pool(config.parallel) as pool:
        for lang in config.langs:
            pool.apply_async(download_single, prepare(config, lang))
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
