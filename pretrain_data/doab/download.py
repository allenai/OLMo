import json
import multiprocessing
import re
import tempfile
from functools import partial
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import springs as sp
import tqdm
from bs4 import BeautifulSoup
from bs4.element import Tag
from cached_path import cached_path
from fake_useragent import UserAgent
from smashed.utils.io_utils import (
    MultiPath,
    open_file_for_write,
    recursively_list_files,
)

UA = UserAgent()


def is_pdf_link(tag: Tag) -> bool:
    # Check if the tag is a link and has 'href' attribute
    if not (tag.name == "a" and "href" in tag.attrs):
        return False

    href = str(tag["href"]).lower()
    if href.startswith("#"):
        return False

    # parse the URL
    url = urlparse(href)

    # Check if the URL is for a PDF file
    if url.path.endswith(".pdf") or "pdf" in url.query.lower():
        return True

    # Check if the link text is PDF or contains PDF
    if "pdf" in tag.get_text(strip=True, separator=" ").lower():
        return True

    # Check if the link has a type attribute with a value of “application/pdf”:
    if "type" in tag.attrs and str(tag["type"]).lower() == "application/pdf":
        return True

    # Check if the link has a parent element with a class or an ID that
    # suggests it is a PDF link. For example, sometimes websites may wrap the
    # PDF link in a <div> or other elements with a class or ID like
    # “pdf-download”, “pdf-viewer”, etc.:
    c = tag.parent.get("class", []) if tag.parent else []
    parent_classes = [c] if isinstance(c, str) else ([] if not c else c)
    parent_id = tag.parent.get("id", "") if tag.parent else ""
    parent_attr = [*parent_classes, parent_id]
    if any(("pdf" in attr.lower()) for attr in parent_attr):
        return True

    return False


@sp.dataclass
class Config:
    metadata: str = sp.field(
        default=("s3://ai2-llm/pretraining-data/sources/doab/raw/doabooks_repository-export_20230331.csv"),
        help="URL or path to metadata file.",
    )
    destination: str = sp.field(
        default=("s3://ai2-llm/pretraining-data/sources/doab/raw"),
        help="URL or path to location where to store files.",
    )
    from_scratch: bool = sp.field(default=False, help="Whether to download from scratch.")
    debug: Optional[str] = sp.field(default=None, help="Provide url to download debug for.")
    parallel: int = sp.field(default=1, help="Number of parallel downloads.")


def process_url(
    url: str,
    id_: str,
    base_path: MultiPath,
    _depth: int = 1,
    _processed_links: Optional[Set[str]] = None,
) -> Tuple[str, bool, List[str]]:
    if _depth < 0:
        return "depth exceeded", False, []

    _processed_links = _processed_links or set()

    if not isinstance(url, str):
        return "invalid url", False, []

    if "||" in url:
        success = False
        urls = sorted(set(u.strip() for u in url.split("||") if u.strip()))
        for i, sub_url in enumerate(urls):
            if sub_url in _processed_links:
                continue

            _, sub_success, sub_urls = process_url(
                url=sub_url,
                id_=f"{id_}_{i}",
                base_path=base_path,
                _depth=_depth,  # do not decrement depth
                _processed_links=_processed_links,
            )
            success = success or sub_success

            _processed_links.update([sub_url, *sub_urls])

        return "url list", success, sorted(_processed_links)

    try:
        response = requests.get(url, headers={"User-Agent": UA.random})
    except Exception:
        return "request error", False, []

    if response.status_code >= 400:
        return f"status code {response.status_code}", False, []

    content_type = response.headers.get("content-type", "unknown")

    if content_type.startswith("application/pdf"):
        with open_file_for_write(base_path / f"{id_}.pdf", "wb") as f:
            f.write(response.content)
        return "pdf", True, [url]

    if content_type.startswith("application"):
        name = (
            match[0]
            if (match := re.findall("filename=(.+)", str(response.headers.get("content-disposition"))))
            else (match[-1] if (match := urlparse(response.url).path.split("/")) else "")
        )
        if name.endswith(".pdf"):
            with open_file_for_write(base_path / f"{id_}.pdf", "wb") as f:
                f.write(response.content)
            return "pdf", True, [url]

    if content_type.startswith("text/html"):
        soup = BeautifulSoup(response.content, "html.parser")
        pdf_links = sorted(set(url for a in soup.find_all(is_pdf_link) if (url := a.get("href"))))
        success = False

        for i, pdf_link in enumerate(sorted(pdf_links)):
            if pdf_link.startswith("/"):
                # combine urls
                pdf_link = urljoin(response.url, pdf_link)

            if pdf_link in _processed_links:
                continue

            _, sub_success, sub_urls = process_url(
                url=pdf_link,
                id_=f"{id_}_{i}",
                base_path=base_path,
                _depth=_depth - 1,
                _processed_links=_processed_links,
            )
            success = success or sub_success
            _processed_links.update([*sub_urls, url])

        return "pdf list", success, sorted(_processed_links)

    return "unknown", False, []


def process_single(config: dict, base_path: MultiPath):
    id_ = config.pop("id", None)
    base_url = config.pop("BITSTREAM Download URL", None)
    content_type = "unknown"
    success = False

    if not id_:
        return

    meta_path = base_path / "metadata" / id_[:2]
    data_path = base_path / "data" / id_[:2]

    properties = {k: v for k, v in config.items() if not pd.isna(v) and v}
    metadata = {"properties": properties, "extra": {}, "id": id_, "url": base_url}

    if base_url and id_:
        content_type, success, urls = process_url(url=base_url, id_=id_, base_path=data_path)
        metadata["extra"]["urls"] = urls

    metadata.update({"type": content_type, "success": success})

    with open_file_for_write(meta_path / f"{id_}.json", "w") as f:
        json.dump(metadata, f)


@sp.cli(Config)
def main(config: Config):
    if config.debug:
        url = config.debug
        base_path = MultiPath.parse(tempfile.gettempdir())
        print(process_url(url, "debug", base_path))
        return

    df = pd.read_csv(cached_path(config.metadata))
    data = df.to_dict(orient="records")
    base_path = MultiPath.parse(config.destination)

    if not config.from_scratch:
        # filter out already processed
        metadata_path = base_path / "metadata"
        already_processed = [
            (MultiPath.parse(path) - metadata_path).as_str.lstrip("/").rstrip(".json")
            for path in recursively_list_files(metadata_path)
        ]
    else:
        already_processed = []

    data = [d for d in data if d["id"] not in already_processed]

    if config.parallel > 1:
        fn = partial(process_single, base_path=base_path)
        with multiprocessing.Pool(config.parallel) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(fn, data),
                total=len(data) + len(already_processed),
                desc="Downloading DOAB books",
                initial=len(already_processed),
                unit=" books",
                unit_scale=True,
            ):
                ...

            pool.close()
            pool.join()

    else:
        for elem in tqdm.tqdm(
            data,
            desc="Downloading DOAB books",
            initial=len(already_processed),
            total=len(data) + len(already_processed),
            unit=" books",
            unit_scale=True,
        ):
            process_single(elem, base_path=base_path)


if __name__ == "__main__":
    main()
