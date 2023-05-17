import gzip
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import boto3
from pathos import multiprocessing as mp
from smart_open import open as smopen

s3 = boto3.client("s3")

import c4
import gopher

WORK_DIR = Path(os.environ.get("WORK_DIR", "/data"))
BUCKET = os.environ.get("BUCKET", "ai2-llm")
PARALLELISM = int(os.environ.get("PARALLELISM", mp.cpu_count()))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def get_attributes(text: str) -> Dict:
    attrs = c4.get_attributes(text)
    attrs.update(gopher.get_attributes(text))
    return attrs


def process_file(path: str) -> Tuple[str, Optional[Exception]]:
    log.info(f"Processing {path}")
    input_url = f"s3://{BUCKET}/{path}"
    output_path = path.replace("documents", "attributes/c4_and_gopher")
    output_file = WORK_DIR / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_url = input_url.replace("documents", "attributes/c4_and_gopher")
    try:
        with smopen(output_url, "rb"):
            log.info(f"{output_path} exists. Skipping.")
            return path, None
    except OSError:
        log.info(f"Writing to {output_file}")
        pass
    try:
        with smopen(output_file, "w") as w:
            for line in smopen(input_url, "rb"):
                obj = json.loads(line)
                text = obj["text"]
                attrs = get_attributes(text)
                out_obj = {
                    "id": obj["id"],
                    "attributes": attrs,
                }
                out_json = json.dumps(out_obj)
                w.write(out_json + "\n")
                w.flush()
        log.info(f"Uploading {output_file} to {output_path}")
        s3.upload_file(str(output_file), "ai2-llm", output_path)
        output_file.unlink()
        return path, None
    except Exception as e:
        return path, e


if __name__ == "__main__":
    paths = [l.strip() for l in sys.stdin.readlines()]
    log.info(f"Processing {len(paths)} files")
    success = 0
    with mp.Pool(PARALLELISM) as p:
        for path, err in p.imap_unordered(process_file, paths):
            if err is None:
                log.info(f"Finished writing {path}")
                success += 1
            else:
                log.error(f"Failed to write {path}: {err}")
    if success == len(paths):
        log.info("Done!")
    else:
        log.warning(f"{len(paths) - success} shards failed")
        exit(1)
