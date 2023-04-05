import json
import logging
import sys

from merger import streams
from merger.config import Config
from pathos import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def run(config: Config):
    log.info(f"Running with config={config.json()}")
    input_streams = [(s, streams.document_inputs(s)) for s in config.streams]
    shards = [shard for s, i in input_streams for shard in streams.split_into_shards(s, i, config.output)]
    missing_shards = list(filter(lambda s: not s.output.exists(), shards))
    log.info(f"Processing {len(missing_shards)} shards from sources [{','.join(s.name for s,_ in input_streams)}]")
    n_processes = config.processes or multiprocessing.cpu_count()
    log.info(f"Using {n_processes} processes")
    success = 0
    with multiprocessing.Pool(n_processes) as p:
        for s, err in p.imap_unordered(streams.process, missing_shards):
            if err is None:
                if s.output.exists():
                    log.info(f"Finished writing {s.output}")
                    success += 1
                else:
                    log.warning(f"Finished processing {s.output} but file does not exist!")
            else:
                log.error(f"Failed writing {s.output}", err)
    if success == len(missing_shards):
        log.info("Done!")
    else:
        log.warning(f"{len(missing_shards) - success} shards failed to write")
        exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m merger <config-file>")
        exit(1)
    with open(sys.argv[1]) as f:
        config = Config(**json.load(f))
    run(config)
