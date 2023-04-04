import argparse
import os
import json
from uuid import uuid4
import datetime
import gzip

class Formatter():
    @staticmethod
    def c4_100_domains(args):
        data = json.load(gzip.open(os.path.join(args.in_dir, args.filename)))
        return [{'text':doc} for doc in data]
    
    @staticmethod
    def pile(args):
        with gzip.open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            for line in f:
                doc = json.loads(line)
                doc['metadata'] = doc['meta']
                del doc['meta']
                yield doc
    
    @staticmethod
    def twitterAAE_helm(args):
        return [{'text':doc} for doc in open(os.path.join(args.in_dir, args.filename))]

    @staticmethod
    def ice(args):
        '''
        This method assumes that the data has already been parsed using ICEScenario.get_instances() from HELMs codebase. This requires unzipping the 
        data into the right place and fixing some path names.
        '''
        with open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            for line in f:
                doc = json.loads(line)
                yield doc

def main():
    parse = argparse.ArgumentParser("")

    parse.add_argument("--in_dir", type=str)
    parse.add_argument("--out_dir", type=str)
    parse.add_argument("--filename", type=str)
    parse.add_argument("--in_format", type=str)

    args = parse.parse_args()

    data = getattr(Formatter, args.in_format)(args)
    with gzip.open(os.path.join(args.out_dir, os.path.splitext(args.filename)[0]+'.jsonl.gz'), 'wt') as fout:
        for doc in data:
            doc['id'] = str(uuid4())
            doc['added'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            doc['source'] = args.in_format
            fout.write(json.dumps(doc) + '\n')


if __name__ == "__main__":
    main()
