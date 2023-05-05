import argparse
import os
import json
from uuid import uuid4
import datetime
import gzip
import re
from collections import defaultdict
from tqdm import tqdm
from bs4 import BeautifulSoup
import html

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
    def c4_en(args):
        with gzip.open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            for line in f:
                doc = json.loads(line)
                yield {"text":doc["text"], "metadata":{"url":doc["url"], "date":doc["date"]}}
    
    @staticmethod
    def c4_en(args):
        with gzip.open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            for line in f:
                doc = json.loads(line)
                yield {"text":doc["text"], "metadata":{"url":doc["url"], "date":doc["date"]}}
    
    @staticmethod
    def twitterAAE_helm(args):
        return [{'text':doc} for doc in open(os.path.join(args.in_dir, args.filename))]

    @staticmethod
    def ice(args):
        '''
        This method assumes that the data has already been preprocessed by the HELM code. You can do this by following the instructions in LLM/eval_data/format_conversion/get_ice/readme.md
        '''
        with open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            for line in f:
                doc = json.loads(line)
                yield doc
                
    @staticmethod
    def wikitext_103(args):
        with open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            text = ''
            for line in f:
                if re.search("^\s*=\s[^=]+\s=\s*$", line) and text.strip() != '':
                    yield {'text':text}
                    text = ''
                text += line
    
    @staticmethod
    def m2d2_wiki(args):
        with open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            doc = ''
            separator = '\n\n\n'
            for char in f.read():
                doc += char
                if doc.endswith(separator):
                    yield {'text':doc[:-len(separator)].strip()}
                    doc = ''
            if doc:
                yield {'text':doc.strip()}

    @staticmethod
    def m2d2_s2orc(args):
        with open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8", errors='ignore') as f:
            for line in f:
                yield {'text':line}
    
    @staticmethod
    def manosphere(args):
        thread2number_post2text = defaultdict(dict)
        with open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8", errors='ignore') as f:
            for line in tqdm(f):
                doc = json.loads(line)
                if not (doc['author'] and doc["text_post"] and doc["number_post"]):
                    continue
                author = doc['author']
                text_post = doc["text_post"]
                thread2number_post2text[doc['thread']][doc["number_post"]] = author +': ' + text_post +'\n\n\n'

        for thread, number_post2text in thread2number_post2text.items():
            # posts = [number_post2text[i+1] for i in range(len(number_post2text))]
            posts = [number_post2text[i] for i in sorted(number_post2text)]
            text = ''.join(posts)
            yield {"text":text, "metadata":{"thread":thread}}

    @staticmethod
    def gab(args):
        with open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            for line in f:
                doc = json.loads(line)
                metadata = {
                    "original_id":doc["id"],
                    "category":doc["post"]["category_details"]["title"] if doc["post"]["category_details"] else None,
                    "topic":doc["post"]["topic"]["title"] if "topic" in doc["post"] and doc["post"]["topic"] else None,
                    "user":doc["post"]["user"]["username"],
                    "created_at":doc["post"]["created_at"]
                }
                yield {"text":doc["post"]["body"], "metadata":metadata}
    @staticmethod
    def four_chan(args):
        def html2text(data):
            soup = BeautifulSoup(html.unescape(data), "html.parser")
            return soup.get_text('\n')
        
        def post_header(doc):
            user = (doc['name'] if 'name' in doc else 'Anonymous')
            return ' '.join([user, doc['now'], 'no.' + str(doc['no'])]) +': '
        
        with open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            for line in tqdm(f):
                thread = json.loads(line)['posts']
                metadata = {
                    "semantic_url": thread[0]["semantic_url"],
                    "original_ids":[doc["no"] for doc in thread],
                    "original_times":[doc["time"] for doc in thread]
                }
                posts = (post_header(doc) + html2text(doc['com']) +'\n\n\n' for doc in thread if 'com' in doc)
                yield {"text":''.join(posts), "metadata":metadata}
    
    @staticmethod
    def PTB(args):
        # PTB has no document seperation so just treat it as one big document
        with open(os.path.join(args.in_dir, args.filename), "rt", encoding="UTF8") as f:
            yield {"text":f.read()}
    


def main():
    parse = argparse.ArgumentParser("")

    parse.add_argument("--in_dir", type=str)
    parse.add_argument("--out_dir", type=str)
    parse.add_argument("--filename", type=str)
    parse.add_argument("--in_format", type=str)

    args = parse.parse_args()

    data = getattr(Formatter, args.in_format)(args)
    basename = os.path.splitext(args.filename)[0]
    if basename.endswith(".json"):
        basename = basename[:-len(".json")]

    with gzip.open(os.path.join(args.out_dir, basename +'.jsonl.gz'), 'wt') as fout:
        for doc in data:
            doc['id'] = str(uuid4()) if 'id' not in doc else doc['id']
            doc['added'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            doc['source'] = args.in_format if 'source' not in doc else doc['source']
            fout.write(json.dumps(doc) + '\n')


if __name__ == "__main__":
    main()
