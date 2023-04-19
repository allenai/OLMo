import json
from pathlib import Path
from .v0 import format_arxiv, format_c4

arxiv_path = Path(__file__).parent / "arxiv_sample.jsonl"
c4_path = Path(__file__).parent / "c4_sample.jsonl"


def main():
    with open(arxiv_path, "r") as f:
        arxiv_data = [json.loads(ln) for ln in f]
        arxiv_formatted = [format_arxiv(row) for row in arxiv_data]
        assert 'id' in arxiv_formatted[0]
        assert 'text' in arxiv_formatted[0]
        assert 'source' in arxiv_formatted[0]
        assert 'version' in arxiv_formatted[0]
        assert 'added' in arxiv_formatted[0]
        assert 'created' in arxiv_formatted[0]
        assert 'metadata' in arxiv_formatted[0]

    with open(c4_path, "r") as f:
        c4_data = [json.loads(ln) for ln in f]
        c4_formatted = [format_c4(row) for row in c4_data]
        assert 'id' in c4_formatted[0]
        assert 'text' in c4_formatted[0]
        assert 'source' in c4_formatted[0]
        assert 'version' in c4_formatted[0]
        assert 'added' in c4_formatted[0]
        assert 'created' in c4_formatted[0]
        assert 'metadata' in c4_formatted[0]
        assert 'url' in c4_formatted[0]['metadata']
        assert 'length' in c4_formatted[0]['metadata']



if __name__ == "__main__":
    main()
