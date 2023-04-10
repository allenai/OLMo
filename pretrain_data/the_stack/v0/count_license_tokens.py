from ast import literal_eval
import pandas as pd
import sys

def count_license_tokens(token_lengths, licenses):
    result = {}
    for i in range(len(token_lengths)):
        token_length = token_lengths[i]
        for li in licenses[i]:
            if li in result:
                result[li] += token_length
            else:
                result[li] = token_length
    return result


def run(path: str):
	converters = {"document_licenses": literal_eval, "document_token_length": literal_eval}
	df = pd.read_csv(path, sep="\t", converters=converters)
	df["license_tokens"] = [df.apply(lambda x: count_license_tokens(x.document_token_length, x.document_licenses), axis=1).values[0]]
	df.to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
	run(sys.argv[1])

