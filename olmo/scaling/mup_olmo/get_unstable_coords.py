import sys

import pandas as pd


def avg_diff(rows):
    # return rows["l1"].mean()
    # return rows["l1"].max() - rows["l1"].min()
    # return rows["l1"].var()
    return (
        rows[rows["width"] == rows["width"].max()]["l1"].mean()
        - rows[rows["width"] == rows["width"].min()]["l1"].mean()
    )


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    print(df.groupby(by="module").apply(avg_diff).sort_values(ascending=False))
