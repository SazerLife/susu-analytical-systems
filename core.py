import argparse
from pathlib import Path
import pandas as pd
from apyori import apriori, RelationRecord
from typing import List


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("supp_threshold", type=float)

    parser.add_argument(
        "--sort-by-support",
        action="store_true",
        help="отсортироввать по значению поддержки",
    )
    parser.add_argument(
        "--sort-by-name",
        action="store_true",
        help="отсортироввать лексикографически",
    )

    return parser


def load_data(path: Path):
    data = list()
    for row in path.read_text().split("\n"):
        row = list(map(int, row.split()))
        data.append(row)
    return data  # [0:10000]


def create_df(output: List[RelationRecord], sort_type: str = "by_name"):
    data = [
        [tuple(result.items), result.support, result.ordered_statistics[0].confidence]
        for result in output
    ]
    columns = ["Items", "Support", "Confidence"]
    df = pd.DataFrame(data, columns=columns)

    if sort_type == "by_name":
        df = df.sort_values(by=["Items"])
    elif sort_type == "by_support":
        df = df.sort_values(by=["Support"])
    else:
        raise NotImplementedError(f"Unknown sort type: {sort_type}")

    return df


def apriori_search(
    data: List[List[int]],
    min_support: float,
    min_confidence: float,
    min_length: int = 2,
    max_length: int = None,
):
    output = apriori(
        transactions=data,
        min_support=min_support,
        min_confidence=min_confidence,
        min_length=min_length,
        max_length=max_length,
    )

    return list(output)


def main(
    data_path: Path, supp_threshold: float, sort_by_support: bool, sort_by_name: bool
):
    data = load_data(data_path)
    output = apriori_search(data, supp_threshold)
    df = create_df(output)

    return df


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert args.data_path.exists()

    main(args.data_path, args.supp_threshold, args.sort_by_support, args.sort_by_name)
