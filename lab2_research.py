from pathlib import Path
from time import time
from core import load_data, apriori_search, create_df
import matplotlib.pyplot as plt
from tqdm import tqdm

support = 0.1
confidences = [round(i / 100, 2) for i in range(70, 100, 5)]


def compare_performance(data_path: Path):
    data = load_data(data_path)
    durations = list()
    for confidence in tqdm(confidences):
        start = time()
        _ = apriori_search(data, support, confidence, 2, 3)
        durations.append(time() - start)

    plt.plot(confidences, durations)
    plt.savefig("lab2-performance_comparation.png")
    plt.close()


def compare_rules_count(data_path: Path):
    data = load_data(data_path)
    counts = list()
    for confidence in tqdm(confidences):
        rules_count = len(apriori_search(data, support, confidence, 2, 3))
        counts.append(rules_count)

    plt.plot(confidences, counts)
    plt.savefig("lab2-rules_count-comparation.png")
    plt.close()


def custom_config(data_path: Path):
    data = load_data(data_path)
    output = apriori_search(data, 0.1, 0.1, 2, 3)
    df = create_df(output)
    df.to_csv("custom_rules.csv", index=None)


if __name__ == "__main__":
    path = Path("data/accidents.dat")
    # path = Path("data/retail.dat")

    compare_performance(path)
    compare_rules_count(path)
    # custom_config(path)
