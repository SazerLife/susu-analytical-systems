from pathlib import Path
from time import time
from core import load_data, apriori_search, create_df
import matplotlib.pyplot as plt
from tqdm import tqdm

supports = [0.01, 0.03, 0.05, 0.10, 0.15]
confidence = 0.0


def compare_performance(data_path: Path):
    data = load_data(data_path)
    durations = list()
    for support_threshold in tqdm(supports):
        start = time()
        _ = apriori_search(data, support_threshold, confidence)
        durations.append(time() - start)

    plt.plot(supports, durations)
    plt.savefig("lab1-performance_comparation.png")
    plt.close()


def compare_rules_count(data_path: Path):
    data = load_data(data_path)
    counts = list()
    for support_threshold in tqdm(supports):
        rules_count = len(apriori_search(data, support_threshold, confidence))
        counts.append(rules_count)

    plt.plot(supports, counts)
    plt.savefig("lab1-rules_count-comparation.png")
    plt.close()


if __name__ == "__main__":
    # path = Path("data/accidents.dat")
    path = Path("data/retail.dat")
    compare_performance(path)
    compare_rules_count(path)
