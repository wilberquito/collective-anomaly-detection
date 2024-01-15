import matplotlib.pyplot as plt
import pandas as pd


def plot(
    time_series: pd.DataFrame,
) -> None:

    plt.subplots(figsize=(12, 4))
    plt.plot(time_series["timestamp"], time_series["value"], label="value")

    plt.xlabel("Timestamp")
    plt.ylabel("Value")

    plt.legend()
    plt.show()
