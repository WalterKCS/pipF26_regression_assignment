"""Chart baseline vs v1 on the 72 test cars.

Left: predicted sale price against the actual sale price.
Right: R² and typical dollar error (RMSE) for each predictor.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

TRAIN_PATH = Path(__file__).with_name("used_car_train.csv")
TEST_PATH = Path(__file__).with_name("used_car_test.csv")
OUT_PATH = Path(__file__).with_name("test_fit_comparison.png")
FORMULA = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident_Yes"

BASELINE_COLOR = "#6b6b6b"
V1_COLOR = "#1f4e79"
LINE_COLOR = "#111111"


def load(path: Path) -> pd.DataFrame:
    cars = pd.read_csv(path)
    cars["Accident_Yes"] = (cars["Accident"] == "yes").astype(int)
    return cars


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main() -> None:
    train = load(TRAIN_PATH)
    test = load(TEST_PATH)
    model = smf.ols(FORMULA, data=train).fit()

    actual = test["Sale_Price"].to_numpy(dtype=float)
    train_mean = float(train["Sale_Price"].mean())
    baseline = np.full_like(actual, train_mean)
    predicted = model.predict(test).to_numpy(dtype=float)

    scores = {
        "Baseline": {"r2": r2_score(actual, baseline), "rmse": rmse(actual, baseline)},
        "v1": {"r2": r2_score(actual, predicted), "rmse": rmse(actual, predicted)},
    }

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
        }
    )

    fig = plt.figure(figsize=(11.2, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1], wspace=0.35, hspace=0.45)
    ax = fig.add_subplot(grid[:, 0])
    ax_rmse = fig.add_subplot(grid[0, 1])
    ax_r2 = fig.add_subplot(grid[1, 1])
    low = min(actual.min(), predicted.min(), train_mean) - 1500
    high = max(actual.max(), predicted.max(), train_mean) + 1500
    ax.plot([low, high], [low, high], color=LINE_COLOR, linewidth=1, label="Perfect prediction")
    ax.axhline(train_mean, color=BASELINE_COLOR, linewidth=1.5, linestyle="--", label="Baseline")
    ax.scatter(actual, predicted, s=28, color=V1_COLOR, alpha=0.85, label="v1", zorder=3)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual sale price")
    ax.set_ylabel("Predicted sale price")
    ax.set_title("Test cars: predicted vs actual")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    labels = ["Baseline", "v1"]
    colors = [BASELINE_COLOR, V1_COLOR]
    y_pos = np.arange(len(labels))[::-1]

    def style_bar_axis(axis) -> None:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_yticks(y_pos, labels)
        axis.tick_params(length=0, axis="y")

    rmse_values = [scores[name]["rmse"] for name in labels]
    rmse_bars = ax_rmse.barh(y_pos, rmse_values, color=colors, height=0.55)
    ax_rmse.set_xlabel("Typical error (RMSE)")
    ax_rmse.set_xlim(0, max(rmse_values) * 1.38)
    ax_rmse.set_title("On 72 cars the model never saw")
    ax_rmse.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    for bar, value in zip(rmse_bars, rmse_values):
        ax_rmse.text(
            value + 150,
            bar.get_y() + bar.get_height() / 2,
            f"${value:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    style_bar_axis(ax_rmse)

    r2_values = [scores[name]["r2"] for name in labels]
    r2_bars = ax_r2.barh(y_pos, r2_values, color=colors, height=0.55)
    ax_r2.set_xlabel("R²")
    ax_r2.set_xlim(-0.2, 1.15)
    ax_r2.axvline(0, color="#222222", linewidth=0.8)
    for bar, value in zip(r2_bars, r2_values):
        ax_r2.text(
            max(value, 0) + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    style_bar_axis(ax_r2)

    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_PATH.name}")
    for name in labels:
        print(f"{name}: R²={scores[name]['r2']:.3f}  RMSE=${scores[name]['rmse']:,.0f}")


if __name__ == "__main__":
    main()
