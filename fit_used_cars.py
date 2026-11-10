"""Fit sale-price regressions on the training cars only."""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

TRAIN_PATH = Path(__file__).with_name("used_car_train.csv")


def load_train() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH)
    train["Accident_Yes"] = (train["Accident"] == "yes").astype(int)
    return train


def main() -> None:
    train = load_train()
    print(f"Training cars: {len(train)}")
    print(train[["Sale_Price", "Mileage_K", "Age_Years", "Horsepower", "Num_Owners"]].describe().round(1))
    print("Accident counts:\n", train["Accident"].value_counts().to_string())
    print()

    full = smf.ols(
        "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident_Yes + Num_Owners",
        data=train,
    ).fit()
    print("=" * 72)
    print("FULL MODEL")
    print("=" * 72)
    print(full.summary())
    print()

    # Drop any feature whose p-value is above 0.05 and refit.
    candidates = ["Mileage_K", "Age_Years", "Horsepower", "Accident_Yes", "Num_Owners"]
    weak = [name for name in candidates if full.pvalues[name] > 0.05]
    print("Features with p > 0.05:", weak or "none")

    for name in weak:
        reduced = smf.ols(
            f"Sale_Price ~ {' + '.join(c for c in candidates if c != name)}",
            data=train,
        ).fit()
        print()
        print("=" * 72)
        print(f"WITHOUT {name}")
        print("=" * 72)
        print(reduced.summary())
        print(
            f"Adj R2 full={full.rsquared_adj:.4f}  "
            f"without {name}={reduced.rsquared_adj:.4f}  "
            f"change={reduced.rsquared_adj - full.rsquared_adj:+.4f}"
        )


if __name__ == "__main__":
    main()
