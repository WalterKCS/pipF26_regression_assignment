"""Dealer buy/pass rules for the 40 auction lots.

v1 predicts sale price from mileage, age, horsepower, and accident history.
Getting a car onto the lot costs $1,500. The dealer buys when the predicted
sale price is above the auction price plus that cost. The baseline uses the
training-set average sale price in place of the prediction.
Profit on a buy is later sale price minus auction price minus $1,500.
A pass earns nothing.
"""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

TRAIN_PATH = Path(__file__).with_name("used_car_train.csv")
LOTS_PATH = Path(__file__).with_name("auction_lots.csv")
FORMULA = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident_Yes"
LOT_COST = 1500


def load_training() -> tuple[pd.DataFrame, float, object]:
    train = pd.read_csv(TRAIN_PATH)
    train["Accident_Yes"] = (train["Accident"] == "yes").astype(int)
    model = smf.ols(FORMULA, data=train).fit()
    return train, float(train["Sale_Price"].mean()), model


def dealer_rule(mileage_k: float, age_years: float, horsepower: float, accident: str, auction_price: float, model) -> str:
    """Return 'buy' or 'pass' for one car using v1."""
    accident_yes = 1 if str(accident).strip().lower() == "yes" else 0
    car = pd.DataFrame(
        {
            "Mileage_K": [mileage_k],
            "Age_Years": [age_years],
            "Horsepower": [horsepower],
            "Accident_Yes": [accident_yes],
        }
    )
    predicted = float(model.predict(car).iloc[0])
    return "buy" if predicted > auction_price + LOT_COST else "pass"


def baseline_rule(auction_price: float, training_average: float) -> str:
    """Return 'buy' or 'pass' using the training-set average sale price."""
    return "buy" if training_average > auction_price + LOT_COST else "pass"


def score(lots: pd.DataFrame, decisions: pd.Series) -> dict:
    bought = lots.loc[decisions.eq("buy")].copy()
    bought["profit"] = bought["Later_Sale_Price"] - bought["Auction_Price"] - LOT_COST
    return {
        "buys": int(len(bought)),
        "losses": int((bought["profit"] < 0).sum()),
        "total_profit": float(bought["profit"].sum()),
    }


def main() -> None:
    _, training_average, model = load_training()
    lots = pd.read_csv(LOTS_PATH)

    v1_decisions = lots.apply(
        lambda row: dealer_rule(
            row["Mileage_K"],
            row["Age_Years"],
            row["Horsepower"],
            row["Accident"],
            row["Auction_Price"],
            model,
        ),
        axis=1,
    )
    baseline_decisions = lots["Auction_Price"].map(lambda price: baseline_rule(price, training_average))

    print(f"Training-set average: ${training_average:,.0f}")
    print(f"Auction lots: {len(lots)}")
    print()
    print(f"{'Rule':<28} {'Buys':>6} {'Lose money':>12} {'Total profit':>14}")
    for name, decisions in (("Baseline", baseline_decisions), ("v1", v1_decisions)):
        result = score(lots, decisions)
        print(
            f"{name:<28} {result['buys']:>6} {result['losses']:>12} "
            f"${result['total_profit']:>12,.0f}"
        )


if __name__ == "__main__":
    main()
