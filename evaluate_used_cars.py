"""Score the training-set average and v1 on the held-out test cars.

v1 is sale price on mileage, age, horsepower, and accident history,
fit on the training cars only. R² uses the test-set mean as the
reference. RMSE is the root mean squared dollar error on the test cars.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

TRAIN_PATH = Path(__file__).with_name("used_car_train.csv")
TEST_PATH = Path(__file__).with_name("used_car_test.csv")
FORMULA = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident_Yes"


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
    v1 = smf.ols(FORMULA, data=train).fit()

    y = test["Sale_Price"].to_numpy(dtype=float)
    train_mean = float(train["Sale_Price"].mean())
    baseline = np.full_like(y, train_mean)
    pred = v1.predict(test).to_numpy(dtype=float)

    print(f"Test cars: {len(test)}")
    print(f"Training-set average sale price: ${train_mean:,.0f}")
    print()
    print(f"{'Predictor':<28} {'R²':>8} {'RMSE':>12}")
    print(f"{'Baseline (training average)':<28} {r2_score(y, baseline):>8.3f} ${rmse(y, baseline):>10,.0f}")
    print(f"{'v1':<28} {r2_score(y, pred):>8.3f} ${rmse(y, pred):>10,.0f}")


if __name__ == "__main__":
    main()
