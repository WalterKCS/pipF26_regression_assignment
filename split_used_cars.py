"""Split used_car_sales.csv into a 70% training set and a 30% test set.

All model fitting uses the training cars only. The test cars are held out
for evaluating fit.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).with_name("used_car_sales.csv")
TRAIN_PATH = Path(__file__).with_name("used_car_train.csv")
TEST_PATH = Path(__file__).with_name("used_car_test.csv")

# Fixed seed so later steps reuse this same split.
RANDOM_STATE = 42
TEST_SIZE = 0.30


def split_used_cars(random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    cars = pd.read_csv(DATA_PATH)
    train, test = train_test_split(
        cars,
        test_size=TEST_SIZE,
        random_state=random_state,
        shuffle=True,
    )
    train = train.sort_index()
    test = test.sort_index()
    return train, test


def main() -> None:
    cars = pd.read_csv(DATA_PATH)
    train, test = split_used_cars()
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    n = len(cars)
    print(f"All cars:      {n}")
    print(f"Training (70%): {len(train)}  ({len(train) / n:.1%})  -> {TRAIN_PATH.name}")
    print(f"Test (30%):     {len(test)}  ({len(test) / n:.1%})  -> {TEST_PATH.name}")
    print(f"Random state:   {RANDOM_STATE}")
    print("No row appears in both sets:", set(train.index).isdisjoint(set(test.index)))


if __name__ == "__main__":
    main()
