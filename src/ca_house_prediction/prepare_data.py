import os
import pandas as pd
from sklearn.model_selection import train_test_split

output_dir = os.path.join(os.path.abspath("../../"), "data", "ca_house_prediction")    
try:

    data = pd.read_csv(f"{output_dir}/housing.csv")
    print(data.head())
    # Features and target
    X = data[['median_income']]   # Median income (single feature for simplicity)
    y = data['median_house_value']  # Median house value

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'Split data: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.')

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    print(f"Saved train/test splits to {output_dir}")
except Exception as e:
    print(f'Error: {e}')
finally:
    pass
