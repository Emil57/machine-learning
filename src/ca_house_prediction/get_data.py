from sklearn.datasets import fetch_california_housing

OUTPUT_FILE = 'data/california_housing.csv'
try:

    # Load dataset
    housing = fetch_california_housing(as_frame=True)
    data = housing.frame

    # Features and target
    X = data[['MedInc']]   # Median income (single feature for simplicity)
    y = data['MedHouseVal']  # Median house value

    print(data.head())

    # Save to CSV
    data.to_csv(OUTPUT_FILE, index=False)
except Exception as e:
    print(f"Error fetching California housing data: {e}")

finally:
    print(f'Finally block')
