import os
import pandas as pd
import numpy as np

def generate_classification_data(filepath, size=300):
    np.random.seed(42)
    customer_ids = [f"CUST-{i:04d}" for i in range(1, size + 1)]
    age = np.random.normal(40, 12, size).astype(int)
    age = np.clip(age, 18, 85)
    
    gender = np.random.choice(["Male", "Female", None], size, p=[0.48, 0.48, 0.04]) # 4% missing gender
    tenure = np.random.randint(1, 72, size)
    monthly_charges = np.random.normal(65, 25, size)
    monthly_charges = np.clip(monthly_charges, 15, 120)
    
    # Introduce target leakage test: TotalCharges = tenure * monthly_charges
    # We will compute it with some minor noise and add missing values (e.g. 5%)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, size)
    total_charges = np.clip(total_charges, 15, 72 * 120)
    missing_idx = np.random.choice(size, int(size * 0.05), replace=False)
    total_charges[missing_idx] = np.nan
    
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size, p=[0.5, 0.3, 0.2])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size, p=[0.4, 0.4, 0.2])
    
    # Calculate churn probability logic (correlation)
    # probability increases if Month-to-month, Fiber optic, age is high, tenure is low, and charges are high
    prob = 0.1
    prob += np.where(contract == "Month-to-month", 0.3, 0.0)
    prob += np.where(internet_service == "Fiber optic", 0.2, 0.0)
    prob += np.where(tenure < 12, 0.3, 0.0)
    prob += (monthly_charges - 15) / 105 * 0.2
    prob = np.clip(prob, 0.05, 0.95)
    
    churn = np.random.binomial(1, prob)
    
    df = pd.DataFrame({
        "CustomerId": customer_ids,
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "Churn": churn
    })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Generated classification data: {filepath}")

def generate_regression_data(filepath, size=300):
    np.random.seed(42)
    house_ids = [f"HOUSE-{i:04d}" for i in range(1, size + 1)]
    sq_ft = np.random.normal(2000, 500, size).astype(int)
    sq_ft = np.clip(sq_ft, 800, 5000)
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], size, p=[0.2, 0.2, 0.3, 0.2, 0.1])
    
    neighborhood = np.random.choice(["Urban", "Suburban", "Rural", None], size, p=[0.3, 0.5, 0.15, 0.05]) # 5% missing
    year_built = np.random.randint(1950, 2024, size)
    
    # Introduce missing values in bedrooms (e.g. 3%)
    bedrooms = bedrooms.astype(float)
    missing_bed_idx = np.random.choice(size, int(size * 0.03), replace=False)
    bedrooms[missing_bed_idx] = np.nan
    
    # Price function
    price = (
        120 * sq_ft +
        20000 * np.nan_to_num(bedrooms, nan=3) +
        15000 * bathrooms +
        np.select(
            [neighborhood == "Urban", neighborhood == "Suburban", neighborhood == "Rural"],
            [40000, 20000, -10000],
            default=15000
        ) +
        (year_built - 1950) * 800 +
        np.random.normal(0, 15000, size)
    )
    price = np.clip(price, 50000, 1000000)
    
    df = pd.DataFrame({
        "HouseId": house_ids,
        "SqFt": sq_ft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Neighborhood": neighborhood,
        "YearBuilt": year_built,
        "Price": price
    })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Generated regression data: {filepath}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, "datasets", "sample")
    generate_classification_data(os.path.join(datasets_dir, "churn_sample.csv"))
    generate_regression_data(os.path.join(datasets_dir, "housing_sample.csv"))
