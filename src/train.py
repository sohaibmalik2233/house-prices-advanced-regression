import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    print(" Starting ML Pipeline...")

    # --- 1. Load Data ---
    print("Loading data...")
    # Assumes train.csv and test.csv are in the main folder (one level up from src)
    df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    # --- 2. Data Cleaning ---
    print("Cleaning data...")
    columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu']
    df_cleaned = df.drop(columns=columns_to_drop)

    num_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df_cleaned.select_dtypes(include=['object']).columns

    for col in num_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    for col in cat_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

    # --- 3. Feature Engineering ---
    print("Encoding categorical variables...")
    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

    # --- 4. Train/Test Split ---
    X = df_encoded.drop('SalePrice', axis=1)
    y = df_encoded['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. Model Training ---
    print("Training Random Forest model (this might take a few seconds)...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate Validation Score
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f" Validation RMSE: ${rmse:,.2f}")

    # --- 6. Process Kaggle Test Data ---
    print("Processing Kaggle test data...")
    test_ids = test_df['Id']
    test_cleaned = test_df.drop(columns=columns_to_drop)

    for col in num_cols:
        if col in test_cleaned.columns:
            test_cleaned[col] = test_cleaned[col].fillna(df_cleaned[col].median())
    for col in cat_cols:
        if col in test_cleaned.columns:
            test_cleaned[col] = test_cleaned[col].fillna(df_cleaned[col].mode()[0])

    test_encoded = pd.get_dummies(test_cleaned, drop_first=True)
    test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)

    # --- 7. Final Output ---
    print("Generating Kaggle submission file...")
    final_predictions = rf_model.predict(test_encoded)
    submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_predictions})
    submission_df.to_csv('outputs/rf_predictions.csv', index=False)
    
    print(" Pipeline Complete! Saved to 'rf_predictions.csv'")

if __name__ == "__main__":
    main()