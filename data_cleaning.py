
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Step 1-2: Data Collection and Loading."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {path}")
    return df


def preprocessing_report(df: pd.DataFrame) -> None:
    """Step 3: Data Preprocessing summary."""
    print("\n=== PREPROCESSING REPORT ===")
    print("Data types:")
    print(df.dtypes)
    print("\nNon-null counts:")
    df.info(verbose=True, show_counts=True)
    print("\nDescriptive statistics (numeric):")
    print(df.describe(include=["number"]))
    print("\nMissing values per column:")
    print(df.isna().sum())


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Remove duplicates."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Missing values handling."""
    # Numeric columns: median
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Categorical columns: mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)

    total_na = df.isna().sum().sum()
    print(f"Missing values after imputation: {total_na}")
    return df


def detect_and_correct_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Outlier detection and correction (IQR capping)."""
    numeric_cols = df.select_dtypes(include="number").columns
    capped_counts = {}

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        outliers_low = (df[col] < lower).sum()
        outliers_high = (df[col] > upper).sum()
        capped_counts[col] = outliers_low + outliers_high

        if outliers_low or outliers_high:
            df[col] = df[col].clip(lower=lower, upper=upper)

    print("Outlier capping counts (per numeric column):")
    for col, count in capped_counts.items():
        if count > 0:
            print(f"  {col}: {count}")

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Encoding categorical variables."""
    cat_cols = ["gender", "country", "device_type", "subscription_type", "marketing_source"]
    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)
    print(f"Encoded categorical cols: {existing_cat_cols}")
    return df


def remove_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Remove columns with all unique values (e.g., IDs)."""
    unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    print(f"Unique-value columns to drop: {unique_cols}")
    df = df.drop(columns=unique_cols, errors='ignore')
    return df


def run_data_cleaning(
    input_csv: str = "data/app_user_behavior_dataset.csv",
    output_csv: str = "cleaned_data/app_user_behavior_cleaned.csv",
) -> pd.DataFrame:
    df = load_data(input_csv)
    preprocessing_report(df)
    df = remove_duplicates(df)
    df = impute_missing_values(df)
    df = detect_and_correct_outliers(df)
    df = encode_categorical(df)
    df = remove_unique_columns(df)
    df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv} ({len(df)} rows, {len(df.columns)} columns)")
    return df


if __name__ == "__main__":
    run_data_cleaning()
