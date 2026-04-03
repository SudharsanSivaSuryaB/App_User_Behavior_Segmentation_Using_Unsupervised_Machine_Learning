import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data/app_user_behavior_cleaned.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print("\nMissing values:\n", df.isna().sum())

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
print("\nNumeric cols:", num_cols)

def plot_in_groups(col_list, plot_fn, title_prefix, ncols=3):
    for i in range(0, len(col_list), ncols):
        group = col_list[i : i + ncols]
        plt.figure(figsize=(5 * len(group), 4))
        for j, col in enumerate(group, 1):
            plt.subplot(1, len(group), j)
            plot_fn(col)
            plt.title(col, fontsize=11)
            plt.tight_layout()
        plt.suptitle(f"{title_prefix} (columns {i+1}-{i+len(group)})", y=1.02, fontsize=14)
        plt.show()

# Histograms in small groups
plot_in_groups(
    num_cols,
    lambda col: sns.histplot(df[col].dropna(), kde=False, bins=15, color="steelblue"),
    "Histogram"
)

# Boxplots in small groups
plot_in_groups(
    num_cols,
    lambda col: sns.boxplot(x=df[col].dropna(), color="lightgreen"),
    "Boxplot"
)

# Correlation heatmap (all numeric in one figure)
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation matrix (numeric features)")
plt.tight_layout()
plt.show()

# Categorical counts (if any)
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        order = df[col].value_counts().index[:10]
        sns.countplot(data=df, x=col, order=order, palette="muted")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Top 10 categories for {col}")
        plt.tight_layout()
        plt.show()
