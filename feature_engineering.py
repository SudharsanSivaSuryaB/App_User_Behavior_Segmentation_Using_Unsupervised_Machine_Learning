
import pandas as pd

df = pd.read_csv("cleaned_data/app_user_behavior_cleaned.csv")

# optional id removal
if 'user_id' in df.columns:
    df.drop(columns=['user_id'], inplace=True)

# Selected numerical and behavior features influencing user behavior
selected_features = [
    'sessions_per_week',
    'avg_session_duration_min',
    'daily_active_minutes',
    'feature_clicks_per_session',
    'notifications_opened_per_week',
    'in_app_search_count',
    'pages_viewed_per_session',
    'engagement_score',
    'churn_risk_score',
    'days_since_last_login',
    'account_age_days'
]

# If categorical dummies exist and may still be useful, keep them as well
# (can be adjusted based on model requirements)
categorical_dummies = [c for c in df.columns if c.startswith('gender_') or c.startswith('country_') or c.startswith('device_type_') or c.startswith('subscription_type_') or c.startswith('marketing_source_')]

keep_columns = [c for c in selected_features + categorical_dummies if c in df.columns]

df_selected = df[keep_columns].copy()

df_selected.to_csv("featured_data/featured_data.csv", index=False)
print("Feature Engineering Done: selected features saved to featured_data/featured_data.csv")
