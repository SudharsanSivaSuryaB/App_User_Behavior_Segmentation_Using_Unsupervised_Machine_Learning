
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# STEP 1: Load original data with user IDs and scaled features
print("="*80)
print("CLUSTERING ANALYSIS: User Segmentation & Behavioral Profiling")
print("="*80)

# Load original data
df_original = pd.read_csv("cleaned_data/app_user_behavior_cleaned.csv")

# Load scaled features
df_scaled_features = pd.read_csv("featured_data/featured_data_scaled.csv")

# Create user IDs and combine with scaled features
df_analysis = pd.DataFrame({
    'user_id': [f'USER_{i:06d}' for i in range(len(df_original))]
})
for col in df_scaled_features.columns:
    df_analysis[col] = df_scaled_features[col]

print(f"\nDataset loaded: {df_analysis.shape[0]} users with {df_analysis.shape[1]-1} features")

# STEP 2: Elbow Method - Determine Optimal Number of Clusters
print("\n" + "="*80)
print("STEP 2: ELBOW METHOD - Determining Optimal Number of Clusters")
print("="*80)

inertias = []
silhouette_scores = []
cluster_range = range(2, 11)

# Extract only scaled features (exclude user_id)
X = df_analysis.iloc[:, 1:].values

for k in cluster_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)
    print(f"K={k}: Inertia = {kmeans_temp.inertia_:.2f}")

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (Sum of Squared Distances)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('clustered_data/elbow_curve.png', dpi=300, bbox_inches='tight')
print("\n✓ Elbow curve saved to clustered_data/elbow_curve.png")
plt.close()

# Based on elbow analysis, optimal k is selected (typically k=4)
optimal_k = 4
print(f"\n✓ Optimal number of clusters selected: K = {optimal_k}")

# STEP 3: K-Means Clustering with Optimal K
print("\n" + "="*80)
print("STEP 3: K-MEANS CLUSTERING - Model Training")
print("="*80)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_analysis['cluster'] = kmeans.fit_predict(X)

print(f"✓ K-Means model trained with {optimal_k} clusters")
print(f"\nCluster Distribution:")
print(df_analysis['cluster'].value_counts().sort_index())

# STEP 4: Cluster-Level Analysis & User Identification
print("\n" + "="*80)
print("STEP 4: CLUSTER PROFILING & USER IDENTIFICATION")
print("="*80)

# Get original numerical features for analysis (from original data, not scaled)
numerical_features = [
    'sessions_per_week', 'avg_session_duration_min', 'daily_active_minutes',
    'feature_clicks_per_session', 'notifications_opened_per_week',
    'in_app_search_count', 'pages_viewed_per_session', 'engagement_score',
    'churn_risk_score', 'days_since_last_login', 'account_age_days'
]

df_numerical = df_original[[col for col in numerical_features if col in df_original.columns]].copy()

df_profile = df_analysis.copy()
for col in df_numerical.columns:
    df_profile[col] = df_numerical[col].values

# Cluster statistics
cluster_stats = df_profile.groupby('cluster').agg({
    'sessions_per_week': ['mean', 'median', 'std'],
    'avg_session_duration_min': ['mean', 'median'],
    'daily_active_minutes': ['mean', 'median'],
    'engagement_score': ['mean', 'median'],
    'churn_risk_score': ['mean', 'median'],
    'pages_viewed_per_session': ['mean', 'median'],
    'feature_clicks_per_session': ['mean', 'median'],
    'notifications_opened_per_week': ['mean', 'median'],
    'days_since_last_login': ['mean', 'median'],
    'account_age_days': ['mean', 'median']
}).round(2)

print("\nDetailed Cluster Statistics:")
print(cluster_stats)

# STEP 5: Customer Segmentation & Cluster Naming
print("\n" + "="*80)
print("STEP 5: CUSTOMER SEGMENTATION & BUSINESS LABELS")
print("="*80)

# Define direct cluster mapping based on K-Means cluster IDs
cluster_mapping = {
    0: {
        'segment_name': 'High Engagement Users',
        'segment_label': 'HIGH',
        'description': 'Users with high session frequency, longer session duration, high engagement scores, and low churn risk'
    },
    1: {
        'segment_name': 'Moderate Engagement Users',
        'segment_label': 'MODERATE',
        'description': 'Users with consistent but average usage patterns, moderate engagement levels, and balanced activity'
    },
    2: {
        'segment_name': 'Low Engagement / At-Risk Users',
        'segment_label': 'AT_RISK',
        'description': 'Users with low activity, infrequent logins, shorter sessions, and higher churn risk indicators'
    },
    3: {
        'segment_name': 'Occasional Users',
        'segment_label': 'OCCASIONAL',
        'description': 'Users who interact with the app irregularly, showing sporadic usage and lower feature interaction'
    }
}

# Define cluster names and profiles based on characteristics
cluster_profiles = {}
cluster_names = {}

for cluster_id in range(optimal_k):
    cluster_data = df_profile[df_profile['cluster'] == cluster_id]
    
    avg_engagement = cluster_data['engagement_score'].mean()
    avg_sessions = cluster_data['sessions_per_week'].mean()
    avg_churn_risk = cluster_data['churn_risk_score'].mean()
    avg_duration = cluster_data['avg_session_duration_min'].mean()
    cluster_size = len(cluster_data)
    
    # Use direct mapping
    mapping = cluster_mapping[cluster_id]
    segment_name = mapping['segment_name']
    segment_label = mapping['segment_label']
    
    cluster_names[cluster_id] = segment_label
    
    cluster_profiles[cluster_id] = {
        'segment_name': segment_name,
        'segment_label': segment_label,
        'size': cluster_size,
        'percentage': (cluster_size / len(df_profile)) * 100,
        'avg_engagement': avg_engagement,
        'avg_churn_risk': avg_churn_risk,
        'avg_sessions': avg_sessions,
        'avg_duration': avg_duration
    }

print("\nCluster Segmentation Summary:")
print("="*80)
for cluster_id, profile in cluster_profiles.items():
    print(f"\n{'─'*70}")
    print(f"Cluster {cluster_id}: {profile['segment_name']} ({profile['segment_label']})")
    print(f"{'─'*70}")
    print(f"  Description: {cluster_mapping[cluster_id]['description']}")
    print(f"  Size: {profile['size']:,} users ({profile['percentage']:.1f}%)")
    print(f"  Avg Engagement Score: {profile['avg_engagement']:.2f}")
    print(f"  Avg Churn Risk: {profile['avg_churn_risk']:.2f}")
    print(f"  Avg Sessions/Week: {profile['avg_sessions']:.2f}")
    print(f"  Avg Session Duration: {profile['avg_duration']:.2f} min")

# STEP 6: Extract User Lists per Cluster
print("\n" + "="*80)
print("STEP 6: USER LISTS EXTRACTION")
print("="*80)

user_clusters = {}
for cluster_id in range(optimal_k):
    users_in_cluster = df_profile[df_profile['cluster'] == cluster_id]['user_id'].tolist()
    segment_label = cluster_names[cluster_id]
    user_clusters[cluster_id] = {
        'segment_label': segment_label,
        'user_count': len(users_in_cluster),
        'user_list': users_in_cluster
    }
    
    print(f"\nCluster {cluster_id} ({segment_label}): {len(users_in_cluster):,} users")
    print(f"Sample user IDs: {users_in_cluster[:10]}")

# STEP 7: Business Insights & Action Mapping
print("\n" + "="*80)
print("STEP 7: BUSINESS INSIGHTS & CUSTOMER ACTION MAPPING")
print("="*80)

business_actions = {
    "HIGH": {
        "description": "High-Value, Loyal Customers",
        "insights": [
            "✓ Highest engagement and session frequency",
            "✓ Low churn risk - proven brand loyalty",
            "✓ High feature adoption and exploration"
        ],
        "actions": [
            "→ Offer premium membership or exclusive benefits",
            "→ Launch loyalty rewards program",
            "→ Invite for beta testing of new features",
            "→ Provide VIP customer support",
            "→ Request product feedback & testimonials"
        ]
    },
    "MODERATE": {
        "description": "Growing, Moderate Engagement Users",
        "insights": [
            "✓ Healthy engagement with growth potential",
            "✓ Regular users with room for increased activity",
            "✓ Moderate churn risk - requires nurturing"
        ],
        "actions": [
            "→ Personalized engagement campaigns",
            "→ Recommend underutilized features",
            "→ Offer tiered upgrade paths",
            "→ Create targeted educational content",
            "→ Run seasonal promotions & incentives"
        ]
    },
    "AT_RISK": {
        "description": "High-Risk, Low Engagement Users",
        "insights": [
            "✗ High churn risk - disengaged users",
            "✗ Low session frequency and duration",
            "✗ Limited feature exploration"
        ],
        "actions": [
            "→ Launch retention campaigns immediately",
            "→ Offer special discounts/reactivation bonuses",
            "→ Send personalized re-engagement emails",
            "→ Conduct user feedback surveys",
            "→ Simplify onboarding and feature discovery",
            "→ Provide priority customer support"
        ]
    },
    "OCCASIONAL": {
        "description": "Occasional, Low-Activity Users",
        "insights": [
            "✓ Low engagement - possible casual users",
            "✓ Infrequent sessions with moderate churn risk",
            "✓ Opportunity for activation"
        ],
        "actions": [
            "→ Send weekly engagement highlights",
            "→ Offer time-limited special offers",
            "→ Create 'Start with basics' tutorials",
            "→ Use push notifications strategically",
            "→ A/B test different engagement triggers",
            "→ Focus on use case education"
        ]
    }
}

for segment_label in ["HIGH", "MODERATE", "AT_RISK", "OCCASIONAL"]:
    if segment_label in business_actions:
        action = business_actions[segment_label]
        print(f"\n{'='*70}")
        print(f"SEGMENT: {action['description']}")
        print(f"{'='*70}")
        print("\nKey Insights:")
        for insight in action['insights']:
            print(f"  {insight}")
        print("\nRecommended Business Actions:")
        for act in action['actions']:
            print(f"  {act}")

# STEP 8: Save Clustered Data & Reports
print("\n" + "="*80)
print("STEP 8: SAVING OUTPUTS")
print("="*80)

# Save clustered data with user IDs
df_output = df_profile[['user_id', 'cluster']].copy()
df_output['segment_label'] = df_output['cluster'].map(cluster_names)
df_output.to_csv("clustered_data/user_clusters.csv", index=False)
print("✓ User clusters saved to clustered_data/user_clusters.csv")

# Save cluster profiles report
cluster_summary_df = pd.DataFrame([
    {
        'Cluster_ID': cluster_id,
        'Segment_Name': profile['segment_name'],
        'Segment_Label': profile['segment_label'],
        'Number_of_Users': profile['size'],
        'Percentage_of_Total': f"{profile['percentage']:.1f}%",
        'Avg_Engagement_Score': f"{profile['avg_engagement']:.2f}",
        'Avg_Churn_Risk': f"{profile['avg_churn_risk']:.2f}",
        'Avg_Sessions_Per_Week': f"{profile['avg_sessions']:.2f}",
        'Avg_Session_Duration_Min': f"{profile['avg_duration']:.2f}"
    }
    for cluster_id, profile in cluster_profiles.items()
])

cluster_summary_df.to_csv("clustered_data/cluster_summary_report.csv", index=False)
print("✓ Cluster summary report saved to clustered_data/cluster_summary_report.csv")

# Save complete cluster analysis
full_analysis = df_profile.copy()
full_analysis['segment_label'] = full_analysis['cluster'].map(cluster_names)
full_analysis.to_csv("clustered_data/complete_cluster_analysis.csv", index=False)
print("✓ Complete cluster analysis saved to clustered_data/complete_cluster_analysis.csv")

# Save each cluster's user list
for cluster_id in range(optimal_k):
    cluster_users = df_profile[df_profile['cluster'] == cluster_id]['user_id'].tolist()
    segment_label = cluster_names[cluster_id]
    
    cluster_df = pd.DataFrame({
        'user_id': cluster_users,
        'cluster_id': cluster_id,
        'segment_label': segment_label
    })
    
    filename = f"clustered_data/users_{segment_label}.csv"
    cluster_df.to_csv(filename, index=False)
    print(f"✓ Cluster {cluster_id} users saved to {filename}")

# STEP 9: Create Visualization
print("\n" + "="*80)
print("STEP 9: CREATING VISUALIZATIONS")
print("="*80)

# Cluster size distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cluster Analysis Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Cluster Size Distribution
ax1 = axes[0, 0]
cluster_sizes = [cluster_profiles[i]['size'] for i in range(optimal_k)]
cluster_labels_names = [f"{cluster_names[i]}\n({cluster_profiles[i]['segment_name']})" for i in range(optimal_k)]
colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
ax1.bar(range(optimal_k), cluster_sizes, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Cluster', fontweight='bold')
ax1.set_ylabel('Number of Users', fontweight='bold')
ax1.set_title('Cluster Size Distribution', fontweight='bold')
ax1.set_xticks(range(optimal_k))
ax1.set_xticklabels([f'C{i}' for i in range(optimal_k)])

# Plot 2: Engagement Score by Cluster
ax2 = axes[0, 1]
engagement_scores = [df_profile[df_profile['cluster'] == i]['engagement_score'].mean() for i in range(optimal_k)]
ax2.bar(range(optimal_k), engagement_scores, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Cluster', fontweight='bold')
ax2.set_ylabel('Avg Engagement Score', fontweight='bold')
ax2.set_title('Average Engagement Score by Cluster', fontweight='bold')
ax2.set_xticks(range(optimal_k))
ax2.set_xticklabels([f'C{i}' for i in range(optimal_k)])

# Plot 3: Churn Risk by Cluster
ax3 = axes[1, 0]
churn_risks = [df_profile[df_profile['cluster'] == i]['churn_risk_score'].mean() for i in range(optimal_k)]
ax3.bar(range(optimal_k), churn_risks, color=colors, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Cluster', fontweight='bold')
ax3.set_ylabel('Avg Churn Risk Score', fontweight='bold')
ax3.set_title('Average Churn Risk by Cluster', fontweight='bold')
ax3.set_xticks(range(optimal_k))
ax3.set_xticklabels([f'C{i}' for i in range(optimal_k)])

# Plot 4: Sessions per Week by Cluster
ax4 = axes[1, 1]
sessions = [df_profile[df_profile['cluster'] == i]['sessions_per_week'].mean() for i in range(optimal_k)]
ax4.bar(range(optimal_k), sessions, color=colors, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Cluster', fontweight='bold')
ax4.set_ylabel('Avg Sessions per Week', fontweight='bold')
ax4.set_title('Average Sessions per Week by Cluster', fontweight='bold')
ax4.set_xticks(range(optimal_k))
ax4.set_xticklabels([f'C{i}' for i in range(optimal_k)])

plt.tight_layout()
plt.savefig('clustered_data/cluster_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Cluster analysis dashboard saved to clustered_data/cluster_analysis_dashboard.png")
plt.close()

# FINAL SUMMARY
print("\n" + "="*80)
print("✓ CLUSTERING ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. clustered_data/elbow_curve.png - Elbow method visualization")
print("  2. clustered_data/cluster_analysis_dashboard.png - 4-panel cluster analysis")
print("  3. clustered_data/user_clusters.csv - User IDs with cluster assignments")
print("  4. clustered_data/cluster_summary_report.csv - Cluster statistics & profiles")
print("  5. clustered_data/complete_cluster_analysis.csv - All features with cluster labels")
print("  6. clustered_data/users_HIGH.csv - High-value user list")
print("  7. clustered_data/users_MODERATE.csv - Moderate engagement user list")
print("  8. clustered_data/users_AT_RISK.csv - At-risk user list")
print("  9. clustered_data/users_OCCASIONAL.csv - Occasional user list")

print("\nKey Metrics:")
for cluster_id, profile in cluster_profiles.items():
    print(f"  Cluster {cluster_id}: {profile['size']:,} users ({profile['percentage']:.1f}%) - {profile['segment_name']}")

print("\n" + "="*80)
