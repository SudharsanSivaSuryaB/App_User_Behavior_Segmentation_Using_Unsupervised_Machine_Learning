
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(
    page_title="User Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-style {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .subheader-style {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    user_clusters = pd.read_csv("clustered_data/user_clusters.csv")
    cluster_summary = pd.read_csv("clustered_data/cluster_summary_report.csv")
    complete_analysis = pd.read_csv("clustered_data/complete_cluster_analysis.csv")
    return user_clusters, cluster_summary, complete_analysis

user_clusters, cluster_summary, complete_analysis = load_data()

# Define cluster colors
cluster_colors = {
    'HIGH': '#2ecc71',
    'MODERATE': '#3498db',
    'AT_RISK': '#e74c3c',
    'OCCASIONAL': '#f39c12'
}

# ============================================================================
# HEADER & KEY METRICS
# ============================================================================
st.markdown('<div class="header-style">📊 User Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown("Strategic insights from K-Means clustering analysis of 50,000 app users")

# Key Metrics Row 1
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Users", f"{len(user_clusters):,}", "50,000 segmented")
with col2:
    st.metric("Active Clusters", "4", "behavioral groups")
with col3:
    high_engagement = len(user_clusters[user_clusters['segment_label'] == 'HIGH'])
    st.metric("High Engagement", f"{high_engagement:,}", f"{high_engagement/len(user_clusters)*100:.1f}%")
with col4:
    at_risk = len(user_clusters[user_clusters['segment_label'] == 'AT_RISK'])
    st.metric("At-Risk Users", f"{at_risk:,}", f"{at_risk/len(user_clusters)*100:.1f}%")

st.divider()

# ============================================================================
# SECTION 1: CLUSTER OVERVIEW
# ============================================================================
st.markdown('<div class="subheader-style">🎯 Cluster Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    # Cluster Distribution - Pie Chart
    fig_pie = px.pie(
        cluster_summary,
        values='Number_of_Users',
        names='Segment_Label',
        title='User Distribution Across Segments',
        color='Segment_Label',
        color_discrete_map=cluster_colors,
        hover_data=['Number_of_Users', 'Percentage_of_Total']
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Summary Table
    st.dataframe(
        cluster_summary[['Segment_Label', 'Number_of_Users', 'Percentage_of_Total']].rename(
            columns={
                'Segment_Label': 'Segment',
                'Number_of_Users': 'Users',
                'Percentage_of_Total': '% of Total'
            }
        ),
        hide_index=True,
        use_container_width=True
    )

# ============================================================================
# SECTION 2: BEHAVIORAL METRICS BY CLUSTER
# ============================================================================
st.markdown('<div class="subheader-style">📈 Behavioral Metrics by Cluster</div>', unsafe_allow_html=True)

# Top row - Full width Engagement Score
fig_engagement = px.bar(
    cluster_summary,
    x='Segment_Label',
    y='Avg_Engagement_Score',
    title='Average Engagement Score (Primary Metric)',
    color='Segment_Label',
    color_discrete_map=cluster_colors,
    labels={'Segment_Label': 'Segment', 'Avg_Engagement_Score': 'Score'},
    height=450
)
st.plotly_chart(fig_engagement, use_container_width=True)

# Bottom row - Two metrics side by side
col1, col2 = st.columns([1.1, 0.9])

with col1:
    fig_churn = px.bar(
        cluster_summary,
        x='Segment_Label',
        y='Avg_Churn_Risk',
        title='Average Churn Risk',
        color='Segment_Label',
        color_discrete_map=cluster_colors,
        labels={'Segment_Label': 'Segment', 'Avg_Churn_Risk': 'Risk Score'},
        height=350
    )
    st.plotly_chart(fig_churn, use_container_width=True)

with col2:
    fig_sessions = px.bar(
        cluster_summary,
        x='Segment_Label',
        y='Avg_Sessions_Per_Week',
        title='Average Sessions Per Week',
        color='Segment_Label',
        color_discrete_map=cluster_colors,
        labels={'Segment_Label': 'Segment', 'Avg_Sessions_Per_Week': 'Sessions'},
        height=350
    )
    st.plotly_chart(fig_sessions, use_container_width=True)

st.divider()

# ============================================================================
# SECTION 3: DETAILED CLUSTER PROFILES
# ============================================================================
st.markdown('<div class="subheader-style">👥 Detailed Cluster Profiles</div>', unsafe_allow_html=True)

# Segment descriptions
profiles = {
    'HIGH': {
        'description': 'High Engagement Users - Premium Value Segment',
        'characteristics': [
            '✓ High session frequency & longer session duration',
            '✓ High engagement scores & low churn risk',
            '✓ Most valuable users - Best retention prospects',
            '✓ Ideal for loyalty programs & premium features'
        ],
        'action': '🎁 Strategy: Loyalty programs, exclusive features, VIP benefits'
    },
    'MODERATE': {
        'description': 'Moderate Engagement Users - Stable Segment',
        'characteristics': [
            '✓ Consistent average usage patterns',
            '✓ Moderate engagement levels & balanced activity',
            '✓ Steady user base with growth potential',
            '✓ Good candidates for upselling'
        ],
        'action': '📈 Strategy: Engagement campaigns, feature recommendations, personalized content'
    },
    'AT_RISK': {
        'description': 'Low Engagement / At-Risk Users - Retention Priority',
        'characteristics': [
            '⚠ Low activity & infrequent logins',
            '⚠ Shorter sessions & higher churn risk',
            '⚠ Declining engagement indicators',
            '⚠ Require immediate retention efforts'
        ],
        'action': '🔔 Strategy: Win-back campaigns, personalized offers, re-engagement incentives'
    },
    'OCCASIONAL': {
        'description': 'Occasional Users - Activation Focus',
        'characteristics': [
            '◇ Irregular interaction & sporadic usage',
            '◇ Lower feature interaction rates',
            '◇ Untapped potential for activation',
            '◇ Need engagement triggers'
        ],
        'action': '🚀 Strategy: Habit-forming notifications, onboarding improvements, feature highlights'
    }
}

# Create tabs for each segment
tab1, tab2, tab3, tab4 = st.tabs(['HIGH', 'MODERATE', 'AT_RISK', 'OCCASIONAL'])

for tab, segment in zip([tab1, tab2, tab3, tab4], ['HIGH', 'MODERATE', 'AT_RISK', 'OCCASIONAL']):
    with tab:
        profile = profiles[segment]
        col1, col2 = st.columns([1, 1.2])
        
        # Segment Statistics
        with col1:
            segment_data = cluster_summary[cluster_summary['Segment_Label'] == segment].iloc[0]
            st.metric(
                f"Users in {segment}",
                f"{segment_data['Number_of_Users']:,}",
                f"{segment_data['Percentage_of_Total']} of total"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Engagement", f"{segment_data['Avg_Engagement_Score']:.2f}", "/ 100")
            with col_b:
                st.metric("Churn Risk", f"{segment_data['Avg_Churn_Risk']:.2f}", "/ 1.0")
            
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("Sessions/Week", f"{segment_data['Avg_Sessions_Per_Week']:.2f}")
            with col_d:
                st.metric("Duration", f"{segment_data['Avg_Session_Duration_Min']:.2f}m", "per session")
        
        # Profile Description & Recommendations
        with col2:
            st.markdown(f"**{profile['description']}**")
            for char in profile['characteristics']:
                st.write(char)
            st.divider()
            st.markdown(f"**{profile['action']}**")

st.divider()

# ============================================================================
# SECTION 4: DETAILED STATISTICS TABLE
# ============================================================================
st.markdown('<div class="subheader-style">📊 Comprehensive Cluster Statistics</div>', unsafe_allow_html=True)

# Format the cluster summary for display
display_summary = cluster_summary.copy()
display_summary['Avg_Engagement_Score'] = display_summary['Avg_Engagement_Score'].round(2)
display_summary['Avg_Churn_Risk'] = display_summary['Avg_Churn_Risk'].round(2)
display_summary['Avg_Sessions_Per_Week'] = display_summary['Avg_Sessions_Per_Week'].round(2)
display_summary['Avg_Session_Duration_Min'] = display_summary['Avg_Session_Duration_Min'].round(2)

st.dataframe(
    display_summary.rename(columns={
        'Segment_Label': 'Segment',
        'Number_of_Users': 'User Count',
        'Percentage_of_Total': '% Total',
        'Avg_Engagement_Score': 'Avg Engagement',
        'Avg_Churn_Risk': 'Avg Churn Risk',
        'Avg_Sessions_Per_Week': 'Sessions/Week',
        'Avg_Session_Duration_Min': 'Duration (min)'
    }).drop(columns=['Cluster_ID', 'Segment_Name']),
    hide_index=True,
    use_container_width=True
)

st.divider()

# ============================================================================
# SECTION 5: ACTIONABLE INSIGHTS & RECOMMENDATIONS
# ============================================================================
st.markdown('<div class="subheader-style">💡 Business Insights & Recommendations</div>', unsafe_allow_html=True)

insights_data = {
    'Key Insight': [
        '🏆 High-Value Segment',
        '⚠️ Retention Priority',
        '📈 Growth Opportunity',
        '🔍 Clear Segmentation'
    ],
    'Details': [
        f"45.1% of users show HIGH engagement - Focus on retention via VIP programs",
        f"15.1% AT_RISK users identified - Immediate win-back campaigns recommended",
        f"24.9% MODERATE users can be upsold - Personalized feature recommendations",
        f"PCA validation confirms clear behavioral separation - Reliable segmentation"
    ],
    'Priority': ['High', 'Critical', 'High', 'Medium']
}

insights_df = pd.DataFrame(insights_data)

for idx, row in insights_df.iterrows():
    with st.container():
        col1, col2, col3 = st.columns([0.5, 3, 1])
        with col1:
            st.write("◆")
        with col2:
            st.markdown(f"**{row['Key Insight']}**: {row['Details']}")
        with col3:
            priority_color = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡'}
            st.write(priority_color.get(row['Priority'], '⚪') + f" {row['Priority']}")

st.divider()

# ============================================================================
# SECTION 6: INTERACTIVE SEGMENT ANALYSIS
# ============================================================================
st.markdown('<div class="subheader-style">🔍 Interactive Segment Explorer</div>', unsafe_allow_html=True)

selected_segment = st.selectbox(
    "Select a segment to view detailed metrics:",
    ['HIGH', 'MODERATE', 'AT_RISK', 'OCCASIONAL'],
    format_func=lambda x: f"{x} Engagement Users"
)

# Get segment data
segment_users = complete_analysis[complete_analysis['segment_label'] == selected_segment]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        f"Users in '{selected_segment}'",
        f"{len(segment_users):,}",
        f"{len(segment_users)/len(complete_analysis)*100:.2f}% of total"
    )
with col2:
    st.metric(
        "Avg Engagement Score",
        f"{segment_users['engagement_score'].mean():.2f}",
        f"Min: {segment_users['engagement_score'].min():.2f}"
    )
with col3:
    st.metric(
        "Avg Churn Risk",
        f"{segment_users['churn_risk_score'].mean():.2f}",
        f"Max: {segment_users['churn_risk_score'].max():.2f}"
    )

# Behavioral metrics for selected segment
col1, col2 = st.columns(2)
with col1:
    st.metric("Avg Sessions/Week", f"{segment_users['sessions_per_week'].mean():.2f}")
    st.metric("Avg Session Duration", f"{segment_users['avg_session_duration_min'].mean():.2f}m")
    st.metric("Avg Daily Active Minutes", f"{segment_users['daily_active_minutes'].mean():.2f}m")
    
with col2:
    st.metric("Avg Feature Clicks/Session", f"{segment_users['feature_clicks_per_session'].mean():.2f}")
    st.metric("Avg Pages Viewed/Session", f"{segment_users['pages_viewed_per_session'].mean():.2f}")
    st.metric("Avg Days Since Last Login", f"{segment_users['days_since_last_login'].mean():.1f}d")

st.divider()

# ============================================================================
# SECTION 7: BUSINESS SUMMARY
# ============================================================================
st.markdown('<div class="subheader-style">🎯 Summary & Next Steps</div>', unsafe_allow_html=True)

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown("""
    **Key Achievements:**
    - ✓ Successfully segmented 50,000 users into 4 behavioral clusters
    - ✓ Achieved clear separation using PCA validation
    - ✓ Identified exact customer-level cluster assignments
    - ✓ Enabled data-driven personalization strategies
    """)

with summary_col2:
    st.markdown("""
    **Recommended Actions:**
    1. Deploy loyalty programs for HIGH engagement users
    2. Launch retention campaigns for AT_RISK segment
    3. Create upsell strategies for MODERATE users
    4. Implement reactivation flows for OCCASIONAL users
    5. Monitor churn metrics weekly across clusters
    """)

st.markdown("---")
st.markdown("*Dashboard powered by K-Means Clustering | Data refreshed from clustered_data outputs*")
