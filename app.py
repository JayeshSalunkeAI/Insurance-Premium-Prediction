# ============================================================================
# STREAMLIT APP - Insurance Charges Prediction (Dark Theme)
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr, spearmanr, f_oneway, pointbiserialr
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION & THEME
# ============================================================================

st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
dark_theme_css = """
<style>
    [data-testid="stApp"] {
        background-color: #1a1a2e;
        color: #eaeaea;
    }
    
    .stMetric {
        background-color: #0f3460;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #16c784;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #16c784;
        margin: 10px 0;
    }
    
    .nav-button {
        background-color: #0f3460;
        border: 2px solid #16c784;
        color: #eaeaea;
        padding: 15px 30px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
        cursor: pointer;
        margin: 5px;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        background-color: #16c784;
        color: #1a1a2e;
        transform: translateY(-2px);
    }
    
    .active-button {
        background-color: #16c784;
        color: #1a1a2e;
        border-color: #16c784;
    }
    
    .section-header {
        color: #16c784;
        border-bottom: 2px solid #16c784;
        padding-bottom: 10px;
        margin: 20px 0 15px 0;
    }
    
    .info-box {
        background-color: #0f3460;
        border-left: 4px solid #16c784;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .important-feature {
        background-color: #1a4d2e;
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .less-important-feature {
        background-color: #4a3220;
        border-left: 4px solid #ff9800;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
"""

st.markdown(dark_theme_css, unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

@st.cache_data
def load_data():
    return pd.read_csv('insurance_data.csv')

@st.cache_resource
def load_model():
    with open('insurance_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def prepare_eda_data():
    """Prepare EDA calculations and statistics"""
    df = load_data()
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    df['sex_encoded'] = le_sex.fit_transform(df['sex'])
    df['smoker_encoded'] = le_smoker.fit_transform(df['smoker'])
    df['region_encoded'] = le_region.fit_transform(df['region'])
    df['smoker_binary'] = (df['smoker'] == 'yes').astype(int)
    df['sex_binary'] = (df['sex'] == 'male').astype(int)
    
    return df, le_sex, le_smoker, le_region

# Load resources
df = load_data()
model_package = load_model()
eda_df, le_sex, le_smoker, le_region = prepare_eda_data()

# Extract model components
model = model_package['model']
model_name = model_package['model_name']
scaler = model_package['scaler']
feature_names = model_package['feature_names']
use_scaling = model_package['use_scaling']
performance = model_package['performance']

# ============================================================================
# NAVIGATION
# ============================================================================

st.sidebar.markdown("## 🏥 Navigation")
st.sidebar.markdown("---")

col1, col2, col3 = st.sidebar.columns(3)

with col1:
    home_btn = st.button("🏠 Home", use_container_width=True)
with col2:
    eda_btn = st.button("📊 E.D.A", use_container_width=True)
with col3:
    pred_btn = st.button("🔮 Predict", use_container_width=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if home_btn:
    st.session_state.page = 'home'
if eda_btn:
    st.session_state.page = 'eda'
if pred_btn:
    st.session_state.page = 'prediction'

# ============================================================================
# HOME PAGE
# ============================================================================

if st.session_state.page == 'home':
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #16c784; font-size: 3em;'>🏥 Insurance Premium Predictor</h1>
        <p style='font-size: 1.2em; color: #cccccc;'>Predict medical insurance charges using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Project Overview")
        st.markdown("""
        <div class='info-box'>
        <p><strong>Objective:</strong> Predict individual medical insurance charges based on personal 
        and health information using Machine Learning.</p>
        
        <p><strong>Model Used:</strong> Advanced regression algorithms trained on real insurance data.</p>
        
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>Age of the individual</li>
            <li>Biological sex</li>
            <li>Body Mass Index (BMI)</li>
            <li>Number of dependents</li>
            <li>Smoking status</li>
            <li>Geographic region</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Model Performance")
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("Accuracy (R²)", f"{performance['test_r2']:.4f}", 
                     delta="86.12%", delta_color="off")
        with perf_col2:
            st.metric("Average Error", f"${performance['mae']:,.0f}", 
                     delta="MAE", delta_color="off")
        with perf_col3:
            st.metric("RMSE", f"${performance['rmse']:,.0f}", 
                     delta="Error", delta_color="off")
    
    st.markdown("---")
    
    # Dataset Overview
    st.markdown("### 📊 Dataset Overview")
    
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    
    with overview_col1:
        st.markdown(f"""
        <div class='metric-card'>
        <h3>Total Records</h3>
        <p style='font-size: 2em; color: #16c784; font-weight: bold;'>{len(df):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col2:
        st.markdown(f"""
        <div class='metric-card'>
        <h3>Features</h3>
        <p style='font-size: 2em; color: #16c784; font-weight: bold;'>{df.shape[1]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col3:
        st.markdown(f"""
        <div class='metric-card'>
        <h3>Avg Charge</h3>
        <p style='font-size: 2em; color: #16c784; font-weight: bold;'>${df['charges'].mean():,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col4:
        st.markdown(f"""
        <div class='metric-card'>
        <h3>Max Charge</h3>
        <p style='font-size: 2em; color: #16c784; font-weight: bold;'>${df['charges'].max():,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Viewer
    st.markdown("### 👁️ View Dataset")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Search or filter data", "")
    with col2:
        rows_to_show = st.slider("Rows", 5, 50, 10)
    
    if search_term:
        filtered_df = df[df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
        st.dataframe(filtered_df.head(rows_to_show), use_container_width=True, height=400)
    else:
        st.dataframe(df.head(rows_to_show), use_container_width=True, height=400)
    
    # Quick Statistics
    st.markdown("### 📊 Quick Statistics")
    
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.markdown("**Age Statistics**")
        st.write(f"Min: {df['age'].min()} | Max: {df['age'].max()} | Mean: {df['age'].mean():.1f}")
    
    with stats_col2:
        st.markdown("**Charges Statistics**")
        st.write(f"Min: ${df['charges'].min():,.2f} | Max: ${df['charges'].max():,.2f} | Mean: ${df['charges'].mean():,.2f}")

# ============================================================================
# EDA PAGE
# ============================================================================

elif st.session_state.page == 'eda':
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #16c784;'>📊 Exploratory Data Analysis</h1>
        <p style='color: #cccccc;'>Statistical Analysis & Feature Importance</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Correlations", "📊 Feature Impact", "🎯 Feature Importance", "🔬 Statistical Tests", "📉 Distributions"]
    )
    
    # TAB 1: CORRELATION ANALYSIS
    with tab1:
        st.markdown("### Correlation Analysis")
        st.markdown("""
        <div class='info-box'>
        <p><strong>What is Correlation?</strong> Measures the linear relationship between features and insurance charges.
        Values range from -1 (negative) to +1 (positive). Closer to ±1 means stronger relationship.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate correlations
        numerical_features = ['age', 'bmi', 'children']
        correlation_results = []
        
        for feature in numerical_features:
            pearson_corr, pearson_p = pearsonr(eda_df[feature], eda_df['charges'])
            spearman_corr, spearman_p = spearmanr(eda_df[feature], eda_df['charges'])
            
            correlation_results.append({
                'Feature': feature,
                'Pearson_Corr': pearson_corr,
                'Pearson_P': pearson_p,
                'Spearman_Corr': spearman_corr,
                'Spearman_P': spearman_p
            })
        
        # Display results
        for result in correlation_results:
            feature = result['Feature'].upper()
            st.markdown(f"#### {feature}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='info-box'>
                <strong>Pearson Correlation:</strong> <span style='color: #16c784; font-size: 1.3em;'>{result['Pearson_Corr']:.4f}</span><br>
                <small>P-value: {result['Pearson_P']:.2e}</small><br>
                <small style='color: #ffb700;'>Measures linear relationship</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='info-box'>
                <strong>Spearman Correlation:</strong> <span style='color: #16c784; font-size: 1.3em;'>{result['Spearman_Corr']:.4f}</span><br>
                <small>P-value: {result['Spearman_P']:.2e}</small><br>
                <small style='color: #ffb700;'>Measures monotonic relationship</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Correlation Heatmap
        st.markdown("#### Correlation Heatmap")
        corr_data = eda_df[['age', 'sex_encoded', 'bmi', 'children', 
                            'smoker_encoded', 'region_encoded', 'charges']]
        corr_data.columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
        corr_matrix = corr_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            width=700,
            height=600,
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='#eaeaea'),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: CATEGORICAL FEATURE ANALYSIS
    with tab2:
        st.markdown("### Categorical Features Impact on Charges")
        st.markdown("""
        <div class='info-box'>
        <p><strong>ANOVA Test:</strong> Statistical test to determine if categorical features significantly 
        affect insurance charges. Lower p-value = stronger impact.</p>
        </div>
        """, unsafe_allow_html=True)
        
        categorical_features = ['sex', 'smoker', 'region']
        
        for feature in categorical_features:
            groups = [eda_df[eda_df[feature] == cat]['charges'].values 
                      for cat in eda_df[feature].unique()]
            f_stat, p_value = f_oneway(*groups)
            
            st.markdown(f"#### {feature.upper()}")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.markdown(f"""
                <div class='info-box'>
                <strong>F-statistic:</strong> <span style='color: #16c784;'>{f_stat:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='info-box'>
                <strong>P-value:</strong> <span style='color: #16c784;'>{p_value:.2e}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='info-box'>
                <strong>Significant:</strong> {'✅ YES' if p_value < 0.05 else '❌ NO'}
                </div>
                """, unsafe_allow_html=True)
            
            # Mean charges by category
            st.markdown(f"**Mean Charges by {feature.title()}:**")
            category_data = []
            for cat in sorted(eda_df[feature].unique()):
                mean_val = eda_df[eda_df[feature] == cat]['charges'].mean()
                count = len(eda_df[eda_df[feature] == cat])
                category_data.append({
                    'Category': cat,
                    'Mean Charge': f"${mean_val:,.2f}",
                    'Count': count
                })
            
            st.dataframe(pd.DataFrame(category_data), use_container_width=True)
            st.markdown("---")
    
    # TAB 3: FEATURE IMPORTANCE
    with tab3:
        st.markdown("### Random Forest Feature Importance")
        st.markdown("""
        <div class='info-box'>
        <p><strong>What is Feature Importance?</strong> Measures how much each feature contributes to 
        predicting insurance charges. Higher percentage = more important for predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate Feature Importance
        from sklearn.ensemble import RandomForestRegressor
        X = eda_df[['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']]
        y = eda_df['charges']
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        feature_names_list = ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region']
        importance_df = pd.DataFrame({
            'Feature': feature_names_list,
            'Importance': rf.feature_importances_,
            'Importance_Pct': rf.feature_importances_ * 100
        }).sort_values('Importance', ascending=False)
        
        # Display as table
        st.markdown("**Feature Importance Rankings:**")
        display_df = importance_df.copy()
        display_df['Importance_Pct'] = display_df['Importance_Pct'].apply(lambda x: f"{x:.2f}%")
        display_df['Importance'] = display_df['Importance'].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True)
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=importance_df['Importance_Pct'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['Importance_Pct'],
                    colorscale='Greens',
                    showscale=True,
                    colorbar=dict(title="Importance %")
                ),
                text=[f"{x:.2f}%" for x in importance_df['Importance_Pct']],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Feature Importance Distribution",
            xaxis_title="Importance (%)",
            yaxis_title="Feature",
            height=400,
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='#eaeaea'),
            xaxis=dict(zeroline=False),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown("#### 📌 Key Insights")
        top_feature = importance_df.iloc[0]
        st.markdown(f"""
        <div class='important-feature'>
        <strong>Most Important Feature:</strong> {top_feature['Feature']}<br>
        <span style='font-size: 1.3em; color: #4caf50;'>{top_feature['Importance_Pct']:.2f}%</span> importance<br>
        <small>This feature has the strongest influence on insurance charges predictions.</small>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 4: STATISTICAL TESTS
    with tab4:
        st.markdown("### Statistical Significance Testing")
        st.markdown("""
        <div class='info-box'>
        <p><strong>P-Value Interpretation:</strong> Values < 0.05 indicate statistically significant features.
        These features have a real, meaningful impact on insurance charges (not due to chance).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Collect all p-values
        p_values = {
            'Smoker': pearsonr(eda_df['smoker_binary'], eda_df['charges'])[1],
            'Age': correlation_results[0]['Pearson_P'],
            'BMI': correlation_results[1]['Pearson_P'],
            'Children': correlation_results[2]['Pearson_P']
        }
        
        p_value_df = pd.DataFrame(list(p_values.items()), 
                                  columns=['Feature', 'P_Value']).sort_values('P_Value')
        p_value_df['Significant'] = p_value_df['P_Value'].apply(lambda x: '✅ Yes' if x < 0.05 else '❌ No')
        p_value_df['P_Value_Display'] = p_value_df['P_Value'].apply(lambda x: f"{x:.2e}")
        
        display_cols = ['Feature', 'P_Value_Display', 'Significant']
        st.dataframe(p_value_df[display_cols].rename(columns={'P_Value_Display': 'P-Value'}), 
                    use_container_width=True)
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(
                y=p_value_df['Feature'],
                x=-np.log10(p_value_df['P_Value']),
                orientation='h',
                marker=dict(color=['#4caf50' if p < 0.05 else '#ff9800' for p in p_value_df['P_Value']])
            )
        ])
        
        fig.add_vline(x=1.3, line_dash="dash", line_color="red", 
                     annotation_text="Significance Threshold (α=0.05)")
        
        fig.update_layout(
            title="-log10(P-Value) - Statistical Significance",
            xaxis_title="-log10(P-Value)",
            yaxis_title="Feature",
            height=400,
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='#eaeaea'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: DISTRIBUTIONS
    with tab5:
        st.markdown("### Feature Distributions vs Charges")
        
        # Distribution visualizations
        fig = go.Figure()
        
        # Age distribution
        fig.add_trace(go.Histogram(
            x=eda_df['age'],
            name='Age Distribution',
            marker=dict(color='#16c784', opacity=0.7),
            nbinsx=30
        ))
        
        fig.update_layout(
            title="Age Distribution",
            xaxis_title="Age",
            yaxis_title="Frequency",
            height=400,
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='#eaeaea'),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Charges distribution
        fig2 = go.Figure()
        
        fig2.add_trace(go.Histogram(
            x=eda_df['charges'],
            name='Charges Distribution',
            marker=dict(color='#16c784', opacity=0.7),
            nbinsx=30
        ))
        
        fig2.update_layout(
            title="Insurance Charges Distribution",
            xaxis_title="Charges ($)",
            yaxis_title="Frequency",
            height=400,
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='#eaeaea'),
            showlegend=True
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Smoker impact
        fig3 = go.Figure()
        
        for smoker_status in ['yes', 'no']:
            data = eda_df[eda_df['smoker'] == smoker_status]['charges']
            fig3.add_trace(go.Histogram(
                x=data,
                name=f'Smoker: {smoker_status.upper()}',
                opacity=0.7,
                nbinsx=30
            ))
        
        fig3.update_layout(
            title="Charges Distribution by Smoking Status",
            xaxis_title="Charges ($)",
            yaxis_title="Frequency",
            height=400,
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='#eaeaea'),
            barmode='overlay'
        )
        
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# PREDICTION PAGE
# ============================================================================

elif st.session_state.page == 'prediction':
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #16c784;'>🔮 Insurance Charge Predictor</h1>
        <p style='color: #cccccc;'>Enter your details to get an insurance charge estimate</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main prediction form
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📋 Your Information")
        
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            help="Your age in years"
        )
        
        sex = st.selectbox(
            "Sex",
            options=['male', 'female'],
            help="Biological sex"
        )
        
        children = st.selectbox(
            "Number of Children/Dependents",
            options=[0, 1, 2, 3, 4, 5],
            help="Number of dependents"
        )
        
        smoker = st.selectbox(
            "Smoking Status",
            options=['no', 'yes'],
            help="Do you smoke?"
        )
        
        region = st.selectbox(
            "Region",
            options=['northeast', 'northwest', 'southeast', 'southwest'],
            help="Geographic region in USA"
        )
        
        # BMI Section
        # BMI Section (allow direct BMI input or calculate from height & weight)
        st.markdown("### 📏 BMI")
        st.markdown("""
        <div class='info-box'>
        <p><strong>BMI (Body Mass Index)</strong> = Weight (kg) ÷ Height² (m²)</p>
        </div>
        """, unsafe_allow_html=True)

        bmi_input_method = st.radio("BMI Input Method", ["Calculate from height & weight", "Enter BMI directly"])

        if bmi_input_method == "Calculate from height & weight":
            bmi_col1, bmi_col2 = st.columns(2)
            with bmi_col1:
                height_feet = st.number_input(
                    "Height (feet)",
                    min_value=3.0,
                    max_value=9.0,
                    value=5.5,
                    step=0.1,
                    help="Your height in feet (e.g., 5.5 for 5'6\")"
                )
            with bmi_col2:
                weight_kg = st.number_input(
                    "Weight (kg)",
                    min_value=30.0,
                    max_value=250.0,
                    value=70.0,
                    step=0.5,
                    help="Your weight in kilograms"
                )
            height_meters = height_feet * 0.3048
            bmi = weight_kg / (height_meters ** 2) if height_meters > 0 else 0.0
        else:
            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=80.0,
                value=24.0,
                step=0.1,
                help="Enter your BMI directly if you already know it"
            )

        # BMI Category display
        if bmi < 18.5:
            bmi_category_label = "Underweight"
            bmi_color = "#2196F3"
        elif bmi < 25:
            bmi_category_label = "Normal Weight"
            bmi_color = "#4caf50"
        elif bmi < 30:
            bmi_category_label = "Overweight"
            bmi_color = "#ff9800"
        else:
            bmi_category_label = "Obese"
            bmi_color = "#f44336"

        st.markdown(f"""
        <div style='background-color: #0f3460; border: 2px solid {bmi_color}; 
                    padding: 20px; border-radius: 10px; text-align: center;'>
        <p style='margin: 0; color: #cccccc;'><strong>Your BMI:</strong></p>
        <p style='font-size: 2.5em; color: {bmi_color}; margin: 10px 0; font-weight: bold;'>{bmi:.2f}</p>
        <p style='margin: 0; color: {bmi_color}; font-weight: bold;'>{bmi_category_label}</p>
        </div>
        """, unsafe_allow_html=True)

        # Prediction column (unchanged visual layout)
        with col2:
            st.markdown("### 📊 Prediction Results")

            if st.button("🔮 Calculate Insurance Cost", type="primary", use_container_width=True, key="predict_button"):
                # Build input_data and engineered features exactly as model expects
                input_data = pd.DataFrame({
                    'age': [age],
                    'sex': [sex],
                    'bmi': [bmi],
                    'children': [children],
                    'smoker': [smoker],
                    'region': [region]
                })

                # Choose label encoders from model package if provided, else fall back to local ones
                label_encs = model_package.get('label_encoders', {}) if isinstance(model_package, dict) else {}
                sex_enc = label_encs.get('sex', le_sex)
                smoker_enc = label_encs.get('smoker', le_smoker)
                region_enc = label_encs.get('region', le_region)

                input_data['sex_encoded'] = sex_enc.transform(input_data['sex'])
                input_data['smoker_encoded'] = smoker_enc.transform(input_data['smoker'])
                input_data['region_encoded'] = region_enc.transform(input_data['region'])
                input_data['smoker_binary'] = (input_data['smoker'] == 'yes').astype(int)

                # Numeric BMI category used by the model (0=Underweight,1=Normal,2=Overweight,3=Obese)
                def categorize_bmi_num(b):
                    if b < 18.5:
                        return 0
                    elif b < 25:
                        return 1
                    elif b < 30:
                        return 2
                    else:
                        return 3

                input_data['bmi_category'] = input_data['bmi'].apply(categorize_bmi_num)

                # Age groups consistent with training
                if input_data.loc[0, 'age'] <= 30:
                    input_data['age_group'] = 0
                elif input_data.loc[0, 'age'] <= 50:
                    input_data['age_group'] = 1
                else:
                    input_data['age_group'] = 2

                input_data['smoker_bmi'] = input_data['smoker_binary'] * input_data['bmi']
                input_data['age_bmi'] = input_data['age'] * input_data['bmi']
                input_data['is_obese_smoker'] = int((input_data.loc[0, 'bmi'] > 30) and (input_data.loc[0, 'smoker'] == 'yes'))
                # Risk score calculation (same formula used elsewhere)
                smoker_bin = int(input_data.loc[0, 'smoker_binary'])
                input_data['risk_score'] = input_data.loc[0, 'age'] / 64 * 0.3 + input_data.loc[0, 'bmi'] / 54 * 0.3 + smoker_bin * 0.4

                # Ensure the input columns are in the same order/names as model expects
                X_input = input_data[feature_names]

                # Scale using the saved scaler -- pass a DataFrame with matching column names
                if use_scaling:
                    X_input = scaler.transform(X_input)

                # Predict
                prediction = model.predict(X_input)[0]

                # Display prediction and profile summary (same UI as before)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #0f3460 0%, #16213e 100%); 
                            border: 3px solid #16c784; padding: 30px; border-radius: 15px; 
                            text-align: center; margin-top: 20px;'>
                <p style='color: #cccccc; font-size: 1.1em; margin: 0;'><strong>Estimated Annual Insurance Cost</strong></p>
                <p style='font-size: 3em; color: #16c784; margin: 15px 0; font-weight: bold;'>${prediction:,.2f}</p>
                <p style='color: #ffb700; margin: 0;'>Based on your personal information</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📈 Your Profile Summary")
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.markdown(f"""
                    <div class='info-box'>
                    <p><strong>Age:</strong> {age} years</p>
                    <p><strong>Sex:</strong> {sex.capitalize()}</p>
                    <p><strong>BMI:</strong> {bmi:.2f} ({bmi_category_label})</p>
                    <p><strong>Children:</strong> {children}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with summary_col2:
                    st.markdown(f"""
                    <div class='info-box'>
                    <p><strong>Smoker:</strong> {'Yes ❌' if smoker == 'yes' else 'No ✅'}</p>
                    <p><strong>Region:</strong> {region.capitalize()}</p>
                    <p><strong>Model Confidence:</strong> {performance['test_r2']*100:.1f}%</p>
                    <p><strong>Expected Error:</strong> ±${performance['mae']:,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Risk factors (same logic as before)
                risk_factors = []
                if age > 50:
                    risk_factors.append("🔴 High Age - Increases costs significantly")
                if smoker == 'yes':
                    risk_factors.append("🔴 Smoking - Major cost driver")
                if bmi >= 30:
                    risk_factors.append("🟡 High BMI - Moderate cost increase")
                if bmi < 18.5:
                    risk_factors.append("🟢 Low BMI - Reduces costs")

                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"<div class='info-box'>{factor}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='important-feature'>
                    ✅ Your profile shows low risk factors - you qualify for better rates!
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='info-box' style='text-align: center; padding: 40px;'>
                <p style='font-size: 1.2em; color: #cccccc;'>👆 Click the button to calculate your insurance cost</p>
                </div>
                """, unsafe_allow_html=True)
# Sidebar with additional information
st.sidebar.header("ℹ️ About")
st.sidebar.info(f"""
**Model Information:**
- Algorithm: {model_name}
- Accuracy: {performance['test_r2']:.2%}
- Average Error: ${performance['mae']:,.0f}

**Features Used:**
- Age, BMI, Smoking Status
- Number of Children
- Engineered risk factors
""")

st.sidebar.header("📚 BMI Reference")
st.sidebar.markdown("""
- **Underweight:** < 18.5
- **Normal:** 18.5 - 24.9
- **Overweight:** 25 - 29.9
- **Obese:** ≥ 30
""")

st.sidebar.header("🎯 Tips to Reduce Charges")
st.sidebar.markdown("""
1. **Quit Smoking** - Biggest factor
2. **Maintain Healthy BMI** (18.5-24.9)
3. **Regular Exercise**
4. **Balanced Diet**
5. **Regular Health Check-ups**
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    Made with ❤️ using Streamlit | Insurance Charges Predictor
</div>
""", unsafe_allow_html=True)
