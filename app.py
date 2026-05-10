"""
NeuralForge AI - Advanced ML Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    KFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet, SGDClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, mean_squared_error,
    mean_absolute_error, r2_score, mean_absolute_percentage_error,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import joblib
import io
import time
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NeuralForge AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a2e;
        --accent-1: #6c63ff;
        --accent-2: #ff6584;
        --accent-3: #00d4aa;
        --accent-4: #ffd166;
        --text-primary: #e8e8f0;
        --text-secondary: #9898b0;
        --border: #2a2a45;
    }
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--bg-primary); color: var(--text-primary); }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #161628 100%);
        border-right: 1px solid var(--border);
    }
    h1, h2, h3, h4 { color: var(--text-primary) !important; font-weight: 700 !important; }
    .stMetric {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 20px rgba(108,99,255,0.08);
    }
    .stMetric [data-testid="stMetricValue"] {
        color: var(--accent-1) !important; font-size: 2rem !important; font-weight: 800 !important;
    }
    .glow-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, #1e1e35 100%);
        border: 1px solid var(--border); border-radius: 16px;
        padding: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 16px;
    }
    .gradient-text {
        background: linear-gradient(135deg, #6c63ff, #ff6584, #00d4aa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-weight: 800;
    }
    .hero-section { text-align: center; padding: 30px 0 20px; }
    .hero-title {
        font-size: 3rem; font-weight: 900;
        background: linear-gradient(135deg, #6c63ff 0%, #ff6584 50%, #00d4aa 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 8px;
    }
    .hero-subtitle { font-size: 1.1rem; color: var(--text-secondary); }
    .step-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 36px; height: 36px; border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
        color: white; font-weight: 700; font-size: 0.9rem; margin-right: 12px;
    }
    .feature-chip {
        display: inline-block; padding: 6px 14px; border-radius: 20px;
        background: rgba(108,99,255,0.15); color: var(--accent-1);
        font-size: 0.8rem; font-weight: 600; margin: 3px;
        border: 1px solid rgba(108,99,255,0.3);
    }
    .info-banner {
        background: linear-gradient(90deg, rgba(108,99,255,0.1), rgba(0,212,170,0.1));
        border-left: 4px solid var(--accent-1); padding: 12px 16px;
        border-radius: 0 8px 8px 0; margin: 12px 0;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important; border-radius: 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: var(--bg-secondary); border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 10px 20px; color: var(--text-secondary); font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important; color: var(--accent-1) !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        'df': None, 'target_col': None, 'task_type': None,
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
        'trained_models': {}, 'best_model_name': None, 'best_model': None,
        'label_encoders': {}, 'scaler': None, 'feature_names': None,
        'data_cleaned': False, 'models_trained': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ══════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════

def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.5)',
        font=dict(color='#e8e8f0', family='Inter'),
        xaxis=dict(gridcolor='rgba(42,42,69,0.6)', zerolinecolor='rgba(42,42,69,0.6)',
                   title_font=dict(size=13, color='#9898b0'),
                   tickfont=dict(size=11, color='#9898b0'), linecolor='rgba(42,42,69,0.6)'),
        yaxis=dict(gridcolor='rgba(42,42,69,0.6)', zerolinecolor='rgba(42,42,69,0.6)',
                   title_font=dict(size=13, color='#9898b0'),
                   tickfont=dict(size=11, color='#9898b0'), linecolor='rgba(42,42,69,0.6)'),
        legend=dict(bgcolor='rgba(26,26,46,0.9)', bordercolor='rgba(42,42,69,0.8)',
                    borderwidth=1, font=dict(size=11, color='#9898b0')),
    )
    return fig


def get_model_registry(task_type):
    if task_type == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42, algorithm='SAMME'),
            "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
            "Support Vector Machine": SVC(random_state=42, probability=True),
            "Naive Bayes": GaussianNB(),
            "MLP Neural Network": MLPClassifier(random_state=42, max_iter=500),
            "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False,
                                          eval_metric='logloss', verbosity=0),
            "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
            "SGD Classifier": SGDClassifier(random_state=42, loss='modified_huber'),
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "ElasticNet": ElasticNet(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
            "Support Vector Machine": SVR(),
            "MLP Neural Network": MLPRegressor(random_state=42, max_iter=500),
            "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=0),
            "LightGBM": lgb.LGBMRegressor(random_state=42, verbose=-1),
        }


def get_param_grid(model_name, task_type):
    grids = {
        "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None],
                          'min_samples_split': [2, 5, 10]},
        "XGBoost": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]},
        "LightGBM": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, -1],
                     'learning_rate': [0.01, 0.1, 0.3]},
        "Gradient Boosting": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
                              'learning_rate': [0.01, 0.1, 0.3]},
        "Logistic Regression": {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'],
                                'solver': ['liblinear', 'saga']},
        "SVM": {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        "KNN": {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']},
        "MLP Neural Network": {'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                               'alpha': [0.0001, 0.001, 0.01]},
        "Decision Tree": {'max_depth': [3, 5, 10, 20, None], 'min_samples_split': [2, 5, 10]},
        "Ridge": {'alpha': [0.01, 0.1, 1, 10, 100]},
        "Lasso": {'alpha': [0.001, 0.01, 0.1, 1, 10]},
        "ElasticNet": {'alpha': [0.01, 0.1, 1], 'l1_ratio': [0.2, 0.5, 0.8]},
        "Linear Regression": {},
        "Naive Bayes": {'var_smoothing': [1e-9, 1e-8, 1e-7]},
        "Extra Trees": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None]},
        "AdaBoost": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
        "SGD Classifier": {'alpha': [0.0001, 0.001, 0.01]},
    }
    return grids.get(model_name, {})


def calculate_metrics(y_test, y_pred, task_type, model=None, X_test=None):
    metrics = {}
    if task_type == "Classification":
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
        except:
            metrics['mcc'] = 0
        try:
            metrics['kappa'] = cohen_kappa_score(y_test, y_pred)
        except:
            metrics['kappa'] = 0
        if hasattr(model, 'predict_proba') and X_test is not None:
            try:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = 0.5
        else:
            metrics['roc_auc'] = 0.5
    else:
        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)
        except:
            metrics['mape'] = 0
    return metrics


def build_results_df(results, task_type):
    rows = []
    for name, m in results.items():
        row = {'Model': name}
        if task_type == "Classification":
            row.update({
                'Accuracy': f"{m.get('accuracy',0):.4f}", 'F1': f"{m.get('f1',0):.4f}",
                'Precision': f"{m.get('precision',0):.4f}", 'Recall': f"{m.get('recall',0):.4f}",
                'ROC-AUC': f"{m.get('roc_auc',0):.4f}", 'MCC': f"{m.get('mcc',0):.4f}",
                'CV Mean': f"{m.get('cv_mean',0):.4f}", 'CV Std': f"{m.get('cv_std',0):.4f}",
                'Time (s)': f"{m.get('train_time',0):.2f}",
            })
        else:
            row.update({
                'R2': f"{m.get('r2',0):.4f}", 'RMSE': f"{m.get('rmse',0):.4f}",
                'MAE': f"{m.get('mae',0):.4f}", 'MAPE': f"{m.get('mape',0):.4f}",
                'CV Mean': f"{m.get('cv_mean',0):.4f}", 'CV Std': f"{m.get('cv_std',0):.4f}",
                'Time (s)': f"{m.get('train_time',0):.2f}",
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px;">
            <div style="font-size:2.5rem;">🧠</div>
            <h2 style="margin:5px 0; background: linear-gradient(135deg,#6c63ff,#ff6584);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:800;">
                NeuralForge AI
            </h2>
            <p style="color:#9898b0; font-size:0.85rem; margin-top:4px;">Advanced ML Platform</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Home", "Data Explorer", "Preprocessing",
             "Model Training", "Model Analysis",
             "Predictions", "Export Center"],
            key="nav_radio"
        )

        st.markdown("---")

        if st.session_state.df is not None:
            st.markdown(f"""
            <div style="background:rgba(108,99,255,0.1); border-radius:10px; padding:12px;
                border: 1px solid rgba(108,99,255,0.2);">
                <p style="color:#6c63ff; font-weight:600; margin:0 0 6px; font-size:0.85rem;">Session Status</p>
                <p style="color:#9898b0; font-size:0.8rem; margin:2px 0;">
                    Dataset: <span style="color:#e8e8f0">{st.session_state.df.shape[0]}x{st.session_state.df.shape[1]}</span></p>
                <p style="color:#9898b0; font-size:0.8rem; margin:2px 0;">
                    Target: <span style="color:#e8e8f0">{st.session_state.target_col or 'Not set'}</span></p>
                <p style="color:#9898b0; font-size:0.8rem; margin:2px 0;">
                    Task: <span style="color:#e8e8f0">{st.session_state.task_type or 'Not set'}</span></p>
                <p style="color:#9898b0; font-size:0.8rem; margin:2px 0;">
                    Models: <span style="color:#e8e8f0">{len(st.session_state.trained_models)}</span></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; padding:10px 0;">
            <p style="color:#9898b0; font-size:0.75rem;">Built with Streamlit v2.0</p>
        </div>
        """, unsafe_allow_html=True)

        return page


# ══════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════

def render_home():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">NeuralForge AI</div>
        <div class="hero-subtitle">End-to-end Machine Learning Platform</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("📊", "Smart EDA", "Automated exploratory\ndata analysis"),
        ("🧹", "Preprocessing", "Missing values, encoding,\nscaling, balancing"),
        ("🤖", "AutoML", "13+ algorithms with\nauto model selection"),
        ("📉", "Deep Analysis", "ROC curves, feature\nimportance, CV"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="glow-card" style="text-align:center; min-height:140px;">
                <div style="font-size:2.2rem; margin-bottom:8px;">{icon}</div>
                <h4 style="margin:0 0 6px; font-size:1rem;">{title}</h4>
                <p style="color:#9898b0; font-size:0.8rem; margin:0; white-space:pre-line;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glow-card">
        <h3 style="margin:0 0 16px;">Quick Start</h3>
        <div style="display:flex; flex-direction:column; gap:12px;">
            <div style="display:flex; align-items:center;">
                <span class="step-badge">1</span>
                <span style="color:#e8e8f0;">Upload your CSV dataset or generate sample data</span>
            </div>
            <div style="display:flex; align-items:center;">
                <span class="step-badge">2</span>
                <span style="color:#e8e8f0;">Explore and preprocess your data</span>
            </div>
            <div style="display:flex; align-items:center;">
                <span class="step-badge">3</span>
                <span style="color:#e8e8f0;">Train multiple ML models with one click</span>
            </div>
            <div style="display:flex; align-items:center;">
                <span class="step-badge">4</span>
                <span style="color:#e8e8f0;">Analyze, compare, and export your best model</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown('<div class="glow-card">', unsafe_allow_html=True)
        st.markdown("### Upload Dataset")
        uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"],
                                         label_visibility="collapsed", key="file_uploader")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_cleaned = False
                st.session_state.trained_models = {}
                st.session_state.models_trained = False
                st.success(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
                st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glow-card">', unsafe_allow_html=True)
        st.markdown("### Sample Data")
        sample_choice = st.selectbox(
            "Choose a sample dataset",
            ["None", "Titanic (Classification)", "California Housing (Regression)",
             "Heart Disease (Classification)", "Wine Quality (Classification)"],
            key="sample_select"
        )
        if sample_choice != "None" and st.button("Generate", key="gen_sample"):
            generate_sample_data(sample_choice)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glow-card">
        <h3 style="margin:0 0 12px;">Supported Algorithms</h3>
        <div style="display:flex; flex-wrap:wrap; gap:4px;">
            <span class="feature-chip">Logistic Regression</span>
            <span class="feature-chip">Linear Regression</span>
            <span class="feature-chip">Ridge / Lasso</span>
            <span class="feature-chip">Decision Tree</span>
            <span class="feature-chip">Random Forest</span>
            <span class="feature-chip">Extra Trees</span>
            <span class="feature-chip">Gradient Boosting</span>
            <span class="feature-chip">AdaBoost</span>
            <span class="feature-chip">XGBoost</span>
            <span class="feature-chip">LightGBM</span>
            <span class="feature-chip">SVM</span>
            <span class="feature-chip">KNN</span>
            <span class="feature-chip">Naive Bayes</span>
            <span class="feature-chip">MLP Neural Network</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def generate_sample_data(choice):
    np.random.seed(42)
    if choice == "Titanic (Classification)":
        n = 891
        df = pd.DataFrame({
            'PassengerId': range(1, n+1),
            'Survived': np.random.choice([0, 1], n, p=[0.62, 0.38]),
            'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
            'Age': np.random.normal(29.7, 14.5, n).clip(0.42, 80),
            'SibSp': np.random.choice(range(4), n, p=[0.68, 0.17, 0.06, 0.09]),
            'Parch': np.random.choice(range(4), n, p=[0.76, 0.13, 0.06, 0.05]),
            'Fare': np.abs(np.random.exponential(32, n)),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09]),
        })
        df.loc[np.random.choice(n, 177, replace=False), 'Age'] = np.nan
        df.loc[np.random.choice(n, 2, replace=False), 'Embarked'] = np.nan
        st.session_state.target_col = 'Survived'
        st.session_state.task_type = 'Classification'
    elif choice == "California Housing (Regression)":
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['MedHouseVal'] = housing.target
        st.session_state.target_col = 'MedHouseVal'
        st.session_state.task_type = 'Regression'
    elif choice == "Heart Disease (Classification)":
        n = 303
        df = pd.DataFrame({
            'age': np.random.normal(54, 9, n).clip(29, 77),
            'sex': np.random.choice([0, 1], n, p=[0.32, 0.68]),
            'cp': np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.16, 0.29, 0.08]),
            'trestbps': np.random.normal(131, 17, n).clip(94, 200),
            'chol': np.random.normal(246, 51, n).clip(126, 564),
            'fbs': np.random.choice([0, 1], n, p=[0.85, 0.15]),
            'thalach': np.random.normal(149, 22, n).clip(71, 202),
            'exang': np.random.choice([0, 1], n, p=[0.67, 0.33]),
            'target': np.random.choice([0, 1], n, p=[0.46, 0.54]),
        })
        st.session_state.target_col = 'target'
        st.session_state.task_type = 'Classification'
    elif choice == "Wine Quality (Classification)":
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        st.session_state.target_col = 'target'
        st.session_state.task_type = 'Classification'

    st.session_state.df = df
    st.session_state.data_cleaned = False
    st.session_state.trained_models = {}
    st.session_state.models_trained = False
    st.success(f"Generated {choice} - {df.shape[0]} rows x {df.shape[1]} columns")
    st.rerun()


# ══════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════

def render_data_explorer():
    st.markdown("## Data Explorer")

    if st.session_state.df is None:
        st.markdown("""
        <div class="glow-card" style="text-align:center; padding:60px;">
            <div style="font-size:3rem; margin-bottom:12px;">📂</div>
            <h3 style="color:#9898b0;">No Dataset Loaded</h3>
            <p style="color:#9898b0;">Go to Home page to upload or generate data.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df

    col1, col2 = st.columns([3, 2])
    with col1:
        target = st.selectbox(
            "Select Target Column",
            options=["None"] + list(df.columns),
            index=(["None"] + list(df.columns)).index(st.session_state.target_col)
            if st.session_state.target_col in df.columns else 0,
            key="target_select"
        )
        if target != "None":
            st.session_state.target_col = target
            n_unique = df[target].nunique()
            dtype = df[target].dtype
            if dtype == 'object' or n_unique <= 15:
                st.session_state.task_type = "Classification"
            else:
                st.session_state.task_type = "Regression"

    with col2:
        task = st.selectbox(
            "Task Type",
            ["Classification", "Regression"],
            index=["Classification", "Regression"].index(st.session_state.task_type)
            if st.session_state.task_type else 0,
            key="task_select"
        )
        st.session_state.task_type = task

    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics_data = [
        ("Rows", f"{df.shape[0]:,}"),
        ("Columns", f"{df.shape[1]}"),
        ("Missing", f"{df.isnull().sum().sum():,}"),
        ("Duplicates", f"{df.duplicated().sum()}"),
        ("Memory", f"{df.memory_usage(deep=True).sum()/1024:.1f} KB"),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics_data):
        with col:
            st.metric(label=label, value=value)

    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Distributions", "Correlations", "Feature Info"])

    with tab1:
        st.markdown(f"**Shape:** `{df.shape[0]}` rows x `{df.shape[1]}` columns")
        st.dataframe(df.head(20), use_container_width=True, height=400)

    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect(
                "Select columns to visualize", numeric_cols,
                default=numeric_cols[:min(6, len(numeric_cols))], key="dist_cols"
            )
            if selected_cols:
                n_cols = min(3, len(selected_cols))
                fig = make_subplots(
                    rows=(len(selected_cols) - 1) // n_cols + 1, cols=n_cols,
                    subplot_titles=selected_cols,
                )
                for idx, col_name in enumerate(selected_cols):
                    row = idx // n_cols + 1
                    col_pos = idx % n_cols + 1
                    fig.add_trace(
                        go.Histogram(
                            x=df[col_name].dropna(), name=col_name,
                            marker_color=['#6c63ff', '#ff6584', '#00d4aa', '#ffd166', '#ff8a65', '#ab47bc'][idx % 6],
                            opacity=0.8, showlegend=False,
                        ), row=row, col=col_pos
                    )
                fig.update_layout(height=300 * ((len(selected_cols)-1)//3+1), bargap=0.1)
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale=[[0, '#ff6584'], [0.5, '#1a1a2e'], [1, '#00d4aa']],
                zmin=-1, zmax=1, text=np.round(corr.values, 2),
                texttemplate='%{text}', textfont={"size": 9}, hoverongaps=False,
            ))
            fig.update_layout(height=600, title="Correlation Matrix")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            if st.session_state.target_col and st.session_state.target_col in numeric_df.columns:
                target_corr = corr[st.session_state.target_col].drop(
                    st.session_state.target_col, errors='ignore').abs().sort_values(ascending=False)
                st.markdown("#### Top Correlated Features with Target")
                top_n = min(10, len(target_corr))
                fig2 = go.Figure(go.Bar(
                    x=target_corr.head(top_n).values, y=target_corr.head(top_n).index,
                    orientation='h',
                    marker_color=['#6c63ff' if v < 0.5 else '#00d4aa' for v in target_corr.head(top_n).values],
                ))
                fig2.update_layout(height=400, xaxis_title="|Correlation|", yaxis_title="", margin=dict(l=150))
                apply_theme(fig2)
                st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        col_info = pd.DataFrame({
            'Column': df.columns, 'Type': df.dtypes.values,
            'Non-Null': df.notnull().sum().values,
            'Missing': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values,
            'Unique': df.nunique().values,
        })
        st.dataframe(col_info.style.format({'Missing %': '{:.2f}%'}), use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════
#  PAGE: PREPROCESSING (UPDATED WITH MULTI-SELECT + AUTO)
# ══════════════════════════════════════════════════════════

def render_preprocessing():
    st.markdown("## Smart Preprocessing")

    if st.session_state.df is None:
        st.warning("Please load a dataset first.")
        return

    if not st.session_state.target_col:
        st.warning("Please select a target column in Data Explorer first.")
        return

    df = st.session_state.df.copy()
    target = st.session_state.target_col

    # ─── ONE-CLICK AUTO PREPROCESS ───
    st.markdown("""
    <div class="glow-card" style="border-left: 4px solid #00d4aa;">
        <h3 style="margin:0 0 8px; color:#00d4aa;">⚡ One-Click Auto Preprocess</h3>
        <p style="color:#9898b0; font-size:0.9rem; margin:0 0 12px;">
            Handles missing values, encodes categoricals, scales features, and splits data automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    auto_col1, auto_col2, auto_col3 = st.columns([2, 1, 1])
    with auto_col1:
        auto_scaler = st.selectbox("Scaling Method",
            ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "No Scaling"], key="auto_scaler_type")
    with auto_col2:
        auto_balance = st.selectbox("Balance (Classification)",
            ["None", "SMOTE", "SMOTETomek"], key="auto_balance")
    with auto_col3:
        auto_test_size = st.slider("Test Size %", 10, 40, 20, key="auto_test_size")

    if st.button("🚀 Auto Preprocess All", type="primary", key="auto_preprocess_btn", use_container_width=True):
        with st.spinner("Running auto preprocessing..."):
            df = st.session_state.df.copy()
            le_dict = {}

            # 1. Missing values
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if target in num_cols:
                num_cols.remove(target)
            if target in cat_cols:
                cat_cols.remove(target)
            if len(num_cols) > 0:
                df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            for col in cat_cols:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "Unknown")
            if df[target].isnull().sum() > 0:
                if df[target].dtype == 'object':
                    mode_val = df[target].mode()
                    df[target] = df[target].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "Unknown")
                else:
                    df[target] = df[target].fillna(df[target].median())
            st.session_state.data_cleaned = True

            # 2. Encode categoricals
            all_cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            for col in all_cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
            st.session_state.label_encoders.update(le_dict)

            # 3. Drop remaining NaN
            df = df.dropna()
            st.session_state.df = df

            # 4. Split
            X = df.drop(columns=[target])
            y = df[target]
            st.session_state.feature_names = X.columns.tolist()

            # 5. Scale
            if auto_scaler != "No Scaling":
                scalers_map = {
                    "Standard Scaler": StandardScaler(),
                    "MinMax Scaler": MinMaxScaler(),
                    "Robust Scaler": RobustScaler(),
                }
                scaler = scalers_map[auto_scaler]
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                st.session_state.scaler = scaler
            else:
                X_scaled = X
                st.session_state.scaler = None

            # 6. Train/test split
            test_size = auto_test_size / 100.0
            if st.session_state.task_type == "Classification":
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42, stratify=y)
                except ValueError:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            # 7. Balance
            if st.session_state.task_type == "Classification" and auto_balance != "None":
                try:
                    if auto_balance == "SMOTE":
                        sampler = SMOTE(random_state=42)
                    elif auto_balance == "SMOTETomek":
                        sampler = SMOTETomek(random_state=42)
                    else:
                        sampler = None
                    if sampler:
                        X_res, y_res = sampler.fit_resample(X_train, y_train)
                        st.session_state.X_train = pd.DataFrame(X_res, columns=X_train.columns)
                        st.session_state.y_train = pd.Series(y_res)
                except Exception as e:
                    st.warning(f"Balancing failed: {str(e)[:80]}")

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Train Size", f"{st.session_state.X_train.shape[0]}")
        with sc2:
            st.metric("Test Size", f"{st.session_state.X_test.shape[0]}")
        with sc3:
            st.metric("Features", f"{st.session_state.X_train.shape[1]}")
        with sc4:
            st.metric("Encoded Cols", f"{len(le_dict)}")
        st.success("Auto preprocessing complete!")
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; margin: 8px 0 16px;">
        <span style="color:#9898b0; font-size:0.85rem;">── Or customize step-by-step below ──</span>
    </div>
    """)

    # ─── MANUAL STEP-BY-STEP ───
    tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Encoding", "Scaling & Split", "Balancing"])

    # ═══════ TAB 1: MISSING VALUES ═══════
    with tab1:
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]

        if len(missing_cols) > 0:
            st.markdown("#### Missing Values Overview")
            fig = go.Figure(go.Bar(
                x=missing_cols.values, y=missing_cols.index, orientation='h',
                marker_color=['#ff6584' if v/len(df) > 0.3 else '#ffd166' if v/len(df) > 0.1 else '#6c63ff'
                              for v in missing_cols.values],
                text=[f"{v} ({v/len(df)*100:.1f}%)" for v in missing_cols.values],
                textposition='outside', textfont=dict(color='#e8e8f0', size=11),
            ))
            fig.update_layout(height=max(300, len(missing_cols) * 40),
                              xaxis_title="Missing Count", yaxis_title="", margin=dict(l=180))
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Select Columns to Fix")
            selected_missing_cols = st.multiselect(
                "Choose columns with missing values",
                list(missing_cols.index),
                default=list(missing_cols.index),
                key="mv_cols_multi"
            )

            st.markdown("#### Imputation Strategy")
            strategy_mode = st.radio("Apply strategy",
                ["Same strategy for all selected columns", "Custom strategy per column"],
                key="mv_strategy_mode", horizontal=True)

            if strategy_mode == "Same strategy for all selected columns":
                strategy = st.selectbox("Strategy for all", [
                    "Drop rows with missing values",
                    "Median (numeric) / Mode (categorical)",
                    "Mean (numeric) / Mode (categorical)",
                    "KNN Imputation (numeric only)",
                    "Forward Fill",
                    "Backward Fill",
                    "Constant: 0 / Unknown",
                ], key="mv_same_strategy")
            else:
                strategy_options = [
                    "Median / Mode", "Mean / Mode", "Drop rows",
                    "KNN Imputation", "Forward Fill", "Backward Fill", "Constant",
                ]
                per_cols = st.columns(2)
                for idx, col in enumerate(selected_missing_cols):
                    with per_cols[idx % 2]:
                        col_type = "Numeric" if df[col].dtype in [np.number] else "Categorical"
                        miss_pct = missing_cols[col] / len(df) * 100
                        st.selectbox(f"{col} ({col_type}, {miss_pct:.1f}% missing)",
                            strategy_options, key=f"mv_per_{col}")

            if st.button("Apply Imputation", key="apply_impute_v2", use_container_width=True):
                if not selected_missing_cols:
                    st.warning("Please select at least one column.")
                else:
                    df = st.session_state.df.copy()
                    if strategy_mode == "Same strategy for all selected columns":
                        if strategy == "Drop rows with missing values":
                            df = df.dropna(subset=selected_missing_cols)
                        elif "Median" in strategy:
                            for col in selected_missing_cols:
                                if df[col].dtype in [np.number]:
                                    df[col] = df[col].fillna(df[col].median())
                                else:
                                    mode_val = df[col].mode()
                                    df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "Unknown")
                        elif "Mean" in strategy:
                            for col in selected_missing_cols:
                                if df[col].dtype in [np.number]:
                                    df[col] = df[col].fillna(df[col].mean())
                                else:
                                    mode_val = df[col].mode()
                                    df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "Unknown")
                        elif "KNN" in strategy:
                            num_selected = [c for c in selected_missing_cols if df[c].dtype in [np.number]]
                            if num_selected:
                                imputer = KNNImputer(n_neighbors=5)
                                df[num_selected] = imputer.fit_transform(df[num_selected])
                            for col in [c for c in selected_missing_cols if c not in num_selected]:
                                mode_val = df[col].mode()
                                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "Unknown")
                        elif "Forward" in strategy:
                            df[selected_missing_cols] = df[selected_missing_cols].ffill()
                        elif "Backward" in strategy:
                            df[selected_missing_cols] = df[selected_missing_cols].bfill()
                        elif "Constant" in strategy:
                            for col in selected_missing_cols:
                                if df[col].dtype in [np.number]:
                                    df[col] = df[col].fillna(0)
                                else:
                                    df[col] = df[col].fillna("Unknown")
                    else:
                        for col in selected_missing_cols:
                            per_strategy = st.session_state.get(f"mv_per_{col}", "Median / Mode")
                            if per_strategy == "Drop rows":
                                df = df.dropna(subset=[col])
                            elif "Median" in per_strategy:
                                if df[col].dtype in [np.number]:
                                    df[col] = df[col].fillna(df[col].median())
                                else:
                                    mode_val = df[col].mode()
                                    df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "Unknown")
                            elif "Mean" in per_strategy:
                                if df[col].dtype in [np.number]:
                                    df[col] = df[col].fillna(df[col].mean())
                                else:
                                    mode_val = df[col].mode()
                                    df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "Unknown")
                            elif "KNN" in per_strategy:
                                if df[col].dtype in [np.number]:
                                    imputer = KNNImputer(n_neighbors=5)
                                    df[[col]] = imputer.fit_transform(df[[col]])
                                else:
                                    mode_val = df[col].mode()
                                    df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "Unknown")
                            elif "Forward" in per_strategy:
                                df[col] = df[col].ffill()
                            elif "Backward" in per_strategy:
                                df[col] = df[col].bfill()
                            elif "Constant" in per_strategy:
                                if df[col].dtype in [np.number]:
                                    df[col] = df[col].fillna(0)
                                else:
                                    df[col] = df[col].fillna("Unknown")

                    df = df.dropna()
                    st.session_state.df = df
                    st.session_state.data_cleaned = True
                    st.success(f"Imputation applied to {len(selected_missing_cols)} columns!")
                    st.rerun()
        else:
            st.markdown("""
            <div class="glow-card" style="text-align:center; padding:40px; border: 1px solid #00d4aa;">
                <div style="font-size:2.5rem; margin-bottom:8px;">✅</div>
                <h3 style="color:#00d4aa;">No Missing Values!</h3>
                <p style="color:#9898b0;">Your data is clean.</p>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.data_cleaned = True

    # ═══════ TAB 2: ENCODING ═══════
    with tab2:
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if target in cat_cols:
            cat_cols.remove(target)

        if len(cat_cols) > 0:
            st.markdown(f"#### Categorical Columns ({len(cat_cols)} found)")

            cat_info = pd.DataFrame({
                'Column': cat_cols,
                'Unique Values': [df[c].nunique() for c in cat_cols],
                'Sample Values': [', '.join(map(str, df[c].unique()[:4])) for c in cat_cols],
            })
            st.dataframe(cat_info, use_container_width=True, height=200)

            selected_cat_cols = st.multiselect(
                "Select columns to encode", cat_cols,
                default=cat_cols, key="encode_cols_v2"
            )

            encoding_method = st.selectbox("Encoding Method", [
                "Label Encoding", "One-Hot Encoding (drop first)",
                "One-Hot Encoding (keep all)", "Frequency Encoding",
            ], key="encode_method_v2")

            encode_target = False
            if df[target].dtype == 'object':
                encode_target = st.checkbox(f"Also encode target '{target}'", value=True, key="encode_target_check")

            if st.button("Apply Encoding", key="apply_encode_v2", use_container_width=True):
                if not selected_cat_cols:
                    st.warning("Please select at least one column.")
                else:
                    df = st.session_state.df.copy()
                    le_dict = {}

                    if encoding_method == "Label Encoding":
                        for col in selected_cat_cols:
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                            le_dict[col] = le
                    elif "drop first" in encoding_method:
                        ohe_cols = [c for c in selected_cat_cols if df[c].nunique() <= 15]
                        le_cols = [c for c in selected_cat_cols if c not in ohe_cols]
                        for col in le_cols:
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                            le_dict[col] = le
                        if ohe_cols:
                            df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
                    elif "keep all" in encoding_method:
                        ohe_cols = [c for c in selected_cat_cols if df[c].nunique() <= 15]
                        le_cols = [c for c in selected_cat_cols if c not in ohe_cols]
                        for col in le_cols:
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                            le_dict[col] = le
                        if ohe_cols:
                            df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
                    elif "Frequency" in encoding_method:
                        for col in selected_cat_cols:
                            freq_map = df[col].value_counts(normalize=True).to_dict()
                            df[col] = df[col].map(freq_map)

                    if encode_target and df[target].dtype == 'object':
                        le_target = LabelEncoder()
                        df[target] = le_target.fit_transform(df[target].astype(str))
                        le_dict[target] = le_target

                    st.session_state.df = df
                    st.session_state.label_encoders.update(le_dict)
                    st.success(f"Encoding applied to {len(selected_cat_cols)} columns!")
                    st.rerun()
        else:
            st.markdown("""
            <div class="glow-card" style="text-align:center; padding:40px; border: 1px solid #00d4aa;">
                <div style="font-size:2.5rem; margin-bottom:8px;">✅</div>
                <h3 style="color:#00d4aa;">No Categorical Columns!</h3>
            </div>
            """, unsafe_allow_html=True)

    # ═══════ TAB 3: SCALING & SPLIT ═══════
    with tab3:
        st.markdown("#### Feature Scaling & Train/Test Split")

        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_features:
            numeric_features.remove(target)

        selected_scale_cols = st.multiselect(
            "Select columns to scale", numeric_features,
            default=numeric_features, key="scale_cols_v2"
        )

        col_sc1, col_sc2, col_sc3 = st.columns(3)
        with col_sc1:
            scaler_type = st.selectbox("Scaling Method",
                ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "No Scaling"], key="scaler_type_v2")
        with col_sc2:
            test_size = st.slider("Test Size %", 10, 40, 20, key="test_size_v2")
        with col_sc3:
            random_state = st.number_input("Random State", value=42, key="random_state_v2")

        if st.button("Apply Scaling & Split Data", key="apply_scale_v2", use_container_width=True):
            df = st.session_state.df.copy()

            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if target in non_numeric:
                non_numeric.remove(target)
            if len(non_numeric) > 0:
                le_dict = {}
                for col in non_numeric:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    le_dict[col] = le
                if df[target].dtype == 'object':
                    le_target = LabelEncoder()
                    df[target] = le_target.fit_transform(df[target].astype(str))
                    le_dict[target] = le_target
                st.session_state.label_encoders.update(le_dict)

            df = df.dropna()
            st.session_state.df = df

            X = df.drop(columns=[target])
            y = df[target]
            st.session_state.feature_names = X.columns.tolist()

            if scaler_type != "No Scaling" and len(selected_scale_cols) > 0:
                scalers_map = {
                    "Standard Scaler": StandardScaler(),
                    "MinMax Scaler": MinMaxScaler(),
                    "Robust Scaler": RobustScaler(),
                }
                scaler = scalers_map[scaler_type]
                X_scaled = X.copy()
                X_scaled[selected_scale_cols] = scaler.fit_transform(X[selected_scale_cols])
                st.session_state.scaler = scaler
            else:
                X_scaled = X
                st.session_state.scaler = None

            ts = test_size / 100.0
            if st.session_state.task_type == "Classification":
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=ts, random_state=random_state, stratify=y)
                except ValueError:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=ts, random_state=random_state)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=ts, random_state=random_state)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.success(f"Data split! Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
            st.rerun()

    # ═══════ TAB 4: BALANCING ═══════
    with tab4:
        if st.session_state.task_type == "Classification":
            target_counts = df[target].value_counts()
            st.markdown("#### Class Distribution")

            fig = go.Figure(data=[
                go.Bar(x=target_counts.index.astype(str), y=target_counts.values,
                       marker_color=['#6c63ff', '#ff6584', '#00d4aa', '#ffd166'][:len(target_counts)],
                       text=target_counts.values, textposition='outside',
                       textfont=dict(color='#e8e8f0'))
            ])
            fig.update_layout(xaxis_title="Class", yaxis_title="Count")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            imbalance_ratio = target_counts.max() / max(target_counts.min(), 1)
            st.metric("Imbalance Ratio", f"{imbalance_ratio:.1f} : 1")

            if imbalance_ratio > 1.5:
                st.warning(f"Dataset is imbalanced ({imbalance_ratio:.1f}:1). Consider balancing.")
                balance_method = st.selectbox("Balancing Method", [
                    "SMOTE (oversample minority)", "ADASYN (adaptive oversample)",
                    "SMOTETomek (oversample + undersample)", "Random Undersample (reduce majority)"
                ], key="balance_method_v2")

                if st.button("Apply Balancing", key="apply_balance_v2", use_container_width=True):
                    if st.session_state.X_train is not None:
                        X_train = st.session_state.X_train
                        y_train = st.session_state.y_train
                        method_map = {
                            "SMOTE (oversample minority)": SMOTE(random_state=42),
                            "ADASYN (adaptive oversample)": ADASYN(random_state=42),
                            "SMOTETomek (oversample + undersample)": SMOTETomek(random_state=42),
                            "Random Undersample (reduce majority)": RandomUnderSampler(random_state=42),
                        }
                        sampler = method_map.get(balance_method)
                        if sampler:
                            try:
                                X_res, y_res = sampler.fit_resample(X_train, y_train)
                                st.session_state.X_train = pd.DataFrame(X_res, columns=X_train.columns)
                                st.session_state.y_train = pd.Series(y_res)
                                st.success(f"Balanced! New train size: {len(y_res)}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Balancing failed: {str(e)[:100]}")
                    else:
                        st.error("Please apply scaling & split first.")
            else:
                st.success("Classes are reasonably balanced. No balancing needed.")
        else:
            st.info("Class balancing is only available for classification tasks.")


# ══════════════════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════

def render_model_training():
    st.markdown("## Model Training")

    if st.session_state.X_train is None:
        st.warning("Please preprocess and split your data first (Preprocessing page).")
        return

    task_type = st.session_state.task_type
    registry = get_model_registry(task_type)

    st.markdown("### Select Models to Train")

    col1, col2 = st.columns([1, 1])
    with col1:
        select_all = st.button("Select All Models", key="select_all")
    with col2:
        select_top = st.button("Select Top 5", key="select_top")

    default_selected = list(registry.keys()) if select_all else (
        ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting", "Extra Trees"]
        if select_top else ["Random Forest", "XGBoost"]
    )

    selected_models = st.multiselect(
        "Choose algorithms", list(registry.keys()),
        default=default_selected if (select_all or select_top) else ["Random Forest", "XGBoost"],
        key="model_select"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5, key="cv_folds")
    with col2:
        scoring = st.selectbox("Scoring Metric",
            ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
            if task_type == "Classification"
            else ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
            key="scoring_metric")
    with col3:
        n_jobs = st.selectbox("Parallel Jobs", [-1, 1, 2, 4], index=0, key="n_jobs")

    with st.expander("Hyperparameter Tuning (Optional)"):
        enable_tuning = st.checkbox("Enable Hyperparameter Tuning", key="enable_tuning")
        if enable_tuning:
            tuning_method = st.selectbox("Tuning Method",
                ["RandomizedSearchCV", "GridSearchCV"], key="tuning_method")
            tuning_iter = st.slider("Max Iterations (Randomized)", 5, 50, 10, key="tuning_iter")

    st.markdown("---")
    if st.button("Train Models", type="primary", key="train_btn", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model.")
            return

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        total = len(selected_models)
        for i, name in enumerate(selected_models):
            status_text.markdown(f"Training **{name}** ({i+1}/{total})...")
            model = registry[name]
            if hasattr(model, 'n_jobs'):
                model.n_jobs = n_jobs

            try:
                start_time = time.time()
                if enable_tuning:
                    param_grid = get_param_grid(name, task_type)
                    if tuning_method == "RandomizedSearchCV":
                        tuner = RandomizedSearchCV(model, param_grid, n_iter=tuning_iter,
                            cv=cv_folds, scoring=scoring, random_state=42, n_jobs=n_jobs)
                    else:
                        tuner = GridSearchCV(model, param_grid, cv=cv_folds,
                            scoring=scoring, n_jobs=n_jobs)
                    tuner.fit(X_train, y_train)
                    model = tuner.best_estimator_
                else:
                    model.fit(X_train, y_train)

                train_time = time.time() - start_time

                if task_type == "Classification":
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                else:
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred, task_type, model, X_test)
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                metrics['train_time'] = train_time
                metrics['model'] = model
                metrics['y_pred'] = y_pred
                results[name] = metrics
                st.session_state.trained_models[name] = metrics
            except Exception as e:
                st.warning(f"{name} failed: {str(e)[:100]}")

            progress_bar.progress((i + 1) / total)

        status_text.empty()
        progress_bar.empty()

        if results:
            if task_type == "Classification":
                best_name = max(results, key=lambda x: results[x].get('accuracy', 0))
            else:
                best_name = max(results, key=lambda x: results[x].get('r2', -999))
            st.session_state.best_model_name = best_name
            st.session_state.best_model = results[best_name]['model']
            st.session_state.models_trained = True
            st.success(f"Training complete! Best model: **{best_name}**")

    if st.session_state.trained_models:
        st.markdown("---")
        st.markdown("### Training Results")
        results_df = build_results_df(st.session_state.trained_models, st.session_state.task_type)
        st.dataframe(results_df, use_container_width=True)

        task_type = st.session_state.task_type
        metric_col = 'accuracy' if task_type == "Classification" else 'r2'
        names = list(st.session_state.trained_models.keys())
        values = [st.session_state.trained_models[n].get(metric_col, 0) for n in names]
        colors = ['#00d4aa' if n == st.session_state.best_model_name else '#6c63ff' for n in names]

        fig = go.Figure(go.Bar(
            x=values, y=names, orientation='h', marker_color=colors,
            text=[f"{v:.4f}" for v in values], textposition='outside',
            textfont=dict(color='#e8e8f0', size=12),
        ))
        fig.update_layout(title=f"Model Comparison - {metric_col.upper()}",
                          xaxis_title=metric_col.upper(), margin=dict(l=200),
                          height=max(400, len(names) * 45))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  PAGE: MODEL ANALYSIS
# ══════════════════════════════════════════════════════════

def render_model_analysis():
    st.markdown("## Model Analysis")

    if not st.session_state.trained_models:
        st.warning("No trained models. Please train models first.")
        return

    model_names = list(st.session_state.trained_models.keys())
    selected_model_name = st.selectbox("Select Model to Analyze", model_names,
        index=model_names.index(st.session_state.best_model_name)
        if st.session_state.best_model_name in model_names else 0, key="analysis_model")

    model_data = st.session_state.trained_models[selected_model_name]
    model = model_data['model']
    y_pred = model_data['y_pred']
    y_test = st.session_state.y_test
    X_test = st.session_state.X_test
    task_type = st.session_state.task_type

    st.markdown("### Model Performance")
    if task_type == "Classification":
        col1, col2, col3, col4, col5 = st.columns(5)
        metric_items = [
            ("Accuracy", model_data.get('accuracy', 0)),
            ("F1 Score", model_data.get('f1', 0)),
            ("Precision", model_data.get('precision', 0)),
            ("Recall", model_data.get('recall', 0)),
            ("ROC-AUC", model_data.get('roc_auc', 0)),
        ]
    else:
        col1, col2, col3, col4 = st.columns(4)
        metric_items = [
            ("R2 Score", model_data.get('r2', 0)),
            ("RMSE", model_data.get('rmse', 0)),
            ("MAE", model_data.get('mae', 0)),
            ("MAPE", model_data.get('mape', 0)),
        ]

    for col, (label, value) in zip([col1, col2, col3, col4, col5][:len(metric_items)], metric_items):
        with col:
            st.metric(label=label, value=f"{value:.4f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Confusion / Residuals", "ROC Curve", "Feature Importance", "Cross-Validation"])

    with tab1:
        if task_type == "Classification":
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(y_test.unique())
            fig = ff.create_annotated_heatmap(
                z=cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                annotation_text=cm, colorscale=[[0, '#1a1a2e'], [0.5, '#6c63ff'], [1, '#ff6584']],
                showscale=True,
            )
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted",
                              yaxis_title="Actual", height=500)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        else:
            residuals = y_test.values - y_pred
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=["Residuals Distribution", "Residual Plot"])
            fig.add_trace(go.Histogram(x=residuals, name="Residuals",
                marker_color='#6c63ff', opacity=0.8), row=1, col=1)
            fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                marker=dict(color='#ff6584', size=5, opacity=0.6),
                name="Residual", showlegend=False), row=1, col=2)
            fig.add_hline(y=0, line_dash="dash", line_color="#9898b0", row=1, col=2)
            fig.update_layout(height=450)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=y_test.values, y=y_pred, mode='markers',
                marker=dict(color='#6c63ff', size=5, opacity=0.6), name="Predictions"))
            min_val = min(y_test.values.min(), y_pred.min())
            max_val = max(y_test.values.max(), y_pred.max())
            fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', line=dict(color='#ff6584', dash='dash'), name="Perfect"))
            fig2.update_layout(title="Predicted vs Actual", xaxis_title="Actual",
                               yaxis_title="Predicted", height=500)
            apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if task_type == "Classification":
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                labels = sorted(y_test.unique())
                fig = go.Figure()
                if len(labels) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    auc_score = roc_auc_score(y_test, y_proba[:, 1])
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                        line=dict(color='#6c63ff', width=3), name=f"ROC (AUC={auc_score:.3f})"))
                else:
                    for i, label in enumerate(labels):
                        y_binary = (y_test == label).astype(int)
                        if y_proba.shape[1] > i:
                            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
                            auc_score = roc_auc_score(y_binary, y_proba[:, i])
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                line=dict(width=2), name=f"Class {label} (AUC={auc_score:.3f})"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                    line=dict(color='#9898b0', dash='dash'), name="Random"))
                fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate", height=500)
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("This model doesn't support probability predictions.")
        else:
            st.info("ROC curves are only for classification tasks.")

    with tab3:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = st.session_state.feature_names or X_test.columns.tolist()
            imp_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)], 'Importance': importances,
            }).sort_values('Importance', ascending=True)
            top_n = st.slider("Show top N features", 5, min(30, len(imp_df)), 15, key="top_n_feat")
            fig = go.Figure(go.Bar(
                x=imp_df['Importance'].tail(top_n), y=imp_df['Feature'].tail(top_n),
                orientation='h', marker_color='#6c63ff',
            ))
            fig.update_layout(title=f"Feature Importance - {selected_model_name}",
                              xaxis_title="Importance", margin=dict(l=200),
                              height=max(400, top_n * 30))
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            feature_names = st.session_state.feature_names or X_test.columns.tolist()
            if len(coef.shape) > 1:
                coef = coef[0]
            coef_df = pd.DataFrame({
                'Feature': feature_names[:len(coef)], 'Coefficient': coef,
            }).sort_values('Coefficient', ascending=True)
            top_n = st.slider("Show top N features", 5, min(30, len(coef_df)), 15, key="top_n_coef")
            fig = go.Figure(go.Bar(
                x=coef_df['Coefficient'].tail(top_n), y=coef_df['Feature'].tail(top_n),
                orientation='h',
                marker_color=['#00d4aa' if c > 0 else '#ff6584' for c in coef_df['Coefficient'].tail(top_n)],
            ))
            fig.update_layout(title=f"Coefficients - {selected_model_name}",
                              xaxis_title="Coefficient Value", margin=dict(l=200),
                              height=max(400, top_n * 30))
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    with tab4:
        cv_mean = model_data.get('cv_mean', 0)
        cv_std = model_data.get('cv_std', 0)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CV Mean", f"{cv_mean:.4f}")
        with col2:
            st.metric("CV Std", f"{cv_std:.4f}")


# ══════════════════════════════════════════════════════════
#  PAGE: PREDICTIONS
# ══════════════════════════════════════════════════════════

def render_predictions():
    st.markdown("## Predictions")

    if not st.session_state.trained_models:
        st.warning("No trained models available.")
        return

    model_names = list(st.session_state.trained_models.keys())
    selected = st.selectbox("Select Model", model_names,
        index=model_names.index(st.session_state.best_model_name)
        if st.session_state.best_model_name in model_names else 0, key="pred_model")

    model = st.session_state.trained_models[selected]['model']
    feature_names = st.session_state.feature_names or st.session_state.X_train.columns.tolist()
    task_type = st.session_state.task_type

    tab1, tab2 = st.tabs(["Manual Input", "Batch Predictions"])

    with tab1:
        st.markdown("### Enter Feature Values")
        input_values = {}
        n_cols = 3
        cols = st.columns(n_cols)
        for idx, feat in enumerate(feature_names):
            with cols[idx % n_cols]:
                X_train = st.session_state.X_train
                if feat in X_train.columns:
                    col_data = X_train[feat]
                    if pd.api.types.is_integer_dtype(col_data):
                        input_values[feat] = st.number_input(feat, value=int(col_data.median()),
                                                              step=1, key=f"input_{feat}")
                    else:
                        input_values[feat] = st.number_input(feat, value=float(col_data.median()),
                                                              format="%.4f", key=f"input_{feat}")
                else:
                    input_values[feat] = st.number_input(feat, value=0.0, key=f"input_{feat}")

        if st.button("Predict", type="primary", key="predict_btn", use_container_width=True):
            input_df = pd.DataFrame([input_values])
            if st.session_state.scaler:
                input_scaled = pd.DataFrame(
                    st.session_state.scaler.transform(input_df), columns=input_df.columns)
            else:
                input_scaled = input_df

            prediction = model.predict(input_scaled)[0]
            st.markdown("---")
            st.markdown("### Prediction Result")

            if task_type == "Classification":
                if st.session_state.target_col in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[st.session_state.target_col]
                    try:
                        pred_label = le.inverse_transform([int(prediction)])[0]
                    except:
                        pred_label = prediction
                else:
                    pred_label = prediction

                st.markdown(f"""
                <div class="glow-card" style="text-align:center; padding:40px;">
                    <p style="color:#9898b0; font-size:1rem; margin-bottom:8px;">Predicted Class</p>
                    <h1 style="font-size:3rem; margin:0;">
                        <span class="gradient-text">{pred_label}</span>
                    </h1>
                </div>
                """, unsafe_allow_html=True)

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_scaled)[0]
                    classes = model.classes_
                    if st.session_state.target_col in st.session_state.label_encoders:
                        le = st.session_state.label_encoders[st.session_state.target_col]
                        try:
                            classes = le.inverse_transform(classes.astype(int))
                        except:
                            pass
                    fig = go.Figure(go.Bar(
                        x=proba, y=[str(c) for c in classes], orientation='h',
                        marker_color=['#6c63ff', '#ff6584', '#00d4aa', '#ffd166'][:len(classes)],
                        text=[f"{p:.2%}" for p in proba], textposition='outside',
                    ))
                    fig.update_layout(title="Class Probabilities", xaxis_title="Probability",
                                      xaxis_range=[0, 1.1], margin=dict(l=100), height=300)
                    apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"""
                <div class="glow-card" style="text-align:center; padding:40px;">
                    <p style="color:#9898b0; font-size:1rem; margin-bottom:8px;">Predicted Value</p>
                    <h1 style="font-size:3rem; margin:0;">
                        <span class="gradient-text">{prediction:.4f}</span>
                    </h1>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Upload CSV for Batch Predictions")
        batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
        if batch_file:
            try:
                batch_df = pd.read_csv(batch_file)
                missing_cols = set(feature_names) - set(batch_df.columns)
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    batch_input = batch_df[feature_names]
                    if st.session_state.scaler:
                        batch_scaled = pd.DataFrame(
                            st.session_state.scaler.transform(batch_input), columns=feature_names)
                    else:
                        batch_scaled = batch_input
                    predictions = model.predict(batch_scaled)
                    batch_df['Prediction'] = predictions
                    if task_type == "Classification" and hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(batch_scaled)
                        for i, cls in enumerate(model.classes_):
                            batch_df[f'Prob_Class_{cls}'] = proba[:, i]
                    st.dataframe(batch_df, use_container_width=True, height=400)
                    csv = batch_df.to_csv(index=False)
                    st.download_button("Download Predictions", csv, "predictions.csv",
                                       "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════
#  PAGE: EXPORT CENTER
# ══════════════════════════════════════════════════════════

def render_export():
    st.markdown("## Export Center")

    if not st.session_state.trained_models:
        st.warning("No trained models to export.")
        return

    model_names = list(st.session_state.trained_models.keys())
    selected = st.selectbox("Select Model to Export", model_names,
        index=model_names.index(st.session_state.best_model_name)
        if st.session_state.best_model_name in model_names else 0, key="export_model")

    model = st.session_state.trained_models[selected]['model']
    model_data = st.session_state.trained_models[selected]

    st.markdown(f"### Model Summary: {selected}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", type(model).__name__)
    with col2:
        st.metric("Training Time", f"{model_data.get('train_time', 0):.2f}s")
    with col3:
        if st.session_state.task_type == "Classification":
            st.metric("Accuracy", f"{model_data.get('accuracy', 0):.4f}")
        else:
            st.metric("R2 Score", f"{model_data.get('r2', 0):.4f}")

    with st.expander("Model Parameters"):
        params = model.get_params()
        st.json({k: str(v) for k, v in params.items()})

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Export Model")
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        st.download_button("Download Model (.joblib)", buffer,
            f"{selected.replace(' ', '_')}_model.joblib", "application/octet-stream",
            use_container_width=True)

    with col2:
        st.markdown("### Export Results")
        results_df = build_results_df(st.session_state.trained_models, st.session_state.task_type)
        csv = results_df.to_csv(index=False)
        st.download_button("Download Results (.csv)", csv,
            "model_comparison_results.csv", "text/csv", use_container_width=True)

    if st.session_state.scaler:
        st.markdown("---")
        st.markdown("### Export Scaler")
        scaler_buffer = io.BytesIO()
        joblib.dump(st.session_state.scaler, scaler_buffer)
        scaler_buffer.seek(0)
        st.download_button("Download Scaler (.joblib)", scaler_buffer,
            "scaler.joblib", "application/octet-stream", use_container_width=True)

    st.markdown("---")
    st.markdown("### Pipeline Configuration")
    pipeline_info = {
        "task_type": st.session_state.task_type,
        "target_column": st.session_state.target_col,
        "feature_names": st.session_state.feature_names,
        "best_model": st.session_state.best_model_name,
        "scaler_type": type(st.session_state.scaler).__name__ if st.session_state.scaler else None,
        "label_encoded_columns": list(st.session_state.label_encoders.keys()),
        "timestamp": datetime.now().isoformat(),
    }
    st.json(pipeline_info)
    json_str = json.dumps(pipeline_info, indent=2)
    st.download_button("Download Pipeline Config (.json)", json_str,
        "pipeline_config.json", "application/json", use_container_width=True)


# ══════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    pages = {
        "Home": render_home,
        "Data Explorer": render_data_explorer,
        "Preprocessing": render_preprocessing,
        "Model Training": render_model_training,
        "Model Analysis": render_model_analysis,
        "Predictions": render_predictions,
        "Export Center": render_export,
    }

    render_fn = pages.get(page)
    if render_fn:
        render_fn()
    else:
        render_home()


if __name__ == "__main__":
    main()