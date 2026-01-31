import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dengue Shock Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #dc3545;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dengue dataset"""
    df = pd.read_csv('DENGUE_2_200321.csv')
    
    # Drop completely empty columns
    df = df.drop(['Unnamed: 14', 'Unnamed: 22'], axis=1, errors='ignore')
    
    # Handle problematic string values in numeric columns
    for col in ['RATE OF PLATELET DROP', 'RATE OF HCT RISE']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def prepare_features(df):
    """Prepare features for modeling"""
    # Define feature columns (exclude target and non-predictive columns)
    exclude_cols = ['st_no', 'SHOCK#', 'bleed_hos', 'doi_shock']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['SHOCK#'].copy()
    
    # Convert target to binary (1 = shock, 0 = no shock)
    # Assuming SHOCK# = 1 means shock, SHOCK# = 2 means no shock
    y = (y == 1).astype(int)
    
    return X, y, feature_cols

def train_models(X_train, X_test, y_train, y_test):
    """Train Random Forest and Logistic Regression models"""
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    models = {}
    predictions = {}
    probabilities = {}
    
    # ===== RANDOM FOREST =====
    st.write("🌲 Training Random Forest Classifier...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',  # Handle imbalanced classes
        n_jobs=-1
    )
    rf_model.fit(X_train_imputed, y_train)
    
    rf_pred = rf_model.predict(X_test_imputed)
    rf_prob = rf_model.predict_proba(X_test_imputed)[:, 1]
    
    models['Random Forest'] = {
        'model': rf_model,
        'imputer': imputer,
        'scaler': None
    }
    predictions['Random Forest'] = rf_pred
    probabilities['Random Forest'] = rf_prob
    
    # ===== LOGISTIC REGRESSION =====
    st.write("📊 Training Logistic Regression...")
    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # Handle imbalanced classes
        solver='lbfgs'
    )
    lr_model.fit(X_train_scaled, y_train)
    
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    models['Logistic Regression'] = {
        'model': lr_model,
        'imputer': imputer,
        'scaler': scaler
    }
    predictions['Logistic Regression'] = lr_pred
    probabilities['Logistic Regression'] = lr_prob
    
    return models, predictions, probabilities, y_test

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive performance metrics"""
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall (Sensitivity)': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
        'Average Precision': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm

def cross_validate_model(model, X, y, model_name, use_scaling=False):
    """Perform cross-validation"""
    
    # Prepare data
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    if use_scaling:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_imputed)
    else:
        X_processed = X_imputed
    
    # Stratified K-Fold for imbalanced data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='roc_auc')
    
    return cv_scores

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix"""
    
    labels = ['No Shock (2)', 'Shock (1)']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Reds',
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500
    )
    
    return fig

def plot_roc_curve(y_true, y_prob, model_name):
    """Plot ROC curve"""
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc:.3f})',
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate (Recall)',
        width=600,
        height=500,
        showlegend=True
    )
    
    return fig

def plot_precision_recall_curve(y_true, y_prob, model_name):
    """Plot Precision-Recall curve"""
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'{model_name} (AP = {avg_precision:.3f})',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=f'Precision-Recall Curve - {model_name}',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=600,
        height=500,
        showlegend=True
    )
    
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance for Random Forest"""
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker_color='indianred'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Importance',
        yaxis_title='Feature',
        width=700,
        height=500,
        showlegend=False
    )
    
    return fig

def predict_new_patient(models, patient_data, feature_cols):
    """Make predictions for a new patient"""
    
    # Create dataframe with patient data
    patient_df = pd.DataFrame([patient_data])
    
    # Ensure all features are present
    for col in feature_cols:
        if col not in patient_df.columns:
            patient_df[col] = np.nan
    
    patient_df = patient_df[feature_cols]
    
    results = {}
    
    for model_name, model_dict in models.items():
        model = model_dict['model']
        imputer = model_dict['imputer']
        scaler = model_dict['scaler']
        
        # Impute missing values
        patient_imputed = imputer.transform(patient_df)
        
        # Scale if needed
        if scaler is not None:
            patient_processed = scaler.transform(patient_imputed)
        else:
            patient_processed = patient_imputed
        
        # Predict
        prediction = model.predict(patient_processed)[0]
        probability = model.predict_proba(patient_processed)[0]
        
        results[model_name] = {
            'prediction': 'SHOCK' if prediction == 1 else 'NO SHOCK',
            'shock_probability': probability[1] * 100,
            'no_shock_probability': probability[0] * 100
        }
    
    return results

def main():
    st.markdown('<p class="main-header">🏥 Dengue Shock Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-Based Risk Assessment: Random Forest vs Logistic Regression</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading dengue patient data...'):
        df = load_data()
    
    st.success(f"✅ Loaded {len(df)} patient records with {df.shape[1]} features")
    
    # Show data overview
    with st.expander("📊 Dataset Overview"):
        col1, col2, col3, col4 = st.columns(4)
        
        shock_counts = df['SHOCK#'].value_counts()
        shock_cases = shock_counts.get(1, 0)
        no_shock_cases = shock_counts.get(2, 0)
        
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("Shock Cases", shock_cases, delta=f"{(shock_cases/len(df)*100):.1f}%")
        with col3:
            st.metric("No Shock Cases", no_shock_cases, delta=f"{(no_shock_cases/len(df)*100):.1f}%")
        with col4:
            st.metric("Features", df.shape[1])
        
        st.dataframe(df.head(10), use_container_width=True)
    
    # Prepare data
    X, y, feature_cols = prepare_features(df)
    
    # Show class distribution
    st.markdown("### 📊 Class Distribution")
    
    class_dist = pd.DataFrame({
        'Class': ['Shock (1)', 'No Shock (2)'],
        'Count': [sum(y == 1), sum(y == 0)],
        'Percentage': [sum(y == 1)/len(y)*100, sum(y == 0)/len(y)*100]
    })
    
    fig_dist = px.pie(
        class_dist,
        values='Count',
        names='Class',
        title='Target Variable Distribution (SHOCK#)',
        color_discrete_sequence=['#dc3545', '#28a745']
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown(f"""
    <div class="warning-box">
        ⚠️ <strong>Class Imbalance Detected:</strong> 
        The dataset has {sum(y == 0)} no-shock cases vs {sum(y == 1)} shock cases 
        (ratio: {sum(y == 0)/sum(y == 1):.1f}:1). Both models use balanced class weights to handle this.
    </div>
    """, unsafe_allow_html=True)
    
    # Train-Test Split
    st.markdown("### 🔄 Model Training")
    
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    st.info(f"📊 Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")
    
    # Train models
    with st.spinner('Training models...'):
        models, predictions, probabilities, y_test = train_models(X_train, X_test, y_train, y_test)
    
    st.success("✅ Models trained successfully!")
    
    # Model Comparison
    st.markdown("---")
    st.markdown("## 🎯 Model Performance Comparison")
    
    # Calculate metrics for both models
    all_metrics = {}
    all_cm = {}
    
    for model_name in ['Random Forest', 'Logistic Regression']:
        metrics, cm = calculate_metrics(
            y_test,
            predictions[model_name],
            probabilities[model_name]
        )
        all_metrics[model_name] = metrics
        all_cm[model_name] = cm
    
    # Display metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌲 Random Forest")
        rf_metrics = all_metrics['Random Forest']
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Accuracy", f"{rf_metrics['Accuracy']:.3f}")
            st.metric("Precision", f"{rf_metrics['Precision']:.3f}")
            st.metric("Recall", f"{rf_metrics['Recall (Sensitivity)']:.3f}")
        with metric_col2:
            st.metric("Specificity", f"{rf_metrics['Specificity']:.3f}")
            st.metric("F1-Score", f"{rf_metrics['F1-Score']:.3f}")
            st.metric("ROC-AUC", f"{rf_metrics['ROC-AUC']:.3f}")
    
    with col2:
        st.markdown("### 📊 Logistic Regression")
        lr_metrics = all_metrics['Logistic Regression']
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Accuracy", f"{lr_metrics['Accuracy']:.3f}")
            st.metric("Precision", f"{lr_metrics['Precision']:.3f}")
            st.metric("Recall", f"{lr_metrics['Recall (Sensitivity)']:.3f}")
        with metric_col2:
            st.metric("Specificity", f"{lr_metrics['Specificity']:.3f}")
            st.metric("F1-Score", f"{lr_metrics['F1-Score']:.3f}")
            st.metric("ROC-AUC", f"{lr_metrics['ROC-AUC']:.3f}")
    
    # Metrics comparison table
    st.markdown("### 📋 Detailed Metrics Comparison")
    
    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df = comparison_df.round(4)
    
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, color='lightgreen'),
        use_container_width=True
    )
    
    # Cross-Validation
    st.markdown("### 🔄 Cross-Validation Scores (5-Fold)")
    
    with st.spinner('Performing cross-validation...'):
        rf_cv = cross_validate_model(
            RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            X, y, 'Random Forest', use_scaling=False
        )
        
        lr_cv = cross_validate_model(
            LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            X, y, 'Logistic Regression', use_scaling=True
        )
    
    cv_col1, cv_col2 = st.columns(2)
    
    with cv_col1:
        st.metric("Random Forest CV Mean", f"{rf_cv.mean():.4f}", 
                 delta=f"±{rf_cv.std():.4f}")
    
    with cv_col2:
        st.metric("Logistic Regression CV Mean", f"{lr_cv.mean():.4f}",
                 delta=f"±{lr_cv.std():.4f}")
    
    # Confusion Matrices
    st.markdown("---")
    st.markdown("## 📊 Confusion Matrices")
    
    cm_col1, cm_col2 = st.columns(2)
    
    with cm_col1:
        fig_cm_rf = plot_confusion_matrix(all_cm['Random Forest'], 'Random Forest')
        st.plotly_chart(fig_cm_rf, use_container_width=True)
    
    with cm_col2:
        fig_cm_lr = plot_confusion_matrix(all_cm['Logistic Regression'], 'Logistic Regression')
        st.plotly_chart(fig_cm_lr, use_container_width=True)
    
    # ROC Curves
    st.markdown("---")
    st.markdown("## 📈 ROC Curves")
    
    roc_col1, roc_col2 = st.columns(2)
    
    with roc_col1:
        fig_roc_rf = plot_roc_curve(y_test, probabilities['Random Forest'], 'Random Forest')
        st.plotly_chart(fig_roc_rf, use_container_width=True)
    
    with roc_col2:
        fig_roc_lr = plot_roc_curve(y_test, probabilities['Logistic Regression'], 'Logistic Regression')
        st.plotly_chart(fig_roc_lr, use_container_width=True)
    
    # Precision-Recall Curves
    st.markdown("---")
    st.markdown("## 📉 Precision-Recall Curves")
    
    pr_col1, pr_col2 = st.columns(2)
    
    with pr_col1:
        fig_pr_rf = plot_precision_recall_curve(y_test, probabilities['Random Forest'], 'Random Forest')
        st.plotly_chart(fig_pr_rf, use_container_width=True)
    
    with pr_col2:
        fig_pr_lr = plot_precision_recall_curve(y_test, probabilities['Logistic Regression'], 'Logistic Regression')
        st.plotly_chart(fig_pr_lr, use_container_width=True)
    
    # Feature Importance (Random Forest only)
    st.markdown("---")
    st.markdown("## 🎯 Feature Importance Analysis (Random Forest)")
    
    fig_importance = plot_feature_importance(
        models['Random Forest']['model'],
        feature_cols,
        top_n=15
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Classification Reports
    st.markdown("---")
    st.markdown("## 📝 Detailed Classification Reports")
    
    report_col1, report_col2 = st.columns(2)
    
    with report_col1:
        st.markdown("### Random Forest")
        rf_report = classification_report(
            y_test,
            predictions['Random Forest'],
            target_names=['No Shock', 'Shock'],
            output_dict=True
        )
        st.dataframe(pd.DataFrame(rf_report).transpose(), use_container_width=True)
    
    with report_col2:
        st.markdown("### Logistic Regression")
        lr_report = classification_report(
            y_test,
            predictions['Logistic Regression'],
            target_names=['No Shock', 'Shock'],
            output_dict=True
        )
        st.dataframe(pd.DataFrame(lr_report).transpose(), use_container_width=True)
    
    # Patient Prediction Tool
    st.markdown("---")
    st.markdown("## 🩺 Individual Patient Risk Assessment")
    
    st.markdown("""
    Enter patient details below to predict shock risk using both models.
    """)
    
    with st.form("patient_form"):
        st.markdown("### Patient Information")
        
        form_col1, form_col2, form_col3 = st.columns(3)
        
        with form_col1:
            age = st.number_input("Age (years)", min_value=5, max_value=16, value=12)
            wt = st.number_input("Weight (kg)", min_value=10, max_value=88, value=35)
            day_ill = st.number_input("Day of Illness", min_value=1, max_value=10, value=3)
            hct_bsl = st.number_input("Baseline Hematocrit (%)", min_value=20, max_value=60, value=40)
        
        with form_col2:
            plt_bsl = st.number_input("Baseline Platelet (×10³/μL)", min_value=10, max_value=400, value=150)
            liver = st.number_input("Liver Enzyme Level", min_value=10, max_value=500, value=50)
            serology = st.selectbox("Serology (DENV)", [1, 2, 3, 4], index=0)
            sex = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        
        with form_col3:
            vomit = st.selectbox("Vomiting", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
            ttest = st.selectbox("Tourniquet Test", [1, 2], format_func=lambda x: "Positive" if x == 1 else "Negative")
            muc_bld = st.selectbox("Mucosal Bleeding", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
            abd_pain = st.selectbox("Abdominal Pain", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
        
        submitted = st.form_submit_button("🔍 Predict Shock Risk", type="primary")
        
        if submitted:
            patient_data = {
                'age': age,
                'wt': wt,
                'day_ill': day_ill,
                'hct_bsl': hct_bsl,
                'plt_bsl': plt_bsl,
                'liver': liver,
                'SEROLOGY#DENV1,DENV2,DENV3,DENV4': serology,
                'SEX#': sex,
                'VOMIT#': vomit,
                'TTEST#': ttest,
                'MUC BLD#': muc_bld,
                'ABD.PAIN#': abd_pain
            }
            
            results = predict_new_patient(models, patient_data, feature_cols)
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown("#### 🌲 Random Forest Prediction")
                rf_result = results['Random Forest']
                
                if rf_result['prediction'] == 'SHOCK':
                    st.error(f"⚠️ **HIGH RISK - SHOCK PREDICTED**")
                else:
                    st.success(f"✅ **LOW RISK - NO SHOCK**")
                
                st.metric("Shock Probability", f"{rf_result['shock_probability']:.1f}%")
                st.metric("No Shock Probability", f"{rf_result['no_shock_probability']:.1f}%")
                
                # Probability gauge
                fig_rf_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rf_result['shock_probability'],
                    title={'text': "Shock Risk Level"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_rf_gauge.update_layout(height=300)
                st.plotly_chart(fig_rf_gauge, use_container_width=True)
            
            with result_col2:
                st.markdown("#### 📊 Logistic Regression Prediction")
                lr_result = results['Logistic Regression']
                
                if lr_result['prediction'] == 'SHOCK':
                    st.error(f"⚠️ **HIGH RISK - SHOCK PREDICTED**")
                else:
                    st.success(f"✅ **LOW RISK - NO SHOCK**")
                
                st.metric("Shock Probability", f"{lr_result['shock_probability']:.1f}%")
                st.metric("No Shock Probability", f"{lr_result['no_shock_probability']:.1f}%")
                
                # Probability gauge
                fig_lr_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=lr_result['shock_probability'],
                    title={'text': "Shock Risk Level"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_lr_gauge.update_layout(height=300)
                st.plotly_chart(fig_lr_gauge, use_container_width=True)
            
            # Clinical Interpretation
            st.markdown("---")
            st.markdown("### 🩺 Clinical Interpretation")
            
            avg_prob = (rf_result['shock_probability'] + lr_result['shock_probability']) / 2
            
            if avg_prob < 30:
                st.success("""
                **Low Risk Assessment**
                - Both models indicate low probability of dengue shock
                - Continue monitoring with standard protocols
                - Regular vital signs assessment recommended
                """)
            elif avg_prob < 70:
                st.warning("""
                **Moderate Risk Assessment**
                - Increased vigilance required
                - Close monitoring of hematocrit and platelet levels
                - Consider more frequent vital signs checks
                - Prepare for potential fluid resuscitation
                """)
            else:
                st.error("""
                **High Risk Assessment**
                - **Immediate medical attention required**
                - Intensive monitoring protocol
                - Prepare for shock management
                - Alert senior medical staff
                - Consider ICU admission
                """)
    
    # Model Information
    st.markdown("---")
    st.markdown("## ℹ️ Model Information")
    
    with st.expander("📚 About the Models"):
        st.markdown("""
        ### Random Forest Classifier
        - **Type**: Ensemble learning method
        - **Parameters**: 100 trees, max depth 10
        - **Advantages**: 
            - Handles non-linear relationships
            - Provides feature importance
            - Robust to outliers
        - **Best for**: Complex pattern recognition
        
        ### Logistic Regression
        - **Type**: Linear classification model
        - **Parameters**: L-BFGS solver, max iterations 1000
        - **Advantages**:
            - Interpretable coefficients
            - Probability estimates
            - Computationally efficient
        - **Best for**: Linear relationships and clinical interpretation
        
        ### Class Balancing
        Both models use `class_weight='balanced'` to handle the imbalanced dataset (more no-shock than shock cases).
        
        ### Performance Metrics
        - **Accuracy**: Overall correct predictions
        - **Precision**: When model predicts shock, how often is it correct?
        - **Recall (Sensitivity)**: Of all actual shock cases, how many did we catch?
        - **Specificity**: Of all actual no-shock cases, how many did we correctly identify?
        - **F1-Score**: Harmonic mean of precision and recall
        - **ROC-AUC**: Area under ROC curve (discrimination ability)
        """)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Medical Disclaimer:</strong><br>
        This tool is for educational and research purposes only. It should not replace professional medical judgment.
        All clinical decisions should be made by qualified healthcare professionals considering the complete
        clinical picture, patient history, and current medical guidelines.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
