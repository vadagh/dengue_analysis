# 🏥 Dengue Shock Prediction System

A comprehensive machine learning application for predicting dengue shock syndrome using **Random Forest** and **Logistic Regression** classifiers.

## 🎯 Project Overview

This Streamlit application analyzes dengue patient data to predict the risk of developing dengue shock syndrome (DSS), a life-threatening complication of dengue infection. The system compares two different machine learning approaches to provide robust risk assessment.

## 📊 Dataset Information

- **Total Records**: 2,168 dengue patients
- **Features**: 25 clinical and laboratory parameters
- **Target Variable**: SHOCK# (1 = Shock, 2 = No Shock)
- **Class Distribution**: Imbalanced (123 shock cases vs 2,045 no-shock cases)

### Key Features

**Patient Demographics:**
- Age (years)
- Weight (kg)
- Sex

**Clinical Symptoms:**
- Day of illness
- Vomiting status
- Abdominal pain
- Mucosal bleeding
- Tourniquet test result

**Laboratory Parameters:**
- Baseline hematocrit (HCT)
- Baseline platelet count (PLT)
- Liver enzyme levels
- Serology (DENV 1-4)
- Minimum platelet (days 3-8)
- Maximum hematocrit (days 3-8)
- Rate of platelet drop
- Rate of hematocrit rise

## 🤖 Machine Learning Models

### 1. Random Forest Classifier 🌲

**Configuration:**
- 100 decision trees
- Max depth: 10
- Balanced class weights
- Min samples split: 5
- Min samples leaf: 2

**Strengths:**
- Excellent at capturing non-linear relationships
- Provides feature importance rankings
- Robust to outliers and missing data
- High accuracy on complex patterns

**Use Case:** Best for comprehensive risk assessment with detailed feature analysis

### 2. Logistic Regression 📊

**Configuration:**
- L-BFGS solver
- Max iterations: 1000
- Balanced class weights
- Standardized features

**Strengths:**
- Interpretable probability outputs
- Computationally efficient
- Clear coefficient interpretation
- Works well for linear relationships

**Use Case:** Best for quick assessments and clinical interpretation

## ✨ Features

### 📈 Model Performance Analysis
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC curves** for discrimination ability
- **Precision-Recall curves** for imbalanced data
- **Confusion matrices** for error analysis
- **5-Fold Cross-Validation** for reliability assessment

### 🎯 Feature Importance
- Visual ranking of most predictive clinical features
- Helps identify key risk factors for dengue shock

### 🩺 Individual Patient Risk Assessment
- Real-time prediction for new patients
- Dual model predictions with probability estimates
- Risk level gauges (Low/Moderate/High)
- Clinical interpretation guidelines

### 📊 Interactive Visualizations
- Class distribution charts
- Performance comparison tables
- ROC and Precision-Recall curves
- Confusion matrices
- Feature importance plots

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 2GB RAM minimum

### Setup

1. **Install dependencies:**
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

2. **Verify file structure:**
```
project/
│
├── dengue_shock_predictor.py
├── DENGUE_2_200321.csv
├── requirements.txt
└── README.md
```

3. **Run the application:**
```bash
streamlit run dengue_shock_predictor.py
```

4. **Access the app:**
Open your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Data Overview
- Review dataset statistics and class distribution
- Understand the imbalanced nature of the data
- Examine sample patient records

### 2. Model Training
- Adjust test set size (10-40%)
- Models train automatically with balanced class weights
- View training progress

### 3. Performance Comparison
- Compare metrics side-by-side
- Analyze confusion matrices
- Review ROC and Precision-Recall curves
- Check cross-validation scores

### 4. Feature Analysis
- Identify most important clinical predictors
- Understand which features drive predictions

### 5. Patient Risk Assessment
Fill in patient details:
- **Demographics**: Age, weight, sex
- **Clinical**: Day of illness, symptoms
- **Laboratory**: Hematocrit, platelet count, liver enzymes
- **Serology**: DENV type

Get instant predictions:
- Shock probability from both models
- Risk level visualization
- Clinical interpretation

## 📊 Performance Metrics Explained

### Accuracy
Overall percentage of correct predictions (both shock and no-shock)

### Precision
When the model predicts shock, how often is it correct?
- **High precision** = Few false alarms
- **Critical for**: Resource allocation

### Recall (Sensitivity)
Of all actual shock cases, how many did we catch?
- **High recall** = Few missed cases
- **Critical for**: Patient safety

### Specificity
Of all no-shock cases, how many were correctly identified?
- **High specificity** = Few false positives
- **Critical for**: Avoiding unnecessary interventions

### F1-Score
Harmonic mean of precision and recall
- Balances both metrics
- Useful for imbalanced datasets

### ROC-AUC
Area under the ROC curve
- **0.5** = Random guessing
- **0.7-0.8** = Acceptable
- **0.8-0.9** = Excellent
- **>0.9** = Outstanding

### Average Precision
Summarizes Precision-Recall curve
- Better for imbalanced datasets than ROC-AUC
- Focus on positive class (shock) performance

## 🎯 Clinical Interpretation

### Risk Levels

**Low Risk (<30% probability)**
- Continue standard monitoring
- Regular vital signs
- Follow dengue fever protocols

**Moderate Risk (30-70% probability)**
- Increased vigilance
- More frequent monitoring
- Close attention to warning signs
- Prepare for potential complications

**High Risk (>70% probability)**
- **Immediate medical attention**
- Intensive monitoring
- Shock management protocol
- Consider ICU admission
- Alert senior medical staff

### Warning Signs to Monitor
- Persistent vomiting
- Severe abdominal pain
- Mucosal bleeding
- Rising hematocrit with falling platelets
- Lethargy or restlessness
- Cold, clammy extremities

## 🔬 Technical Details

### Data Preprocessing
1. **Missing Value Handling**: Median imputation
2. **Feature Scaling**: StandardScaler for Logistic Regression
3. **Class Balancing**: Balanced class weights (compensates for 16.6:1 imbalance)
4. **Train-Test Split**: Stratified sampling to maintain class distribution

### Model Training Pipeline
```
Raw Data → Missing Value Imputation → Feature Scaling (LR only) →
Model Training → Cross-Validation → Performance Evaluation →
Predictions & Probabilities
```

### Cross-Validation Strategy
- **Method**: Stratified 5-Fold
- **Why Stratified**: Maintains class distribution in each fold
- **Metric**: ROC-AUC (handles imbalanced data well)

### Handling Class Imbalance
- **Ratio**: 2045 no-shock : 123 shock (16.6:1)
- **Strategy**: `class_weight='balanced'`
- **Effect**: Penalizes misclassifying minority class (shock) more heavily

## 📈 Expected Performance

### Typical Metrics (may vary by data split)

**Random Forest:**
- Accuracy: 0.92-0.96
- Precision: 0.40-0.70
- Recall: 0.60-0.85
- ROC-AUC: 0.85-0.95

**Logistic Regression:**
- Accuracy: 0.90-0.94
- Precision: 0.35-0.65
- Recall: 0.55-0.80
- ROC-AUC: 0.80-0.90

**Note**: High accuracy is expected due to class imbalance. Focus on recall (catching shock cases) and ROC-AUC for true performance assessment.

## 🎓 Understanding the Results

### Why is Accuracy High but Precision Low?
The dataset is heavily imbalanced (94.3% no-shock). A model that always predicts "no-shock" would have 94.3% accuracy but be clinically useless. That's why we also look at:
- **Recall**: Are we catching the shock cases?
- **ROC-AUC**: How well can we discriminate?
- **Precision-Recall curve**: Performance on the minority class

### Which Model is Better?
**It depends on the clinical context:**

**Choose Random Forest if:**
- You need highest overall performance
- You want to understand feature importance
- You have computational resources
- You're doing comprehensive risk stratification

**Choose Logistic Regression if:**
- You need interpretable results
- You want faster predictions
- You need to explain the model to clinicians
- You prefer simpler, more transparent models

**Best Practice:**
Use both models and look for agreement. When both predict high risk, confidence is highest.

## 🔧 Troubleshooting

### CSV File Not Found
- Ensure `DENGUE_2_200321.csv` is in the same directory
- Check file name spelling exactly

### Module Import Errors
- Run: `pip install -r requirements.txt`
- Ensure Python 3.8+

### Low Performance Metrics
- Check class imbalance handling
- Verify cross-validation scores
- Ensure sufficient training data

### Predictions Don't Make Sense
- Verify input data ranges
- Check for missing/invalid values
- Review feature importance

## 📚 Research Background

### Dengue Shock Syndrome (DSS)
- Severe complication of dengue infection
- Characterized by plasma leakage and circulatory failure
- Occurs in 2-5% of dengue cases
- Mortality rate: 1-5% with proper treatment, up to 20% without

### Critical Period
- Usually occurs during defervescence (when fever breaks)
- Days 3-7 of illness are most critical
- Warning signs precede shock by 12-24 hours

### Risk Factors
- Secondary dengue infection
- Young age (5-15 years)
- High baseline hematocrit
- Rapidly falling platelet count
- Plasma leakage indicators

## ⚠️ Limitations

1. **Model is trained on specific population** (may not generalize to all regions)
2. **Requires complete clinical data** (missing values are imputed)
3. **Not a replacement for clinical judgment**
4. **Performance depends on data quality**
5. **Class imbalance** affects precision metrics

## 🔮 Future Enhancements

- [ ] Add more ML models (XGBoost, Neural Networks)
- [ ] SHAP values for individual prediction explanations
- [ ] Time-series analysis for disease progression
- [ ] Integration with electronic health records
- [ ] Real-time monitoring dashboard
- [ ] Multi-center validation
- [ ] External dataset validation

## 📄 Citation

If you use this system in research, please cite:
```
Dengue Shock Prediction System
Machine Learning Application for Clinical Decision Support
[Your Institution/Research Group]
2025
```

## ⚖️ Medical Disclaimer

**IMPORTANT: This tool is for educational and research purposes only.**

- Not FDA approved or clinically validated
- Should NOT replace professional medical judgment
- All clinical decisions must be made by qualified healthcare professionals
- Consider complete clinical picture, patient history, and current guidelines
- Use only as a supplementary decision support tool

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional features/biomarkers
- Model optimization
- Validation studies
- UI/UX enhancements
- Documentation

## 📧 Support

For issues, questions, or suggestions:
1. Check troubleshooting section
2. Review documentation
3. Verify data format
4. Check model performance metrics

## 📜 License

This project is provided for educational and research purposes.

---

**Built with ❤️ for improving dengue patient care through AI**

🏥 Remember: Always prioritize patient safety and clinical judgment! 🩺
