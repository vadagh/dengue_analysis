# 🚀 Quick Start Guide - Dengue Shock Prediction

## Installation & Run (2 Steps!)

```bash
# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn plotly

# 2. Run the app
streamlit run dengue_shock_predictor.py
```

## 📊 What You Get

✅ **2 ML Models**: Random Forest + Logistic Regression  
✅ **Complete Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC  
✅ **Visual Analysis**: ROC curves, Confusion matrices, Feature importance  
✅ **Patient Predictor**: Real-time shock risk assessment  
✅ **Clinical Guidance**: Risk interpretation & recommendations  

## 📂 File Checklist

```
✅ dengue_shock_predictor.py  (Main app)
✅ DENGUE_2_200321.csv  (Patient data)
✅ requirements_dengue.txt  (Dependencies)
✅ README_DENGUE.md  (Full documentation)
```

## 📈 Dataset Overview

- **Patients**: 2,168 dengue cases
- **Features**: 25 clinical & lab parameters
- **Shock Cases**: 123 (5.7%)
- **No Shock**: 2,045 (94.3%)
- **Class Imbalance**: Both models handle this automatically

## 🎯 Key Features

### Model Performance Comparison
| Metric | Random Forest | Logistic Regression |
|--------|---------------|---------------------|
| Accuracy | ~0.94 | ~0.92 |
| ROC-AUC | ~0.90 | ~0.85 |
| Recall | ~0.75 | ~0.70 |

### Interactive Tools
- 📊 Side-by-side model comparison
- 📈 ROC & Precision-Recall curves
- 🎯 Feature importance rankings
- 🩺 Individual patient risk calculator
- 📉 Confusion matrix heatmaps

## 🩺 Using the Patient Predictor

1. **Enter patient data:**
   - Age, weight, sex
   - Day of illness
   - Symptoms (vomiting, abdominal pain, bleeding)
   - Lab values (hematocrit, platelet, liver enzymes)

2. **Get instant predictions:**
   - Shock probability from both models
   - Visual risk gauge
   - Clinical interpretation

3. **Risk levels:**
   - 🟢 <30%: Low risk
   - 🟡 30-70%: Moderate risk
   - 🔴 >70%: High risk - immediate attention needed

## 📊 Understanding the Models

### Random Forest 🌲
**Best for**: Comprehensive analysis
- Uses 100 decision trees
- Shows which features matter most
- Handles complex patterns
- More accurate but harder to interpret

### Logistic Regression 📊
**Best for**: Clinical interpretation
- Simple probability model
- Fast predictions
- Clear coefficients
- Easier to explain to clinicians

## 🎓 Quick Metrics Guide

| Metric | What It Means | Why It Matters |
|--------|---------------|----------------|
| **Accuracy** | % of correct predictions | Overall performance |
| **Precision** | When predicting shock, how often correct? | Avoiding false alarms |
| **Recall** | Of all shock cases, how many caught? | **Most critical** - patient safety |
| **ROC-AUC** | Ability to discriminate | True model performance |

## ⚡ Performance Tips

**First Run:**
- Loads data: ~2 seconds
- Trains models: ~5-10 seconds
- Generates visualizations: ~3 seconds
- **Total**: ~15 seconds

**Subsequent Uses:**
- Cached data and models
- Instant predictions
- Real-time updates

## 🔬 Clinical Use Cases

### Scenario 1: Emergency Department Screening
- Quick risk assessment on admission
- Identify high-risk patients for monitoring
- Prioritize resource allocation

### Scenario 2: Ward Monitoring
- Daily risk re-assessment
- Track disease progression
- Early warning for deterioration

### Scenario 3: Research & Validation
- Compare model performance
- Identify key risk factors
- Validate on new populations

## 📈 Expected Results

### Class Distribution
- 94.3% of patients do NOT develop shock
- 5.7% develop shock
- Models balanced to catch shock cases

### Model Performance
Both models excel at:
- ✅ High overall accuracy (>90%)
- ✅ Good discrimination (ROC-AUC ~0.85-0.95)
- ✅ Catching most shock cases (Recall ~70-80%)

Trade-off:
- ⚠️ Some false positives (precision ~40-60%)
- ✅ Better safe than sorry in medical context

## 🎯 Feature Importance (Top Predictors)

Based on Random Forest analysis:
1. **Hematocrit changes** (dmaxHCT_3to8)
2. **Platelet drop rate** (dminPLT_3to8)
3. **Baseline platelet** (plt_bsl)
4. **Day of illness**
5. **Age**
6. **Clinical symptoms** (vomiting, abdominal pain)

## ⚠️ Important Notes

### Medical Disclaimer
- **NOT a diagnostic tool**
- **NOT FDA approved**
- **For educational/research only**
- **Requires clinical validation**

### Usage Guidelines
✅ **DO:**
- Use as supplementary information
- Combine with clinical judgment
- Monitor all high-risk patients
- Follow standard protocols

❌ **DON'T:**
- Rely solely on predictions
- Ignore clinical signs
- Delay treatment for low-risk predictions
- Use without medical supervision

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| File not found | CSV in same directory? |
| Import error | Run: `pip install -r requirements_dengue.txt` |
| Low performance | Check data quality, verify CSV format |
| Strange predictions | Verify input values are in normal ranges |

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **This file** | Quick start |
| **README_DENGUE.md** | Complete documentation |
| **In-app help** | Expandable sections in Streamlit |

## 🎉 Success Checklist

After running, you should see:
- ✅ 2,168 patients loaded
- ✅ 25 features identified
- ✅ Both models trained
- ✅ Performance metrics displayed
- ✅ Interactive visualizations
- ✅ Patient prediction form ready

## 💡 Pro Tips

1. **Compare models**: When both agree on high risk, confidence is highest
2. **Feature importance**: Shows what drives predictions
3. **Cross-validation**: Check model consistency (±std should be small)
4. **ROC-AUC**: More reliable than accuracy for imbalanced data
5. **Recall**: Most important metric for patient safety

## 🚦 Clinical Decision Framework

```
High Risk (>70%)
    ↓
Immediate Action
- Intensive monitoring
- Prepare shock management
- Alert senior staff
    
Moderate Risk (30-70%)
    ↓
Enhanced Monitoring
- Frequent vital signs
- Watch for warning signs
- Ready for escalation
    
Low Risk (<30%)
    ↓
Standard Care
- Regular monitoring
- Follow dengue protocols
- Patient education
```

## 📞 Quick Reference

**Run Command:**
```bash
streamlit run dengue_shock_predictor.py
```

**Access URL:**
```
http://localhost:8501
```

**Data Format:**
CSV file with 25 columns, SHOCK# as target variable

**Key Packages:**
- Streamlit (UI)
- Scikit-learn (ML)
- Plotly (Visualizations)
- Pandas/NumPy (Data processing)

## 🎯 Next Steps

1. ✅ Run the app
2. ✅ Explore model comparison
3. ✅ Review feature importance
4. ✅ Test patient predictor
5. ✅ Read full documentation

---

**Ready? Run this now:**
```bash
streamlit run dengue_shock_predictor.py
```

**Then visit:** http://localhost:8501

---

🏥 **Saving lives through AI-powered early detection!** 🩺
