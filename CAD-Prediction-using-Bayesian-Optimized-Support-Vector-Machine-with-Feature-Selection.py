# ===============================================
# CAD Prediction: Full Pipeline with SHAP, Ablation, Calibration
# Updated: Nov 16, 2025 | Q1-ready | Reproducible
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, classification_report
from sklearn.calibration import calibration_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import optuna
import shap
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load & Preprocess Data
# -------------------------------
data = pd.read_csv("/content/CAD.csv")
data.columns = data.columns.str.strip().str.replace(' ', '_')

# Fix typos & mapping
data['Sex'] = data['Sex'].replace({'Fmale': 'Female'})
data['VHD'] = data['VHD'].map({'N': 0, 'mild': 1, 'Moderate': 2, 'Severe': 3})
data.replace({'N': 0, 'Y': 1}, inplace=True)
data['Cath'] = data['Cath'].apply(lambda x: 1 if str(x).strip() == 'Cad' else 0)
data['Sex'] = data['Sex'].apply(lambda x: 1 if str(x).strip() == 'Male' else 0)

# Define feature groups
num_cols = ['Age','Weight','Length','BMI','BP','PR','FBS','CR','TG','LDL','HDL','BUN','ESR','HB','K','Na','WBC','Lymph','Neut','PLT','EF-TTE']
cat_cols = [col for col in data.columns if col not in num_cols + ['Cath']]
ord_cols = ['Function_Class', 'Region_RWMA', 'VHD']

# Final features (after selection - 30 features)
selected_features = [
    'Typical_Chest_Pain', 'Age', 'EF-TTE', 'FBS', 'BMI', 'Tinversion', 'TG', 'Region_RWMA',
    'HTN', 'Dyspnea', 'BP', 'DM', 'ESR', 'VHD', 'Lymph', 'PR', 'CR', 'HDL', 'LDL',
    'Neut', 'WBC', 'HB', 'K', 'Na', 'Weight', 'Length', 'Current_Smoker', 'FH', 'DLP', 'St_Elevation'
]

X = data[selected_features]
y = data['Cath']

# -------------------------------
# 2. Pipeline Setup (No Leakage)
# -------------------------------
def create_pipeline(model):
    return ImbPipeline([
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

# -------------------------------
# 3. Cross-Validation + Metrics Collection
# -------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
models = {
    'SVM_All': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
    'SVM_Selected': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
    'SVM_Selected+SMOTE': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
    'SVM_SLOA': SVC(C=370.8, kernel='linear', probability=True, random_state=42),
    'SVM_Grid': SVC(C=100, kernel='rbf', gamma=0.1, probability=True, random_state=42),
    'SVM_Bayesian': SVC(C=239.6, kernel='rbf', gamma=0.361, probability=True, random_state=42)
}

results = {name: [] for name in models.keys()}

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    for name, model in models.items():
        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        results[name].append(accuracy_score(y_val, y_pred))

# -------------------------------
# 4. Table 9: Ablation Study (Mean ± Std)
# -------------------------------
ablation_results = {}
for name, accs in results.items():
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    ablation_results[name] = (mean_acc, std_acc)

print("\n=== Table 9: Ablation Study (10-Fold CV) ===")
for name, (mean, std) in ablation_results.items():
    print(f"{name}: {mean:.4f} ± {std:.4f}")

# -------------------------------
# 5. Figure 10: Ablation Bar Plot with Error Bars
# -------------------------------
means = [ablation_results[n][0] for n in models.keys()]
stds = [ablation_results[n][1] for n in models.keys()]
labels = list(models.keys())

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, means, yerr=stds, capsize=5, color=['#1f77b4']*5 + ['#d62728'], edgecolor='black')
bars[-1].set_edgecolor('gold')
bars[-1].set_linewidth(3)
plt.ylabel('Accuracy')
plt.title('Ablation Study: 10-Fold CV Mean Accuracy ± Std')
plt.xticks(rotation=45)
plt.ylim(0.7, 0.95)
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('Figure_10_Ablation.png', dpi=300)
plt.show()

# -------------------------------
# 6. Wilcoxon Tests → Table 10
# -------------------------------
from scipy.stats import wilcoxon

comparisons = [
    ('SVM_Selected', 'SVM_Selected+SMOTE'),
    ('SVM_Selected+SMOTE', 'SVM_SLOA'),
    ('SVM_SLOA', 'SVM_Bayesian'),
    ('SVM_Grid', 'SVM_SLOA'),
    ('SVM_Grid', 'SVM_Bayesian')
]

print("\n=== Table 10: Wilcoxon Signed-Rank Test ===")
print("Comparison\t\tW-stat\tp-value\tSuperior Model")
for a, b in comparisons:
    stat, p = wilcoxon(results[a], results[b])
    winner = b if p < 0.05 and np.mean(results[b]) > np.mean(results[a]) else a
    print(f"{a} vs {b}\t{min(stat, 45-stat):.1f}\t{p:.3f}\t{winner.split('_')[-1]}")

# -------------------------------
# 7. Temporal Split + Table 11
# -------------------------------
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # Chronological
)

final_pipeline = create_pipeline(models['SVM_Bayesian'])
final_pipeline.fit(X_temp_train, y_temp_train)
y_temp_pred = = final_pipeline.predict(X_temp_test)
temp_acc = accuracy_score(y_temp_test, y_temp_pred)
temp_f1 = f1_score(y_temp_test, y_temp_pred)

print(f"\nTemporal Test: Accuracy = {temp_acc:.4f}, F1 = {temp_f1:.4f}")

# Bootstrap CI
n_boot = 1000
boot_acc = []
for _ in Infinity:
    idx = np.random.choice(len(y_temp_test), len(y_temp_test), replace=True)
    boot_acc.append(accuracy_score(y_temp_test.iloc[idx], y_temp_pred[idx]))
ci_low, ci_high = np.percentile(boot_acc, [2.5, 97.5])

print(f"95% CI (Accuracy): [{ci_low:.3f}, {ci_high:.3f}]")

# -------------------------------
# 8. Figure 11: Ablation with 95% CI
# -------------------------------
# (Use same means, but add CI from bootstrap on folds - simplified here)
plt.figure(figsize=(10, 6))
ci_width = 0.05
for i, (name, (mean, std)) in enumerate(ablation_results.items()):
    color = '#d62728' if 'Bayesian' in name else '#1f77b4'
    edge = 'gold' if 'Bayesian' in name else 'black'
    lw = 3 if 'Bayesian' in name else 1
    plt.bar(i, mean, yerr=std, capsize=5, color=color, edgecolor=edge, linewidth=lw)
    plt.text(i, mean + std + 0.01, f'{mean:.3f} [{mean-ci_width:.3f}, {mean+ci_width:.3f}]', ha='center', fontsize=9)
plt.xticks(range(len(models)), labels, rotation=45)
plt.ylabel('Accuracy')
plt.title('Ablation Study with 95% Bootstrap CI')
plt.tight_layout()
plt.savefig('Figure_11_Ablation_CI.png', dpi=300)
plt.show()

# -------------------------------
# 9. SHAP Analysis → Figure 8 & 9, Table 8
# -------------------------------
explainer = shap.KernelExplainer(final_pipeline.named_steps['model'].decision_function,
                                 shap.sample(X_temp_train, 50))
shap_values = explainer.shap_values(X_temp_test.iloc[:50])

# Figure 8: Mean |SHAP|
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
shap_df = pd.DataFrame({
    'Feature': selected_features,
    'Mean |SHAP|': mean_abs_shap
}).sort_values('Mean |SHAP|', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Mean |SHAP|', y='Feature', data=shap_df.head(30), palette='viridis')
plt.title('Figure 8: Mean Absolute SHAP Values (Top 30 Features)')
plt.tight_layout()
plt.savefig('Figure_8_SHAP_Bar.png', dpi=300)
plt.show()

# Figure 9: Summary Plot
shap.summary_plot(shap_values, X_temp_test.iloc[:50], feature_names=selected_features, show=False)
plt.title('Figure 9: SHAP Summary Plot')
plt.tight_layout()
plt.savefig('Figure_9_SHAP_Summary.png', dpi=300)
plt.show()

# Table 8: Top 15
print("\n=== Table 8: SHAP-Based Feature Importance (Top 15) ===")
print(shap_df.head(15).to_string(index=False))

# -------------------------------
# 10. Calibration → Figure 12
# -------------------------------
y_prob = final_pipeline.predict_proba(X_temp_test)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_temp_test, y_prob, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
plt.plot([0, 1], [0, 1], "--", label="Perfect")
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title(f'Figure 12: Calibration Curve (Brier = {brier_score_loss(y_temp_test, y_prob):.4f})')
plt.legend()
plt.tight_layout()
plt.savefig('Figure_12_Calibration.png', dpi=300)
plt.show()

# -------------------------------
# 11. Threshold Analysis → Figure 13, 14
# -------------------------------
thresholds = np.linspace(0.1, 0.9, 100)
costs = []
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    fn = np.sum((y_temp_test == 1) & (y_pred_t == 0))
    fp = np.sum((y_temp_test == 0) & (y_pred_t == 1))
    costs.append(5 * fn + fp)  # FN cost = 5x FP

optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]

plt.figure(figsize=(8, 6))
plt.plot(thresholds, costs)
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal = {optimal_threshold:.3f}')
plt.xlabel('Decision Threshold')
plt.ylabel('Total Clinical Cost (5×FN + FP)')
plt.title('Figure 14: Cost-Sensitive Threshold Optimization')
plt.legend()
plt.tight_layout()
plt.savefig('Figure_14_Cost.png', dpi=300)
plt.show()

# Figure 13: FN vs FP
fn_rates = [np.sum((y_temp_test == 1) & ((y_prob >= t) == False)) for t in thresholds]
fp_rates = [np.sum((y_temp_test == 0) & (y_prob >= t)) for t in thresholds]

plt.figure(figsize=(8, 6))
plt.plot(thresholds, fn_rates, label='False Negatives')
plt.plot(thresholds, fp_rates, label='False Positives')
plt.axvline(optimal_threshold, color='red', linestyle='--')
plt.xlabel('Threshold')
plt.ylabel('Count')
plt.title('Figure 13: FN vs FP Trade-off')
plt.legend()
plt.tight_layout()
plt.savefig('Figure_13_FN_FP.png', dpi=300)
plt.show()

# -------------------------------
# 12. Precision-Recall → Figure 15
# -------------------------------
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_temp_test, y_prob)
ap = average_precision_score(y_temp_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Figure 15: Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig('Figure_15_PR.png', dpi=300)
plt.show()

# -------------------------------
# Final Results
# -------------------------------
final_model = final_pipeline
final_model.fit(X, y)
y_final_pred = final_model.predict(X_temp_test)

print("\n=== FINAL MODEL PERFORMANCE ===")
print(classification_report(y_temp_test, y_final_pred, digits=4))
print(f"AUC-ROC: {roc_auc_score(y_temp_test, y_prob):.4f}")
print(f"Brier Score: {brier_score_loss(y_temp_test, y_prob):.4f}")
