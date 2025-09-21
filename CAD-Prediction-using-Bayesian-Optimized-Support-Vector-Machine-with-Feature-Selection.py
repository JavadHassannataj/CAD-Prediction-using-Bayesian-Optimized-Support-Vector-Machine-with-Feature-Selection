import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
data = pd.read_csv("/content/CAD.csv")
data.head(10)
data.info()
data.describe().T
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(' ', '_')
data.columns

target = data['Cath'].value_counts()

target.plot.bar()

data.isna().sum().sum()

print(f'There are {data.duplicated().sum()} duplicate rows')

from matplotlib import pyplot as plt
import seaborn as sns
plt.figure(figsize=(40,40))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title("Correlation Matrix Heatmap", fontsize=20)
plt.show()

# Numerical variables:
num_cols = ['Age','Weight', 'Length','BMI', 'BP', 'PR', 'FBS', 'CR', 'TG', 'LDL', 'HDL', 'BUN', 'ESR', 'HB', 'K', 'Na', 'WBC', 'Lymph', 'Neut', 'PLT', 'EF-TTE']

# Categorical variables:
cat_cols = ['Sex', 'DM', 'HTN', 'Current_Smoker' ,'EX-Smoker', 'FH', 'Obesity', 'CRF', 'CVA', 'Airway_disease', 'Thyroid_Disease', 'CHF', 'DLP', 'Edema', 'Weak_Peripheral_Pulse', 'Lung_rales', 'Systolic_Murmur', 'Diastolic_Murmur', 'Typical_Chest_Pain', 'Dyspnea', 'Atypical', 'Nonanginal', 'Exertional_CP', 'LowTH_Ang', 'Q_Wave', 'St_Elevation', 'St_Depression', 'Tinversion', 'LVH', 'Poor_R_Progression', 'Cath']

# Ordinal variables
ord_cols = ['Function_Class', "Region_RWMA", "VHD"]

print(f"[Unique Values in {len(cat_cols)} Categorical Variables]\n")

for cat_col in cat_cols:
    print(f"{cat_col}:{data[cat_col].nunique()} Unique Values => {data[cat_col].unique()}")

print(f"[Unique Values in {len(ord_cols)} Ordinal Variables]\n")

for ord_col in ord_cols:
    print("* {} : {} Unique Values =>".format(ord_col, data[ord_col].nunique()), data[ord_col].unique())

print(f"[Unique Values in {len(num_cols)} Numerical Variables]\n")

for num_col in num_cols:
    print("* {} : {} Unique Values".format(num_col, data[num_col].nunique()))

data[num_cols].describe(percentiles=[0.1, 0.25, 0.75, 0.9, 0.95])

vhd = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
sex = {"Male": "Male", "Fmale": "Female"}

data['VHD'] = data['VHD'].map(vhd)
data['Sex'] = data['Sex'].map(sex)

data.replace('N', 0, inplace=True)
data.replace('Y', 1, inplace=True)

print(f"[Unique Values in {len(cat_cols)} Categorical Variables]\n")
for cat_col in cat_cols:
    print(f"{cat_col}:{data[cat_col].nunique()} Unique Values => {data[cat_col].unique()}")

print(f"[Unique Values in {len(ord_cols)} Ordinal Variables]\n")

data['Cath'] = [1 if i.strip() == "Cad" else 0 for i in data['Cath']]
data['Sex'] = [1 if i.strip() == "Male" else 0 for i in data['Sex']]
data.head(5)
label = data['Cath'].value_counts()
label.plot.pie()
X_1 = data.drop("Cath", axis=1)
y_1 = data['Cath']
columns = X_1.columns.tolist()
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
print(scaler.fit(X_1))
X_1 = scaler.transform(X_1)
X_1
new_df = pd.DataFrame(X_1, columns = columns)
new_df["Cath"] = y_1
new_df

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

ada_model = AdaBoostClassifier()
ada_model.fit(X_1, y_1)

# Step 2: Extract feature importance from Ada
ada_feature_importance = ada_model.feature_importances_

# Step 3: Train an dt model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_1, y_1)

# Step 4: Extract feature importance from dt
dt_feature_importance = dt_model.feature_importances_

combined_feature_importance = (ada_feature_importance + dt_feature_importance) / 2

# Step 6: Rank the features
sorted_indices = np.argsort(combined_feature_importance)[::-1]  # Sort indices in descending order
sorted_features = [columns[i] for i in sorted_indices]

# Print feature importance scores and ranked features
print("Feature Importance Scores:")
for feature, importance in zip(columns, combined_feature_importance):
    print(f"{feature}: {importance}")

print("\nRanked Features:")
for rank, feature in enumerate(sorted_features, start=1):
    print(f"Rank {rank}: {feature}")
feature_importance_dict = {}
for feature, importance in zip(columns, combined_feature_importance):
    feature_importance_dict[feature] = importance

# Print feature importance scores
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")
feature_importance_list = [(feature, importance) for feature, importance in feature_importance_dict.items()]

# Print the list
print("Feature Importance Scores:")
for feature, importance in feature_importance_list:
    print(f"{feature}: {importance}")
from matplotlib import pyplot as plt

# Unpack feature names and importance scores from the list of tuples
features, importance_scores = zip(*feature_importance_list)

plt.figure(figsize=(15, 6))
plt.bar(range(len(features)), importance_scores, align='center')
plt.xticks(range(len(features)), features, rotation=90)
plt.xlabel('Features')
plt.ylabel('Features Importance')
plt.title('Features Importances')
plt.tight_layout()
plt.show()

columns_to_drop  = ['Sex','Diastolic_Murmur','Systolic_Murmur','Dyspnea','Function_Class','Obesity','LVH','PLT','HB','St_Depression','Exertional_CP','Lung_rales','Thyroid_Disease','Edema','EX-Smoker','CVA','Airway_disease','Q_Wave','Lung_rales','Poor_R_Progression','CRF','LowTH_Ang','Weak_Peripheral_Pulse','LowTH_Ang','Exertional_CP','WBC','CHF','Cath']
X = new_df.drop(columns_to_drop, axis=1)
y = new_df['Cath']
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
print(scaler.fit(X))
X = scaler.transform(X)
X
import smote_variants as sv
oversampler= sv.distance_SMOTE()
X_res,y_res = oversampler.sample(X, y)
X_res.shape
!pip install smote_variants
X_res.shape
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10,shuffle=True, random_state = 42)

for train_ix, test_ix in kfold.split(X_res, y_res):
    x_train, x_test = X_res[train_ix], X_res[test_ix]
    y_train, y_test = y_res[train_ix], y_res[test_ix]


print('X_train:',x_train.shape)

print('X_test:',x_test.shape)

print('y_train:',y_train.shape)

print('y_test:',y_test.shape)
from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted))
!pip install optuna
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def objective(trial):
    # Define the hyperparameters to optimize
    C = trial.suggest_loguniform('C', 1e-3, 1e3)
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else None
    gamma = trial.suggest_loguniform('gamma', 1e-3, 1e3) if kernel in ['rbf', 'poly', 'sigmoid'] else 'scale'

    if kernel == 'poly':
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    else:
        model = SVC(C=C, kernel=kernel, gamma=gamma)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_accuracy = study.best_value

print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)

best_model = SVC(**best_params)
best_model.fit(x_train, y_train)

y_pred = best_model.predict(x_test)
final_accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:", final_accuracy)


from sklearn.linear_model import LogisticRegression

def objective(trial):
    # Define the hyperparameters to optimize
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    C = trial.suggest_loguniform('C', 1e-3, 1e3)

    # Create and train the Logistic Regression model with the specified hyperparameters
    model_lg = LogisticRegression(penalty=penalty, C=C, solver='liblinear')
    model_lg.fit(x_train, y_train)

    # Evaluate the model
    y_pred_lg = model_lg.predict(x_test)
    accuracy_lg = accuracy_score(y_test, y_pred_lg)

    return accuracy_lg

# Create and optimize the study using Optuna
study_lg = optuna.create_study(direction='maximize')
study_lg.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params_lg = study_lg.best_params
best_accuracy_lg = study_lg.best_value

print("Best Hyperparameters:", best_params_lg)
print("Best Accuracy:", best_accuracy_lg)



def objective(trial):
    # Define the hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Create and train the Random Forest model with the specified hyperparameters
    model_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      criterion=criterion, random_state=42)
    model_rf.fit(x_train, y_train)

    # Evaluate the model
    y_pred_rf = model_rf.predict(x_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    return accuracy_rf

# Create and optimize the study using Optuna
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params_rf = study_rf.best_params
best_accuracy_rf = study_rf.best_value

print("Best Hyperparameters:", best_params_rf)
print("Best Accuracy:", best_accuracy_rf)
!pip install mealpy
from sklearn import metrics
from mealpy import FloatVar, StringVar, SLO, Problem
from sklearn.svm import SVC
class SvmOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        C_paras, kernel_paras = x_decoded["C_paras"], x_decoded["kernel_paras"]

        svc = SVC(C=C_paras, kernel=kernel_paras, random_state=1)
        svc.fit(x_train, y_train)
        y_predict = svc.predict(x_test)
        return metrics.accuracy_score(y_test, y_predict)


data = [x_train, y_train,y_test,x_test]
my_bounds = [
    FloatVar(lb=0.01, ub=1000., name="C_paras"),
    FloatVar(lb=0.01, ub=1000., name="gama"),
    StringVar(valid_sets=('linear', 'poly', 'rbf', 'sigmoid'), name="kernel_paras")
]
problem = SvmOptimizedProblem(bounds=my_bounds, minmax="max", data=data)

model = SLO.ImprovedSLO(epoch=50, pop_size=50)
model.solve(problem)

print(f"Best agent: {model.g_best}")
print(f"Best solution: {model.g_best.solution}")
print(f"Best accuracy: {model.g_best.target.fitness}")
print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")


from xgboost import XGBClassifier as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
estimators = []
estimators.append(('LogisticRegression', LogisticRegression(penalty = 'l2', C = 58.48737264443094)))
estimators.append(('RandomForest', RandomForestClassifier(n_estimators = 380, max_depth = 22, min_samples_split = 2, min_samples_leaf = 13, criterion = 'gini')))
estimators.append(('Ada Boost Classifier', AdaBoostClassifier(n_estimators = 339, learning_rate = 0.009492811483791653)))
estimators.append(('Support Vector Machine', SVC(C = 239.59501536334488, kernel = 'rbf', gamma = 0.36055928693321015) ))
SVM_OPT = SVC(C = 239.59501536334488, kernel = 'rbf', gamma = 0.36055928693321015)

from sklearn.ensemble import StackingClassifier
SC_smote = StackingClassifier(estimators=estimators,final_estimator = SVM_OPT,cv=5)
SC_smote.fit(x_train, y_train)
y_pred = SC_smote.predict(x_test)

print(f"\nStacking classifier training Accuracy: {SC_smote.score(x_train, y_train):0.2f}")
print(f"Stacking classifier test Accuracy: {SC_smote.score(x_test, y_test):0.2f}")
models={
    "LRG": LogisticRegression(penalty = 'l2', C = 58.48737264443094),
    "SVM_optuna": SVC(C = 239.59501536334488, kernel = 'rbf', gamma = 0.36055928693321015),
    "SVM_slo": SVC(C = 370.79954615548917, kernel = 'linear'),
    "RFC": RandomForestClassifier(n_estimators = 380, max_depth = 22, min_samples_split = 2, min_samples_leaf = 13, criterion = 'gini'),
}
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def test_model_performance(models, x_train, x_test, y_train, y_test):
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)

        accuracy = accuracy_score(y_test, predicted) * 100
        sensitivity = recall_score(y_test, predicted) * 100
        precision = precision_score(y_test, predicted) * 100
        f1score = f1_score(y_test, predicted) * 100

        print(f'{model_name}:  Accuracy - {accuracy:.2f}%, Recall - {sensitivity:.2f}%, '
              f'Precision - {precision:.2f}%, F1_Score - {f1score:.2f}%')

test_model_performance(models, x_train, x_test, y_train, y_test)

SVM_B = SVC(C = 239.59501536334488, kernel = 'rbf', gamma = 0.36055928693321015,probability=True)
SVM_B.fit(x_train, y_train)
from sklearn.metrics import roc_auc_score, roc_curve, auc

y_pred_prob1 = SVM_B.predict_proba(x_test)[:, 1]


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob1)

roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
SVM_slo = SVC(C = 370.79954615548917, kernel = 'linear',probability=True)
SVM_slo.fit(x_train, y_train)
from sklearn.metrics import roc_auc_score, roc_curve, auc

y_pred_prob2 = SVM_slo.predict_proba(x_test)[:, 1]


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob2)

roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()