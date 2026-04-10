import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic operational risk dataset
n_samples = 5000
data = {
    'transaction_amount': np.random.exponential(10000, n_samples),
    'transaction_frequency': np.random.poisson(50, n_samples),
    'employee_tenure_months': np.random.randint(1, 240, n_samples),
    'num_failed_controls': np.random.poisson(2, n_samples),
    'system_downtime_hours': np.random.exponential(5, n_samples),
    'audit_findings_count': np.random.poisson(3, n_samples),
    'compliance_violations': np.random.poisson(1, n_samples),
    'vendor_risk_score': np.random.uniform(0, 100, n_samples),
    'process_complexity_score': np.random.uniform(1, 10, n_samples),
    'manual_intervention_rate': np.random.uniform(0, 1, n_samples),
    'staff_turnover_rate': np.random.uniform(0, 0.5, n_samples),
    'training_hours_deficit': np.random.exponential(10, n_samples),
    'incident_history_count': np.random.poisson(2, n_samples),
    'department': np.random.choice(['Trading', 'Operations', 'IT', 'Compliance', 'HR'], n_samples),
    'region': np.random.choice(['North America', 'Europe', 'Asia', 'LATAM'], n_samples)
}

df = pd.DataFrame(data)

risk_probability = (
    0.1 * (df['num_failed_controls'] > 2).astype(int) +
    0.15 * (df['compliance_violations'] > 1).astype(int) +
    0.1 * (df['audit_findings_count'] > 4).astype(int) +
    0.1 * (df['vendor_risk_score'] > 70).astype(int) +
    0.1 * (df['manual_intervention_rate'] > 0.6).astype(int) +
    0.1 * (df['system_downtime_hours'] > 10).astype(int) +
    0.05 * (df['staff_turnover_rate'] > 0.3).astype(int) +
    0.1 * (df['incident_history_count'] > 3).astype(int) +
    0.05 * (df['process_complexity_score'] > 7).astype(int) +
    np.random.uniform(0, 0.15, n_samples)
)
df['risk_event'] = (risk_probability > 0.35).astype(int)

print("="*60)
print("OPERATIONAL RISK DATASET OVERVIEW")
print("="*60)
print(f"\nDataset Shape: {df.shape}")
print(f"\nTarget Distribution:")
print(df['risk_event'].value_counts(normalize=True).round(3))

le_dept = LabelEncoder()
le_region = LabelEncoder()
df['department_encoded'] = le_dept.fit_transform(df['department'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['transaction_amount', 'transaction_frequency', 'employee_tenure_months',
                'num_failed_controls', 'system_downtime_hours', 'audit_findings_count',
                'compliance_violations', 'vendor_risk_score', 'process_complexity_score',
                'manual_intervention_rate', 'staff_turnover_rate', 'training_hours_deficit',
                'incident_history_count', 'department_encoded', 'region_encoded']

X = df[feature_cols]
y = df['risk_event']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")