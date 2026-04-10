from model_training import *

def calculate_metrics(y_true, y_pred, y_proba, model_name):
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'Specificity': recall_score(y_true, y_pred, pos_label=0)
    }
    return metrics

rf_metrics = calculate_metrics(y_test, rf_predictions, rf_proba, 'Random Forest')
xgb_metrics = calculate_metrics(y_test, xgb_predictions, xgb_proba, 'XGBoost')

metrics_df = pd.DataFrame([rf_metrics, xgb_metrics])
metrics_df = metrics_df.set_index('Model')

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)
print("\n" + metrics_df.round(4).to_string())

print("\n" + "-"*60)
print("METRIC-BY-METRIC WINNER:")
print("-"*60)
for col in metrics_df.columns:
    rf_val = metrics_df.loc['Random Forest', col]
    xgb_val = metrics_df.loc['XGBoost', col]
    winner = 'Random Forest' if rf_val > xgb_val else 'XGBoost' if xgb_val > rf_val else 'Tie'
    diff = abs(rf_val - xgb_val)
    print(f"{col:15} → {winner} (difference: {diff:.4f})")