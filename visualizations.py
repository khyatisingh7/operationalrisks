from metrics_comparison import *

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ROC Curves
ax1 = axes[0, 0]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
ax1.plot(rf_fpr, rf_tpr, 'b-', linewidth=2, label=f'Random Forest (AUC = {rf_metrics["ROC-AUC"]:.4f})')
ax1.plot(xgb_fpr, xgb_tpr, 'r-', linewidth=2, label=f'XGBoost (AUC = {xgb_metrics["ROC-AUC"]:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax1.fill_between(xgb_fpr, xgb_tpr, alpha=0.2, color='red')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Metrics Bar Chart
ax2 = axes[0, 1]
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics_to_plot))
width = 0.35
rf_vals = [rf_metrics[m] for m in metrics_to_plot]
xgb_vals = [xgb_metrics[m] for m in metrics_to_plot]
bars1 = ax2.bar(x - width/2, rf_vals, width, label='Random Forest', color='steelblue', edgecolor='black')
bars2 = ax2.bar(x + width/2, xgb_vals, width, label='XGBoost', color='coral', edgecolor='black')
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_to_plot, fontsize=10)
ax2.legend(fontsize=10)
ax2.set_ylim(0.6, 1.0)
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    height = bar.get_height()
    ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

# Confusion Matrices
ax3 = axes[1, 0]
rf_cm = confusion_matrix(y_test, rf_predictions)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
ax3.set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
ax3.set_xlabel('Predicted', fontsize=12)
ax3.set_ylabel('Actual', fontsize=12)

ax4 = axes[1, 1]
xgb_cm = confusion_matrix(y_test, xgb_predictions)
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges', ax=ax4,
            xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
ax4.set_title('XGBoost - Confusion Matrix', fontsize=14, fontweight='bold')
ax4.set_xlabel('Predicted', fontsize=12)
ax4.set_ylabel('Actual', fontsize=12)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()