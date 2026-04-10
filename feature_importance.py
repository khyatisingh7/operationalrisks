from model_training import *

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], color='steelblue')
axes[0].set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Importance', fontsize=12)

xgb_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'], color='coral')
axes[1].set_title('XGBoost - Feature Importance', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importance', fontsize=12)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("TOP 5 MOST IMPORTANT FEATURES")
print("="*60)
print("\nRandom Forest:")
for i, row in rf_importance.tail(5).iloc[::-1].iterrows():
    print(f"  • {row['Feature']}: {row['Importance']:.4f}")

print("\nXGBoost:")
for i, row in xgb_importance.tail(5).iloc[::-1].iterrows():
    print(f"  • {row['Feature']}: {row['Importance']:.4f}")