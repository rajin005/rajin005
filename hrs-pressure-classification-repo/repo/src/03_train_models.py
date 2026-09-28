#!/usr/bin/env python3
"""
Model Training & Evaluation
Trains MLP classifier + 3 baselines and evaluates on test set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

print("=" * 80)
print("STEP 3: MODEL TRAINING & EVALUATION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\nLoading data...")
train_df = pd.read_csv('data/processed/train_set.csv')
val_df = pd.read_csv('data/processed/val_set.csv')
test_df = pd.read_csv('data/processed/test_set.csv')

feature_cols = [col for col in train_df.columns if col not in ['time', 'pressure_state']]
target_col = 'pressure_state'

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_val = val_df[feature_cols].values
y_val = val_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

print(f"✓ Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"✓ Val:   {X_val.shape[0]} samples")
print(f"✓ Test:  {X_test.shape[0]} samples")

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)
class_names = le.classes_

print(f"✓ Classes: {list(class_names)}")

# ============================================================================
# STEP 2: TRAIN MLP CLASSIFIER
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING MLP CLASSIFIER")
print("=" * 80)

model_mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42,
    verbose=0
)

print("Model architecture:")
print(f"  Input features: {X_train.shape[1]}")
print(f"  Hidden layers: (64, 32)")
print(f"  Output classes: {len(class_names)}")
print(f"  Activation: relu")

model_mlp.fit(X_train, y_train_encoded)

print(f"✓ Training complete. Epochs: {model_mlp.n_iter_}")

# ============================================================================
# STEP 3: TRAIN BASELINE MODELS
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING BASELINE MODELS")
print("=" * 80)

# Logistic Regression
print("\n1. Logistic Regression...")
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train_encoded)
print("   ✓ Done")

# Random Forest
print("2. Random Forest...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train_encoded)
print("   ✓ Done")

# Decision Tree
print("3. Decision Tree...")
model_dt = DecisionTreeClassifier(max_depth=10, random_state=42)
model_dt.fit(X_train, y_train_encoded)
print("   ✓ Done")

# ============================================================================
# STEP 4: EVALUATE ALL MODELS ON TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION ON TEST SET")
print("=" * 80)

models = {
    'MLP': model_mlp,
    'Logistic Regression': model_lr,
    'Random Forest': model_rf,
    'Decision Tree': model_dt
}

results = {}

for model_name, model in models.items():
    print(f"\n{model_name}:")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Per-class metrics
    print(f"  Per-class metrics:")
    for i, class_name in enumerate(class_names):
        class_mask = y_test_encoded == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_test_encoded[class_mask], y_pred[class_mask])
            class_prec = precision_score(y_test_encoded[class_mask], y_pred[class_mask], 
                                        average='binary', zero_division=0) if len(class_names) == 2 else \
                        precision_score(y_test_encoded, y_pred, labels=[i], average='micro', zero_division=0)
            class_recall = recall_score(y_test_encoded[class_mask], y_pred[class_mask],
                                       average='binary', zero_division=0) if len(class_names) == 2 else \
                         recall_score(y_test_encoded, y_pred, labels=[i], average='micro', zero_division=0)
            
            print(f"    {class_name}: acc={class_acc:.3f}, prec={class_prec:.3f}, recall={class_recall:.3f}")
    
    results[model_name] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'cm': confusion_matrix(y_test_encoded, y_pred)
    }

# ============================================================================
# STEP 5: SAVE METRICS TO CSV
# ============================================================================

print("\n" + "=" * 80)
print("SAVING METRICS")
print("=" * 80)

metrics_data = []
for model_name, metrics in results.items():
    metrics_data.append({
        'Model': model_name,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1']
    })

metrics_df = pd.DataFrame(metrics_data)
os.makedirs('results', exist_ok=True)
metrics_file = 'results/model_comparison.csv'
metrics_df.to_csv(metrics_file, index=False)
print(f"✓ Saved metrics: {metrics_file}")

# ============================================================================
# STEP 6: GENERATE CONFUSION MATRICES (2D Heatmaps)
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING CONFUSION MATRICES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Confusion Matrices - Test Set', fontsize=14, fontweight='bold')

for idx, (model_name, metrics) in enumerate(results.items()):
    ax = axes[idx // 2, idx % 2]
    cm = metrics['cm']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=False)
    ax.set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
cm_file = 'results/confusion_matrices.png'
plt.savefig(cm_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved confusion matrices: {cm_file}")

# ============================================================================
# STEP 7: SAVE MODEL COMPARISON VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING MODEL COMPARISON PLOTS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy comparison
models_list = metrics_df['Model'].tolist()
accuracies = metrics_df['Accuracy'].tolist()
axes[0].bar(models_list, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison (Test Set)')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# F1-score comparison
f1_scores = metrics_df['F1-Score'].tolist()
axes[1].bar(models_list, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[1].set_ylabel('F1-Score')
axes[1].set_title('Model F1-Score Comparison (Test Set)')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
curves_file = 'results/model_comparison_plots.png'
plt.savefig(curves_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved model comparison plots: {curves_file}")

# ============================================================================
# STEP 8: SAVE MODELS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Save MLP
pickle.dump(model_mlp, open(os.path.join(model_dir, 'mlp_classifier.pkl'), 'wb'))
print(f"✓ Saved MLP model")

# Save baselines
pickle.dump(model_lr, open(os.path.join(model_dir, 'logistic_regression.pkl'), 'wb'))
print(f"✓ Saved Logistic Regression")

pickle.dump(model_rf, open(os.path.join(model_dir, 'random_forest.pkl'), 'wb'))
print(f"✓ Saved Random Forest")

pickle.dump(model_dt, open(os.path.join(model_dir, 'decision_tree.pkl'), 'wb'))
print(f"✓ Saved Decision Tree")

# Save label encoder
pickle.dump(le, open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb'))
print(f"✓ Saved Label Encoder")

# ============================================================================
# STEP 9: GENERATE DETAILED CLASSIFICATION REPORT
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT (MLP on Test Set)")
print("=" * 80)

y_pred_mlp = results['MLP']['y_pred']
print("\n" + classification_report(y_test_encoded, y_pred_mlp, 
                                   target_names=class_names, digits=4))

# ============================================================================
# STEP 10: SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)

print(f"\nTest Set Performance:")
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 70)
for _, row in metrics_df.iterrows():
    print(f"{row['Model']:<25} {row['Accuracy']:<12.4f} {row['Precision']:<12.4f} {row['Recall']:<12.4f} {row['F1-Score']:<12.4f}")

print(f"\nBest Model: {metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']}")
print(f"Best Accuracy: {metrics_df['Accuracy'].max():.4f}")

print(f"\nOutput Files:")
print(f"  1. {metrics_file}")
print(f"  2. {cm_file}")
print(f"  3. {curves_file}")
print(f"  4. {model_dir}/mlp_classifier.pkl")
print(f"  5. {model_dir}/logistic_regression.pkl")
print(f"  6. {model_dir}/random_forest.pkl")
print(f"  7. {model_dir}/decision_tree.pkl")

print("\n" + "=" * 80)
print("✓ MODEL TRAINING COMPLETE")
print("=" * 80)

print("\nNext steps:")
print("  1. Review model comparison metrics")
print("  2. Review confusion matrices")
print("  3. Select best model (MLP)")
print("  4. Prepare revised manuscript with results")

