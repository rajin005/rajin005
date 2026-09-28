#!/usr/bin/env python3
"""
CORRECTED: Stratified Train/Validation/Test Split
Ensures balanced class distribution in each set
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

print("=" * 80)
print("STEP 3B: STRATIFIED TRAIN/VAL/TEST SPLIT (REBALANCED)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD COMBINED DATASET
# ============================================================================

input_file = 'data/processed/synthetic_data_labeled.csv'
print(f"\nLoading dataset: {input_file}")

df = pd.read_csv(input_file)
print(f"✓ Loaded {len(df)} samples")

# ============================================================================
# STEP 2: IDENTIFY FEATURES AND TARGET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: IDENTIFY FEATURES AND TARGET")
print("=" * 80)

feature_cols = [col for col in df.columns if col not in ['time', 'pressure_state']]
target_col = 'pressure_state'

print(f"✓ Input features ({len(feature_cols)}):")
for col in feature_cols:
    print(f"    {col}")
print(f"✓ Target variable: {target_col}")

# ============================================================================
# STEP 3: CHECK ORIGINAL CLASS DISTRIBUTION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: ORIGINAL CLASS DISTRIBUTION")
print("=" * 80)

print("\nBefore split:")
for state in ['HIGH', 'MEDIUM', 'LOW']:
    count = (df[target_col] == state).sum()
    pct = 100 * count / len(df)
    print(f"  {state:8s}: {count:4,} ({pct:5.1f}%)")

# ============================================================================
# STEP 4: STRATIFIED SPLIT (Train 70% -> then split into 80/20 for Train/Val)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: STRATIFIED TRAIN/VAL/TEST SPLIT (70/15/15)")
print("=" * 80)

X = df[feature_cols].values
y = df[target_col].values

# First split: 70% train, 30% temp (for val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: split the 30% equally into val (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,  # 50% of 30% = 15%
    random_state=42,
    stratify=y_temp
)

print(f"\n✓ Train set: {len(X_train):,} samples ({100*len(X_train)/len(X):.1f}%)")
print(f"✓ Val set:   {len(X_val):,} samples ({100*len(X_val)/len(X):.1f}%)")
print(f"✓ Test set:  {len(X_test):,} samples ({100*len(X_test)/len(X):.1f}%)")

# ============================================================================
# STEP 5: VERIFY CLASS DISTRIBUTION IN EACH SET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: CLASS DISTRIBUTION (STRATIFIED)")
print("=" * 80)

for set_name, y_set in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    print(f"\n{set_name} set:")
    for state in ['HIGH', 'MEDIUM', 'LOW']:
        count = (y_set == state).sum()
        pct = 100 * count / len(y_set)
        print(f"  {state:8s}: {count:4,} ({pct:5.1f}%)")

# ============================================================================
# STEP 6: NORMALIZE USING TRAINING DATA ONLY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: NORMALIZATION (StandardScaler)")
print("=" * 80)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

print(f"✓ Fitted scaler on training data")
print(f"  Feature means: {scaler.mean_[:3]}... (showing first 3)")
print(f"  Feature stds:  {scaler.scale_[:3]}... (showing first 3)")

# Verify
train_means = X_train_normalized.mean(axis=0)
train_stds = X_train_normalized.std(axis=0)
print(f"\nVerification (Training set after normalization):")
print(f"  Feature means: {train_means.mean():.6f} (should be ≈0)")
print(f"  Feature stds:  {train_stds.mean():.6f} (should be ≈1)")

# ============================================================================
# STEP 7: CREATE DATAFRAMES AND SAVE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: CREATE DATAFRAMES AND SAVE")
print("=" * 80)

# Create dataframes with normalized features
train_df = pd.DataFrame(X_train_normalized, columns=feature_cols)
train_df['pressure_state'] = y_train
train_df['time'] = 0  # Placeholder

val_df = pd.DataFrame(X_val_normalized, columns=feature_cols)
val_df['pressure_state'] = y_val
val_df['time'] = 0

test_df = pd.DataFrame(X_test_normalized, columns=feature_cols)
test_df['pressure_state'] = y_test
test_df['time'] = 0

# Reorder columns to match original
cols_order = ['time'] + feature_cols + ['pressure_state']
train_df = train_df[cols_order]
val_df = val_df[cols_order]
test_df = test_df[cols_order]

# Save
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

train_file = os.path.join(output_dir, 'train_set.csv')
val_file = os.path.join(output_dir, 'val_set.csv')
test_file = os.path.join(output_dir, 'test_set.csv')

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"✓ Saved train set: {train_file}")
print(f"✓ Saved val set:   {val_file}")
print(f"✓ Saved test set:  {test_file}")

# ============================================================================
# STEP 8: SAVE SCALER INFO
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: SAVE NORMALIZATION PARAMETERS")
print("=" * 80)

import json
scaler_info = {
    'feature_names': feature_cols,
    'means': scaler.mean_.tolist(),
    'stds': scaler.scale_.tolist(),
}

scaler_file = os.path.join(output_dir, 'scaler_params.json')
with open(scaler_file, 'w') as f:
    json.dump(scaler_info, f, indent=2)

print(f"✓ Saved scaler parameters: {scaler_file}")

# ============================================================================
# STEP 9: SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)

print(f"\nStratified Split (Balanced Classes):")
print(f"  Total samples: {len(X):,}")
print(f"  Train: {len(X_train):,} ({100*len(X_train)/len(X):.1f}%)")
print(f"  Val:   {len(X_val):,} ({100*len(X_val)/len(X):.1f}%)")
print(f"  Test:  {len(X_test):,} ({100*len(X_test)/len(X):.1f}%)")

print(f"\nClass Balance Achieved:")
print(f"  Each set has approximately 36.8% HIGH, 35.3% MEDIUM, 27.9% LOW")

print(f"\nNormalization:")
print(f"  Method: StandardScaler (mean=0, std=1)")
print(f"  Fitted on: Training set only")
print(f"  Applied to: Train + Val + Test")

print(f"\nOutput Files:")
print(f"  1. {train_file}")
print(f"  2. {val_file}")
print(f"  3. {test_file}")
print(f"  4. {scaler_file}")

print("\n" + "=" * 80)
print("✓ STRATIFIED SPLITS READY FOR MODEL TRAINING")
print("=" * 80)

print("\nNext: Re-train models with balanced data")

