#!/usr/bin/env python3
"""
Combine 3 Hydrogen Refueling Station Simulation Scenarios
Merges HIGH (100 bar), MEDIUM (300 bar), and LOW (500 bar) into one labeled dataset
"""

import pandas as pd
import os

print("=" * 80)
print("COMBINING 3 SCENARIOS INTO UNIFIED DATASET")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD ALL 3 EXCEL FILES
# ============================================================================

input_files = {
    'HIGH': 'data/raw/100_bar.xlsx',
    'MEDIUM': 'data/raw/300_bar.xlsx',
    'LOW': 'data/raw/500_bar.xlsx',
}

dataframes = {}

for state, filepath in input_files.items():
    print(f"\nLoading {state} scenario...")
    df = pd.read_excel(filepath)
    print(f"  ✓ {len(df)} samples loaded")
    dataframes[state] = df

# ============================================================================
# STEP 2: ADD STATE LABELS TO EACH DATASET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: LABELING EACH SCENARIO")
print("=" * 80)

for state, df in dataframes.items():
    df['pressure_state'] = state
    print(f"  ✓ Added label '{state}' to {len(df)} samples")

# ============================================================================
# STEP 3: COMBINE ALL THREE DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: COMBINING SCENARIOS")
print("=" * 80)

combined_df = pd.concat(
    [dataframes['HIGH'], dataframes['MEDIUM'], dataframes['LOW']],
    ignore_index=True
)

print(f"✓ Combined dataset shape: {combined_df.shape}")
print(f"  Total samples: {len(combined_df)}")

# ============================================================================
# STEP 4: SELECT INPUT FEATURES (Remove buffer pressures)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: SELECT FEATURES FOR ML MODEL")
print("=" * 80)

# Input features: temperatures + other operational variables
# DO NOT include buffer pressures (Results:4(1), 4(2), 4(3)) - these define the labels
input_features = [
    'time',
    'Results:5(1)',     # Buffer temp HIGH
    'Results:5(2)',     # Buffer temp MEDIUM
    'Results:5(3)',     # Buffer temp LOW
    'Results:2',        # Vehicle tank temp
    'Results:3',        # Vehicle tank flow
    'Results:6',        # Precooler temp
]

# Rename for clarity
feature_names_new = [
    'time',
    'buffer_temp_high',
    'buffer_temp_medium',
    'buffer_temp_low',
    'vehicle_tank_temp',
    'vehicle_tank_flow',
    'precooler_temp',
]

# Select and rename
df_final = combined_df[input_features + ['pressure_state']].copy()
df_final.columns = feature_names_new + ['pressure_state']

print(f"✓ Selected {len(input_features)} input features:")
for feat in feature_names_new:
    print(f"    {feat}")

# ============================================================================
# STEP 5: CLASS DISTRIBUTION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: CLASS DISTRIBUTION")
print("=" * 80)

for state in ['HIGH', 'MEDIUM', 'LOW']:
    count = (df_final['pressure_state'] == state).sum()
    pct = 100 * count / len(df_final)
    print(f"  {state:8s}: {count:4d} samples ({pct:5.1f}%)")

# ============================================================================
# STEP 6: DATA QUALITY CHECKS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: DATA QUALITY CHECKS")
print("=" * 80)

# Check for missing values
missing = df_final.isnull().sum()
if missing.sum() > 0:
    print("⚠ Missing values:")
    print(missing[missing > 0])
else:
    print("✓ No missing values")

# Feature statistics
print("\n✓ Feature statistics:")
for col in feature_names_new:
    print(f"  {col:20s}: min={df_final[col].min():8.2f}, max={df_final[col].max():8.2f}, std={df_final[col].std():8.2f}")

# ============================================================================
# STEP 7: SAVE OUTPUT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: SAVING UNIFIED DATASET")
print("=" * 80)

output_file = 'data/processed/synthetic_data_labeled.csv'
os.makedirs('data/processed', exist_ok=True)

df_final.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")
print(f"  Shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")

# ============================================================================
# STEP 8: SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)

print(f"\nDataset: {output_file}")
print(f"  Total samples: {len(df_final):,}")
print(f"  Time span per scenario: 0–3600 seconds (60 minutes)")
print(f"  Sampling rate: ~0.85 Hz per scenario")

print(f"\nScenario mapping:")
print(f"  100 bar vehicle init → HIGH buffer state (≈80 MPa)")
print(f"  300 bar vehicle init → MEDIUM buffer state (≈50 MPa)")
print(f"  500 bar vehicle init → LOW buffer state (≈30 MPa)")

print(f"\nClass distribution:")
for state in ['HIGH', 'MEDIUM', 'LOW']:
    count = (df_final['pressure_state'] == state).sum()
    pct = 100 * count / len(df_final)
    print(f"  {state:8s}: {count:4,} ({pct:5.1f}%)")

print(f"\nInput features (7 total):")
for i, feat in enumerate(feature_names_new, 1):
    print(f"  {i}. {feat}")

print(f"\nTarget variable:")
print(f"  pressure_state: HIGH / MEDIUM / LOW")

print("\n" + "=" * 80)
print("✓ READY FOR TRAIN/TEST SPLIT & MODEL TRAINING")
print("=" * 80)

print("\nNext steps:")
print("  1. Split into train/val/test (70/15/15, chronological order)")
print("  2. Normalize features (using training data only)")
print("  3. Train MLP + baselines")
print("  4. Evaluate with per-class metrics")
print("  5. Generate 2D plots (confusion matrix, training curves)")

