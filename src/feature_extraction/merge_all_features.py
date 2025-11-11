#!/usr/bin/env python3
"""
Merge static features + LLM features + profiling data
Creates comprehensive datasets for model training
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def encode_llm_categorical(df):
    """Encode LLM categorical features"""
    print("\nEncoding LLM categorical features...")
    
    categorical_llm_features = [
        'memory_access_pattern',
        'spatial_locality',
        'temporal_locality',
        'data_dependency_type',
        'false_sharing_risk',
        'load_balance_characteristic',
        'algorithmic_complexity',
        'dominant_operation',
        'vectorization_potential',
        'numa_sensitivity',
        'cache_behavior_pattern',
        'parallelization_overhead',
        'scalability_bottleneck'
    ]
    
    for feat in categorical_llm_features:
        if feat in df.columns:
            le = LabelEncoder()
            df[f'{feat}_encoded'] = le.fit_transform(df[feat].astype(str))
            print(f"  ✓ {feat}: {len(le.classes_)} categories")
    
    return df

def merge_features():
    """Merge all feature sources"""
    
    print("="*60)
    print("MERGING ALL FEATURES")
    print("="*60)
    
    # Load static features
    print("\n1. Loading static features...")
    static_file = Path('data/extracted_features/static_features.json')
    with open(static_file, 'r') as f:
        static_data = json.load(f)
    df_static = pd.DataFrame(static_data)
    print(f"   Static features: {len(df_static)} programs, {len(df_static.columns)} columns")
    
    # Load LLM features
    print("\n2. Loading LLM features...")
    llm_file = Path('data/extracted_features/llm_features.json')
    with open(llm_file, 'r') as f:
        llm_data = json.load(f)
    df_llm = pd.DataFrame(llm_data)
    print(f"   LLM features: {len(df_llm)} programs, {len(df_llm.columns)} columns")
    
    # Remove metadata columns from feature DataFrames (keep only features)
    static_feature_cols = [col for col in df_static.columns 
                          if col not in ['filename', 'application_id', 'benchmark']]
    llm_feature_cols = [col for col in df_llm.columns 
                       if col not in ['filename', 'application_id', 'benchmark', 
                                     'iteration_count_analysis']]  # Remove explanation text
    
    print(f"\n3. Feature columns:")
    print(f"   Static: {len(static_feature_cols)} features")
    print(f"   LLM: {len(llm_feature_cols)} features")
    
    # Prepare feature dataframes with only application_id + features
    df_static_clean = df_static[['application_id'] + static_feature_cols]
    df_llm_clean = df_llm[['application_id'] + llm_feature_cols]
    
    # Encode LLM categorical features
    df_llm_encoded = encode_llm_categorical(df_llm_clean.copy())
    
    # Drop original categorical columns (keep only encoded versions)
    categorical_to_drop = [col for col in df_llm_encoded.columns 
                          if col in llm_feature_cols and f'{col}_encoded' in df_llm_encoded.columns]
    df_llm_encoded = df_llm_encoded.drop(columns=categorical_to_drop)
    
    # Merge static + LLM features
    print("\n4. Merging static + LLM features...")
    df_combined_features = df_static_clean.merge(df_llm_encoded, on='application_id', how='left')
    print(f"   Combined features: {len(df_combined_features.columns) - 1} features")  # -1 for application_id
    
    return df_combined_features, static_feature_cols, llm_feature_cols

def merge_with_profiling(df_features, profiling_file, output_suffix):
    """Merge combined features with profiling data"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {profiling_file.name}")
    print(f"{'='*60}")
    
    # Load profiling data
    df_profiling = pd.read_csv(profiling_file)
    print(f"Profiling data: {len(df_profiling)} rows, {len(df_profiling.columns)} columns")
    
    # Merge with features
    df_merged = df_profiling.merge(df_features, on='application_id', how='left')
    print(f"Merged data: {len(df_merged)} rows, {len(df_merged.columns)} columns")
    
    # Check for missing
    missing = df_merged[[col for col in df_features.columns if col != 'application_id']].isna().any(axis=1).sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} rows missing features")
    else:
        print("✓ All rows successfully merged!")
    
    # Create output directory
    output_dir = Path('data/merged')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # VERSION 1: WITH benchmark identity
    output_v1 = output_dir / f'all_features_with_benchmark_{output_suffix}.csv'
    df_merged.to_csv(output_v1, index=False)
    print(f"\n✓ Version 1 (WITH benchmark): {output_v1}")
    print(f"   Columns: {len(df_merged.columns)}")
    
    # VERSION 2: WITHOUT benchmark identity
    df_no_benchmark = df_merged.drop(columns=['application_id', 'benchmark'])
    output_v2 = output_dir / f'all_features_no_benchmark_{output_suffix}.csv'
    df_no_benchmark.to_csv(output_v2, index=False)
    print(f"\n✓ Version 2 (NO benchmark): {output_v2}")
    print(f"   Columns: {len(df_no_benchmark.columns)}")
    
    return df_merged, df_no_benchmark

def main():
    # Merge static + LLM features
    df_combined_features, static_cols, llm_cols = merge_features()
    
    # Save combined features for reference
    output_dir = Path('data/extracted_features')
    combined_file = output_dir / 'combined_features.json'
    df_combined_features.to_json(combined_file, orient='records', indent=2)
    print(f"\n✓ Saved combined features: {combined_file}")
    
    # Merge with both profiling datasets
    profiling_files = [
        (Path('data/raw/profiling_data_tx2_v2.csv'), 'tx2'),
        (Path('data/raw/profiling_data_rubikpi_v2.csv'), 'rubikpi')
    ]
    
    for profiling_file, suffix in profiling_files:
        if profiling_file.exists():
            df_with, df_without = merge_with_profiling(df_combined_features, profiling_file, suffix)
            
            # Print summary
            print("\n" + "-"*60)
            print("Summary:")
            print("-"*60)
            print(f"Total samples: {len(df_with)}")
            print(f"Static features: {len(static_cols)}")
            print(f"LLM features: {len(llm_cols)}")
            print(f"Total extracted features: {len(df_combined_features.columns) - 1}")
            print(f"\nTarget (time_elapsed):")
            print(f"  Mean: {df_with['time_elapsed'].mean():.4f} s")
            print(f"  Std:  {df_with['time_elapsed'].std():.4f} s")
            print(f"  Range: {df_with['time_elapsed'].min():.4f} - {df_with['time_elapsed'].max():.4f} s")
    
    print("\n" + "="*60)
    print("ALL FEATURES MERGED SUCCESSFULLY!")
    print("="*60)
    print("\nDatasets created in data/merged/:")
    print("  - all_features_with_benchmark_tx2.csv")
    print("  - all_features_no_benchmark_tx2.csv")
    print("  - all_features_with_benchmark_rubikpi.csv")
    print("  - all_features_no_benchmark_rubikpi.csv")
    print("\nReady for model training! 🚀")

if __name__ == '__main__':
    main()
