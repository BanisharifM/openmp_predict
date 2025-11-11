#!/usr/bin/env python3
"""
Merge extracted static features with profiling data
Creates two versions:
1. With benchmark identity (for configuration prediction)
2. Without benchmark identity (for pure feature-based prediction)
"""

import json
import pandas as pd
from pathlib import Path

def main():
    # Load profiling data
    print("Loading profiling data...")
    profiling_file = Path('data/raw/profiling_data_rubikpi_v2.csv')
    df_profiling = pd.read_csv(profiling_file)
    print(f"  Profiling data: {len(df_profiling)} rows, {len(df_profiling.columns)} columns")
    
    # Load extracted features
    print("\nLoading extracted static features...")
    features_file = Path('data/extracted_features/static_features.json')
    with open(features_file, 'r') as f:
        features_list = json.load(f)
    
    df_features = pd.DataFrame(features_list)
    print(f"  Static features: {len(df_features)} programs, {len(df_features.columns)} features")
    
    # Extract only the feature columns (exclude filename, application_id, benchmark)
    feature_cols = [col for col in df_features.columns 
                   if col not in ['filename', 'application_id', 'benchmark']]
    
    print(f"\nFeature columns to merge: {len(feature_cols)}")
    print(f"  {feature_cols}")
    
    df_features_only = df_features[['application_id'] + feature_cols]
    
    # Merge on application_id
    print("\nMerging datasets...")
    df_merged = df_profiling.merge(
        df_features_only, 
        on='application_id', 
        how='left'
    )
    
    print(f"  Merged data: {len(df_merged)} rows, {len(df_merged.columns)} columns")
    
    # Check for missing merges
    missing = df_merged[df_merged['loop_depth'].isna()]
    if len(missing) > 0:
        print(f"\n  ⚠️  Warning: {len(missing)} rows missing extracted features")
        print(f"  Missing application_ids: {missing['application_id'].unique()}")
    else:
        print("  ✓ All rows successfully merged!")
    
    # ========================================================================
    # VERSION 1: WITH BENCHMARK IDENTITY (keep application_id and benchmark)
    # ========================================================================
    output_v1 = Path('data/merged_features_with_benchmark.csv')
    df_merged.to_csv(output_v1, index=False)
    print(f"\n✓ Version 1 (WITH benchmark): {output_v1}")
    print(f"   Columns: {len(df_merged.columns)}")
    print(f"   Includes: application_id, benchmark")
    
    # ========================================================================
    # VERSION 2: WITHOUT BENCHMARK IDENTITY (drop application_id and benchmark)
    # ========================================================================
    df_no_benchmark = df_merged.drop(columns=['application_id', 'benchmark'])
    output_v2 = Path('data/merged_features_no_benchmark.csv')
    df_no_benchmark.to_csv(output_v2, index=False)
    print(f"\n✓ Version 2 (NO benchmark): {output_v2}")
    print(f"   Columns: {len(df_no_benchmark.columns)}")
    print(f"   Dropped: application_id, benchmark")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df_merged)}")
    print(f"Unique benchmarks: {df_merged['benchmark'].nunique()}")
    print(f"\nVersion 1 features: {len(df_merged.columns)}")
    print(f"Version 2 features: {len(df_no_benchmark.columns)}")
    print(f"Extracted features added: {len(feature_cols)}")
    
    print(f"\nTarget variable (time_elapsed):")
    print(f"  Mean:  {df_merged['time_elapsed'].mean():.4f} seconds")
    print(f"  Std:   {df_merged['time_elapsed'].std():.4f} seconds")
    print(f"  Min:   {df_merged['time_elapsed'].min():.4f} seconds")
    print(f"  Max:   {df_merged['time_elapsed'].max():.4f} seconds")
    print(f"  Range: {df_merged['time_elapsed'].max() - df_merged['time_elapsed'].min():.4f} seconds")
    
    print(f"\nExtracted feature statistics:")
    print(f"  Loop depth:           {df_merged['loop_depth'].min():.0f} - {df_merged['loop_depth'].max():.0f}")
    print(f"  Loop count:           {df_merged['loop_count'].min():.0f} - {df_merged['loop_count'].max():.0f}")
    print(f"  Parallel regions:     {df_merged['parallel_regions'].min():.0f} - {df_merged['parallel_regions'].max():.0f}")
    print(f"  Synchronization pts:  {df_merged['synchronization_points'].min():.0f} - {df_merged['synchronization_points'].max():.0f}")
    print(f"  Tasks:                {df_merged['tasks'].min():.0f} - {df_merged['tasks'].max():.0f}")
    print(f"  Shared vars:          {df_merged['shared_vars'].min():.0f} - {df_merged['shared_vars'].max():.0f}")
    print(f"  Lines of code:        {df_merged['lines_of_code'].min():.0f} - {df_merged['lines_of_code'].max():.0f}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("Version 1 (WITH benchmark): Use for configuration prediction")
    print("  → Split: Regular KFold or StratifiedKFold")
    print("  → Question: Given benchmark X, predict performance on config Y")
    print("\nVersion 2 (NO benchmark): Use for feature-based generalization")
    print("  → Split: GroupKFold by benchmark")
    print("  → Question: Predict performance of completely new program")

if __name__ == '__main__':
    main()
