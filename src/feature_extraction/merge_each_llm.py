#!/usr/bin/env python3
"""
Create separate merged datasets for each LLM model
This allows comparing ML model performance with different LLM features
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def encode_llm_categorical(df):
    """Encode LLM categorical features"""
    
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
    
    return df

def process_llm_model(llm_name, static_features):
    """Process a single LLM model's features"""
    
    print(f"\n{'='*70}")
    print(f"Processing: {llm_name.upper()}")
    print(f"{'='*70}")
    
    # Load LLM features
    llm_file = Path(f'data/extracted_features/llm_features_{llm_name}.json')
    with open(llm_file, 'r') as f:
        llm_data = json.load(f)
    df_llm = pd.DataFrame(llm_data)
    
    # Keep only features (remove metadata)
    llm_feature_cols = [col for col in df_llm.columns 
                       if col not in ['filename', 'application_id', 'benchmark', 
                                     'iteration_count_analysis']]
    
    df_llm_clean = df_llm[['application_id'] + llm_feature_cols]
    
    # Encode categorical
    df_llm_encoded = encode_llm_categorical(df_llm_clean.copy())
    
    # Drop original categorical columns
    categorical_to_drop = [col for col in df_llm_encoded.columns 
                          if col in llm_feature_cols and f'{col}_encoded' in df_llm_encoded.columns]
    df_llm_encoded = df_llm_encoded.drop(columns=categorical_to_drop)
    
    # Merge with static features
    df_combined = static_features.merge(df_llm_encoded, on='application_id', how='left')
    
    print(f"✓ Static features: 17")
    print(f"✓ LLM features: {len(llm_feature_cols)}")
    print(f"✓ Total features: {len(df_combined.columns) - 1}")
    
    return df_combined

def merge_with_profiling(df_features, profiling_file, llm_name, platform):
    """Merge features with profiling data"""
    
    df_profiling = pd.read_csv(profiling_file)
    df_merged = df_profiling.merge(df_features, on='application_id', how='left')
    
    output_dir = Path('data/merged')
    
    # Version 1: WITH benchmark
    output_v1 = output_dir / f'{llm_name}_features_with_benchmark_{platform}.csv'
    df_merged.to_csv(output_v1, index=False)
    
    # Version 2: WITHOUT benchmark
    df_no_benchmark = df_merged.drop(columns=['application_id', 'benchmark'])
    output_v2 = output_dir / f'{llm_name}_features_no_benchmark_{platform}.csv'
    df_no_benchmark.to_csv(output_v2, index=False)
    
    print(f"  ✓ WITH benchmark: {output_v1.name} ({len(df_merged.columns)} cols)")
    print(f"  ✓ NO benchmark: {output_v2.name} ({len(df_no_benchmark.columns)} cols)")
    
    return df_merged

def main():
    print("="*70)
    print("CREATING DATASETS FOR EACH LLM MODEL")
    print("="*70)
    
    # Load static features
    static_file = Path('data/extracted_features/static_features.json')
    with open(static_file, 'r') as f:
        static_data = json.load(f)
    df_static = pd.DataFrame(static_data)
    
    # Keep only features
    static_feature_cols = [col for col in df_static.columns 
                          if col not in ['filename', 'application_id', 'benchmark']]
    df_static_clean = df_static[['application_id'] + static_feature_cols]
    
    print(f"\n✓ Loaded static features: {len(static_feature_cols)} features")
    
    # Process each LLM
    llm_models = ['deepseek', 'claude', 'gpt4o']
    
    for llm_name in llm_models:
        df_combined = process_llm_model(llm_name, df_static_clean)
        
        # Merge with both platforms
        print(f"\n  Merging with profiling data:")
        
        # TX2
        tx2_file = Path('data/raw/profiling_data_tx2_v2.csv')
        if tx2_file.exists():
            merge_with_profiling(df_combined, tx2_file, llm_name, 'tx2')
        
        # RubikPi
        rubikpi_file = Path('data/raw/profiling_data_rubikpi_v2.csv')
        if rubikpi_file.exists():
            merge_with_profiling(df_combined, rubikpi_file, llm_name, 'rubikpi')
    
    print("\n" + "="*70)
    print("ALL LLM DATASETS CREATED!")
    print("="*70)
    print("\nCreated datasets:")
    print("  For each LLM (deepseek, claude, gpt4o) × 2 platforms × 2 versions:")
    print("    - {llm}_features_with_benchmark_{platform}.csv")
    print("    - {llm}_features_no_benchmark_{platform}.csv")
    print("\nTotal: 12 new datasets (3 LLMs × 2 platforms × 2 versions)")
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR YOUR PAPER")
    print("="*70)
    print("1. Train models with DeepSeek features")
    print("2. Train models with Claude features")
    print("3. Train models with GPT-4o features")
    print("4. Compare which LLM features lead to best performance")
    print("5. Optionally: Create ensemble using majority voting")

if __name__ == '__main__':
    main()
