#!/usr/bin/env python3
"""
Compare feature extraction results from DeepSeek, Claude, and GPT-4o
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

def load_llm_features(model_name):
    """Load LLM features from JSON file"""
    file_path = Path(f'data/extracted_features/llm_features_{model_name}.json')
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_categorical_features(df_deep, df_claude, df_gpt):
    """Compare categorical feature distributions"""
    
    categorical_features = [
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
    
    print("\n" + "="*70)
    print("CATEGORICAL FEATURE COMPARISON")
    print("="*70)
    
    for feature in categorical_features:
        print(f"\n{feature.upper()}:")
        print("-" * 70)
        
        deep_counts = Counter(df_deep[feature])
        claude_counts = Counter(df_claude[feature])
        gpt_counts = Counter(df_gpt[feature])
        
        # Get all unique values
        all_values = set(deep_counts.keys()) | set(claude_counts.keys()) | set(gpt_counts.keys())
        
        print(f"{'Value':<25} {'DeepSeek':<15} {'Claude':<15} {'GPT-4o':<15}")
        print("-" * 70)
        for value in sorted(all_values):
            print(f"{value:<25} {deep_counts.get(value, 0):<15} "
                  f"{claude_counts.get(value, 0):<15} {gpt_counts.get(value, 0):<15}")

def calculate_agreement(df_deep, df_claude, df_gpt):
    """Calculate inter-model agreement"""
    
    categorical_features = [
        'memory_access_pattern',
        'spatial_locality',
        'temporal_locality',
        'data_dependency_type',
        'false_sharing_risk',
        'load_balance_characteristic',
        'algorithmic_complexity',
        'dominant_operation',
        'vectorization_potential',
        'cache_behavior_pattern',
        'parallelization_overhead',
        'scalability_bottleneck'
    ]
    
    print("\n" + "="*70)
    print("INTER-MODEL AGREEMENT")
    print("="*70)
    
    agreements = []
    
    for feature in categorical_features:
        # DeepSeek vs Claude
        dc_agreement = (df_deep[feature] == df_claude[feature]).sum() / len(df_deep) * 100
        
        # DeepSeek vs GPT
        dg_agreement = (df_deep[feature] == df_gpt[feature]).sum() / len(df_deep) * 100
        
        # Claude vs GPT
        cg_agreement = (df_claude[feature] == df_gpt[feature]).sum() / len(df_claude) * 100
        
        # All three agree
        all_agree = ((df_deep[feature] == df_claude[feature]) & 
                     (df_claude[feature] == df_gpt[feature])).sum() / len(df_deep) * 100
        
        agreements.append({
            'feature': feature,
            'deep_vs_claude': dc_agreement,
            'deep_vs_gpt': dg_agreement,
            'claude_vs_gpt': cg_agreement,
            'all_three': all_agree
        })
        
        print(f"\n{feature}:")
        print(f"  DeepSeek vs Claude:  {dc_agreement:.1f}%")
        print(f"  DeepSeek vs GPT-4o:  {dg_agreement:.1f}%")
        print(f"  Claude vs GPT-4o:    {cg_agreement:.1f}%")
        print(f"  All three agree:     {all_agree:.1f}%")
    
    # Overall statistics
    df_agreements = pd.DataFrame(agreements)
    
    print("\n" + "="*70)
    print("OVERALL AGREEMENT STATISTICS")
    print("="*70)
    print(f"Average DeepSeek vs Claude:  {df_agreements['deep_vs_claude'].mean():.1f}%")
    print(f"Average DeepSeek vs GPT-4o:  {df_agreements['deep_vs_gpt'].mean():.1f}%")
    print(f"Average Claude vs GPT-4o:    {df_agreements['claude_vs_gpt'].mean():.1f}%")
    print(f"Average All Three Agree:     {df_agreements['all_three'].mean():.1f}%")
    
    return df_agreements

def find_disagreements(df_deep, df_claude, df_gpt):
    """Find cases where models strongly disagree"""
    
    print("\n" + "="*70)
    print("EXAMPLES OF DISAGREEMENTS")
    print("="*70)
    
    features_to_check = ['algorithmic_complexity', 'scalability_bottleneck', 
                         'load_balance_characteristic']
    
    for feature in features_to_check:
        print(f"\n{feature.upper()}:")
        print("-" * 70)
        
        # Find rows where all three disagree
        disagree_mask = ((df_deep[feature] != df_claude[feature]) & 
                        (df_claude[feature] != df_gpt[feature]) & 
                        (df_deep[feature] != df_gpt[feature]))
        
        if disagree_mask.sum() > 0:
            for idx in df_deep[disagree_mask].head(3).index:
                benchmark = df_deep.loc[idx, 'benchmark']
                print(f"  {benchmark}:")
                print(f"    DeepSeek: {df_deep.loc[idx, feature]}")
                print(f"    Claude:   {df_claude.loc[idx, feature]}")
                print(f"    GPT-4o:   {df_gpt.loc[idx, feature]}")

def main():
    print("="*70)
    print("LLM FEATURE EXTRACTION COMPARISON")
    print("="*70)
    
    # Load all three models
    print("\nLoading LLM features...")
    deep_data = load_llm_features('deepseek')
    claude_data = load_llm_features('claude')
    gpt_data = load_llm_features('gpt4o')
    
    df_deep = pd.DataFrame(deep_data).sort_values('application_id').reset_index(drop=True)
    df_claude = pd.DataFrame(claude_data).sort_values('application_id').reset_index(drop=True)
    df_gpt = pd.DataFrame(gpt_data).sort_values('application_id').reset_index(drop=True)
    
    print(f"  DeepSeek: {len(df_deep)} programs")
    print(f"  Claude:   {len(df_claude)} programs")
    print(f"  GPT-4o:   {len(df_gpt)} programs")
    
    # Compare distributions
    compare_categorical_features(df_deep, df_claude, df_gpt)
    
    # Calculate agreement
    df_agreements = calculate_agreement(df_deep, df_claude, df_gpt)
    
    # Find disagreements
    find_disagreements(df_deep, df_claude, df_gpt)
    
    # Save agreement statistics
    output_file = Path('data/extracted_features/llm_agreement_statistics.csv')
    df_agreements.to_csv(output_file, index=False)
    print(f"\n✓ Saved agreement statistics to: {output_file}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
