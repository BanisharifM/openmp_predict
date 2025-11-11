#!/usr/bin/env python3
"""
Static analysis feature extraction for OpenMP programs
Extracts 8 core features using Tree-sitter
"""

import os
import re
import json
from pathlib import Path
from tree_sitter import Language, Parser

# Load C language
C_LANGUAGE = Language('build/c.so', 'c')
parser = Parser()
parser.set_language(C_LANGUAGE)

def get_max_loop_depth(node, current_depth=0):
    """Recursively find maximum loop nesting depth"""
    max_depth = current_depth
    if node.type in ['for_statement', 'while_statement', 'do_statement']:
        current_depth += 1
        max_depth = current_depth
    
    for child in node.children:
        child_depth = get_max_loop_depth(child, current_depth)
        max_depth = max(max_depth, child_depth)
    
    return max_depth

def count_loops(node):
    """Count total number of loops"""
    count = 1 if node.type in ['for_statement', 'while_statement', 'do_statement'] else 0
    for child in node.children:
        count += count_loops(child)
    return count

def extract_openmp_features(source_code):
    """Extract OpenMP pragma features via regex"""
    pragmas = re.findall(r'#pragma\s+omp\s+([^\n]+)', source_code, re.MULTILINE)
    
    # Count different pragma types
    parallel_count = len([p for p in pragmas if 'parallel' in p])
    critical_count = len([p for p in pragmas if 'critical' in p])
    barrier_count = len([p for p in pragmas if 'barrier' in p])
    atomic_count = len([p for p in pragmas if 'atomic' in p])
    task_count = len([p for p in pragmas if 'task' in p])
    
    # Extract thread count (look for num_threads clause)
    thread_counts = re.findall(r'num_threads\s*\(\s*(\d+)\s*\)', source_code)
    thread_count = int(thread_counts[0]) if thread_counts else -1  # -1 = default/runtime
    
    # Extract schedule type
    schedule_match = re.search(r'schedule\s*\(\s*(\w+)', source_code)
    schedule_type = schedule_match.group(1) if schedule_match else 'none'
    schedule_encoding = {
        'static': 0, 'dynamic': 1, 'guided': 2, 
        'auto': 3, 'runtime': 4, 'none': -1
    }.get(schedule_type, -1)
    
    # Count shared variables
    shared_vars = re.findall(r'shared\s*\(([^)]+)\)', source_code)
    shared_count = 0
    for vars_str in shared_vars:
        shared_count += len([v.strip() for v in vars_str.split(',') if v.strip()])
    
    # Count private variables
    private_vars = re.findall(r'private\s*\(([^)]+)\)', source_code)
    private_count = 0
    for vars_str in private_vars:
        private_count += len([v.strip() for v in vars_str.split(',') if v.strip()])
    
    # Count reduction variables
    reduction_vars = re.findall(r'reduction\s*\([^:]+:([^)]+)\)', source_code)
    reduction_count = 0
    for vars_str in reduction_vars:
        reduction_count += len([v.strip() for v in vars_str.split(',') if v.strip()])
    
    return {
        'total_pragmas': len(pragmas),
        'parallel_regions': parallel_count,
        'synchronization_points': critical_count + barrier_count + atomic_count,
        'critical_sections': critical_count,
        'barriers': barrier_count,
        'atomics': atomic_count,
        'tasks': task_count,
        'thread_count': thread_count,
        'schedule_type': schedule_type,
        'schedule_encoding': schedule_encoding,
        'shared_vars': shared_count,
        'private_vars': private_count,
        'reduction_vars': reduction_count,
    }

def estimate_arithmetic_intensity(source_code):
    """Rough estimation of arithmetic intensity"""
    # Count arithmetic operations
    flops = len(re.findall(r'[+\-*/]', source_code))
    
    # Count array accesses (rough proxy for memory operations)
    array_accesses = len(re.findall(r'\w+\s*\[', source_code))
    
    if array_accesses == 0:
        return 0.0
    
    return flops / array_accesses

def extract_features(filepath):
    """Extract all features from a single C file"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
    
    # Parse with Tree-sitter
    tree = parser.parse(bytes(code, 'utf8'))
    
    # Extract structural features
    loop_depth = get_max_loop_depth(tree.root_node)
    loop_count = count_loops(tree.root_node)
    
    # Extract OpenMP features
    omp_features = extract_openmp_features(code)
    
    # Estimate arithmetic intensity
    arith_intensity = estimate_arithmetic_intensity(code)
    
    # Lines of code
    loc = len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('//')])
    
    # Combine all features
    features = {
        'filename': os.path.basename(filepath),
        'application_id': int(os.path.basename(filepath).split('_')[0]),
        'benchmark': '_'.join(os.path.basename(filepath).split('_')[1:]).replace('.c', ''),
        'loop_depth': loop_depth,
        'loop_count': loop_count,
        'lines_of_code': loc,
        'estimated_arithmetic_intensity': round(arith_intensity, 4),
        **omp_features
    }
    
    return features

def main():
    source_dir = Path('data/source_codes')
    output_dir = Path('data/extracted_features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    
    print("Extracting features from OpenMP source files...")
    print("=" * 60)
    
    for filepath in sorted(source_dir.glob('*.c')):
        print(f"Processing: {filepath.name}...", end=' ')
        try:
            features = extract_features(filepath)
            all_features.append(features)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Save as JSON
    output_file = output_dir / 'static_features.json'
    with open(output_file, 'w') as f:
        json.dump(all_features, f, indent=2)
    
    print("=" * 60)
    print(f"✓ Extracted features for {len(all_features)} programs")
    print(f"✓ Saved to: {output_file}")
    
    # Print summary
    print("\nFeature Summary:")
    print(f"  Average loop depth: {sum(f['loop_depth'] for f in all_features) / len(all_features):.2f}")
    print(f"  Average parallel regions: {sum(f['parallel_regions'] for f in all_features) / len(all_features):.2f}")
    print(f"  Average synchronization points: {sum(f['synchronization_points'] for f in all_features) / len(all_features):.2f}")

if __name__ == '__main__':
    main()
