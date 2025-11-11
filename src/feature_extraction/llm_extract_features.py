#!/usr/bin/env python3
"""
LLM-based semantic feature extraction for OpenMP programs
Uses OpenRouter API to extract features that static analysis cannot easily capture
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# OpenRouter API configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat"  # Cheapest option

def create_extraction_prompt(code, benchmark_name):
    """Create structured prompt for LLM feature extraction"""
    
    prompt = f"""Analyze this OpenMP C program ({benchmark_name}) and extract ONLY the following features as valid JSON.

CRITICAL INSTRUCTIONS:
- Your ENTIRE response must be ONLY a single valid JSON object
- DO NOT include any explanations, markdown, or text outside the JSON
- DO NOT use backticks or code blocks
- If a feature cannot be determined, use -1 or "unknown"

Code to analyze:
```c
{code}
```

Extract these features in JSON format with EXACTLY these keys:

{{
  "estimated_iteration_count": <integer or -1 if dynamic>,
  "iteration_count_analysis": "<brief explanation of how you calculated this>",
  "memory_access_pattern": "<unit_stride|non_unit_stride|random|mixed>",
  "spatial_locality": "<high|medium|low>",
  "temporal_locality": "<high|medium|low>",
  "data_dependency_type": "<none|loop_carried|cross_iteration|complex>",
  "false_sharing_risk": "<high|medium|low|none>",
  "load_balance_characteristic": "<uniform|irregular|dynamic>",
  "algorithmic_complexity": "<O(n)|O(n^2)|O(n^3)|O(nlogn)|other>",
  "dominant_operation": "<arithmetic|memory|logic|mixed>",
  "vectorization_potential": "<high|medium|low>",
  "numa_sensitivity": "<high|medium|low>",
  "cache_behavior_pattern": "<streaming|random|blocked|mixed>",
  "parallelization_overhead": "<low|medium|high>",
  "scalability_bottleneck": "<none|memory_bandwidth|synchronization|load_imbalance|other>"
}}

RESPOND WITH ONLY THE JSON OBJECT, NOTHING ELSE."""

    return prompt

def call_llm_api(prompt, max_retries=3):
    """Call OpenRouter API with retry logic"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    
    return None

def parse_llm_response(response_text):
    """Parse and validate LLM JSON response"""
    
    # Strip markdown code blocks if present
    response_text = response_text.strip()
    if response_text.startswith('```'):
        # Remove markdown code blocks
        response_text = response_text.split('```')[1]
        if response_text.startswith('json'):
            response_text = response_text[4:]
        response_text = response_text.strip()
    
    try:
        features = json.loads(response_text)
        return features, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

def extract_llm_features(filepath):
    """Extract semantic features from a single C file using LLM"""
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
    
    # Truncate very large files
    if len(code) > 15000:
        code = code[:15000] + "\n\n... [file truncated for API call]"
    
    benchmark_name = '_'.join(os.path.basename(filepath).split('_')[1:]).replace('.c', '')
    application_id = int(os.path.basename(filepath).split('_')[0])
    
    # Create prompt
    prompt = create_extraction_prompt(code, benchmark_name)
    
    # Call LLM
    response = call_llm_api(prompt)
    
    if response is None:
        return None, "API call failed"
    
    # Parse response
    features, error = parse_llm_response(response)
    
    if error:
        return None, error
    
    # Add metadata
    features['application_id'] = application_id
    features['benchmark'] = benchmark_name
    features['filename'] = os.path.basename(filepath)
    
    return features, None

def main():
    source_dir = Path('data/source_codes')
    output_dir = Path('data/extracted_features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    errors = []
    total_cost = 0
    
    print("="*60)
    print("LLM-BASED FEATURE EXTRACTION")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    source_files = sorted(source_dir.glob('*.c'))
    print(f"\nProcessing {len(source_files)} files...\n")
    
    for i, filepath in enumerate(source_files, 1):
        print(f"[{i}/{len(source_files)}] {filepath.name}...", end=' ', flush=True)
        
        try:
            features, error = extract_llm_features(filepath)
            
            if error:
                print(f"✗ {error}")
                errors.append({'file': filepath.name, 'error': error})
            else:
                print("✓")
                all_features.append(features)
                
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"✗ Exception: {e}")
            errors.append({'file': filepath.name, 'error': str(e)})
    
    # Save results
    output_file = output_dir / 'llm_features.json'
    with open(output_file, 'w') as f:
        json.dump(all_features, f, indent=2)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"✓ Successfully extracted: {len(all_features)}/{len(source_files)} files")
    print(f"✓ Saved to: {output_file}")
    
    if errors:
        print(f"\n⚠️  Errors: {len(errors)}")
        error_file = output_dir / 'llm_extraction_errors.json'
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"   Error log: {error_file}")
    
    # Estimate cost (approximate)
    avg_tokens_per_file = 3000  # prompt + response
    total_tokens = avg_tokens_per_file * len(all_features)
    estimated_cost = (total_tokens / 1_000_000) * 0.50  # DeepSeek pricing
    
    print(f"\n💰 Estimated cost: ${estimated_cost:.4f}")
    
    # Print sample features
    if all_features:
        print("\n" + "-"*60)
        print("Sample LLM Features (first program):")
        print("-"*60)
        sample = all_features[0]
        for key, value in sample.items():
            if key not in ['filename', 'application_id', 'benchmark']:
                print(f"  {key}: {value}")

if __name__ == '__main__':
    main()
