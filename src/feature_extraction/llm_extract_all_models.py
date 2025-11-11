#!/usr/bin/env python3
"""
Extract features using all 3 LLM models: DeepSeek, Claude, GPT-4o
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Define all models to test
MODELS = {
#    'deepseek': {
#        'name': 'deepseek/deepseek-chat',
#        'output': 'llm_features_deepseek.json',
#       'cost_per_1m': 0.50
#    },
    'claude': {
        'name': 'anthropic/claude-sonnet-4.5',
        'output': 'llm_features_claude.json',
        'cost_per_1m': 3.00
    },
    'gpt4o': {
        'name': 'openai/gpt-4o',
        'output': 'llm_features_gpt4o.json',
        'cost_per_1m': 2.50
    }
}

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

def call_llm_api(prompt, model_name, max_retries=3):
    """Call OpenRouter API with retry logic"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
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
                time.sleep(2 ** attempt)
            else:
                raise
    
    return None

def parse_llm_response(response_text):
    """Parse and validate LLM JSON response"""
    
    response_text = response_text.strip()
    if response_text.startswith('```'):
        response_text = response_text.split('```')[1]
        if response_text.startswith('json'):
            response_text = response_text[4:]
        response_text = response_text.strip()
    
    try:
        features = json.loads(response_text)
        return features, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

def extract_llm_features(filepath, model_name):
    """Extract semantic features from a single C file using LLM"""
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
    
    if len(code) > 15000:
        code = code[:15000] + "\n\n... [file truncated for API call]"
    
    benchmark_name = '_'.join(os.path.basename(filepath).split('_')[1:]).replace('.c', '')
    application_id = int(os.path.basename(filepath).split('_')[0])
    
    prompt = create_extraction_prompt(code, benchmark_name)
    response = call_llm_api(prompt, model_name)
    
    if response is None:
        return None, "API call failed"
    
    features, error = parse_llm_response(response)
    
    if error:
        return None, error
    
    features['application_id'] = application_id
    features['benchmark'] = benchmark_name
    features['filename'] = os.path.basename(filepath)
    
    return features, None

def run_model(model_key, model_config):
    """Run extraction for a single model"""
    
    source_dir = Path('data/source_codes')
    output_dir = Path('data/extracted_features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    errors = []
    
    print("\n" + "="*60)
    print(f"MODEL: {model_key.upper()}")
    print("="*60)
    print(f"API Model: {model_config['name']}")
    print(f"Output: {model_config['output']}")
    print("="*60)
    
    source_files = sorted(source_dir.glob('*.c'))
    print(f"\nProcessing {len(source_files)} files...\n")
    
    start_time = time.time()
    
    for i, filepath in enumerate(source_files, 1):
        print(f"[{i}/{len(source_files)}] {filepath.name}...", end=' ', flush=True)
        
        try:
            features, error = extract_llm_features(filepath, model_config['name'])
            
            if error:
                print(f"✗ {error}")
                errors.append({'file': filepath.name, 'error': error})
            else:
                print("✓")
                all_features.append(features)
                
            time.sleep(0.5)
            
        except Exception as e:
            print(f"✗ Exception: {e}")
            errors.append({'file': filepath.name, 'error': str(e)})
    
    elapsed_time = time.time() - start_time
    
    # Save results
    output_file = output_dir / model_config['output']
    with open(output_file, 'w') as f:
        json.dump(all_features, f, indent=2)
    
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print(f"✓ Successfully extracted: {len(all_features)}/{len(source_files)} files")
    print(f"✓ Saved to: {output_file}")
    print(f"⏱️  Time: {elapsed_time/60:.1f} minutes")
    
    if errors:
        print(f"\n⚠️  Errors: {len(errors)}")
        error_file = output_dir / f'llm_extraction_errors_{model_key}.json'
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"   Error log: {error_file}")
    
    # Estimate cost
    avg_tokens_per_file = 3000
    total_tokens = avg_tokens_per_file * len(all_features)
    estimated_cost = (total_tokens / 1_000_000) * model_config['cost_per_1m']
    
    print(f"\n💰 Estimated cost: ${estimated_cost:.4f}")
    
    return len(all_features), estimated_cost

def main():
    print("="*60)
    print("LLM FEATURE EXTRACTION - ALL MODELS")
    print("="*60)
    print("Models to run:")
    for key, config in MODELS.items():
        print(f"  - {key}: {config['name']}")
    print("="*60)
    
    results = {}
    total_cost = 0
    
    for model_key, model_config in MODELS.items():
        success_count, cost = run_model(model_key, model_config)
        results[model_key] = {
            'success': success_count,
            'cost': cost
        }
        total_cost += cost
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for model_key, stats in results.items():
        print(f"{model_key.upper()}:")
        print(f"  Success: {stats['success']}/42 files")
        print(f"  Cost: ${stats['cost']:.4f}")
    
    print(f"\n💰 TOTAL COST: ${total_cost:.4f}")
    print("\n✓ All models complete!")
    print("\nNext step: Compare results with comparison script")

if __name__ == '__main__':
    main()
