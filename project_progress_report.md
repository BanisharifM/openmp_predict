# OpenMP Performance Prediction Project - Progress Report

**Date:** November 11, 2025  
**Project:** LLM-Enhanced Feature Extraction for OpenMP Performance Prediction  
**Status:** Feature Extraction Complete ✅ | Ready for Model Training 🚀

---

## Executive Summary

Successfully completed **Steps 1-9** of the project pipeline, establishing a comprehensive dataset combining static code analysis with LLM-based semantic feature extraction. Three state-of-the-art LLMs (DeepSeek, Claude Sonnet 4.5, GPT-4o) were used to extract semantic features from 42 OpenMP programs, achieving a unique multi-model comparison framework.

**Key Achievement:** Created 12 production-ready datasets (3 LLMs × 2 platforms × 2 versions) for comprehensive model training and comparison.

---

## Project Structure

```
openmp_predict/
├── data/
│   ├── source_codes/           # 42 OpenMP C programs
│   ├── extracted_features/
│   │   ├── static_features.json              # Tree-sitter analysis (17 features)
│   │   ├── llm_features_deepseek.json       # DeepSeek V3 (~$0.06)
│   │   ├── llm_features_claude.json         # Claude Sonnet 4.5 (~$1.10)
│   │   ├── llm_features_gpt4o.json          # GPT-4o (~$0.52)
│   │   └── llm_agreement_statistics.csv     # Inter-model comparison
│   ├── merged/
│   │   ├── deepseek_features_*_*.csv        # 4 files (2 platforms × 2 versions)
│   │   ├── claude_features_*_*.csv          # 4 files
│   │   └── gpt4o_features_*_*.csv           # 4 files
│   └── raw/
│       ├── profiling_data_tx2_v2.csv        # Jetson TX2: 20,160 samples
│       └── profiling_data_rubikpi_v2.csv    # RubikPi: 26,880 samples
├── src/
│   └── feature_extraction/
│       ├── extract_features.py              # Static analysis (Tree-sitter)
│       ├── llm_extract_all_models.py        # Multi-LLM extraction
│       ├── compare_llm_models.py            # LLM comparison analysis
│       ├── merge_each_llm.py                # Dataset creation per LLM
│       └── [3 other utility scripts]
└── logs/                                     # Execution logs (4 files)
```

---

## Completed Steps (1-9)

### ✅ Steps 1-3: Environment Setup
- **Step 1:** Project structure created
- **Step 2:** Virtual environment configured with all dependencies
- **Step 3:** Tree-sitter C parser installed and verified

**Dependencies Installed:**
- Tree-sitter (0.20.2) for static analysis
- OpenRouter API integration
- Standard ML libraries (pandas, numpy, scikit-learn)

---

### ✅ Steps 4-6: Static Feature Extraction

**Method:** Tree-sitter AST parsing  
**Output:** `data/extracted_features/static_features.json`

**Extracted Features (17):**
1. `loop_depth` (0-4)
2. `loop_count` (0-45)
3. `lines_of_code` (88-4821)
4. `estimated_arithmetic_intensity` (0.0-284.2)
5. `total_pragmas` (1-61)
6. `parallel_regions` (1-2)
7. `synchronization_points` (0-11)
8. `critical_sections` (0-5)
9. `barriers` (0)
10. `atomics` (0-6)
11. `tasks` (0-57)
12. `thread_count` (-1, dynamic)
13. `schedule_type` (static/dynamic/none)
14. `schedule_encoding` (-1, 0, 1)
15. `shared_vars` (0-36)
16. `private_vars` (0-36)
17. `reduction_vars` (0)

**Coverage:** 42/42 programs (100% success rate)

---

### ✅ Steps 7-9: LLM-Based Semantic Feature Extraction

**Approach:** Multi-model comparison using OpenRouter API

#### Models Used:
1. **DeepSeek V3** (`deepseek/deepseek-chat`)
   - Cost: ~$0.06
   - Fastest, most economical
   
2. **Claude Sonnet 4.5** (`anthropic/claude-sonnet-4.5`)
   - Cost: ~$1.10
   - Best for code reasoning (#1 in Technology)
   
3. **GPT-4o** (`openai/gpt-4o-2024-08-06`)
   - Cost: ~$0.52
   - Production-stable

**Total LLM Cost:** ~$1.68

#### LLM-Extracted Features (14 categorical):
1. `memory_access_pattern` (unit_stride/non_unit_stride/random/mixed)
2. `spatial_locality` (high/medium/low)
3. `temporal_locality` (high/medium/low)
4. `data_dependency_type` (none/loop_carried/cross_iteration/complex)
5. `false_sharing_risk` (high/medium/low/none)
6. `load_balance_characteristic` (uniform/irregular/dynamic)
7. `algorithmic_complexity` (O(n)/O(n²)/O(n³)/O(nlogn)/other)
8. `dominant_operation` (arithmetic/memory/logic/mixed)
9. `vectorization_potential` (high/medium/low)
10. `numa_sensitivity` (high/medium/low)
11. `cache_behavior_pattern` (streaming/random/blocked/mixed)
12. `parallelization_overhead` (low/medium/high)
13. `scalability_bottleneck` (none/memory_bandwidth/synchronization/load_imbalance/other)
14. `estimated_iteration_count` (integer or -1)

**All features automatically label-encoded for ML models.**

---

## Key Findings: LLM Comparison

### Inter-Model Agreement Analysis

**Overall Agreement (All 3 LLMs):** 36.3%

**Most Reliable Features:**
- `dominant_operation`: **73.8%** agreement ✅
- `algorithmic_complexity`: **59.5%** agreement ✅
- `temporal_locality`: **47.6%** agreement ✅

**Most Subjective Features:**
- `false_sharing_risk`: **14.3%** agreement ⚠️
- `cache_behavior_pattern`: **16.7%** agreement ⚠️
- `memory_access_pattern`: **21.4%** agreement ⚠️

**Pairwise Model Similarity:**
- DeepSeek ↔ GPT-4o: **61.9%** (most similar)
- DeepSeek ↔ Claude: **54.4%**
- Claude ↔ GPT-4o: **49.6%** (most different)

**Notable Case - Strassen Algorithm Complexity:**
- DeepSeek: O(n³) ❌
- Claude: **O(n^2.807)** ✅ (CORRECT! Strassen's actual complexity)
- GPT-4o: O(n²) ❌

**Implication for Paper:** LLMs show subjective interpretations of code semantics. Agreement varies significantly by feature type, providing rich material for discussion.

---

## Dataset Summary

### Created Datasets

**12 CSV files organized by:**
- **LLM Model:** DeepSeek, Claude, GPT-4o
- **Platform:** TX2 (Jetson TX2), RubikPi
- **Version:** 
  - `with_benchmark`: Includes application_id + benchmark name (for configuration prediction)
  - `no_benchmark`: Excludes identifiers (for pure feature-based generalization)

### Dataset Statistics

**TX2 Platform:**
- Samples: 20,160
- Columns: 94-96 (depending on version)
- Target (time_elapsed): Mean 4.28s, Std 12.07s, Range 0.0063-209.33s

**RubikPi Platform:**
- Samples: 26,880
- Columns: 182-184 (depending on version)
- Target (time_elapsed): Mean 3.97s, Std 18.54s, Range 0.0043-251.43s

### Feature Composition

**Per Dataset:**
- Profiling features: 65 (TX2) / 153 (RubikPi)
- Static features: 17
- LLM features: 14 (encoded as integers)
- **Total features: ~96 (TX2) / ~184 (RubikPi)**

---

## Next Steps (10-25): Model Training Pipeline

### Phase 3: Data Preparation (Steps 10-11)
**Status:** Ready to start ▶️

**Tasks:**
1. Feature selection and importance analysis
2. Feature scaling/normalization
3. Handle missing values (if any)
4. Create engineered features (e.g., IPC, cache hit rate)
5. Split strategies:
   - `with_benchmark`: Standard KFold (5-10 folds)
   - `no_benchmark`: GroupKFold by benchmark (test on unseen programs)

**Expected Time:** 1-2 hours

---

### Phase 4: Baseline Model Training (Steps 12-15)
**Status:** Pending

**Models to Train:**
1. **Random Forest Regressor**
   - Quick baseline
   - Feature importance analysis
   
2. **XGBoost Regressor**
   - Industry standard
   - Excellent performance
   
3. **LightGBM Regressor**
   - Faster than XGBoost
   - Memory efficient
   
4. **Gradient Boosting Regressor**
   - sklearn baseline

**Evaluation Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

**Expected Time:** 2-3 hours

---

### Phase 5: Hyperparameter Optimization (Steps 16-18)
**Status:** Pending

**Approach:**
- RandomizedSearchCV or Optuna
- 5-fold cross-validation
- Optimize for RMSE

**Parameters to Tune:**
- Learning rate
- Max depth
- Number of estimators
- Regularization parameters

**Expected Time:** 3-4 hours

---

### Phase 6: LLM Feature Comparison (Steps 19-21)
**Status:** Pending - **CRITICAL FOR PAPER**

**Comparison Framework:**

Train models on 4 feature combinations:
1. **Static Only** (baseline): 17 features
2. **Static + DeepSeek**: 31 features
3. **Static + Claude**: 31 features
4. **Static + GPT-4o**: 31 features

**Analysis:**
- Which LLM features improve performance most?
- Are expensive LLMs worth it? (Claude/GPT vs DeepSeek)
- Which semantic features are most predictive?
- Feature importance comparison across LLMs

**Expected Time:** 2-3 hours

---

### Phase 7: Advanced Techniques (Steps 22-23)
**Status:** Pending

**Techniques:**
1. **Ensemble Methods:**
   - Stacking (meta-model on top of base models)
   - Weighted averaging
   - Majority voting for LLM features
   
2. **Feature Engineering:**
   - Interaction terms
   - Polynomial features
   - Domain-specific ratios

**Expected Time:** 2-3 hours

---

### Phase 8: Results & Visualization (Steps 24-25)
**Status:** Pending

**Deliverables:**
1. **Performance Tables:**
   - Model comparison by LLM
   - Cross-platform analysis (TX2 vs RubikPi)
   - With/without benchmark comparison
   
2. **Visualizations:**
   - Feature importance plots
   - Prediction vs Actual scatter plots
   - Learning curves
   - LLM agreement vs model accuracy correlation
   - Residual analysis
   
3. **Statistical Tests:**
   - Paired t-tests between models
   - Wilcoxon signed-rank tests
   - Effect size calculations

**Expected Time:** 2-3 hours

---

## Recommended Training Order

### **Priority 1: Quick Validation (1-2 hours)**
1. Train baseline RF model on TX2 with DeepSeek features
2. Verify pipeline works end-to-end
3. Get initial R² baseline

### **Priority 2: LLM Comparison (2-3 hours)**
1. Train on all 3 LLM feature sets (TX2 platform)
2. Compare performance: Static vs Static+LLM
3. Generate comparison table for paper

### **Priority 3: Full Pipeline (4-6 hours)**
1. Hyperparameter optimization
2. Cross-platform validation (apply to RubikPi)
3. Ensemble methods

### **Priority 4: Paper-Ready Results (2-3 hours)**
1. Generate all visualizations
2. Statistical significance tests
3. Create publication-quality tables

---

## Key Commands Reference

### Run Feature Extraction (Already Done ✅)
```bash
# Static features
python src/feature_extraction/extract_features.py

# LLM features (all 3 models)
python src/feature_extraction/llm_extract_all_models.py

# Compare LLMs
python src/feature_extraction/compare_llm_models.py

# Create datasets
python src/feature_extraction/merge_each_llm.py
```

### Next Commands (To Be Created)
```bash
# Data preparation
python src/data_preparation/prepare_data.py

# Train baseline models
python src/modeling/train_baseline.py --platform tx2 --llm deepseek

# Compare LLMs
python src/modeling/compare_llm_features.py --platform tx2

# Hyperparameter tuning
python src/modeling/optimize_hyperparameters.py

# Generate results
python src/visualization/create_publication_plots.py
```

---

## Research Questions for Paper

### Primary Research Questions:
1. **RQ1:** Can LLM-extracted semantic features improve OpenMP performance prediction?
   - Compare: Static-only vs Static+LLM
   
2. **RQ2:** Which LLM provides the most predictive features?
   - Compare: DeepSeek vs Claude vs GPT-4o
   
3. **RQ3:** Is there a correlation between LLM agreement and prediction accuracy?
   - Analyze: Features with high agreement → better predictions?
   
4. **RQ4:** Can models generalize to unseen OpenMP programs?
   - Evaluate: GroupKFold results (no benchmark version)

### Secondary Research Questions:
5. Which semantic features are most predictive of performance?
6. How does feature quality vary by cost (DeepSeek $0.06 vs Claude $1.10)?
7. Can majority voting across LLMs improve feature quality?

---

## Expected Paper Contributions

1. **Novel Feature Extraction:** First use of multi-model LLM comparison for code analysis
2. **Comprehensive Evaluation:** 3 LLMs × 2 platforms × 42 programs
3. **Inter-Model Analysis:** Quantified agreement rates and disagreement patterns
4. **Cost-Benefit Analysis:** Performance vs cost tradeoffs
5. **Generalization Study:** Both configuration prediction and program generalization
6. **Open Dataset:** All features and code released for reproducibility

---

## Technical Specifications

**Hardware Tested:**
- Jetson TX2 (ARM, 6 cores)
- RubikPi (ARM, 8 cores)

**Software Stack:**
- Python 3.11
- Tree-sitter 0.20.2
- OpenRouter API
- scikit-learn, XGBoost, LightGBM

**LLM APIs:**
- DeepSeek V3 via OpenRouter
- Claude Sonnet 4.5 via OpenRouter
- GPT-4o via OpenRouter

---

## Current Blockers & Risks

### Blockers: None ✅
All dependencies installed, all data prepared, ready to proceed.

### Potential Risks:
1. **Overfitting Risk:** High-dimensional data (180+ features)
   - Mitigation: Use regularization, feature selection
   
2. **Class Imbalance:** Wide range in target variable (0.006s - 251s)
   - Mitigation: Log transformation, stratified splits
   
3. **Generalization Challenge:** Only 42 unique programs
   - Mitigation: GroupKFold, leave-one-benchmark-out CV

---

## Estimated Timeline to Completion

**Conservative Estimate:**
- Data Preparation: 2 hours
- Baseline Training: 3 hours
- Optimization: 4 hours
- LLM Comparison: 3 hours
- Results & Visualization: 3 hours
- **Total: 15-20 hours of active work**

**Optimistic Estimate:**
- With parallel execution and efficient coding: 10-12 hours

---

## Contact & Handoff Information

**Current Session:**
- Platform: Claude.ai
- Started: November 11, 2025
- Messages Used: ~85% of context

**To Continue in New Chat:**
1. Upload this report
2. Reference: "Continue from Phase 3: Model Training"
3. Specify platform: TX2 or RubikPi
4. Specify LLM: DeepSeek, Claude, or GPT-4o

**Critical Files to Reference:**
- `data/merged/deepseek_features_no_benchmark_tx2.csv` (recommended starting point)
- `data/extracted_features/llm_agreement_statistics.csv` (for analysis)

---

## Appendix: File Sizes & Statistics

**Dataset Sizes:**
- TX2 datasets: ~15-20 MB each
- RubikPi datasets: ~25-30 MB each
- Total merged data: ~200 MB

**Feature JSONs:**
- static_features.json: ~50 KB
- llm_features_*.json: ~100 KB each

**Memory Requirements:**
- TX2 training: ~2-4 GB RAM
- RubikPi training: ~4-6 GB RAM
- Safe to run on 16GB system

---

## Success Metrics

**Project Success Criteria:**
1. ✅ Static feature extraction: 100% coverage (42/42)
2. ✅ LLM feature extraction: 100% coverage across 3 models
3. ⏳ Baseline model: R² > 0.7
4. ⏳ LLM improvement: >5% R² gain over static-only
5. ⏳ Cross-platform validation: Consistent results TX2 ↔ RubikPi
6. ⏳ Generalization: R² > 0.6 on unseen programs (GroupKFold)

---

## End of Report

**Status:** Ready for Model Training Phase ✅  
**Next Action:** Create `src/modeling/train_baseline.py`  
**Priority:** LLM Feature Comparison for paper contribution

**Total Cost Investment:** $1.68 (LLM APIs)  
**Total Time Investment:** ~8-10 hours (feature extraction phase)  
**Remaining Work:** ~15-20 hours (model training phase)

---

*Report Generated: November 11, 2025*  
*Project: OpenMP Performance Prediction with LLM-Enhanced Features*  
*Status: 36% Complete (Steps 1-9 of 25)*
