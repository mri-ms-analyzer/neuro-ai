"""
MS Framework Expert Evaluation - Integrated Analysis System
============================================================
This script integrates analysis, table generation, and visualization
into a unified workflow that automatically extracts data from evaluation
forms and produces all outputs.

Author: Mahdi Bashiri Bawil
Date: 2025-09-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, pearsonr
import warnings
import glob
import os
from datetime import datetime
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Professional color palette
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60', '#E67E22', '#3498DB']


# ============================================================================
# FIX 1: CORRECT CORRELATION/MAE CALCULATION
# ============================================================================

def calculate_correlation_stats(ai_values, expert_values, case_ids):
    """
    Calculate comprehensive correlation statistics with confidence intervals.
    
    Parameters:
    -----------
    ai_values : np.array
        AI predictions (per-case means)
    expert_values : np.array
        Expert assessments (per-case means)
    case_ids : list
        Case identifiers
        
    Returns:
    --------
    dict : All statistics including CIs
    """
    n = len(ai_values)
    
    # Pearson correlation
    r, p_value = pearsonr(ai_values, expert_values)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(ai_values - expert_values))
    
    # 95% CI for correlation (Fisher's z-transformation)
    r_ci_lower, r_ci_upper = fisher_z_confidence_interval(r, n)
    
    # 95% CI for MAE (bootstrap)
    mae_ci_lower, mae_ci_upper = bootstrap_mae_confidence_interval(ai_values, expert_values)
    
    # Additional statistics
    mean_diff = np.mean(ai_values - expert_values)
    std_diff = np.std(ai_values - expert_values, ddof=1)
    
    # Agreement within ±1 lesion
    agreement_1 = np.sum(np.abs(ai_values - expert_values) <= 1) / n * 100
    
    return {
        'n': n,
        'r': r,
        'p_value': p_value,
        'r_ci_lower': r_ci_lower,
        'r_ci_upper': r_ci_upper,
        'mae': mae,
        'mae_ci_lower': mae_ci_lower,
        'mae_ci_upper': mae_ci_upper,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'agreement_within_1': agreement_1,
        'ai_mean': np.mean(ai_values),
        'ai_std': np.std(ai_values, ddof=1),
        'expert_mean': np.mean(expert_values),
        'expert_std': np.std(expert_values, ddof=1),
        'case_ids': list(case_ids)
    }


# ============================================================================
# FIX 2: CONFIDENCE INTERVAL CALCULATIONS
# ============================================================================

def fisher_z_confidence_interval(r, n, alpha=0.05):
    """
    Calculate 95% confidence interval for Pearson correlation
    using Fisher's z-transformation.
    
    Parameters:
    -----------
    r : float
        Pearson correlation coefficient
    n : int
        Sample size (number of cases)
    alpha : float
        Significance level (default: 0.05 for 95% CI)
        
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    
    Reference:
    ----------
    Fisher, R. A. (1915). Frequency distribution of the values of the 
    correlation coefficient in samples from an indefinitely large population.
    Biometrika, 10(4), 507-521.
    """
    # Fisher's z-transformation
    z = np.arctanh(r)
    
    # Standard error
    se = 1 / np.sqrt(n - 3)
    
    # Critical value
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Confidence interval in z-space
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    
    # Transform back to correlation space
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)
    
    return r_lower, r_upper


def bootstrap_mae_confidence_interval(ai_values, expert_values, 
                                     n_bootstrap=1000, alpha=0.05):
    """
    Calculate 95% confidence interval for MAE using bootstrap resampling.
    
    Parameters:
    -----------
    ai_values : np.array
        AI predictions
    expert_values : np.array
        Expert assessments  
    n_bootstrap : int
        Number of bootstrap iterations (default: 1000)
    alpha : float
        Significance level (default: 0.05 for 95% CI)
        
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    
    Reference:
    ----------
    Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap.
    CRC press.
    """
    n = len(ai_values)
    bootstrap_maes = []
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.randint(0, n, size=n)
        ai_boot = ai_values[indices]
        expert_boot = expert_values[indices]
        
        # Calculate MAE for this bootstrap sample
        mae_boot = np.mean(np.abs(ai_boot - expert_boot))
        bootstrap_maes.append(mae_boot)
    
    # Percentile-based confidence interval
    ci_lower = np.percentile(bootstrap_maes, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_maes, (1 - alpha/2) * 100)
    
    return ci_lower, ci_upper


# ============================================================================
# FIX 3: CORRECT COHEN'S D CALCULATION
# ============================================================================

def calculate_cohens_d_CORRECTED(traditional_time, automated_time):
    """
    CORRECTED Cohen's d using pooled standard deviation.
    
    This should REPLACE the calculation at line 193 in ms_evaluation_integrated.py
    
    Parameters:
    -----------
    traditional_time : list or np.array
        Traditional analysis times
    automated_time : list or np.array
        Automated analysis times
        
    Returns:
    --------
    float : Cohen's d effect size
    
    Reference:
    ----------
    Cohen, J. (1988). Statistical power analysis for the behavioral sciences
    (2nd ed.). Hillsdale, NJ: Lawrence Erlbaum Associates.
    """
    traditional_time = np.array(traditional_time)
    automated_time = np.array(automated_time)
    
    # Mean difference
    mean_diff = np.mean(traditional_time) - np.mean(automated_time)
    
    # Pooled standard deviation
    var_trad = np.var(traditional_time, ddof=1)
    var_auto = np.var(automated_time, ddof=1)
    pooled_std = np.sqrt((var_trad + var_auto) / 2)
    
    # Cohen's d
    cohens_d = mean_diff / pooled_std
    
    return cohens_d


# ============================================================================
# REPLACEMENT FOR MS_TABLE_GENERATOR.PY
# ============================================================================

def create_dataset_comparison_table(self):
    """
    NEW TABLE: Compares performance across institutional vs external datasets.
    Add this as Table 7 or supplementary table.
    """
    print("\nGenerating Dataset Comparison Table...")
    
    if 'correlation_analysis_corrected' not in self.results:
        print("ERROR: Run analyze_lesion_correlations_CORRECTED() first!")
        return None
    
    corr_results = self.results['correlation_analysis_corrected']
    lesion_types = ['Periventricular', 'Paraventricular', 'Juxtacortical', 'Total_Lesions']
    
    comparison_data = []
    
    for lesion_type in lesion_types:
        row = {'Lesion Type': lesion_type.replace('_', ' ')}
        
        for dataset in ['overall', 'institutional', 'external']:
            if lesion_type in corr_results[dataset]:
                stats = corr_results[dataset][lesion_type]
                row[f'{dataset.title()} n'] = stats['n']
                row[f'{dataset.title()} r'] = f"{stats['r']:.3f}"
                row[f'{dataset.title()} MAE'] = f"{stats['mae']:.1f}"
        
        comparison_data.append(row)
    
    table = pd.DataFrame(comparison_data)
    
    csv_path = os.path.join(self.output_dir, 'Table_Dataset_Comparison.csv')
    table.to_csv(csv_path, index=False)
    
    print(f"✓ Saved to {csv_path}")
    return table


class MSEvaluationAnalyzer:
    """
    Main class for MS Framework evaluation analysis.
    Handles data loading, analysis, and output generation.
    """
    
    def __init__(self, overall_csv_path, case_csv_pattern, output_dir='MS_Evaluation_Results'):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        overall_csv_path : str
            Path to the overall evaluation CSV file
        case_csv_pattern : str
            Pattern for case evaluation files (e.g., 'ms_case_evaluations_MSSEG_all_*.csv')
        output_dir : str
            Directory for output files
        """
        self.overall_csv_path = overall_csv_path
        self.case_csv_pattern = case_csv_pattern
        self.output_dir = output_dir
        
        # Data containers
        self.overall_data = None
        self.combined_case_data = None
        self.available_cases = []
        
        # Analysis results
        self.results = {
            'expert_info': {},
            'time_analysis': {},
            'performance_metrics': {},
            'case_summary': {},
            'clinical_utility': {},
            'implementation': {},
            'educational': {},
            'overall_assessment': {}
        }
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Created output directory: {output_dir}")
    
    def load_data(self):
        """Load all evaluation data files."""
        print("\n" + "=" * 70)
        print("LOADING EVALUATION DATA")
        print("=" * 70)
        
        # Load overall evaluation data
        try:
            self.overall_data = pd.read_csv(self.overall_csv_path)
            print(f"✓ Loaded overall evaluation: {self.overall_data.shape}")
        except FileNotFoundError:
            print(f"✗ Error: Could not find {self.overall_csv_path}")
            return False
        
        # Load case evaluation data
        case_files = glob.glob(self.case_csv_pattern)
        all_case_data = []
        
        for case_file in sorted(case_files):
            try:
                case_data = pd.read_csv(case_file)
                all_case_data.append(case_data)
                print(f"✓ Loaded {os.path.basename(case_file)}: {case_data.shape}")
            except Exception as e:
                print(f"✗ Error loading {case_file}: {e}")
        
        if all_case_data:
            self.combined_case_data = pd.concat(all_case_data, ignore_index=True)
            self.available_cases = sorted(self.combined_case_data['Case_ID'].unique())
            print(f"✓ Combined case data: {self.combined_case_data.shape}")
            print(f"✓ Available cases: {', '.join(self.available_cases)}")
            return True
        else:
            print("✗ No case evaluation files found!")
            return False
    
    def analyze_expert_info(self):
        """Extract expert demographic information."""
        print("\n" + "=" * 70)
        print("ANALYZING EXPERT PROFILES")
        print("=" * 70)
        
        expert_info = self.overall_data[self.overall_data['Section'] == 'Expert_Info'].copy()
        
        # Extract expert data
        for idx, row in expert_info.iterrows():
            question = row['Question_Text']
            if 'Experience' in question:
                self.results['expert_info']['years_experience'] = [
                    row['Response_Value_Expert_1'],
                    row['Response_Value_Expert_2'],
                    row['Response_Value_Expert_3']
                ]
            elif 'Position' in question:
                self.results['expert_info']['positions'] = [
                    row['Response_Value_Expert_1'],
                    row['Response_Value_Expert_2'],
                    row['Response_Value_Expert_3']
                ]
            elif 'Institution' in question:
                self.results['expert_info']['institutions'] = [
                    row['Response_Value_Expert_1'],
                    row['Response_Value_Expert_2'],
                    row['Response_Value_Expert_3']
                ]
        
        print("✓ Expert information extracted")
    
    def analyze_time_efficiency(self):
        """Analyze time efficiency metrics."""
        print("\n" + "=" * 70)
        print("ANALYZING TIME EFFICIENCY")
        print("=" * 70)
        
        time_data = self.overall_data[self.overall_data['Section'] == 'Time_Analysis'].copy()
        traditional_time = []
        automated_time = []
        
        for idx, row in time_data.iterrows():
            if 'Traditional Analysis Time' in row['Question_Text']:
                traditional_time = [
                    float(row['Response_Value_Expert_1']),
                    float(row['Response_Value_Expert_2']),
                    float(row['Response_Value_Expert_3'])
                ]
            elif 'Automated Analysis' in row['Question_Text'] and 'Review Time' in row['Question_Text']:
                automated_time = [
                    float(row['Response_Value_Expert_1']),
                    float(row['Response_Value_Expert_2']),
                    float(row['Response_Value_Expert_3'])
                ]
        
        if traditional_time and automated_time:
            time_reduction = np.array(traditional_time) - np.array(automated_time)
            time_reduction_percent = (time_reduction / np.array(traditional_time)) * 100
            
            # Statistical analysis
            t_stat, p_value = stats.ttest_rel(traditional_time, automated_time)
            effect_size = calculate_cohens_d_CORRECTED(traditional_time, automated_time)            
            
            self.results['time_analysis'] = {
                'traditional_time': traditional_time,
                'automated_time': automated_time,
                'time_reduction': time_reduction.tolist(),
                'time_reduction_percent': time_reduction_percent.tolist(),
                'mean_traditional': np.mean(traditional_time),
                'mean_automated': np.mean(automated_time),
                'mean_reduction': np.mean(time_reduction),
                'mean_reduction_percent': np.mean(time_reduction_percent),
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size
            }
            
            print(f"Traditional Time: {traditional_time} minutes")
            print(f"Automated Time: {automated_time} minutes")
            print(f"Mean Reduction: {np.mean(time_reduction):.1f} minutes ({np.mean(time_reduction_percent):.1f}%)")
            print(f"Statistical: t={t_stat:.2f}, p={p_value:.4f}, d={effect_size:.2f}")
            print("✓ Time efficiency analysis complete")
    
    def analyze_performance_metrics(self):
        """Analyze comparative performance metrics."""
        print("\n" + "=" * 70)
        print("ANALYZING PERFORMANCE METRICS")
        print("=" * 70)
        
        comparison_data = self.overall_data[self.overall_data['Section'] == 'Comparison'].copy()
        
        metrics = {}
        for idx, row in comparison_data.iterrows():
            question = row['Question_Text']
            expert1 = float(row['Response_Value_Expert_1']) if pd.notna(row['Response_Value_Expert_1']) else np.nan
            expert2 = float(row['Response_Value_Expert_2']) if pd.notna(row['Response_Value_Expert_2']) else np.nan
            expert3 = float(row['Response_Value_Expert_3']) if pd.notna(row['Response_Value_Expert_3']) else np.nan
            
            if 'Traditional' in question:
                metric_name = question.split(' - Traditional')[0]
                if metric_name not in metrics:
                    metrics[metric_name] = {'Traditional': [], 'Automated': []}
                metrics[metric_name]['Traditional'] = [expert1, expert2, expert3]
            elif 'Automated' in question:
                metric_name = question.split(' - Automated')[0]
                if metric_name not in metrics:
                    metrics[metric_name] = {'Traditional': [], 'Automated': []}
                metrics[metric_name]['Automated'] = [expert1, expert2, expert3]
        
        # Calculate statistics for each metric
        for metric, values in metrics.items():
            if values['Traditional'] and values['Automated']:
                trad = np.array(values['Traditional'])
                auto = np.array(values['Automated'])
                
                metrics[metric]['trad_mean'] = np.nanmean(trad)
                metrics[metric]['trad_std'] = np.nanstd(trad)
                metrics[metric]['auto_mean'] = np.nanmean(auto)
                metrics[metric]['auto_std'] = np.nanstd(auto)
                metrics[metric]['improvement'] = metrics[metric]['auto_mean'] - metrics[metric]['trad_mean']
                
                # Statistical test
                try:
                    stat, p_val = wilcoxon(trad[~np.isnan(trad)], auto[~np.isnan(auto)])
                    metrics[metric]['p_value'] = p_val
                except:
                    metrics[metric]['p_value'] = 1.0
        
        self.results['performance_metrics'] = metrics
        
        print(f"✓ Analyzed {len(metrics)} performance metrics")
        for metric, data in metrics.items():
            if 'trad_mean' in data:
                print(f"  {metric}: Trad={data['trad_mean']:.1f} → Auto={data['auto_mean']:.1f} (Δ{data['improvement']:+.1f})")
    
    def analyze_case_evaluations(self):
        """Analyze individual case evaluations."""
        print("\n" + "=" * 70)
        print("ANALYZING CASE EVALUATIONS")
        print("=" * 70)
        
        case_summary = {}
        
        for case_id in self.available_cases:
            case_data = self.combined_case_data[self.combined_case_data['Case_ID'] == case_id].copy()
            
            case_summary[case_id] = {
                'segmentation_accuracy': [],
                'detection_completeness': [],
                'classification_accuracy': [],
                'lesion_counts': {}
            }
            
            # Segmentation metrics
            seg_accuracy = case_data[case_data['Question_ID'] == 'A1_001']
            detection_completeness = case_data[case_data['Question_ID'] == 'A1_002']
            classification_accuracy = case_data[case_data['Question_ID'] == 'A1_003']
            
            if not seg_accuracy.empty:
                seg_row = seg_accuracy.iloc[0]
                case_summary[case_id]['segmentation_accuracy'] = [
                    float(seg_row['Response_Value_Expert_1']),
                    float(seg_row['Response_Value_Expert_2']),
                    float(seg_row['Response_Value_Expert_3'])
                ]
            
            if not detection_completeness.empty:
                det_row = detection_completeness.iloc[0]
                case_summary[case_id]['detection_completeness'] = [
                    float(det_row['Response_Value_Expert_1']),
                    float(det_row['Response_Value_Expert_2']),
                    float(det_row['Response_Value_Expert_3'])
                ]
            
            if not classification_accuracy.empty:
                class_row = classification_accuracy.iloc[0]
                case_summary[case_id]['classification_accuracy'] = [
                    float(class_row['Response_Value_Expert_1']),
                    float(class_row['Response_Value_Expert_2']),
                    float(class_row['Response_Value_Expert_3'])
                ]
            
            # Lesion count analysis
            lesion_types = ['Periventricular', 'Paraventricular', 'Juxtacortical']
            for lesion_type in lesion_types:
                auto_count_rows = case_data[
                    case_data['Question_Text'].str.contains(f'{lesion_type} - Automated Count', na=False)]
                expert_count_rows = case_data[
                    case_data['Question_Text'].str.contains(f'{lesion_type} - Expert Count', na=False)]
                
                if not auto_count_rows.empty and not expert_count_rows.empty:
                    auto_row = auto_count_rows.iloc[0]
                    expert_row = expert_count_rows.iloc[0]
                    
                    auto_counts = [
                        int(float(auto_row['Response_Value_Expert_1'])),
                        int(float(auto_row['Response_Value_Expert_2'])),
                        int(float(auto_row['Response_Value_Expert_3']))
                    ]
                    expert_counts = [
                        int(float(expert_row['Response_Value_Expert_1'])),
                        int(float(expert_row['Response_Value_Expert_2'])),
                        int(float(expert_row['Response_Value_Expert_3']))
                    ]
                    
                    case_summary[case_id]['lesion_counts'][lesion_type] = {
                        'automated': auto_counts,
                        'expert': expert_counts
                    }
        
        self.results['case_summary'] = case_summary
        print(f"✓ Analyzed {len(case_summary)} cases")

    def analyze_lesion_correlations_CORRECTED(self):
        """
        CORRECTED METHOD: Calculate correlations on per-case means, not individual ratings.
        This replaces the incorrect method in ms_table_generator.py

        This is the CRITICAL fix for manuscript accuracy.
        """
        print("\n" + "=" * 70)
        print("ANALYZING LESION COUNT CORRELATIONS (CORRECTED METHOD)")
        print("=" * 70)

        case_summary = self.results['case_summary']
        lesion_types = ['Periventricular', 'Paraventricular', 'Juxtacortical']

        # Identify dataset types
        institutional_cases = []
        external_cases = []

        for case_id in case_summary.keys():
            if 'msseg' in case_id.lower():
                external_cases.append(case_id)
            else:
                institutional_cases.append(case_id)

        print(f"Dataset breakdown:")
        print(f"  Institutional cases (n={len(institutional_cases)}): {institutional_cases}")
        print(f"  External cases (n={len(external_cases)}): {external_cases}")
        print(f"  Total cases: {len(case_summary)}")

        # Initialize results
        correlation_results = {
            'overall': {},
            'institutional': {},
            'external': {}
        }

        # Calculate for each lesion type
        for lesion_type in lesion_types:
            print(f"\n  Processing {lesion_type} lesions...")

            # Collect per-case means for all cases
            all_ai_means = []
            all_expert_means = []
            all_case_ids = []

            for case_id, data in case_summary.items():
                if lesion_type in data.get('lesion_counts', {}):
                    counts = data['lesion_counts'][lesion_type]
                    # CRITICAL: Average across 3 experts per case
                    ai_mean = np.mean(counts['automated'])
                    expert_mean = np.mean(counts['expert'])
                    all_ai_means.append(ai_mean)
                    all_expert_means.append(expert_mean)
                    all_case_ids.append(case_id)

            # Overall statistics
            if len(all_ai_means) > 1:
                correlation_results['overall'][lesion_type] = calculate_correlation_stats(
                    np.array(all_ai_means),
                    np.array(all_expert_means),
                    all_case_ids
                )

            # Institutional statistics
            inst_ai = []
            inst_expert = []
            inst_ids = []
            for i, case_id in enumerate(all_case_ids):
                if case_id in institutional_cases:
                    inst_ai.append(all_ai_means[i])
                    inst_expert.append(all_expert_means[i])
                    inst_ids.append(case_id)

            if len(inst_ai) > 1:
                correlation_results['institutional'][lesion_type] = calculate_correlation_stats(
                    np.array(inst_ai),
                    np.array(inst_expert),
                    inst_ids
                )

            # External statistics
            ext_ai = []
            ext_expert = []
            ext_ids = []
            for i, case_id in enumerate(all_case_ids):
                if case_id in external_cases:
                    ext_ai.append(all_ai_means[i])
                    ext_expert.append(all_expert_means[i])
                    ext_ids.append(case_id)

            if len(ext_ai) > 1:
                correlation_results['external'][lesion_type] = calculate_correlation_stats(
                    np.array(ext_ai),
                    np.array(ext_expert),
                    ext_ids
                )

        # Calculate total lesions (sum across categories per case)
        for dataset_name in ['overall', 'institutional', 'external']:
            if dataset_name == 'overall':
                cases_to_use = list(case_summary.keys())
            elif dataset_name == 'institutional':
                cases_to_use = institutional_cases
            else:
                cases_to_use = external_cases

            total_ai_means = []
            total_expert_means = []

            for case_id in cases_to_use:
                if case_id in case_summary:
                    case_ai_sum = 0
                    case_expert_sum = 0
                    for lesion_type in lesion_types:
                        if lesion_type in case_summary[case_id].get('lesion_counts', {}):
                            counts = case_summary[case_id]['lesion_counts'][lesion_type]
                            case_ai_sum += np.mean(counts['automated'])
                            case_expert_sum += np.mean(counts['expert'])

                    total_ai_means.append(case_ai_sum)
                    total_expert_means.append(case_expert_sum)

            if len(total_ai_means) > 1:
                correlation_results[dataset_name]['Total_Lesions'] = calculate_correlation_stats(
                    np.array(total_ai_means),
                    np.array(total_expert_means),
                    cases_to_use
                )

        # Store results
        self.results['correlation_analysis_corrected'] = correlation_results

        # Print summary
        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS RESULTS (CORRECTED)")
        print("=" * 70)

        for dataset in ['overall', 'institutional', 'external']:
            if correlation_results[dataset]:
                print(f"\n{dataset.upper()} DATASET:")
                for lesion_type, stats_dict in correlation_results[dataset].items():
                    print(f"  {lesion_type}:")
                    print(f"    n = {stats_dict['n']}")
                    print(
                        f"    r = {stats_dict['r']:.3f}, 95% CI [{stats_dict['r_ci_lower']:.3f}, {stats_dict['r_ci_upper']:.3f}]")
                    print(f"    p = {stats_dict['p_value']:.4f}")
                    print(
                        f"    MAE = {stats_dict['mae']:.1f}, 95% CI [{stats_dict['mae_ci_lower']:.1f}, {stats_dict['mae_ci_upper']:.1f}]")

        return correlation_results

    def analyze_clinical_utility(self):
        """Analyze clinical utility assessment."""
        print("\n" + "=" * 70)
        print("ANALYZING CLINICAL UTILITY")
        print("=" * 70)
        
        clinical_utility = self.overall_data[self.overall_data['Section'] == 'Clinical_Utility'].copy()
        
        # Primary strengths
        strengths = clinical_utility[clinical_utility['Question_ID'].str.contains('C1_00[1-8]', regex=True)]
        strength_summary = {}
        
        for idx, row in strengths.iterrows():
            strength_name = row['Question_Text'].replace('Primary Strengths - ', '')
            e1 = int(row['Response_Value_Expert_1']) if pd.notna(row['Response_Value_Expert_1']) else 0
            e2 = int(row['Response_Value_Expert_2']) if pd.notna(row['Response_Value_Expert_2']) else 0
            e3 = int(row['Response_Value_Expert_3']) if pd.notna(row['Response_Value_Expert_3']) else 0
            strength_summary[strength_name] = {'agreement': e1 + e2 + e3, 'experts': [e1, e2, e3]}
        
        # Primary limitations
        limitations = clinical_utility[clinical_utility['Question_ID'].str.contains('C1_01[0-6]', regex=True)]
        limitation_summary = {}
        
        for idx, row in limitations.iterrows():
            limitation_name = row['Question_Text'].replace('Primary Limitations - ', '')
            e1 = int(row['Response_Value_Expert_1']) if pd.notna(row['Response_Value_Expert_1']) else 0
            e2 = int(row['Response_Value_Expert_2']) if pd.notna(row['Response_Value_Expert_2']) else 0
            e3 = int(row['Response_Value_Expert_3']) if pd.notna(row['Response_Value_Expert_3']) else 0
            limitation_summary[limitation_name] = {'agreement': e1 + e2 + e3, 'experts': [e1, e2, e3]}
        
        self.results['clinical_utility'] = {
            'strengths': strength_summary,
            'limitations': limitation_summary
        }
        
        print(f"✓ Identified {len(strength_summary)} strengths and {len(limitation_summary)} limitations")
    
    def analyze_implementation(self):
        """Analyze implementation readiness."""
        print("\n" + "=" * 70)
        print("ANALYZING IMPLEMENTATION READINESS")
        print("=" * 70)
        
        implementation = self.overall_data[self.overall_data['Section'] == 'Implementation'].copy()
        
        # Clinical readiness
        readiness_rows = implementation[implementation['Question_ID'] == 'C2_001']
        if not readiness_rows.empty:
            readiness = readiness_rows.iloc[0]
            self.results['implementation']['clinical_readiness'] = [
                readiness['Response_Value_Expert_1'],
                readiness['Response_Value_Expert_2'],
                readiness['Response_Value_Expert_3']
            ]
        
        # User benefit ranking
        user_benefits = implementation[implementation['Question_ID'].str.contains('C2_00[9]|C2_01[0-3]', regex=True)]
        benefit_rankings = {}
        
        for idx, row in user_benefits.iterrows():
            user_type = row['Question_Text'].replace('User Benefit Ranking - ', '')
            e1 = float(row['Response_Value_Expert_1'])
            e2 = float(row['Response_Value_Expert_2'])
            e3 = float(row['Response_Value_Expert_3'])
            benefit_rankings[user_type] = {
                'rankings': [e1, e2, e3],
                'mean': np.mean([e1, e2, e3])
            }
        
        self.results['implementation']['user_benefits'] = benefit_rankings
        print(f"✓ Analyzed implementation readiness for {len(benefit_rankings)} user types")
    
    def analyze_educational_value(self):
        """Analyze educational value assessment."""
        print("\n" + "=" * 70)
        print("ANALYZING EDUCATIONAL VALUE")
        print("=" * 70)
        
        educational = self.overall_data[self.overall_data['Section'] == 'Educational'].copy()
        educational_metrics = educational[educational['Question_ID'].str.contains('D1_00[2-5]', regex=True)]
        
        edu_summary = {}
        for idx, row in educational_metrics.iterrows():
            metric_name = row['Question_Text'].replace(' Educational Value (1-5)', '')
            e1 = float(row['Response_Value_Expert_1'])
            e2 = float(row['Response_Value_Expert_2'])
            e3 = float(row['Response_Value_Expert_3'])
            edu_summary[metric_name] = {
                'scores': [e1, e2, e3],
                'mean': np.mean([e1, e2, e3]),
                'std': np.std([e1, e2, e3])
            }
        
        self.results['educational'] = edu_summary
        print(f"✓ Analyzed {len(edu_summary)} educational metrics")
    
    def analyze_overall_assessment(self):
        """Analyze overall framework assessment."""
        print("\n" + "=" * 70)
        print("ANALYZING OVERALL ASSESSMENT")
        print("=" * 70)
        
        overall_assessment = self.overall_data[self.overall_data['Section'] == 'Overall_Assessment'].copy()
        final_rating_rows = overall_assessment[overall_assessment['Question_ID'] == 'F2_001']
        
        if not final_rating_rows.empty:
            final_rating = final_rating_rows.iloc[0]
            e1_rating = float(final_rating['Response_Value_Expert_1'])
            e2_rating = float(final_rating['Response_Value_Expert_2'])
            e3_rating = float(final_rating['Response_Value_Expert_3'])
            
            self.results['overall_assessment'] = {
                'ratings': [e1_rating, e2_rating, e3_rating],
                'mean': np.mean([e1_rating, e2_rating, e3_rating]),
                'std': np.std([e1_rating, e2_rating, e3_rating])
            }
            
            print(f"✓ Final ratings: {[e1_rating, e2_rating, e3_rating]} (Mean: {np.mean([e1_rating, e2_rating, e3_rating]):.1f})")
    
    def run_complete_analysis(self):
        """Run all analysis components."""
        print("\n" + "=" * 70)
        print("MS FRAMEWORK EXPERT EVALUATION - INTEGRATED ANALYSIS")
        print("=" * 70)
        
        if not self.load_data():
            print("\n✗ Failed to load data. Exiting.")
            return False
        
        self.analyze_expert_info()
        self.analyze_time_efficiency()
        self.analyze_performance_metrics()
        self.analyze_case_evaluations()
        
        self.analyze_lesion_correlations_CORRECTED()

        self.analyze_clinical_utility()
        self.analyze_implementation()
        self.analyze_educational_value()
        self.analyze_overall_assessment()

        self.generate_tables()
        self.generate_visualizations()

        print("\n" + "=" * 70)
        print("✓ ANALYSIS COMPLETE")
        print("=" * 70)
        return True
    
    def generate_tables(self):
        """Generate all publication tables."""
        print("\n" + "=" * 70)
        print("GENERATING PUBLICATION TABLES")
        print("=" * 70)
        
        tables_dir = os.path.join(self.output_dir, 'Tables')
        if not os.path.exists(tables_dir):
            os.makedirs(tables_dir)
        
        # Import table generator
        from ms_table_generator import MSTableGenerator
        table_gen = MSTableGenerator(self.results, tables_dir)
        table_gen.generate_all_tables()
        
        print(f"✓ Tables saved to {tables_dir}")
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        viz_dir = os.path.join(self.output_dir, 'Figures')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Import visualization generator
        from ms_visualization_generator import MSVisualizationGenerator
        viz_gen = MSVisualizationGenerator(self.results, viz_dir)
        viz_gen.generate_all_figures()
        
        print(f"✓ Visualizations saved to {viz_dir}")
    
    def save_analysis_report(self):
        """Save a comprehensive analysis report."""
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MS FRAMEWORK EXPERT EVALUATION - ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Time analysis
            if self.results['time_analysis']:
                f.write("TIME EFFICIENCY ANALYSIS\n")
                f.write("-" * 70 + "\n")
                ta = self.results['time_analysis']
                f.write(f"Traditional Time: {ta['mean_traditional']:.1f} ± {np.std(ta['traditional_time']):.1f} min\n")
                f.write(f"Automated Time: {ta['mean_automated']:.1f} ± {np.std(ta['automated_time']):.1f} min\n")
                f.write(f"Time Reduction: {ta['mean_reduction']:.1f} min ({ta['mean_reduction_percent']:.1f}%)\n")
                f.write(f"Statistical: t={ta['t_statistic']:.2f}, p={ta['p_value']:.4f}, d={ta['effect_size']:.2f}\n\n")
            
            # Performance metrics
            if self.results['performance_metrics']:
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 70 + "\n")
                for metric, data in self.results['performance_metrics'].items():
                    if 'trad_mean' in data:
                        f.write(f"{metric}:\n")
                        f.write(f"  Traditional: {data['trad_mean']:.1f} ± {data['trad_std']:.1f}\n")
                        f.write(f"  Automated: {data['auto_mean']:.1f} ± {data['auto_std']:.1f}\n")
                        f.write(f"  Improvement: {data['improvement']:+.1f}\n")
                        f.write(f"  p-value: {data['p_value']:.4f}\n\n")
            
            # Overall assessment
            if self.results['overall_assessment']:
                f.write("OVERALL ASSESSMENT\n")
                f.write("-" * 70 + "\n")
                oa = self.results['overall_assessment']
                f.write(f"Expert Ratings: {oa['ratings']}\n")
                f.write(f"Mean Rating: {oa['mean']:.1f} ± {oa['std']:.1f} / 10\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("Report saved successfully\n")
        
        print(f"✓ Analysis report saved to {report_path}")


def main():
    """Main execution function."""
    # Configuration
    overall_csv = 'ms_overall_evaluation_all.csv'
    case_pattern = 'ms_case_evaluations_MS_all_*.csv'
    output_dir = 'MS_Evaluation_Results'
    
    # Initialize analyzer
    analyzer = MSEvaluationAnalyzer(overall_csv, case_pattern, output_dir)
    
    # Run complete analysis
    if analyzer.run_complete_analysis():
        # Save analysis report
        analyzer.save_analysis_report()
        
        print("\n" + "=" * 70)
        print("ANALYSIS PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to: {output_dir}/")
        print("\nGenerated outputs:")
        print("  - analysis_report.txt")
        print("  - Tables/ (publication-ready tables)")
        print("  - Figures/ (publication-quality visualizations)")
    else:
        print("\n✗ Analysis failed. Please check your data files.")


if __name__ == '__main__':
    main()
