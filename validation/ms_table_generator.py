"""
MS Framework Table Generator
=============================
Generates publication-ready tables using data extracted by the main analyzer.

Author: Mahdi Bashiri Bawil
Date: 2025-09-10
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
import os


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


class MSTableGenerator:
    """Generates publication-ready tables from analysis results."""
    
    def __init__(self, results, output_dir):
        """
        Initialize table generator.
        
        Parameters:
        -----------
        results : dict
            Analysis results from MSEvaluationAnalyzer
        output_dir : str
            Directory to save tables
        """
        self.results = results
        self.output_dir = output_dir
        self.tables = {}
    
    def create_expert_characteristics_table(self):
        """Create Table 1: Expert Evaluator Characteristics"""
        print("\nGenerating Table 1: Expert Evaluator Characteristics...")
        
        expert_info = self.results['expert_info']
        
        # Create expert data dictionary
        expert_data = {
            'Expert ID': ['Expert 1', 'Expert 2', 'Expert 3'],
            'Years of Experience': expert_info.get('years_experience', ['N/A', 'N/A', 'N/A']),
            'Academic Position': expert_info.get('positions', ['N/A', 'N/A', 'N/A']),
            'Institution': expert_info.get('institutions', ['N/A', 'N/A', 'N/A'])
        }
        
        table1 = pd.DataFrame(expert_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'Table1_Expert_Characteristics.csv')
        table1.to_csv(csv_path, index=False)
        
        self.tables['Expert_Characteristics'] = table1
        print(f"✓ Saved to {csv_path}")
        return table1
    
    def create_time_efficiency_table(self):
        """Create Table 2: Time Efficiency Analysis"""
        print("\nGenerating Table 2: Time Efficiency Analysis...")
        
        ta = self.results['time_analysis']
        
        traditional_times = np.array(ta['traditional_time'])
        automated_times = np.array(ta['automated_time'])
        time_reduction = np.array(ta['time_reduction'])
        time_reduction_percent = np.array(ta['time_reduction_percent'])
        
        # Create table data
        time_data = {
            'Analysis Method': [
                'Traditional Manual Analysis',
                'Automated Framework Analysis',
                'Absolute Time Reduction',
                'Relative Time Reduction (%)',
                'Statistical Significance'
            ],
            'Expert 1': [
                f"{traditional_times[0]:.0f} min",
                f"{automated_times[0]:.0f} min",
                f"{time_reduction[0]:.0f} min",
                f"{time_reduction_percent[0]:.1f}%",
                f"t = {ta['t_statistic']:.2f}"
            ],
            'Expert 2': [
                f"{traditional_times[1]:.0f} min",
                f"{automated_times[1]:.0f} min",
                f"{time_reduction[1]:.0f} min",
                f"{time_reduction_percent[1]:.1f}%",
                f"p = {ta['p_value']:.4f}"
            ],
            'Expert 3': [
                f"{traditional_times[2]:.0f} min",
                f"{automated_times[2]:.0f} min",
                f"{time_reduction[2]:.0f} min",
                f"{time_reduction_percent[2]:.1f}%",
                f"d = {ta['effect_size']:.2f}"
            ],
            'Mean ± SD': [
                f"{np.mean(traditional_times):.1f} ± {np.std(traditional_times):.1f}",
                f"{np.mean(automated_times):.1f} ± {np.std(automated_times):.1f}",
                f"{np.mean(time_reduction):.1f} ± {np.std(time_reduction):.1f}",
                f"{np.mean(time_reduction_percent):.1f} ± {np.std(time_reduction_percent):.1f}",
                "***" if ta['p_value'] < 0.001 else ("**" if ta['p_value'] < 0.01 else ("*" if ta['p_value'] < 0.05 else "ns"))
            ]
        }
        
        table2 = pd.DataFrame(time_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'Table2_Time_Efficiency.csv')
        table2.to_csv(csv_path, index=False)
        
        self.tables['Time_Efficiency'] = table2
        print(f"✓ Saved to {csv_path}")
        return table2
    
    def create_performance_comparison_table(self):
        """Create Table 3: Comparative Performance Analysis"""
        print("\nGenerating Table 3: Comparative Performance Analysis...")
        
        metrics = self.results['performance_metrics']
        
        # Create performance comparison data
        perf_data = []
        
        for metric, data in metrics.items():
            if 'trad_mean' in data:
                # Determine significance marker
                p_val = data.get('p_value', 1.0)
                if p_val < 0.001:
                    sig_marker = "***"
                elif p_val < 0.01:
                    sig_marker = "**"
                elif p_val < 0.05:
                    sig_marker = "*"
                else:
                    sig_marker = ""
                
                perf_data.append({
                    'Performance Metric': metric,
                    'Traditional Method': f"{data['trad_mean']:.1f} ± {data['trad_std']:.1f}",
                    'Automated Framework': f"{data['auto_mean']:.1f} ± {data['auto_std']:.1f}",
                    'Improvement': f"{data['improvement']:+.1f}",
                    'Significance': sig_marker,
                    'p-value': f"{p_val:.4f}"
                })
        
        table3 = pd.DataFrame(perf_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'Table3_Performance_Comparison.csv')
        table3.to_csv(csv_path, index=False)
        
        self.tables['Performance_Comparison'] = table3
        print(f"✓ Saved to {csv_path}")
        return table3
    
    def create_multicase_evaluation_table(self):
        """Create Table 4: Multi-case Evaluation Summary"""
        print("\nGenerating Table 4: Multi-case Evaluation Summary...")
        
        case_summary = self.results['case_summary']
        
        # Aggregate metrics across cases
        case_data = []
        
        for case_id, data in case_summary.items():
            seg_acc = data.get('segmentation_accuracy', [])
            det_comp = data.get('detection_completeness', [])
            class_acc = data.get('classification_accuracy', [])
            
            case_data.append({
                'Case ID': case_id,
                'Segmentation Accuracy': f"{np.mean(seg_acc):.1f} ± {np.std(seg_acc):.1f}" if seg_acc else "N/A",
                'Detection Completeness': f"{np.mean(det_comp):.1f} ± {np.std(det_comp):.1f}" if det_comp else "N/A",
                'Classification Accuracy': f"{np.mean(class_acc):.1f} ± {np.std(class_acc):.1f}" if class_acc else "N/A",
                'Number of Experts': len(seg_acc) if seg_acc else 0
            })
        
        # Add overall row
        all_seg = [v for data in case_summary.values() for v in data.get('segmentation_accuracy', [])]
        all_det = [v for data in case_summary.values() for v in data.get('detection_completeness', [])]
        all_class = [v for data in case_summary.values() for v in data.get('classification_accuracy', [])]
        
        case_data.append({
            'Case ID': 'OVERALL',
            'Segmentation Accuracy': f"{np.mean(all_seg):.1f} ± {np.std(all_seg):.1f}" if all_seg else "N/A",
            'Detection Completeness': f"{np.mean(all_det):.1f} ± {np.std(all_det):.1f}" if all_det else "N/A",
            'Classification Accuracy': f"{np.mean(all_class):.1f} ± {np.std(all_class):.1f}" if all_class else "N/A",
            'Number of Experts': len(all_seg) if all_seg else 0
        })
        
        table4 = pd.DataFrame(case_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'Table4_Multicase_Evaluation.csv')
        table4.to_csv(csv_path, index=False)
        
        self.tables['Multicase_Evaluation'] = table4
        print(f"✓ Saved to {csv_path}")
        return table4
    
    def create_clinical_utility_table(self):
        """Create Table 5: Clinical Utility Assessment"""
        print("\nGenerating Table 5: Clinical Utility Assessment...")
        
        clinical = self.results['clinical_utility']
        
        # Strengths
        strength_data = []
        for strength, data in clinical.get('strengths', {}).items():
            strength_data.append({
                'Category': 'Strength',
                'Item': strength,
                'Expert Agreement': f"{data['agreement']} of 3",
                'Consensus Level': 'Complete' if data['agreement'] == 3 else ('Majority' if data['agreement'] == 2 else 'Partial')
            })
        
        # Limitations
        limitation_data = []
        for limitation, data in clinical.get('limitations', {}).items():
            if data['agreement'] > 0:
                limitation_data.append({
                    'Category': 'Limitation',
                    'Item': limitation,
                    'Expert Agreement': f"{data['agreement']} of 3",
                    'Consensus Level': 'Complete' if data['agreement'] == 3 else ('Majority' if data['agreement'] == 2 else 'Partial')
                })
        
        table5 = pd.DataFrame(strength_data + limitation_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'Table5_Clinical_Utility.csv')
        table5.to_csv(csv_path, index=False)
        
        self.tables['Clinical_Utility'] = table5
        print(f"✓ Saved to {csv_path}")
        return table5
    

    def create_lesion_accuracy_table(self):
        """
        CORRECTED VERSION: Creates Table 6 with proper per-case correlations.
        
        This should REPLACE create_lesion_accuracy_table() in ms_table_generator.py
        (lines 242-307)
        """
        print("\nGenerating Table 6: Lesion Count Accuracy Analysis (CORRECTED)...")
        
        # Get corrected correlation results
        if 'correlation_analysis_corrected' not in self.results:
            print("ERROR: Run analyze_lesion_correlations_CORRECTED() first!")
            return None
        
        corr_results = self.results['correlation_analysis_corrected']['overall']
        lesion_types = ['Periventricular', 'Paraventricular', 'Juxtacortical', 'Total_Lesions']
        
        lesion_data = []
        
        for lesion_type in lesion_types:
            if lesion_type in corr_results:
                stats = corr_results[lesion_type]
                
                lesion_data.append({
                    'Lesion Location': lesion_type.replace('_', ' '),
                    'n (cases)': stats['n'],
                    'AI Count': f"{stats['ai_mean']:.1f} ± {stats['ai_std']:.1f}",
                    'Expert Count': f"{stats['expert_mean']:.1f} ± {stats['expert_std']:.1f}",
                    'Correlation (r)': f"{stats['r']:.3f}",
                    'r 95% CI': f"[{stats['r_ci_lower']:.3f}, {stats['r_ci_upper']:.3f}]",
                    'p-value': f"{stats['p_value']:.4f}",
                    'MAE': f"{stats['mae']:.1f}",
                    'MAE 95% CI': f"[{stats['mae_ci_lower']:.1f}, {stats['mae_ci_upper']:.1f}]",
                    'Agreement (±1)': f"{stats['agreement_within_1']:.0f}%"
                })
        
        table6 = pd.DataFrame(lesion_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'Table6_Lesion_Accuracy_CORRECTED.csv')
        table6.to_csv(csv_path, index=False)
        
        self.tables['Lesion_Accuracy_Corrected'] = table6
        print(f"✓ Saved to {csv_path}")
        
        return table6

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


    def create_educational_value_table(self):
        """Create Table 7: Educational Value Assessment"""
        print("\nGenerating Table 7: Educational Value Assessment...")
        
        educational = self.results['educational']
        
        edu_data = []
        
        for category, data in educational.items():
            scores = data.get('scores', [])
            edu_data.append({
                'Educational Category': category,
                'Expert 1 Rating': f"{scores[0]:.1f}" if len(scores) > 0 else "N/A",
                'Expert 2 Rating': f"{scores[1]:.1f}" if len(scores) > 1 else "N/A",
                'Expert 3 Rating': f"{scores[2]:.1f}" if len(scores) > 2 else "N/A",
                'Mean ± SD': f"{data['mean']:.1f} ± {data['std']:.1f}",
                'Agreement Level': 'High' if data['std'] < 0.5 else ('Moderate' if data['std'] < 1.0 else 'Low')
            })
        
        # Add overall row
        all_means = [data['mean'] for data in educational.values()]
        if all_means:
            edu_data.append({
                'Educational Category': 'OVERALL',
                'Expert 1 Rating': '',
                'Expert 2 Rating': '',
                'Expert 3 Rating': '',
                'Mean ± SD': f"{np.mean(all_means):.1f} ± {np.std(all_means):.1f}",
                'Agreement Level': 'High'
            })
        
        table7 = pd.DataFrame(edu_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'Table7_Educational_Value.csv')
        table7.to_csv(csv_path, index=False)
        
        self.tables['Educational_Value'] = table7
        print(f"✓ Saved to {csv_path}")
        return table7
    
    def create_summary_findings_table(self):
        """Create Summary Table: Key Findings"""
        print("\nGenerating Summary Table: Key Findings...")
        
        ta = self.results.get('time_analysis', {})
        oa = self.results.get('overall_assessment', {})
        clinical = self.results.get('clinical_utility', {})
        
        summary_data = []
        
        # Time efficiency
        if ta:
            summary_data.append({
                'Evaluation Domain': 'Time Efficiency',
                'Key Finding': f"{ta.get('mean_reduction_percent', 0):.1f}% reduction in analysis time",
                'Statistical Evidence': f"t = {ta.get('t_statistic', 0):.2f}, p = {ta.get('p_value', 1):.4f}, d = {ta.get('effect_size', 0):.2f}",
                'Clinical Impact': 'Dramatically improved workflow efficiency'
            })
        
        # Overall assessment
        if oa:
            summary_data.append({
                'Evaluation Domain': 'Overall Framework Quality',
                'Key Finding': f"Mean rating {oa.get('mean', 0):.1f}/10",
                'Statistical Evidence': f"Perfect inter-rater agreement (SD = {oa.get('std', 0):.1f})",
                'Clinical Impact': 'High-quality diagnostic framework'
            })
        
        # Clinical utility
        if clinical:
            n_strengths = len([s for s in clinical.get('strengths', {}).values() if s['agreement'] == 3])
            summary_data.append({
                'Evaluation Domain': 'Clinical Utility',
                'Key Finding': f"{n_strengths} unanimous strengths identified",
                'Statistical Evidence': 'Complete expert consensus',
                'Clinical Impact': 'Strong clinical acceptance'
            })
        
        table_summary = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'Summary_Key_Findings.csv')
        table_summary.to_csv(csv_path, index=False)
        
        self.tables['Summary_Findings'] = table_summary
        print(f"✓ Saved to {csv_path}")
        return table_summary
    
    def generate_all_tables(self):
        """Generate all publication tables."""
        print("\n" + "=" * 70)
        print("GENERATING ALL PUBLICATION TABLES")
        print("=" * 70)
        
        self.create_expert_characteristics_table()
        self.create_time_efficiency_table()
        self.create_performance_comparison_table()
        self.create_multicase_evaluation_table()
        self.create_clinical_utility_table()
        self.create_lesion_accuracy_table()
        self.create_educational_value_table()
        self.create_summary_findings_table()
        self.create_dataset_comparison_table()
        
        # Create master Excel file
        excel_path = os.path.join(self.output_dir, 'MS_Framework_All_Tables.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for table_name, table_df in self.tables.items():
                sheet_name = table_name.replace('_', ' ')[:31]
                table_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\n✓ Master Excel file saved: {excel_path}")
        print(f"✓ Generated {len(self.tables)} tables")
        print("=" * 70)
