"""
Test script for MS Framework Evaluation - Integrated System
Generates sample data and runs complete analysis
"""

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("GENERATING SAMPLE DATA FOR TESTING")
print("=" * 70)

# Create sample overall evaluation data
overall_data = {
    'Section': [],
    'Question_ID': [],
    'Question_Text': [],
    'Response_Type': [],
    'Response_Value_Expert_1': [],
    'Notes_Expert_1': [],
    'Response_Value_Expert_2': [],
    'Notes_Expert_2': [],
    'Response_Value_Expert_3': [],
    'Notes_Expert_3': []
}

# Expert Info
expert_info = [
    ('Expert_Info', 'EXP_001', 'Years of Experience in Neuroimaging', 'Numeric', 25, '', 11, '', 3, ''),
    ('Expert_Info', 'EXP_002', 'Current Position/Title', 'Text', 'Professor', '', 'Associate Professor', '', 'Assistant Professor', ''),
    ('Expert_Info', 'EXP_003', 'Institution', 'Text', 'Tabriz University', '', 'Tabriz University', '', 'Tabriz University', ''),
]

# Time Analysis
time_data = [
    ('Time_Analysis', 'B1_001', 'Traditional Analysis Time per Case (minutes)', 'Numeric', 8, '', 10, '', 10, ''),
    ('Time_Analysis', 'B1_002', 'Automated Analysis Review Time per Case (minutes)', 'Numeric', 1, '', 2, '', 2, ''),
]

# Comparison metrics
comparison_data = [
    ('Comparison', 'B2_001', 'Speed of Analysis - Traditional (1-5)', 'Likert_5', 1, '', 3, '', 3, ''),
    ('Comparison', 'B2_002', 'Speed of Analysis - Automated (1-5)', 'Likert_5', 5, '', 5, '', 5, ''),
    ('Comparison', 'B2_003', 'Lesion Detection Completeness - Traditional (1-5)', 'Likert_5', 4, '', 4, '', 4, ''),
    ('Comparison', 'B2_004', 'Lesion Detection Completeness - Automated (1-5)', 'Likert_5', 4, '', 5, '', 5, ''),
    ('Comparison', 'B2_005', 'Classification Accuracy - Traditional (1-5)', 'Likert_5', 5, '', 4, '', 4, ''),
    ('Comparison', 'B2_006', 'Classification Accuracy - Automated (1-5)', 'Likert_5', 4, '', 4, '', 4, ''),
    ('Comparison', 'B2_007', 'Quantitative Detail - Traditional (1-5)', 'Likert_5', 4, '', 4, '', 4, ''),
    ('Comparison', 'B2_008', 'Quantitative Detail - Automated (1-5)', 'Likert_5', 5, '', 5, '', 5, ''),
    ('Comparison', 'B2_009', 'Reproducibility - Traditional (1-5)', 'Likert_5', 4, '', 4, '', 4, ''),
    ('Comparison', 'B2_010', 'Reproducibility - Automated (1-5)', 'Likert_5', 5, '', 5, '', 5, ''),
    ('Comparison', 'B2_011', 'Clinical Utility - Traditional (1-5)', 'Likert_5', 4, '', 4, '', 4, ''),
    ('Comparison', 'B2_012', 'Clinical Utility - Automated (1-5)', 'Likert_5', 5, '', 4, '', 4, ''),
]

# Clinical Utility
clinical_utility_data = [
    ('Clinical_Utility', 'C1_001', 'Primary Strengths - Comprehensive Lesion Detection', 'Binary_Check', 1, '', 1, '', 1, ''),
    ('Clinical_Utility', 'C1_002', 'Primary Strengths - Accurate Lesion Classification', 'Binary_Check', 1, '', 1, '', 1, ''),
    ('Clinical_Utility', 'C1_003', 'Primary Strengths - Detailed Quantitative Analysis', 'Binary_Check', 1, '', 1, '', 1, ''),
    ('Clinical_Utility', 'C1_004', 'Primary Strengths - Time-Efficient Workflow', 'Binary_Check', 1, '', 1, '', 1, ''),
    ('Clinical_Utility', 'C1_005', 'Primary Strengths - Consistent Reproducibility', 'Binary_Check', 1, '', 1, '', 1, ''),
    ('Clinical_Utility', 'C1_006', 'Primary Strengths - Educational Value for Trainees', 'Binary_Check', 1, '', 1, '', 1, ''),
    ('Clinical_Utility', 'C1_007', 'Primary Strengths - Structured Reporting Format', 'Binary_Check', 1, '', 1, '', 1, ''),
    ('Clinical_Utility', 'C1_008', 'Primary Strengths - Clinical Decision Support', 'Binary_Check', 1, '', 0, '', 1, ''),
    ('Clinical_Utility', 'C1_010', 'Primary Limitations - Segmentation Errors', 'Binary_Check', 0, '', 1, '', 0, ''),
    ('Clinical_Utility', 'C1_011', 'Primary Limitations - Classification Inaccuracies', 'Binary_Check', 0, '', 0, '', 1, ''),
]

# Implementation
implementation_data = [
    ('Implementation', 'C2_001', 'Clinical Readiness', 'Categorical', 'Ready with Minor Modifications', '', 'Ready with Minor Modifications', '', 'Ready with Minor Modifications', ''),
    ('Implementation', 'C2_009', 'User Benefit Ranking - Radiology Residents', 'Numeric', 1, '', 2, '', 1, ''),
    ('Implementation', 'C2_010', 'User Benefit Ranking - Researchers', 'Numeric', 2, '', 1, '', 2, ''),
    ('Implementation', 'C2_011', 'User Benefit Ranking - Experienced Neuroradiologists', 'Numeric', 3, '', 3, '', 4, ''),
]

# Educational
educational_data = [
    ('Educational', 'D1_002', 'Training Effectiveness Educational Value (1-5)', 'Likert_5', 4.5, '', 4.0, '', 4.5, ''),
    ('Educational', 'D1_003', 'Learning Acceleration Educational Value (1-5)', 'Likert_5', 4.0, '', 4.5, '', 4.0, ''),
    ('Educational', 'D1_004', 'Knowledge Transfer Educational Value (1-5)', 'Likert_5', 4.0, '', 4.5, '', 4.0, ''),
    ('Educational', 'D1_005', 'Skill Development Educational Value (1-5)', 'Likert_5', 4.5, '', 4.0, '', 4.5, ''),
]

# Overall Assessment
overall_assessment_data = [
    ('Overall_Assessment', 'F2_001', 'Final Framework Rating (1-10)', 'Numeric', 9, '', 9, '', 9, ''),
]

# Combine all data
all_rows = expert_info + time_data + comparison_data + clinical_utility_data + implementation_data + educational_data + overall_assessment_data

for row in all_rows:
    overall_data['Section'].append(row[0])
    overall_data['Question_ID'].append(row[1])
    overall_data['Question_Text'].append(row[2])
    overall_data['Response_Type'].append(row[3])
    overall_data['Response_Value_Expert_1'].append(row[4])
    overall_data['Notes_Expert_1'].append(row[5])
    overall_data['Response_Value_Expert_2'].append(row[6])
    overall_data['Notes_Expert_2'].append(row[7])
    overall_data['Response_Value_Expert_3'].append(row[8])
    overall_data['Notes_Expert_3'].append(row[9])

# Save overall evaluation
df_overall = pd.DataFrame(overall_data)
df_overall.to_csv('ms_overall_evaluation_all_msseg.csv', index=False)
print("✓ Created ms_overall_evaluation_all_msseg.csv")

# Create sample case evaluation data for 3 cases
for case_num in range(1, 4):
    case_id = f"MS-{case_num:03d}"
    
    case_data = {
        'Case_ID': [],
        'Question_ID': [],
        'Question_Text': [],
        'Response_Type': [],
        'Response_Value_Expert_1': [],
        'Notes_Expert_1': [],
        'Response_Value_Expert_2': [],
        'Notes_Expert_2': [],
        'Response_Value_Expert_3': [],
        'Notes_Expert_3': []
    }
    
    # Add some randomness to make each case different
    np.random.seed(case_num)
    
    case_rows = [
        (case_id, 'A1_001', 'Segmentation Accuracy (1-5)', 'Likert_5', 4, '', 5, '', 4, ''),
        (case_id, 'A1_002', 'Lesion Detection Completeness (1-5)', 'Likert_5', 4, '', 4, '', 5, ''),
        (case_id, 'A1_003', 'Lesion Classification Accuracy (1-5)', 'Likert_5', 4, '', 4, '', 4, ''),
        (case_id, 'A2_001', 'Periventricular - Automated Count', 'Numeric', 4+case_num, '', 5+case_num, '', 4+case_num, ''),
        (case_id, 'A2_002', 'Periventricular - Expert Count', 'Numeric', 4+case_num, '', 5+case_num, '', 5+case_num, ''),
        (case_id, 'A2_004', 'Paraventricular - Automated Count', 'Numeric', 8+case_num, '', 10+case_num, '', 9+case_num, ''),
        (case_id, 'A2_005', 'Paraventricular - Expert Count', 'Numeric', 8+case_num, '', 9+case_num, '', 9+case_num, ''),
        (case_id, 'A2_007', 'Juxtacortical - Automated Count', 'Numeric', 3+case_num, '', 4+case_num, '', 3+case_num, ''),
        (case_id, 'A2_008', 'Juxtacortical - Expert Count', 'Numeric', 3+case_num, '', 4+case_num, '', 4+case_num, ''),
    ]
    
    for row in case_rows:
        case_data['Case_ID'].append(row[0])
        case_data['Question_ID'].append(row[1])
        case_data['Question_Text'].append(row[2])
        case_data['Response_Type'].append(row[3])
        case_data['Response_Value_Expert_1'].append(row[4])
        case_data['Notes_Expert_1'].append(row[5])
        case_data['Response_Value_Expert_2'].append(row[6])
        case_data['Notes_Expert_2'].append(row[7])
        case_data['Response_Value_Expert_3'].append(row[8])
        case_data['Notes_Expert_3'].append(row[9])
    
    df_case = pd.DataFrame(case_data)
    filename = f'ms_case_evaluations_MSSEG_all_{case_id}.csv'
    df_case.to_csv(filename, index=False)
    print(f"✓ Created {filename}")

print("\n" + "=" * 70)
print("SAMPLE DATA GENERATION COMPLETE")
print("=" * 70)

# Now run the integrated analysis
print("\n" + "=" * 70)
print("RUNNING INTEGRATED ANALYSIS")
print("=" * 70)

from ms_evaluation_integrated import MSEvaluationAnalyzer

# Initialize analyzer
analyzer = MSEvaluationAnalyzer(
    'ms_overall_evaluation_all_msseg.csv',
    'ms_case_evaluations_MSSEG_all_*.csv',
    'MS_Evaluation_Results_Test'
)

# Run complete analysis
if analyzer.run_complete_analysis():
    analyzer.save_analysis_report()
    
    print("\n" + "=" * 70)
    print("GENERATING TABLES")
    print("=" * 70)
    
    from ms_table_generator import MSTableGenerator
    tables_dir = os.path.join(analyzer.output_dir, 'Tables')
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)
    
    table_gen = MSTableGenerator(analyzer.results, tables_dir)
    table_gen.generate_all_tables()
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    from ms_visualization_generator import MSVisualizationGenerator
    viz_dir = os.path.join(analyzer.output_dir, 'Figures')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    viz_gen = MSVisualizationGenerator(analyzer.results, viz_dir)
    viz_gen.generate_all_figures()
    
    print("\n" + "=" * 70)
    print("✓ TEST COMPLETE - ALL OUTPUTS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nCheck the '{analyzer.output_dir}' directory for all outputs:")
    print(f"  - analysis_report.txt")
    print(f"  - Tables/ (8 CSV files + 1 Excel file)")
    print(f"  - Figures/ (5 PNG files)")
else:
    print("\n✗ Analysis failed")
