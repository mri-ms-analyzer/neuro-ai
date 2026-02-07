# MS Framework Expert Evaluation - Integrated Analysis System

## Overview

This integrated system combines analysis, table generation, and visualization into a unified workflow that automatically extracts data from evaluation forms and produces all outputs in one execution.

## System Architecture

The system consists of three main components:

1. **ms_evaluation_integrated.py** - Main analysis script
   - Loads and analyzes evaluation data
   - Extracts metrics from CSV files
   - Coordinates table and visualization generation
   - Produces comprehensive analysis report

2. **ms_table_generator.py** - Table generation module
   - Creates 8 publication-ready tables
   - Generates master Excel file with all tables
   - Uses data directly from the analyzer

3. **ms_visualization_generator.py** - Visualization module
   - Creates 5 publication-quality figures
   - All visualizations use extracted analysis data
   - Generates high-resolution PNG files

## Key Improvements

### Before (Original Scripts)
- Required manual data entry and hardcoded values
- Tables and visualizations used fixed inputs
- Had to run three separate scripts
- Manual coordination between scripts

### After (Integrated System)
- Fully automated data extraction from CSV files
- Tables and visualizations use live data
- Single script execution for complete analysis
- Automatic data flow between components

## Usage

### Basic Usage

```bash
python ms_evaluation_integrated.py
```

This single command will:
1. Load all evaluation CSV files
2. Perform complete statistical analysis
3. Generate all tables
4. Create all visualizations
5. Save comprehensive report

### File Requirements

The script expects these files in the working directory:

1. **Overall evaluation file**: `ms_overall_evaluation_all_msseg.csv`
2. **Case evaluation files**: `ms_case_evaluations_MSSEG_all_MS-001.csv`, `ms_case_evaluations_MSSEG_all_MS-002.csv`, etc.

### Output Structure

```
MS_Evaluation_Results/
├── analysis_report.txt          # Comprehensive text report
├── Tables/                      # All publication tables
│   ├── Table1_Expert_Characteristics.csv
│   ├── Table2_Time_Efficiency.csv
│   ├── Table3_Performance_Comparison.csv
│   ├── Table4_Multicase_Evaluation.csv
│   ├── Table5_Clinical_Utility.csv
│   ├── Table6_Lesion_Accuracy.csv
│   ├── Table7_Educational_Value.csv
│   ├── Summary_Key_Findings.csv
│   └── MS_Framework_All_Tables.xlsx  # Master Excel file
└── Figures/                     # All visualizations
    ├── figure1_time_efficiency.png
    ├── figure2_performance_radar.png
    ├── figure3_multicase_dashboard.png
    ├── figure4_clinical_utility.png
    └── figure5_summary_assessment.png
```

## Customization

### Changing Input Files

Edit these lines in `ms_evaluation_integrated.py`:

```python
# Configuration
overall_csv = 'your_overall_file.csv'
case_pattern = 'your_case_pattern_*.csv'
output_dir = 'Your_Output_Directory'
```

### Modifying Analysis

The `MSEvaluationAnalyzer` class contains methods for each analysis component:
- `analyze_expert_info()` - Expert demographics
- `analyze_time_efficiency()` - Time analysis with statistics
- `analyze_performance_metrics()` - Comparative performance
- `analyze_case_evaluations()` - Individual case analysis
- `analyze_clinical_utility()` - Strengths and limitations
- `analyze_implementation()` - Readiness assessment
- `analyze_educational_value()` - Educational metrics
- `analyze_overall_assessment()` - Final ratings

### Adding New Tables or Figures

1. Add a new method to `MSTableGenerator` class
2. Call it in `generate_all_tables()` method
3. Similarly for visualizations in `MSVisualizationGenerator`

## Data Flow

```
CSV Files
    ↓
MSEvaluationAnalyzer (loads and analyzes)
    ↓
results dictionary (structured data)
    ↓
    ├→ MSTableGenerator (creates tables)
    └→ MSVisualizationGenerator (creates figures)
```

## Dependencies

```python
pandas
numpy
matplotlib
seaborn
scipy
openpyxl  # For Excel file creation
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl
```

## Features

### Automatic Data Extraction
- Reads all expert responses from CSV files
- Extracts metrics by section and question ID
- Handles missing data gracefully
- Aggregates across multiple experts and cases

### Statistical Analysis
- Paired t-tests for time efficiency
- Wilcoxon signed-rank tests for performance metrics
- Correlation analysis for lesion counts
- Effect size calculations (Cohen's d)

### Publication-Ready Outputs
- High-resolution figures (300 DPI)
- Professional formatting
- Consistent color schemes
- Comprehensive tables with statistics

### Robust Error Handling
- Checks for missing files
- Handles incomplete data
- Reports errors clearly
- Continues with available data

## Troubleshooting

### "File not found" errors
- Ensure CSV files are in the working directory
- Check file naming matches the expected pattern
- Verify file paths in configuration

### "No case evaluation files found"
- Check the case_pattern matches your file names
- Ensure at least one case file exists
- Verify file permissions

### Missing data in outputs
- Check if CSV files have data in all required columns
- Look for empty Response_Value fields
- Review the analysis_report.txt for warnings

## Author

Developed for MS Framework evaluation analysis
Version: 2.0
Date: 2025-09-10
