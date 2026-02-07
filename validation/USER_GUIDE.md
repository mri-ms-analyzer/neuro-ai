# MS Framework Expert Evaluation - User Guide

## Quick Start

### Installation

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

2. Place your evaluation CSV files in the working directory:
   - `ms_overall_evaluation_all_msseg.csv`
   - `ms_case_evaluations_MSSEG_all_MS-001.csv`
   - `ms_case_evaluations_MSSEG_all_MS-002.csv`
   - etc.

3. Run the integrated analysis:
```bash
python ms_evaluation_integrated.py
```

That's it! The system will automatically:
- Load all evaluation data
- Perform statistical analysis
- Generate all tables (CSV and Excel)
- Create all visualizations (PNG)
- Save a comprehensive report

## What's New in the Integrated System?

### Major Improvements

**Before (Original Scripts):**
- ❌ Manual data entry required
- ❌ Hardcoded values in scripts
- ❌ Had to run 3 separate scripts
- ❌ Tables and visualizations disconnected from data
- ❌ Risk of inconsistencies

**After (Integrated System):**
- ✅ Fully automated data extraction
- ✅ Dynamic data loading from CSV files
- ✅ Single script execution
- ✅ Tables and visualizations use live data
- ✅ Guaranteed consistency across all outputs

### What Changed?

1. **evaluation_claude_analysis.py** → **ms_evaluation_integrated.py**
   - Now orchestrates the entire workflow
   - Extracts and stores results in structured format
   - Calls table and visualization generators automatically

2. **evaluation_claude_tables.py** → **ms_table_generator.py**
   - Changed from standalone script to importable module
   - Uses extracted data instead of hardcoded values
   - Generates tables dynamically based on actual evaluation results

3. **evaluation_claude_visualizations.py** → **ms_visualization_generator.py**
   - Changed from standalone script to importable module
   - Creates visualizations from extracted data
   - Adapts to number of cases and experts in dataset

## Detailed Usage

### Basic Workflow

```python
from ms_evaluation_integrated import MSEvaluationAnalyzer

# Initialize
analyzer = MSEvaluationAnalyzer(
    overall_csv_path='ms_overall_evaluation_all_msseg.csv',
    case_csv_pattern='ms_case_evaluations_MSSEG_all_*.csv',
    output_dir='MS_Evaluation_Results'
)

# Run analysis
analyzer.run_complete_analysis()

# Save report
analyzer.save_analysis_report()
```

### Advanced Usage: Custom Analysis

You can run individual analysis components:

```python
# Load data
analyzer.load_data()

# Run specific analyses
analyzer.analyze_time_efficiency()
analyzer.analyze_performance_metrics()
analyzer.analyze_case_evaluations()

# Access results
time_results = analyzer.results['time_analysis']
print(f"Mean time reduction: {time_results['mean_reduction_percent']:.1f}%")
```

### Customizing Output

#### Change Output Directory

```python
analyzer = MSEvaluationAnalyzer(
    overall_csv_path='ms_overall_evaluation_all_msseg.csv',
    case_csv_pattern='ms_case_evaluations_MSSEG_all_*.csv',
    output_dir='Custom_Output_Directory'
)
```

#### Generate Only Tables

```python
from ms_table_generator import MSTableGenerator

# After running analysis
tables_dir = 'Tables_Only'
table_gen = MSTableGenerator(analyzer.results, tables_dir)
table_gen.generate_all_tables()
```

#### Generate Only Visualizations

```python
from ms_visualization_generator import MSVisualizationGenerator

# After running analysis
viz_dir = 'Figures_Only'
viz_gen = MSVisualizationGenerator(analyzer.results, viz_dir)
viz_gen.generate_all_figures()
```

## Understanding the Data Structure

### Overall Evaluation CSV

Contains expert-level data organized by sections:
- **Expert_Info**: Demographics and qualifications
- **Time_Analysis**: Time efficiency metrics
- **Comparison**: Traditional vs. Automated performance
- **Clinical_Utility**: Strengths and limitations
- **Implementation**: Readiness and user benefits
- **Educational**: Educational value assessments
- **Overall_Assessment**: Final ratings

### Case Evaluation CSV

Contains case-specific evaluations:
- **A1_XXX**: Segmentation and detection metrics
- **A2_XXX**: Lesion counts by type and location
- **A3_XXX**: Clinical interpretation quality

## Output Files Explained

### Tables Directory

1. **Table1_Expert_Characteristics.csv**
   - Expert demographics and qualifications
   - Experience levels and positions

2. **Table2_Time_Efficiency.csv**
   - Time comparison analysis
   - Statistical significance testing
   - Percentage reductions

3. **Table3_Performance_Comparison.csv**
   - Traditional vs. Automated metrics
   - Statistical tests (Wilcoxon)
   - Improvement scores

4. **Table4_Multicase_Evaluation.csv**
   - Case-by-case performance
   - Aggregated metrics
   - Overall performance summary

5. **Table5_Clinical_Utility.csv**
   - Primary strengths identified
   - Limitations noted
   - Expert consensus levels

6. **Table6_Lesion_Accuracy.csv**
   - Lesion count comparisons
   - Agreement percentages
   - Correlation coefficients

7. **Table7_Educational_Value.csv**
   - Educational metrics by category
   - Inter-rater agreement
   - Overall educational score

8. **Summary_Key_Findings.csv**
   - High-level summary for abstracts
   - Statistical evidence
   - Clinical impact statements

9. **MS_Framework_All_Tables.xlsx**
   - Master Excel file with all tables
   - Each table in a separate sheet
   - Ready for publication

### Figures Directory

1. **figure1_time_efficiency.png**
   - Time comparison bars
   - Percentage reduction chart
   - Cumulative savings plot
   - Distribution histogram

2. **figure2_performance_radar.png**
   - Radar chart comparing methods
   - Expert agreement visualization
   - 6-metric comparison

3. **figure3_multicase_dashboard.png**
   - Segmentation accuracy by case
   - Detection completeness
   - Classification accuracy
   - Lesion count correlation

4. **figure4_clinical_utility.png**
   - Primary strengths bar chart
   - User benefit assessment
   - Implementation readiness pie
   - Educational value scores

5. **figure5_summary_assessment.png**
   - Final expert ratings
   - Key performance indicators
   - Recommendation donut chart
   - Comprehensive comparison

### Analysis Report

**analysis_report.txt**: Text summary including:
- Time efficiency statistics
- Performance metric improvements
- Overall assessment scores
- Key findings summary

## Troubleshooting

### Common Issues

**Issue**: "File not found" error
- **Solution**: Ensure CSV files are in the working directory
- Check file names match expected patterns
- Verify file permissions

**Issue**: "No case evaluation files found"
- **Solution**: Check case file naming pattern
- Ensure at least one case file exists
- Verify glob pattern in configuration

**Issue**: Empty or missing data in outputs
- **Solution**: Check CSV files have data in Response_Value columns
- Ensure expert responses are not blank
- Review analysis_report.txt for warnings

**Issue**: Import errors
- **Solution**: Ensure all three .py files are in same directory
- Check Python path includes current directory
- Verify all dependencies installed

### Data Quality Checks

The system performs automatic quality checks:
- ✅ Validates file existence
- ✅ Checks for required columns
- ✅ Handles missing data gracefully
- ✅ Reports data quality issues

### Debug Mode

Add verbose output by modifying the main function:

```python
# In ms_evaluation_integrated.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

### Data Preparation

1. **Complete all evaluation forms** before running analysis
2. **Use consistent naming** for case files
3. **Verify data entry** - check for typos or missing values
4. **Back up original files** before processing

### Reproducibility

1. **Document your data sources**
2. **Save the generated CSV files** used for analysis
3. **Keep the output directory** with timestamp
4. **Note any manual adjustments** made to data

### Publication Workflow

1. Run integrated analysis
2. Review analysis_report.txt for overview
3. Check tables for accuracy
4. Examine visualizations for clarity
5. Make any necessary adjustments
6. Re-run if data is updated
7. Use Excel file for final publication tables
8. Use PNG files for manuscript figures

## Extending the System

### Adding New Metrics

1. Add extraction logic to `MSEvaluationAnalyzer`
2. Create corresponding table in `MSTableGenerator`
3. Create visualization in `MSVisualizationGenerator`

Example:

```python
# In MSEvaluationAnalyzer
def analyze_new_metric(self):
    """Analyze a new custom metric."""
    new_data = self.overall_data[self.overall_data['Section'] == 'New_Section']
    # Extract and process
    self.results['new_metric'] = {...}

# In MSTableGenerator
def create_new_metric_table(self):
    """Create table for new metric."""
    data = self.results['new_metric']
    # Create DataFrame
    # Save to CSV

# In MSVisualizationGenerator
def create_new_metric_plot(self):
    """Visualize new metric."""
    data = self.results['new_metric']
    # Create matplotlib figure
    # Save to PNG
```

### Custom Statistical Tests

Add custom tests in the analyzer:

```python
from scipy.stats import mannwhitneyu

def custom_statistical_test(self, data1, data2):
    """Perform Mann-Whitney U test."""
    statistic, p_value = mannwhitneyu(data1, data2)
    return {'statistic': statistic, 'p_value': p_value}
```

## Support and Feedback

For issues or questions:
1. Check this user guide
2. Review README.md
3. Examine sample outputs in MS_Evaluation_Results_Example/
4. Check code comments in .py files

## Version History

**Version 2.0** (Current)
- ✅ Fully integrated system
- ✅ Automatic data extraction
- ✅ Dynamic table generation
- ✅ Live visualization creation
- ✅ Comprehensive error handling

**Version 1.0** (Original)
- ⚠️ Manual data entry
- ⚠️ Hardcoded values
- ⚠️ Separate scripts
- ⚠️ Limited flexibility

## License and Citation

This system was developed for MS Framework evaluation analysis.
When using this system, please cite the MS Framework evaluation study.

---

**Last Updated**: 2026-01-29
**System Version**: 2.0
**Python Version**: 3.7+
