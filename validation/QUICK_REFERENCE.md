# MS Framework Evaluation - Quick Reference Card

## Installation
```bash
pip install -r requirements.txt
```

## Basic Usage
```bash
python ms_evaluation_integrated.py
```

## File Requirements
```
Working Directory:
├── ms_overall_evaluation_all_msseg.csv
├── ms_case_evaluations_MSSEG_all_MS-001.csv
├── ms_case_evaluations_MSSEG_all_MS-002.csv
└── ... (more case files)
```

## Output Structure
```
MS_Evaluation_Results/
├── analysis_report.txt
├── Tables/
│   ├── Table1_Expert_Characteristics.csv
│   ├── Table2_Time_Efficiency.csv
│   ├── Table3_Performance_Comparison.csv
│   ├── Table4_Multicase_Evaluation.csv
│   ├── Table5_Clinical_Utility.csv
│   ├── Table6_Lesion_Accuracy.csv
│   ├── Table7_Educational_Value.csv
│   ├── Summary_Key_Findings.csv
│   └── MS_Framework_All_Tables.xlsx
└── Figures/
    ├── figure1_time_efficiency.png
    ├── figure2_performance_radar.png
    ├── figure3_multicase_dashboard.png
    ├── figure4_clinical_utility.png
    └── figure5_summary_assessment.png
```

## Common Commands

### Run Complete Analysis
```bash
python ms_evaluation_integrated.py
```

### Run with Custom Output Directory
Edit `ms_evaluation_integrated.py`:
```python
output_dir = 'My_Custom_Directory'
```

### Test the System
```bash
python test_integrated_system.py
```

## Quick Customization

### Change Input Files
In `ms_evaluation_integrated.py`, modify:
```python
overall_csv = 'my_overall_file.csv'
case_pattern = 'my_case_files_*.csv'
```

### Generate Only Tables
```python
from ms_evaluation_integrated import MSEvaluationAnalyzer
from ms_table_generator import MSTableGenerator

analyzer = MSEvaluationAnalyzer(...)
analyzer.run_complete_analysis()
table_gen = MSTableGenerator(analyzer.results, 'Tables_Output')
table_gen.generate_all_tables()
```

### Generate Only Figures
```python
from ms_evaluation_integrated import MSEvaluationAnalyzer
from ms_visualization_generator import MSVisualizationGenerator

analyzer = MSEvaluationAnalyzer(...)
analyzer.run_complete_analysis()
viz_gen = MSVisualizationGenerator(analyzer.results, 'Figures_Output')
viz_gen.generate_all_figures()
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| File not found | Check CSV files in working directory |
| Import error | Run `pip install -r requirements.txt` |
| No output | Check file permissions, verify data in CSV |
| Empty tables | Ensure Response_Value columns have data |

## Key Features

✅ Automatic data extraction from CSV  
✅ Single command execution  
✅ Dynamic table generation  
✅ Publication-quality visualizations  
✅ Comprehensive statistical analysis  
✅ Excel file output  
✅ High-resolution figures (300 DPI)  
✅ Error handling and validation  

## System Components

| File | Purpose |
|------|---------|
| `ms_evaluation_integrated.py` | Main orchestrator & analyzer |
| `ms_table_generator.py` | Table creation module |
| `ms_visualization_generator.py` | Figure generation module |
| `requirements.txt` | Python dependencies |
| `README.md` | System overview |
| `USER_GUIDE.md` | Detailed documentation |

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- openpyxl

## Getting Help

1. Check `USER_GUIDE.md` for detailed instructions
2. Review `DEVELOPMENT_SUMMARY.md` for technical details
3. Examine `MS_Evaluation_Results_Example/` for sample outputs
4. Look at code comments in `.py` files

## Version Info

**Current Version**: 2.0  
**Python Required**: 3.7+  
**Last Updated**: 2026-01-29  

---

© 2026 MS Framework Evaluation System
