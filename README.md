# Neuro-AI: AI-Driven MS Lesion Analysis Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

A comprehensive clinical decision support system for automated multiple sclerosis (MS) lesion characterization from T2-FLAIR MRI imaging. This framework provides standardized, reproducible quantitative metrics aligned with McDonald 2017 diagnostic criteria.

> **Reference**: XXX et al. (2026). "AI-Driven Multi-Parametric MS Lesion Analysis from T2-FLAIR Imaging: A Clinical Decision Support Framework for Neuroradiology." *Journal of Imaging Informatics in Medicine* (In Press).

---

## 🎯 Key Features

### AI-Powered Segmentation
- **Dual cGAN Models**: Specialized gray matter segmentation and multi-class white matter hyperintensity detection
- **Context-Aware Refinement**: Intelligent lesion classification based on neuroanatomical knowledge
- **Real-time Processing**: 42.2 ± 14.4 seconds per case end-to-end analysis

### Neuroanatomical Classification
Automated distinction between three clinically significant lesion types:
- **Periventricular lesions** (3-10mm from ventricles)
- **Juxtacortical lesions** (≤3mm from gray-white matter junction)
- **Paraventricular lesions** (deep white matter)

### Multi-Parametric Quantitative Analysis
Five-domain comprehensive assessment:
1. **Lesion Count**: Discrete lesion identification across anatomical regions
2. **Area Quantification**: Precise morphometric measurements
3. **Penetration Depth**: Distance from CSF/ventricular boundaries
4. **Hemispheric Position**: Spatial distribution analysis
5. **Intensity Characterization**: Lesion heterogeneity profiling

### Clinical Validation
- **Expert Correlation**: r > 0.9 across all lesion categories (n=15 cases)
- **Workflow Efficiency**: 74.2% reduction in analysis time (11.3 → 3.0 minutes per case)
- **Diagnostic Accuracy**: 4.3 ± 0.5 / 5.0 expert rating
- **Clinical Readiness**: 9.0 / 10.0 consensus rating from board-certified neuroradiologists
- **Cross-Center Validation**: Tested on institutional and MSSEG2016 datasets

---

## 📊 Performance Metrics

| Metric | Performance |
|--------|-------------|
| **Periventricular Lesions** | r = 0.929, 95% CI [0.796, 0.977], MAE = 3.3 |
| **Paraventricular Lesions** | r = 0.932, 95% CI [0.804, 0.978], MAE = 3.9 |
| **Juxtacortical Lesions** | r = 0.927, 95% CI [0.790, 0.976], MAE = 1.7 |
| **Overall Correlation** | r = 0.955, 95% CI [0.868, 0.985], MAE = 5.7 |
| **Segmentation DSC** | 0.852 ± 0.004 (5-fold CV) |
| **Processing Time** | 42.2 ± 14.4 seconds/case |
| **Time Savings** | 74.2% (p = 0.017) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mri-ms-analyzer/neuro-ai.git
cd neuro-ai

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate neuro-ai

# Or use pip
pip install -r requirements.txt
```

### Basic Usage

```python
from src.pipeline import MSLesionAnalyzer

# Initialize analyzer
analyzer = MSLesionAnalyzer(
    gm_model_path='models/gray_matter_segmentation/model_weights.h5',
    wmh_model_path='models/wmh_segmentation/model_weights.h5'
)

# Process FLAIR image
results = analyzer.analyze('path/to/flair.nii.gz')

# Access results
print(f"Total lesions: {results['total_lesions']}")
print(f"Periventricular: {results['periventricular_count']}")
print(f"Juxtacortical: {results['juxtacortical_count']}")
print(f"Paraventricular: {results['paraventricular_count']}")

# Generate visualization
analyzer.visualize(results, output_path='results/patient_001/')
```

### Command-Line Interface

```bash
# Analyze single case
python run_analysis.py --input data/case_001/flair.nii.gz --output results/case_001/

# Batch processing
python run_analysis.py --input-dir data/cases/ --output-dir results/ --batch

# Generate comprehensive report
python run_analysis.py --input data/case_001/flair.nii.gz --output results/case_001/ --report
```

---

## 📁 Repository Structure

```
neuro-ai/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── setup.py                     # Package installation
│
├── src/                         # Source code
│   ├── preprocessing/           # Image preprocessing pipeline
│   ├── models/                  # cGAN architectures
│   ├── postprocessing/          # Context-aware refinement
│   ├── analysis/                # Multi-parametric feature extraction
│   ├── visualization/           # Result visualization
│   └── utils/                   # Utility functions
│
├── models/                      # Pre-trained model weights
│   ├── gray_matter_segmentation/
│   └── wmh_segmentation/
│
├── data/                        # Sample data and examples
│   ├── sample_cases/            # 3 representative cases
│   ├── validation_cases/        # 15 validation cases
│   └── AI_outputs/           # AI-analyzed results
│
├── validation/                  # Validation scripts and data
│   ├── expert_evaluation_data/  # Expert assessment results
│   ├── evaluation_analysis_results/  # Statistical Analysis Results
│   ├── statistical_analysis.py  # Correlation and metrics
│   └── generate_figures.py      # Reproduce manuscript figures
│
├── notebooks/                   # Jupyter tutorials
│   ├── 01_quick_start.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_inference.ipynb
│   ├── 04_feature_extraction.ipynb
│   └── 05_validation.ipynb
│
├── docs/                        # Documentation
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   ├── ARCHITECTURE.md
│   └── FAQ.md
│
└── scripts/                     # Utility scripts
    ├── download_models.sh       # Download pre-trained weights
    └── process_msseg2016.py     # MSSEG2016 preprocessing
```

---

## 🧠 Architecture Overview

### Preprocessing Pipeline
1. **Noise Removal**: Median filtering + selective Gaussian filtering
2. **Brain Extraction**: Morphology-based elliptical mask approximation
3. **Intensity Normalization**: Slice-wise normalization to [-1, 1]
4. **CSF Extraction**: Otsu thresholding with brain mask refinement

### Deep Learning Models

#### Model 1: Gray Matter Segmentation
- **Architecture**: Conditional GAN (pix2pix) with U-Net generator
- **Input**: T2-FLAIR images
- **Output**: Specialized gray matter masks optimized for lesion-present conditions
- **Performance**: DSC = 0.76 ± 0.04 for juxtacortical lesion classification

#### Model 2: Multi-Class WMH Segmentation
- **Architecture**: Conditional GAN (pix2pix) with attention mechanism
- **Input**: T2-FLAIR images
- **Output**: Four classes (ventricles, CSF, normal WMH, abnormal WMH)
- **Performance**: DSC = 0.852 ± 0.004 (5-fold cross-validation)
- **Innovation**: Distinguishes pathological from normal periventricular hyperintensities

### Post-Processing
1. **Context-Aware Refinement**: Spatial contiguity analysis (1mm threshold)
2. **Neuroanatomical Classification**: Geodesic distance-based lesion categorization
3. **Multi-Parametric Analysis**: Five-domain quantitative feature extraction

---

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions for different platforms
- **[Usage Guide](docs/USAGE.md)**: Comprehensive API documentation and examples
- **[Architecture Details](docs/ARCHITECTURE.md)**: Technical deep-dive into model design
- **[FAQ](docs/FAQ.md)**: Common questions and troubleshooting

---

## 🔬 Clinical Applications

### Diagnostic Support
- Automated lesion detection and classification aligned with McDonald 2017 criteria
- Standardized quantitative metrics for dissemination in space (DIS) assessment
- Objective disease burden quantification for clinical reporting

### Longitudinal Monitoring
- Reproducible baseline and follow-up assessments
- Early detection of subtle lesion changes
- Treatment response evaluation

### Research Applications
- Standardized endpoints for clinical trials
- Multi-center studies with consistent methodology
- Correlation with clinical outcome measures (EDSS, cognitive function)

### Educational Use
- Training tool for radiology residents
- MS imaging protocol standardization
- Benchmark for algorithm development

---

## 📊 Datasets

### Training Dataset
- **Source**: Golgasht Medical Imaging Center, Tabriz, Iran
- **Size**: 300 MS patients (79M/221F, age 18-68, mean 37.8±9.7)
- **Scanner**: 1.5T TOSHIBA Vantage
- **Protocol**: T2-FLAIR (TR=10,000ms, TE=100ms, TI=2,500ms)
- **Voxel Size**: 0.9×0.9×6.0 mm³
- **Ground Truth**: Manual annotation by board-certified neuroradiologist (20+ years experience)
- **Access**: Available through controlled access (see Data Availability)

### External Validation Dataset
- **Source**: MSSEG2016 Challenge Dataset
- **Size**: 6 cases (2 per center × 3 centers)
- **Centers**: Multiple imaging centers with different scanners/protocols
- **Access**: Publicly available at https://portal.fli-iam.irisa.fr/msseg-challenge/

### Sample Data
- **Included**: 9 representative cases with complete annotations
- **Format**: NIfTI (.nii.gz) with DICOM metadata
- **Purpose**: Testing framework without full dataset access

---

## ⚙️ System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 10.15+
- **CPU**: Intel Core i5 or equivalent (4+ cores)
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with 8GB VRAM (optional but recommended)
- **Storage**: 10 GB free space
- **Python**: 3.9+

### Recommended Requirements
- **CPU**: Intel Core i7-7700K or equivalent (8+ cores)
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or better
- **CUDA**: 11.2+ with cuDNN 8.1+
- **Storage**: 50 GB SSD (for full dataset and models)

### Software Dependencies
- Python 3.9+
- TensorFlow 2.8+
- NumPy 1.21+
- SciPy 1.7+
- OpenCV 4.5+
- scikit-image 0.19+
- nibabel 3.2+ (for NIfTI support)
- pydicom 2.3+ (for DICOM support)

---

## 🔄 Workflow Integration

### Standalone Application
```bash
# Process single case
python run_analysis.py --input patient.nii.gz --output results/

# Output includes:
# - Segmentation masks (NIfTI format)
# - Color-coded overlays (PNG)
# - Heat maps (PNG)
# - Quantitative report (PDF/JSON)
# - Statistical plots (PNG)
```

### PACS Integration
The framework can be integrated with Picture Archiving and Communication Systems (PACS):
- DICOM input/output support
- HL7 reporting interface
- Automatic worklist processing
- Results pushed back to PACS

### Research Pipeline
```python
# Batch processing for research studies
from src.batch import BatchProcessor

processor = BatchProcessor(
    input_dir='study_data/',
    output_dir='results/',
    num_workers=4
)

# Process entire cohort
processor.run(generate_report=True, export_csv=True)

# Export to statistical software
processor.export_to_csv('results/cohort_metrics.csv')
```

---

## 📈 Validation Results

### Expert Correlation Analysis

**Institutional Dataset (n=9):**
- Periventricular: r=0.941, MAE=3.3 lesions
- Paraventricular: r=0.943, MAE=3.6 lesions
- Juxtacortical: r=0.969, MAE=1.6 lesions

**External Dataset (MSSEG2016, n=6):**
- Periventricular: r=0.935, MAE=3.4 lesions
- Paraventricular: r=0.925, MAE=4.3 lesions
- Juxtacortical: r=0.906, MAE=1.8 lesions

**Overall (n=15):**
- All lesion types: r=0.955, MAE=5.7 lesions
- Consistent performance across different scanners and protocols

### Workflow Efficiency
- **Analysis Time**: 11.3±2.7 min (manual) → 3.0±1.1 min (AI-assisted)
- **Time Reduction**: 74.2% (p=0.017, Cohen's d=3.27)
- **Expert Agreement**: Perfect inter-rater reliability (9.0±0.0 / 10.0)

### Clinical Utility Ratings (5-point Likert scale)
- Segmentation Accuracy: 4.3 ± 0.5
- Detection Completeness: 4.6 ± 0.6
- Classification Accuracy: 4.5 ± 0.5
- Overall Framework Rating: 9.0 ± 0.0 / 10.0

---

## 🧪 Reproducing Results

### Reproduce Validation Study

```bash
# Download validation data
python scripts/download_validation_data.py

# Run validation analysis
cd validation/
python statistical_analysis.py --cases validation_cases/ --expert-data expert_evaluation_data/

# Generate manuscript figures
python generate_figures.py --output ../figures/
```

### Reproduce Model Training

```bash
# Requires full training dataset (contact authors for access)
cd src/models/

# Train GM segmentation model (Model 1)
python train_gm_segmentation.py --data ../../data/training/ --folds 5

# Train WMH segmentation model (Model 2)
python train_wmh_segmentation.py --data ../../data/training/ --folds 5
```

### Cross-Validation

```python
from src.validation import CrossValidator

validator = CrossValidator(
    data_dir='data/training/',
    n_folds=5,
    output_dir='cv_results/'
)

# Run 5-fold cross-validation
results = validator.run()

# Generate performance report
validator.generate_report(results, 'cv_report.pdf')
```

---

## 🎓 Citation

If you use this framework in your research, please cite:

```bibtex
@article{XXX2026neuroai,
  title={AI-Driven Multi-Parametric MS Lesion Analysis: A Clinical Decision Support Framework for Neuroradiology},
  author={[Authors]},
  journal={Journal of Imaging Informatics in Medicine},
  year={2026},
  note={In Press},
  doi={[DOI to be added upon publication]}
}
```

**Software Citation:**
```bibtex
@software{neuroai2026,
  author={[Authors]},
  title={Neuro-AI: AI-Driven MS Lesion Analysis Framework},
  year={2026},
  publisher={GitHub},
  url={https://github.com/mri-ms-analyzer/neuro-ai}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Summary**:
- ⚠️ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ⚠️ Liability and warranty disclaimers apply

---

## 🤝 Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 📖 Improve documentation
- 🧪 Add test cases
- 🔧 Submit bug fixes or enhancements
- 📊 Share validation results on new datasets

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/neuro-ai.git
cd neuro-ai

# Create development environment
conda env create -f environment-dev.yml
conda activate neuro-ai-dev

# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/ --check
```

---

## 📞 Contact & Support

### Authors
- **Corresponding Author**: Prof. XXX
  - Email: [email]
  - Affiliation: XXX

### Issues & Questions
- **GitHub Issues**: [Report bugs or request features](https://github.com/mri-ms-analyzer/neuro-ai/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/mri-ms-analyzer/neuro-ai/discussions)
- **Email**: For data access requests or collaboration inquiries

### Acknowledgments
- Golgasht Medical Imaging Center, Tabriz, Iran for providing imaging data
- Expert neuroradiologists for validation assessments
- MSSEG2016 challenge organizers for public dataset
- Open-source community for foundational tools (TensorFlow, scikit-image, etc.)

---

## 🔗 Related Resources

### McDonald 2017 Criteria
- Thompson et al. (2018). "Diagnosis of Multiple sclerosis: 2017 Revisions of the McDonald Criteria." *The Lancet Neurology*, 17(2), 162-173.

### MS Imaging Guidelines
- Rovira et al. (2015). "MAGNIMS consensus guidelines on the use of MRI in multiple sclerosis." *Nature Reviews Neurology*, 11(8), 471-482.
- Traboulsee et al. (2016). "Revised Recommendations of the Consortium of MS Centers Task Force." *American Journal of Neuroradiology*, 37(3), 394-401.

### MSSEG2016 Challenge
- Commowick et al. (2018). "Objective Evaluation of Multiple Sclerosis Lesion Segmentation using a Data Management and Processing Infrastructure." *Scientific Reports*, 8(1), 13650.

### Deep Learning for Medical Imaging
- Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*.
- Isola et al. (2017). "Image-to-Image Translation with Conditional Adversarial Networks." *CVPR*.

---

## 🗺️ Roadmap

### Version 1.0 (Current)
- ✅ Dual cGAN segmentation models
- ✅ Multi-parametric feature extraction
- ✅ Clinical validation on 15 cases
- ✅ Sample data and pre-trained weights

### Version 1.1 (Q2 2026)
- 🔄 Longitudinal analysis module (track lesion evolution over time)
- 🔄 Integration with additional MRI sequences (T1, T2, DWI)
- 🔄 Enhanced visualization dashboard
- 🔄 DICOM SR structured reporting

### Version 2.0 (Q4 2026)
- 📋 Full 3D segmentation architecture
- 📋 Multi-sequence fusion (T1, T2, FLAIR, DWI, post-contrast)
- 📋 Black hole and paramagnetic rim lesion detection
- 📋 Cognitive correlation analysis
- 📋 Clinical trial endpoint generation

### Version 2.1 (2027)
- 📋 Real-time PACS integration
- 📋 Cloud-based processing option
- 📋 Mobile app for remote review
- 📋 Multi-language support
- 📋 Federated learning for privacy-preserving model updates

---

## 📝 Changelog

### v1.0.0 (February 2026)
- Initial public release
- Dual cGAN segmentation models
- Multi-parametric analysis framework
- Clinical validation results
- Sample data and documentation

---

## ⚠️ Disclaimer

**Clinical Use Notice**: This software is provided for research and educational purposes. While the framework has been validated by board-certified neuroradiologists, it is NOT approved as a medical device by regulatory agencies (FDA, CE, etc.). Clinical decisions should always be made by qualified healthcare professionals using their clinical judgment. The authors and contributors assume no responsibility for any clinical use of this software.

**Data Privacy**: Users are responsible for ensuring compliance with applicable data privacy regulations (HIPAA, GDPR, etc.) when processing patient data. Remove all protected health information (PHI) before sharing or uploading data.

**Performance Variability**: Model performance may vary with different scanner types, imaging protocols, and patient populations. External validation on your specific data is recommended before clinical deployment.

---

## 📊 Repository Statistics

![GitHub stars](https://img.shields.io/github/stars/mri-ms-analyzer/neuro-ai?style=social)
![GitHub forks](https://img.shields.io/github/forks/mri-ms-analyzer/neuro-ai?style=social)
![GitHub issues](https://img.shields.io/github/issues/mri-ms-analyzer/neuro-ai)
![GitHub pull requests](https://img.shields.io/github/issues-pr/mri-ms-analyzer/neuro-ai)
![Last commit](https://img.shields.io/github/last-commit/mri-ms-analyzer/neuro-ai)

---

**Last Updated**: February 2026 | **Version**: 1.0.0 | **Status**: Active Development

---

*Made with ❤️ for the MS research community*
