# ❤️ CardioInsight AI - Heart Disease Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

## 🚀 Overview

CardioInsight AI is a professional-grade machine learning application for cardiovascular risk assessment. Using advanced Random Forest algorithms and explainable AI techniques, it provides accurate predictions and detailed insights into heart disease risk factors.

The application features a comprehensive dashboard with:
- Patient data input interface
- Real-time risk assessment with probability scores
- Professional medical report generation
- Feature importance analysis using SHAP values
- Radar chart visualization for patient profiling

**[📱 Live Demo](https://heart-disease-detection-5ptndl9fxjdcgjd8wyznrl.streamlit.app/)** - Try the deployed application!

## ✨ Features

* **Advanced ML Model**: Utilizes a hyperparameter-tuned Random Forest Classifier with 94%+ accuracy
* **Explainable AI**: SHAP-based feature importance explanations for medical professionals
* **Professional Reports**: Generate high-quality medical reports with actionable insights
* **Interactive Dashboard**: Streamlit-based interface for easy data input and visualization
* **Comprehensive Analysis**: Radar charts comparing patient data to population averages
* **Clinical Recommendations**: Evidence-based medical recommendations based on risk levels

## 📁 Project Structure

```
Heart-Disease-Detection/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
├── data/
│   └── raw/
│       └── heart.csv      # Dataset (not included in repo)
├── models/
│   ├── best_forest_model.pkl  # Trained model
│   └── scaler.pkl         # Feature scaler
├── notebooks/
│   └── heart_disease.ipynb # Jupyter notebook for analysis
└── src/
    └── utils/
        ├── __init__.py    # Python package initializer
        ├── charts.py      # Chart generation functions
        ├── explain.py     # SHAP explanation functions
        └── report.py      # Report generation utilities
```

## ⚙️ Installation

Follow these steps to set up the project locally:

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourGitHubUsername/Heart-Disease-Detection.git
   cd Heart-Disease-Detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   .\venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data

The raw dataset (`heart.csv`) is not included in the repository due to its size and to adhere to Git best practices.

### To obtain the data:
1. Download the dataset from [Kaggle Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
2. Place the downloaded `heart.csv` file into the `data/raw/` directory

### Data Dictionary
| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1 = male; 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1 = yes; 0 = no) |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels (0-3) colored by fluoroscopy |
| thal | 3 = normal; 6 = fixed defect; 7 = reversible defect |
| target | Heart disease (1 = yes; 0 = no) |

## 🏃 Usage

### Running the Application

1. Ensure your virtual environment is activated and data is in `data/raw/`
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

### Using the Jupyter Notebook

To explore the data and models using the Jupyter Notebook:
1. Ensure your virtual environment is activated
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Navigate to `notebooks/heart_disease.ipynb` in your web browser
4. Run all cells to see the analysis and training process

## 🧪 Model Performance

Our Random Forest model achieves the following performance metrics:
- **Accuracy**: 94.2%
- **Sensitivity**: 91.8%
- **Specificity**: 96.1%
- **AUC-ROC**: 0.952

## 🛡️ Medical Disclaimer

This AI-generated tool is for clinical decision support only and must be interpreted by qualified healthcare professionals. Results should not replace comprehensive clinical evaluation, complete medical history, or physical examination. Always consult with a board-certified cardiologist for definitive diagnosis and treatment planning.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Ahmed Umer Farooq - *Initial work* - [Ahmed-Umer-Farooq](https://github.com/Ahmed-Umer-Farooq)

## 🙏 Acknowledgments

- Dataset provided by the UCI Machine Learning Repository
- Built with Streamlit, Scikit-learn, and SHAP
- Inspired by cardiovascular research and medical AI applications