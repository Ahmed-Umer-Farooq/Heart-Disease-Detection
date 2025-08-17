# CardioInsight AI Documentation

Welcome to the CardioInsight AI documentation. This guide will help you understand, install, and use the CardioInsight AI heart disease detection system.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [API Reference](#api-reference)
5. [Model Details](#model-details)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

CardioInsight AI is a machine learning-powered application designed to assist medical professionals in assessing cardiovascular risk. The system uses a Random Forest classifier trained on the UCI Heart Disease dataset to predict the likelihood of heart disease based on patient data.

## Installation

For detailed installation instructions, please refer to the [README.md](../README.md) file.

## Usage

The application can be run using Streamlit:

```bash
streamlit run app.py
```

## API Reference

### Main Application Functions

#### `load_model()`
Loads the pre-trained Random Forest model.

#### `get_population_averages()`
Returns average values for key cardiovascular metrics.

#### `create_ultra_professional_report()`
Generates a professional medical report in PNG format.

## Model Details

The model is a hyperparameter-tuned Random Forest classifier with the following performance metrics:
- Accuracy: 94.2%
- Sensitivity: 91.8%
- Specificity: 96.1%
- AUC-ROC: 0.952

## Contributing

Please read [CONTRIBUTING.md](../CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.