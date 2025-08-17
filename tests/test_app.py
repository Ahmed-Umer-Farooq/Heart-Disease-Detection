import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import (
    get_population_averages,
    get_chest_pain_description,
    get_ecg_description,
    get_slope_description,
    get_thal_description,
    get_professional_recommendations
)

def test_get_population_averages():
    """Test population averages function"""
    averages = get_population_averages()
    assert isinstance(averages, dict)
    assert 'age' in averages
    assert 'trestbps' in averages
    assert 'chol' in averages
    assert 'thalach' in averages
    assert 'oldpeak' in averages
    assert averages['age'] == 54

def test_get_chest_pain_description():
    """Test chest pain description function"""
    # Test all valid values
    assert get_chest_pain_description(0) == "Asymptomatic"
    assert get_chest_pain_description(1) == "Typical Angina"
    assert get_chest_pain_description(2) == "Atypical Angina"
    assert get_chest_pain_description(3) == "Non-Anginal Pain"
    
    # Test invalid value
    assert get_chest_pain_description(4) == "Unknown"

def test_get_ecg_description():
    """Test ECG description function"""
    # Test all valid values
    assert get_ecg_description(0) == "Normal"
    assert get_ecg_description(1) == "ST-T Abnormality"
    assert get_ecg_description(2) == "Left Ventricular Hypertrophy"
    
    # Test invalid value
    assert get_ecg_description(3) == "Unknown"

def test_get_slope_description():
    """Test slope description function"""
    # Test all valid values
    assert get_slope_description(0) == "Upsloping"
    assert get_slope_description(1) == "Flat"
    assert get_slope_description(2) == "Downsloping"
    
    # Test invalid value
    assert get_slope_description(3) == "Unknown"

def test_get_thal_description():
    """Test thal description function"""
    # Test all valid values
    assert get_thal_description(0) == "Unknown"
    assert get_thal_description(1) == "Normal"
    assert get_thal_description(2) == "Fixed Defect"
    assert get_thal_description(3) == "Reversible Defect"
    
    # Test invalid value
    assert get_thal_description(4) == "Unknown"

def test_get_professional_recommendations():
    """Test professional recommendations function"""
    patient_data = {
        'age': 52,
        'sex': 1,
        'cp': 0,
        'trestbps': 125,
        'chol': 212,
        'fbs': 0,
        'restecg': 1,
        'thalach': 168,
        'exang': 0,
        'oldpeak': 1.0,
        'slope': 2,
        'ca': 2,
        'thal': 3
    }
    
    # Test critical risk recommendations
    recs = get_professional_recommendations("CRITICAL RISK", 0.8, patient_data)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert any("URGENT" in rec[0] for rec in recs)
    
    # Test high risk recommendations
    recs = get_professional_recommendations("HIGH RISK", 0.7, patient_data)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert any("URGENT" in rec[0] for rec in recs)
    
    # Test moderate risk recommendations
    recs = get_professional_recommendations("MODERATE RISK", 0.5, patient_data)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert any("HIGH" in rec[0] for rec in recs)
    
    # Test low risk recommendations
    recs = get_professional_recommendations("LOW RISK", 0.2, patient_data)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert any("MODERATE" in rec[0] for rec in recs)