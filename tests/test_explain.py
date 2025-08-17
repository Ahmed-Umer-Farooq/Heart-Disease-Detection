import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.utils.explain import get_shap_explanation, generate_plain_english_summary

def test_get_shap_explanation():
    """Test SHAP explanation function"""
    # Create a mock model
    mock_model = Mock()
    
    # Create sample patient data
    patient_data = pd.DataFrame([{
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
    }])
    
    # Mock the SHAP explainer
    with patch('src.utils.explain.shap.TreeExplainer') as mock_explainer:
        # Create a mock explainer instance
        mock_explainer_instance = Mock()
        mock_explainer.return_value = mock_explainer_instance
        
        # Create a mock SHAP values object
        mock_shap_values = Mock()
        mock_shap_values_multiclass = Mock()
        mock_shap_values_multiclass.__getitem__ = Mock(return_value=mock_shap_values)
        mock_explainer_instance.return_value = mock_shap_values_multiclass
        
        # Test the function
        result = get_shap_explanation(mock_model, patient_data)
        
        # Verify the explainer was called correctly
        mock_explainer.assert_called_once_with(mock_model)
        mock_explainer_instance.assert_called_once_with(patient_data)

def test_generate_plain_english_summary():
    """Test plain English summary generation"""
    # Create a mock SHAP values object
    mock_shap_values = Mock()
    mock_shap_values.values = np.array([0.1, -0.05, 0.2, -0.15, 0.08])
    mock_shap_values.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol']
    mock_shap_values.data = [52, 1, 0, 125, 212]
    
    # Test the function
    summary = generate_plain_english_summary(mock_shap_values)
    
    # Check that summary is generated
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert 'The model\'s prediction was primarily influenced by these factors:' in summary