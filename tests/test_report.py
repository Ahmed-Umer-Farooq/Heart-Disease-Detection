import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.utils.report import (
    get_chest_pain_type, 
    get_risk_level_and_color, 
    get_recommendations,
    create_clean_pil_report
)

def test_get_chest_pain_type():
    """Test chest pain type conversion"""
    # Test all valid values
    assert get_chest_pain_type(0) == 'Typical Angina'
    assert get_chest_pain_type(1) == 'Atypical Angina'
    assert get_chest_pain_type(2) == 'Non-anginal Pain'
    assert get_chest_pain_type(3) == 'Asymptomatic'
    
    # Test invalid value
    assert get_chest_pain_type(4) == 'Unknown'
    assert get_chest_pain_type(-1) == 'Unknown'

def test_get_risk_level_and_color():
    """Test risk level determination"""
    # Mock colors dictionary
    colors = {
        'danger': '#dc2626',
        'warning': '#d97706', 
        'success': '#059669'
    }
    
    # Test high risk (prediction = 1)
    risk_level, risk_color = get_risk_level_and_color(1, 0.5, colors)
    assert risk_level == "HIGH RISK"
    assert risk_color == colors['danger']
    
    # Test high risk (probability >= 0.7)
    risk_level, risk_color = get_risk_level_and_color(0, 0.8, colors)
    assert risk_level == "HIGH RISK"
    assert risk_color == colors['danger']
    
    # Test moderate risk (probability >= 0.4)
    risk_level, risk_color = get_risk_level_and_color(0, 0.5, colors)
    assert risk_level == "MODERATE RISK"
    assert risk_color == colors['warning']
    
    # Test low risk (probability < 0.4)
    risk_level, risk_color = get_risk_level_and_color(0, 0.2, colors)
    assert risk_level == "LOW RISK"
    assert risk_color == colors['success']

def test_get_recommendations():
    """Test recommendation generation"""
    # Test high risk recommendations
    recs = get_recommendations("HIGH RISK", 0.8)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert "Immediate cardiology consultation" in recs[0]
    
    # Test moderate risk recommendations
    recs = get_recommendations("MODERATE RISK", 0.5)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert "Follow-up with healthcare provider" in recs[0]
    
    # Test low risk recommendations
    recs = get_recommendations("LOW RISK", 0.2)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert "Continue current healthy lifestyle" in recs[0]
    
    # Test unknown risk level (should default to moderate)
    recs = get_recommendations("UNKNOWN", 0.5)
    assert isinstance(recs, list)
    assert len(recs) > 0

def test_create_clean_pil_report():
    """Test PIL report generation"""
    # Sample patient data
    patient_data = {
        'age': 52,
        'sex': 1,
        'cp': 0,
        'trestbps': 125,
        'chol': 212,
        'thalach': 168,
        'oldpeak': 1.0,
        'ca': 2
    }
    
    # Test report generation with mock data
    with patch('src.utils.report.Image.new') as mock_image, \
         patch('src.utils.report.ImageDraw.Draw') as mock_draw, \
         patch('src.utils.report.io.BytesIO') as mock_bytesio:
        
        # Mock the PIL objects
        mock_img_instance = Mock()
        mock_image.return_value = mock_img_instance
        
        mock_draw_instance = Mock()
        mock_draw.return_value = mock_draw_instance
        
        mock_buffer = Mock()
        mock_bytesio.return_value = mock_buffer
        
        # Test the function
        result = create_clean_pil_report(patient_data, 1, 0.8)
        
        # Verify that PIL functions were called
        mock_image.assert_called()
        mock_draw.assert_called()