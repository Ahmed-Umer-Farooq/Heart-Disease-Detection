import pytest
import pandas as pd
import numpy as np
from src.utils.charts import plot_radar_chart

def test_plot_radar_chart():
    """Test that radar chart function runs without error"""
    # Sample patient data
    patient_data = {
        'age': 52,
        'trestbps': 125,
        'chol': 212,
        'thalach': 168,
        'oldpeak': 1.0
    }
    
    # Population averages
    avg_data = {
        'age': 54,
        'trestbps': 131,
        'chol': 246,
        'thalach': 149,
        'oldpeak': 1.0
    }
    
    # Test that function runs without error
    fig = plot_radar_chart(patient_data, avg_data)
    
    # Check that figure is created
    assert fig is not None
    assert hasattr(fig, 'data')
    assert len(fig.data) == 2  # Should have two traces (patient and average)

def test_plot_radar_chart_empty_data():
    """Test radar chart with empty data"""
    patient_data = {}
    avg_data = {}
    
    # Should handle empty data gracefully
    fig = plot_radar_chart(patient_data, avg_data)
    assert fig is not None