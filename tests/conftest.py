import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import joblib
import os

# Sample patient data for testing
@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
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

@pytest.fixture
def sample_patient_df(sample_patient_data):
    """Sample patient data as DataFrame"""
    return pd.DataFrame([sample_patient_data])

# Create a simple model for testing if needed
@pytest.fixture(scope="session")
def dummy_model(tmp_path_factory):
    """Create a dummy model for testing"""
    # Create a simple dummy model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=13, random_state=42)
    
    # Create and train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model to temporary file
    model_path = tmp_path_factory.mktemp("models") / "dummy_model.pkl"
    joblib.dump(model, model_path)
    
    return model_path