# utils/explain.py

import shap
import pandas as pd

def get_shap_explanation(model, patient_data):
    """Generates SHAP values to explain the prediction for the 'High Risk' class."""
    explainer = shap.TreeExplainer(model)
    shap_values_multiclass = explainer(patient_data)
    
    # We explicitly select the explanation for the FIRST patient and for CLASS 1 (High Risk)
    return shap_values_multiclass[0, :, 1]

def generate_plain_english_summary(shap_values):
    """Generates a plain-English summary of the top prediction drivers."""
    abs_shap = abs(shap_values.values)
    sorted_indices = abs_shap.argsort()[::-1]

    summary = "The model's prediction was primarily influenced by these factors:\n\n"
    
    for i in range(min(3, len(sorted_indices))):
        idx = sorted_indices[i]
        feature = shap_values.feature_names[idx]
        value = shap_values.data[idx]
        shap_value = shap_values.values[idx]

        direction = "increased" if shap_value > 0 else "decreased"
        impact_word = "significantly" if abs(shap_value) > 0.05 else "slightly"

        summary += f"- The **{feature}** value of **{value}** {impact_word} {direction} the patient's risk.\n"
        
    return summary