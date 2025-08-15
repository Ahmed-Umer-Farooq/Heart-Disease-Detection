# utils/charts.py

import plotly.graph_objects as go
import pandas as pd

def plot_radar_chart(patient_data, avg_data):
    """Creates a radar chart comparing patient values to population averages."""
    categories = list(avg_data.keys())
    
    max_vals = {'age': 100, 'trestbps': 200, 'chol': 570, 'thalach': 220, 'oldpeak': 6.2}
    patient_normalized = [patient_data[cat] / max_vals[cat] * 100 for cat in categories]
    avg_normalized = [avg_data[cat] / max_vals[cat] * 100 for cat in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=patient_normalized, theta=categories, fill='toself', name='Patient Values',
        line=dict(color='teal')
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_normalized, theta=categories, fill='toself', name='Population Averages',
        line=dict(color='rgba(0,0,0,0.2)')
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, title="Patient vs. Averages (Normalized %)",
        font=dict(size=10), legend=dict(yanchor="top", y=1.15, xanchor="left", x=0.1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
    
def plot_shap_explanation(shap_values, feature_names): # <-- ADDED feature_names HERE
    """Creates a waterfall chart for SHAP values."""
    fig = go.Figure(go.Waterfall(
        name="Prediction Explanation", orientation="h",
        measure=["relative"] * len(shap_values.values),
        y=feature_names, # <-- USE feature_names HERE
        x=shap_values.values,
        base=shap_values.base_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker":{"color":"#FF4136"}}, 
        decreasing={"marker":{"color":"#2ECC40"}}
    ))
    fig.update_layout(
        title="Key Factors Influencing Prediction (SHAP)",
        showlegend=False, yaxis_title="Features", xaxis_title="Contribution to Risk Probability"
    )
    return fig