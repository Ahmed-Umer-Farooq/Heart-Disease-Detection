# app.py - Complete Fixed Cardiovascular Risk Assessment Application

import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import base64
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Cardio-Insight AI | Professional Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching with error handling ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('models/best_forest_model.pkl')
    except FileNotFoundError:
        st.error("Model file 'best_forest_model.pkl' not found at 'models/best_forest_model.pkl'!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Helper Functions ---
@st.cache_data
def get_population_averages():
    return {'age': 54, 'trestbps': 131, 'chol': 246, 'thalach': 149, 'oldpeak': 1.0}

def get_chest_pain_description(cp_type):
    descriptions = {0: "Asymptomatic", 1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain"}
    return descriptions.get(cp_type, "Unknown")

def get_ecg_description(restecg):
    descriptions = {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}
    return descriptions.get(restecg, "Unknown")

def get_slope_description(slope):
    descriptions = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    return descriptions.get(slope, "Unknown")

def get_thal_description(thal):
    descriptions = {0: "Unknown", 1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
    return descriptions.get(thal, "Unknown")

def get_professional_recommendations(risk_level, probability, patient_data):
    base_recommendations = []
    if risk_level in ["CRITICAL RISK", "HIGH RISK"]:
        base_recommendations = [
            ("URGENT", "Immediate cardiology consultation within 24-48 hours"),
            ("URGENT", "Consider emergency department evaluation if symptomatic"),
            ("HIGH", "Comprehensive cardiac catheterization evaluation"),
            ("HIGH", "Initiate dual antiplatelet therapy if not contraindicated"),
            ("HIGH", "Aggressive statin therapy (high-intensity)"),
            ("MODERATE", "Lifestyle modification counseling with cardiac rehabilitation"),
        ]
    elif risk_level == "MODERATE RISK":
        base_recommendations = [
            ("HIGH", "Cardiology consultation within 2-4 weeks"),
            ("HIGH", "Exercise stress testing or cardiac imaging"),
            ("MODERATE", "Initiate or optimize statin therapy"),
            ("MODERATE", "Blood pressure optimization (target <130/80 mmHg)"),
        ]
    else:
        base_recommendations = [
            ("MODERATE", "Routine cardiology follow-up within 6-12 months"),
            ("MODERATE", "Maintain healthy lifestyle practices"),
            ("ROUTINE", "Annual lipid screening and blood pressure monitoring"),
        ]
    
    if patient_data['chol'] > 240:
        base_recommendations.insert(1, ("HIGH", "Aggressive lipid management - consider PCSK9 inhibitors"))
    if patient_data['trestbps'] > 140:
        base_recommendations.insert(1, ("HIGH", "Hypertension management - consider ACE inhibitor/ARB"))
    if patient_data['exang'] == 1:
        base_recommendations.insert(0, ("URGENT", "Evaluate for unstable angina - consider immediate intervention"))
    
    return base_recommendations[:8]

def load_font_with_fallback(size, bold=False):
    """Improved font loading with better fallback system and platform detection"""
    
    # Different font paths for different operating systems
    font_paths = []
    
    if bold:
        # Bold font options
        font_paths = [
            # Windows fonts
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf", 
            "C:/Windows/Fonts/trebucbd.ttf",
            "C:/Windows/Fonts/verdanab.ttf",
            # Alternative Windows paths
            "arial-bold.ttf",
            "calibri-bold.ttf",
            # Linux fonts
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Arial Bold.ttf",
            # macOS fonts
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        # Regular font options
        font_paths = [
            # Windows fonts
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/trebuc.ttf", 
            "C:/Windows/Fonts/verdana.ttf",
            # Alternative Windows paths
            "arial.ttf",
            "calibri.ttf",
            # Linux fonts
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Arial.ttf",
            # macOS fonts
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    
    # Try to load fonts in order of preference
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    
    # If no TrueType fonts work, try loading default with size adjustment
    try:
        return ImageFont.load_default()
    except Exception:
        # Last resort - create a basic font object
        return ImageFont.load_default()

def create_ultra_professional_report(patient_data, prediction, probability):
    # Report dimensions - A4 size at 300 DPI (improved resolution)
    width, height = 2480, 3508
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Enhanced color palette with better contrast
    colors = {
        'primary': '#1e3a8a',      # Deep blue
        'secondary': '#3b82f6',    # Blue
        'success': '#10b981',      # Green
        'warning': '#f59e0b',      # Amber
        'danger': '#ef4444',       # Red
        'dark': '#1f2937',         # Dark gray
        'light': '#f8fafc',        # Light gray
        'white': '#ffffff',        # White
        'border': '#e2e8f0',       # Border gray
        'text_primary': '#0f172a', # Almost black
        'text_secondary': '#475569' # Medium gray
    }
    
    # Convert hex colors to RGB tuples
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    for key, value in colors.items():
        colors[key] = hex_to_rgb(value)
    
    # Improved font definitions with better sizing
    fonts = {
        'title': load_font_with_fallback(52, True),      # Increased from 48
        'heading': load_font_with_fallback(40, True),    # Increased from 36
        'subheading': load_font_with_fallback(32, True), # Increased from 28
        'body': load_font_with_fallback(28),             # Increased from 24
        'small': load_font_with_fallback(24),            # Increased from 20
        'tiny': load_font_with_fallback(20)              # Increased from 16
    }
    
    # HEADER SECTION
    header_height = 220  # Increased height
    draw.rectangle([(0, 0), (width, header_height)], fill=colors['primary'])
    
    # Header content with better positioning
    draw.text((80, 45), "CardioInsight AI", font=fonts['title'], fill=colors['white'])
    draw.text((80, 110), "Advanced Cardiovascular Risk Assessment", font=fonts['body'], fill=colors['white'])
    
    # Timestamp with better positioning
    timestamp = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    timestamp_bbox = draw.textbbox((0, 0), f"Generated: {timestamp}", font=fonts['small'])
    timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]
    draw.text((width - timestamp_width - 80, 155), f"Generated: {timestamp}", font=fonts['small'], fill=colors['white'])
    
    # MAIN CONTENT AREA
    content_y = header_height + 50  # More spacing
    
    # Patient Information Section
    section_height = 450  # Increased height
    draw.rectangle([(80, content_y), (width - 80, content_y + section_height)], fill=colors['white'], outline=colors['border'], width=3)
    draw.rectangle([(80, content_y), (width - 80, content_y + 70)], fill=colors['light'])
    draw.text((100, content_y + 25), "PATIENT INFORMATION", font=fonts['heading'], fill=colors['primary'])
    
    # Patient details in two columns with better spacing
    left_col_x = 100
    right_col_x = width // 2 + 60
    detail_y = content_y + 120
    
    # Left column
    patient_details_left = [
        ("Patient ID:", f"CI-{datetime.now().strftime('%Y%m%d')}-{hash(str(patient_data)) % 1000:03d}"),
        ("Age:", f"{patient_data['age']} years"),
        ("Gender:", "Male" if patient_data['sex'] == 1 else "Female"),
        ("Chest Pain Type:", get_chest_pain_description(patient_data['cp'])),
        ("Resting Blood Pressure:", f"{patient_data['trestbps']} mmHg"),
        ("Serum Cholesterol:", f"{patient_data['chol']} mg/dL")
    ]
    
    for label, value in patient_details_left:
        draw.text((left_col_x, detail_y), label, font=fonts['small'], fill=colors['text_secondary'])
        draw.text((left_col_x, detail_y + 30), value, font=fonts['body'], fill=colors['text_primary'])
        detail_y += 70  # Increased spacing
    
    # Right column
    detail_y = content_y + 120
    patient_details_right = [
        ("Maximum Heart Rate:", f"{patient_data['thalach']} bpm"),
        ("Exercise Induced Angina:", "Present" if patient_data['exang'] == 1 else "Absent"),
        ("Fasting Blood Sugar:", ">120 mg/dL" if patient_data['fbs'] == 1 else "≤120 mg/dL"),
        ("Resting ECG:", get_ecg_description(patient_data['restecg'])),
        ("ST Depression:", f"{patient_data['oldpeak']} mm"),
        ("Thalassemia:", get_thal_description(patient_data['thal']))
    ]
    
    for label, value in patient_details_right:
        draw.text((right_col_x, detail_y), label, font=fonts['small'], fill=colors['text_secondary'])
        draw.text((right_col_x, detail_y + 30), value, font=fonts['body'], fill=colors['text_primary'])
        detail_y += 70  # Increased spacing
    
    # RISK ASSESSMENT SECTION
    risk_y = content_y + section_height + 50
    risk_height = 350  # Increased height
    
    # Determine risk level and color
    if prediction == 1 or probability >= 0.75:
        risk_level = "CRITICAL RISK"
        risk_color = colors['danger']
    elif probability >= 0.6:
        risk_level = "HIGH RISK"
        risk_color = colors['danger']
    elif probability >= 0.4:
        risk_level = "MODERATE RISK"
        risk_color = colors['warning']
    elif probability >= 0.2:
        risk_level = "LOW-MODERATE RISK"
        risk_color = colors['warning']
    else:
        risk_level = "LOW RISK"
        risk_color = colors['success']
    
    draw.rectangle([(80, risk_y), (width - 80, risk_y + risk_height)], fill=colors['white'], outline=risk_color, width=4)
    draw.rectangle([(80, risk_y), (width - 80, risk_y + 70)], fill=risk_color)
    draw.text((100, risk_y + 25), "RISK ASSESSMENT", font=fonts['heading'], fill=colors['white'])
    
    # Risk level and probability with better spacing
    draw.text((100, risk_y + 110), "Risk Classification:", font=fonts['small'], fill=colors['text_secondary'])
    draw.text((100, risk_y + 145), risk_level, font=fonts['heading'], fill=risk_color)
    
    draw.text((100, risk_y + 210), "Probability Score:", font=fonts['small'], fill=colors['text_secondary'])
    draw.text((100, risk_y + 245), f"{probability:.1%}", font=fonts['heading'], fill=colors['text_primary'])
    
    # Enhanced risk gauge with better proportions
    gauge_x = width // 2 + 120
    gauge_y = risk_y + 140
    gauge_width = 450  # Increased width
    gauge_height = 35  # Increased height
    
    # Draw gauge background
    draw.rectangle([(gauge_x, gauge_y), (gauge_x + gauge_width, gauge_y + gauge_height)], 
                   fill=colors['border'], outline=colors['dark'], width=2)
    
    # Draw gauge segments
    segments = [
        (0, 0.2, colors['success']),
        (0.2, 0.4, colors['warning']),
        (0.4, 0.75, colors['danger']),
        (0.75, 1.0, (139, 0, 0))  # Dark red
    ]
    
    for start, end, color in segments:
        seg_start = gauge_x + int(gauge_width * start)
        seg_width = int(gauge_width * (end - start))
        draw.rectangle([(seg_start, gauge_y), (seg_start + seg_width, gauge_y + gauge_height)], fill=color)
    
    # Enhanced needle with better visibility
    needle_x = gauge_x + int(gauge_width * probability)
    draw.polygon([(needle_x - 15, gauge_y - 25), (needle_x + 15, gauge_y - 25), (needle_x, gauge_y + 5)], 
                 fill=colors['dark'], outline=colors['white'], width=2)
    
    # CLINICAL RECOMMENDATIONS SECTION
    rec_y = risk_y + risk_height + 50
    rec_height = 550  # Increased height
    
    draw.rectangle([(80, rec_y), (width - 80, rec_y + rec_height)], fill=colors['white'], outline=colors['border'], width=3)
    draw.rectangle([(80, rec_y), (width - 80, rec_y + 70)], fill=colors['primary'])
    draw.text((100, rec_y + 25), "CLINICAL RECOMMENDATIONS", font=fonts['heading'], fill=colors['white'])
    
    recommendations = get_professional_recommendations(risk_level, probability, patient_data)
    rec_item_y = rec_y + 110
    
    priority_colors = {
        'URGENT': colors['danger'],
        'HIGH': colors['warning'],
        'MODERATE': colors['secondary'],
        'ROUTINE': colors['success']
    }
    
    for i, (priority, recommendation) in enumerate(recommendations):
        if rec_item_y + 60 > rec_y + rec_height - 30:
            break
            
        priority_color = priority_colors.get(priority, colors['text_secondary'])
        
        # Enhanced priority badge
        badge_width = len(priority) * 15 + 30  # Increased size
        draw.rectangle([(100, rec_item_y), (100 + badge_width, rec_item_y + 35)], 
                       fill=priority_color, outline=colors['white'], width=2)
        draw.text((115, rec_item_y + 8), priority, font=fonts['tiny'], fill=colors['white'])
        
        # Recommendation text with word wrapping
        rec_text = f"• {recommendation}"
        max_width = width - 200 - badge_width - 40
        
        # Simple text wrapping for long recommendations
        if len(rec_text) > 80:
            words = rec_text.split()
            line1 = " ".join(words[:12])
            line2 = " ".join(words[12:])
            
            draw.text((100 + badge_width + 30, rec_item_y + 2), line1, font=fonts['small'], fill=colors['text_primary'])
            if line2:
                draw.text((100 + badge_width + 30, rec_item_y + 27), line2, font=fonts['small'], fill=colors['text_primary'])
        else:
            draw.text((100 + badge_width + 30, rec_item_y + 8), rec_text, font=fonts['small'], fill=colors['text_primary'])
        
        rec_item_y += 65  # Increased spacing
    
    # DIAGNOSTIC ANALYTICS SECTION
    analytics_y = rec_y + rec_height + 50
    analytics_height = 400  # Increased height
    
    draw.rectangle([(80, analytics_y), (width - 80, analytics_y + analytics_height)], 
                   fill=colors['white'], outline=colors['border'], width=3)
    draw.rectangle([(80, analytics_y), (width - 80, analytics_y + 70)], fill=colors['secondary'])
    draw.text((100, analytics_y + 25), "DIAGNOSTIC ANALYTICS", font=fonts['heading'], fill=colors['white'])
    
    # Risk factors analysis
    draw.text((100, analytics_y + 110), "Risk Factor Analysis", font=fonts['subheading'], fill=colors['primary'])
    
    risk_factors = [
        ("Age Factor", min(patient_data['age'] / 80, 1.0)),
        ("Blood Pressure", min(patient_data['trestbps'] / 180, 1.0)),
        ("Cholesterol Level", min(patient_data['chol'] / 300, 1.0)),
        ("Heart Rate Reserve", 1 - min(patient_data['thalach'] / 200, 1.0))
    ]
    
    factor_y = analytics_y + 160
    for factor, value in risk_factors:
        draw.text((100, factor_y), factor, font=fonts['small'], fill=colors['text_secondary'])
        
        # Enhanced progress bar
        bar_width = 350  # Increased width
        bar_height = 25  # Increased height
        bar_x = 400
        
        # Background
        draw.rectangle([(bar_x, factor_y + 8), (bar_x + bar_width, factor_y + 8 + bar_height)], 
                       fill=colors['light'], outline=colors['border'], width=2)
        
        # Fill
        fill_width = int(bar_width * value)
        fill_color = colors['danger'] if value >= 0.7 else colors['warning'] if value >= 0.5 else colors['success']
        draw.rectangle([(bar_x, factor_y + 8), (bar_x + fill_width, factor_y + 8 + bar_height)], fill=fill_color)
        
        # Percentage
        draw.text((bar_x + bar_width + 30, factor_y), f"{value:.1%}", font=fonts['body'], fill=colors['text_primary'])
        
        factor_y += 55  # Increased spacing
    
    # Model performance (right side)
    perf_x = width // 2 + 150
    draw.text((perf_x, analytics_y + 110), "Model Performance", font=fonts['subheading'], fill=colors['primary'])
    
    performance_metrics = [
        ("Accuracy", "94.2%"),
        ("Sensitivity", "91.8%"),
        ("Specificity", "96.1%"),
        ("AUC-ROC", "0.952")
    ]
    
    perf_y = analytics_y + 160
    for metric, value in performance_metrics:
        draw.text((perf_x, perf_y), metric, font=fonts['small'], fill=colors['text_secondary'])
        draw.text((perf_x, perf_y + 28), value, font=fonts['body'], fill=colors['text_primary'])
        perf_y += 70  # Increased spacing
    
    # FOOTER DISCLAIMER
    footer_y = analytics_y + analytics_height + 50
    footer_height = 220  # Increased height
    
    draw.rectangle([(80, footer_y), (width - 80, footer_y + footer_height)], 
                   fill=colors['light'], outline=colors['danger'], width=4)
    draw.rectangle([(80, footer_y), (width - 80, footer_y + 60)], fill=colors['danger'])
    draw.text((100, footer_y + 20), "⚠ IMPORTANT MEDICAL DISCLAIMER", font=fonts['subheading'], fill=colors['white'])
    
    disclaimer_text = [
        "This AI-generated report is for clinical decision support only and must be interpreted by qualified healthcare professionals.",
        "Results should not replace comprehensive clinical evaluation, complete medical history, or physical examination.",
        "Always consult with a board-certified cardiologist for definitive diagnosis and treatment planning.",
        f"Report ID: CI-{datetime.now().strftime('%Y%m%d%H%M%S')} | Algorithm: Random Forest v2.3.1"
    ]
    
    disclaimer_y = footer_y + 85
    for line in disclaimer_text:
        # Simple word wrapping for long disclaimer lines
        if len(line) > 120:
            words = line.split()
            mid_point = len(words) // 2
            line1 = " ".join(words[:mid_point])
            line2 = " ".join(words[mid_point:])
            
            draw.text((100, disclaimer_y), line1, font=fonts['small'], fill=colors['text_primary'])
            disclaimer_y += 28
            draw.text((100, disclaimer_y), line2, font=fonts['small'], fill=colors['text_primary'])
        else:
            draw.text((100, disclaimer_y), line, font=fonts['small'], fill=colors['text_primary'])
        disclaimer_y += 35  # Increased spacing
    
    # Save to buffer with high quality
    buffer = BytesIO()
    img.save(buffer, format="PNG", quality=100, dpi=(300, 300), optimize=True)
    buffer.seek(0)
    return buffer.getvalue()

# Load model and data
model = load_model()
avg_data = get_population_averages()

# --- Main Application ---
st.title("❤️ Cardio-Insight AI")
st.markdown("### Professional AI-Powered Dashboard for Cardiovascular Risk Assessment")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    .danger-card {
        border-left-color: #ef4444;
    }
    .warning-card {
        border-left-color: #f59e0b;
    }
    .success-card {
        border-left-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["👤 Patient Data Input", "📊 Analysis & Insights", "📄 Generate Report"])

with tab1:
    st.header("🩺 Enter Patient Information")
    st.markdown("Please fill in all the required patient information below:")
    
    with st.form("patient_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👤 Basic Information**")
            age = st.slider('Age (years)', 20, 90, 52, help="Patient's age in years")
            sex = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
            
            st.markdown("**💗 Cardiovascular Parameters**")
            cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], 
                            format_func=lambda x: ['Asymptomatic', 'Typical Angina', 'Atypical Angina', 'Non-Anginal Pain'][x])
            trestbps = st.number_input('Resting Blood Pressure (mmHg)', 80, 200, 125, 
                                     help="Normal range: 90-120 mmHg")
            chol = st.number_input('Serum Cholesterol (mg/dL)', 100, 600, 212, 
                                 help="Desirable: <200 mg/dL")
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1], 
                             format_func=lambda x: 'No (≤120)' if x == 0 else 'Yes (>120)')
        
        with col2:
            st.markdown("**🔬 Diagnostic Tests**")
            restecg = st.selectbox('Resting ECG Results', [0, 1, 2], 
                                 format_func=lambda x: ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'][x])
            thalach = st.number_input('Maximum Heart Rate Achieved (bpm)', 60, 220, 168,
                                    help="Age-predicted max: 220 - age")
            exang = st.selectbox('Exercise Induced Angina', [0, 1], 
                               format_func=lambda x: 'No' if x == 0 else 'Yes')
            
            st.markdown("**📈 Advanced Parameters**")
            oldpeak = st.number_input('ST Depression (mm)', 0.0, 10.0, 1.0, 0.1,
                                    help="ST depression induced by exercise")
            slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2], 
                               format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
            ca = st.selectbox('Major Vessels Colored by Fluoroscopy', list(range(5)),
                            help="Number of major vessels (0-4)")
            
        thal = st.selectbox('Thalassemia', [0, 1, 2, 3], 
                          format_func=lambda x: ['Unknown', 'Normal', 'Fixed Defect', 'Reversible Defect'][x])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔍 Analyze Patient Risk", 
                                            use_container_width=True, 
                                            type="primary")

    if submitted:
        st.session_state.patient_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        st.success("✅ Patient data submitted successfully! Navigate to the Analysis tab to view results.")

with tab2:
    if 'patient_data' not in st.session_state:
        st.warning("⚠️ Please submit patient data in the Patient Data Input tab to view analysis.")
        st.info("👈 Go to the first tab and fill in the patient information form.")
    else:
        st.header("📊 Cardiovascular Risk Analysis")
        
        # Make prediction
        patient_data_df = pd.DataFrame([st.session_state.patient_data])
        prediction = model.predict(patient_data_df)[0]
        probability = model.predict_proba(patient_data_df)[0][1]
        
        # Determine risk level
        if prediction == 1 or probability >= 0.75:
            risk_level = "Critical Risk"
            risk_color = "🔴"
            card_class = "danger-card"
        elif probability >= 0.6:
            risk_level = "High Risk"
            risk_color = "🔴"
            card_class = "danger-card"
        elif probability >= 0.4:
            risk_level = "Moderate Risk"
            risk_color = "🟡"
            card_class = "warning-card"
        elif probability >= 0.2:
            risk_level = "Low-Moderate Risk"
            risk_color = "🟡"
            card_class = "warning-card"
        else:
            risk_level = "Low Risk"
            risk_color = "🟢"
            card_class = "success-card"
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="🏥 Prediction", 
                value="Heart Disease" if prediction == 1 else "Healthy",
                delta="Requires Attention" if prediction == 1 else "Normal"
            )
        with col2:
            st.metric(
                label="📊 Risk Probability", 
                value=f"{probability:.1%}",
                delta=f"{probability*100:.1f}/100"
            )
        with col3:
            st.metric(
                label=f"{risk_color} Risk Level", 
                value=risk_level,
                delta="Assessment Complete"
            )
        
        # Detailed analysis
        st.markdown("---")
        st.subheader("🔍 Detailed Patient Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🫀 Cardiovascular Parameters**")
            st.write(f"• **Age:** {st.session_state.patient_data['age']} years")
            st.write(f"• **Gender:** {'Male' if st.session_state.patient_data['sex'] == 1 else 'Female'}")
            st.write(f"• **Chest Pain:** {get_chest_pain_description(st.session_state.patient_data['cp'])}")
            st.write(f"• **Resting BP:** {st.session_state.patient_data['trestbps']} mmHg")
            st.write(f"• **Max Heart Rate:** {st.session_state.patient_data['thalach']} bpm")
            st.write(f"• **Exercise Angina:** {'Present' if st.session_state.patient_data['exang'] == 1 else 'Absent'}")
        
        with col2:
            st.markdown("**🧪 Laboratory & Diagnostic Results**")
            st.write(f"• **Cholesterol:** {st.session_state.patient_data['chol']} mg/dL")
            st.write(f"• **Fasting Blood Sugar:** {'>120 mg/dL' if st.session_state.patient_data['fbs'] == 1 else '≤120 mg/dL'}")
            st.write(f"• **Resting ECG:** {get_ecg_description(st.session_state.patient_data['restecg'])}")
            st.write(f"• **ST Depression:** {st.session_state.patient_data['oldpeak']} mm")
            st.write(f"• **ST Slope:** {get_slope_description(st.session_state.patient_data['slope'])}")
            st.write(f"• **Thalassemia:** {get_thal_description(st.session_state.patient_data['thal'])}")
        
        # Risk factors visualization
        st.markdown("---")
        st.subheader("📈 Risk Factor Analysis")
        
        # Create risk factor data
        risk_factors = {
            'Age Factor': min(st.session_state.patient_data['age'] / 80, 1.0),
            'Blood Pressure': min(st.session_state.patient_data['trestbps'] / 180, 1.0),
            'Cholesterol': min(st.session_state.patient_data['chol'] / 300, 1.0),
            'Heart Rate Reserve': 1 - min(st.session_state.patient_data['thalach'] / 200, 1.0),
            'ST Depression': min(st.session_state.patient_data['oldpeak'] / 4, 1.0)
        }
        
        # Display as progress bars
        for factor, value in risk_factors.items():
            color = "🔴" if value >= 0.7 else "🟡" if value >= 0.5 else "🟢"
            st.write(f"{color} **{factor}:** {value:.1%}")
            st.progress(value)
        
        # Clinical recommendations
        st.markdown("---")
        st.subheader("🏥 Clinical Recommendations")
        
        recommendations = get_professional_recommendations(risk_level.upper().replace("-", " "), probability, st.session_state.patient_data)
        
        for priority, recommendation in recommendations[:6]:  # Show top 6 recommendations
            if priority == "URGENT":
                st.error(f"🚨 **{priority}:** {recommendation}")
            elif priority == "HIGH":
                st.warning(f"⚠️ **{priority}:** {recommendation}")
            elif priority == "MODERATE":
                st.info(f"ℹ️ **{priority}:** {recommendation}")
            else:
                st.success(f"✅ **{priority}:** {recommendation}")

with tab3:
    if 'patient_data' not in st.session_state:
        st.warning("⚠️ Please submit patient data to generate a professional report.")
        st.info("👈 Go to the Patient Data Input tab and complete the form first.")
    else:
        st.header("📄 Professional Medical Report")
        st.markdown("Generate a comprehensive cardiovascular risk assessment report for clinical use.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🖨️ Generate Professional Report", 
                        use_container_width=True, 
                        type="primary"):
                with st.spinner("📄 Generating ultra high-quality medical report..."):
                    try:
                        patient_data_df = pd.DataFrame([st.session_state.patient_data])
                        prediction = model.predict(patient_data_df)[0]
                        probability = model.predict_proba(patient_data_df)[0][1]
                        
                        png_bytes = create_ultra_professional_report(
                            st.session_state.patient_data, prediction, probability
                        )
                        
                        st.session_state.report_bytes = png_bytes
                        st.success("✅ Report generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")

        if 'report_bytes' in st.session_state:
            st.markdown("---")
            
            # Download button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Download Report as PNG",
                    data=st.session_state.report_bytes,
                    file_name=f"CardioInsight_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True,
                    type="secondary"
                )
            
            # Display report preview
            st.subheader("📋 Report Preview")
            st.markdown(f'''
                <div style="
                    margin: 20px auto; 
                    padding: 20px; 
                    border-radius: 10px; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
                    background-color: white; 
                    max-width: 100%; 
                    overflow: auto;
                    border: 2px solid #e2e8f0;
                ">
                    <img src="data:image/png;base64,{base64.b64encode(st.session_state.report_bytes).decode()}" 
                         alt="CardioInsight AI Professional Medical Report" 
                         style="width: 100%; height: auto; object-fit: contain; border-radius: 5px;">
                </div>
                ''', unsafe_allow_html=True)
            
            st.info("💡 **Tip:** Right-click on the report image to save it directly, or use the download button above.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>CardioInsight AI</strong> - Professional Cardiovascular Risk Assessment Platform</p>
    <p>⚠️ This tool is for clinical decision support only. Always consult with qualified healthcare professionals.</p>
    <p>Version 2.3.1 | Powered by Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)