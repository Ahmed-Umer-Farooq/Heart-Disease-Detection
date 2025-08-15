# utils/report.py - CLEAN PROFESSIONAL VERSION

import io
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import textwrap

def create_lab_report_png(patient_data, prediction, probability, radar_chart_img, shap_chart_img, logo_path=None):
    """
    Creates a clean, professional medical report with proper spacing and no overlaps.
    """
    
    # Create figure with proper A4 dimensions and high DPI
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    fig.patch.set_facecolor('white')
    
    # Professional color palette
    colors = {
        'primary': '#1e40af',      # Deep blue
        'secondary': '#64748b',    # Slate gray
        'success': '#059669',      # Green
        'warning': '#d97706',      # Orange
        'danger': '#dc2626',       # Red
        'light': '#f8fafc',        # Very light gray
        'border': '#e2e8f0',       # Light border
        'text': '#1f2937'          # Dark text
    }
    
    # Clear any existing plots
    plt.clf()
    
    # Create main layout with no overlapping elements
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)  # Increased height for better spacing
    ax.axis('off')
    
    # 1. HEADER SECTION (Top)
    draw_header(ax, colors)
    
    # 2. PATIENT INFO AND RISK ASSESSMENT (Side by side)
    draw_patient_info(ax, patient_data, colors, y_start=125)
    draw_risk_assessment(ax, prediction, probability, colors, y_start=125)
    
    # 3. CHARTS SECTION (Reserved space, no overlaps)
    chart_success = draw_charts_section(ax, radar_chart_img, shap_chart_img, colors, y_start=80)
    
    # 4. RECOMMENDATIONS SECTION
    draw_recommendations(ax, prediction, probability, colors, y_start=35)
    
    # 5. FOOTER/DISCLAIMER
    draw_footer(ax, colors)
    
    # Save with high quality
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='PNG', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    buffer.seek(0)
    plt.close(fig)  # Clean up memory
    
    return buffer.getvalue()


def draw_header(ax, colors):
    """Draw clean header section"""
    # Header background
    header_bg = Rectangle((2, 140), 96, 8, facecolor=colors['primary'], alpha=0.1, 
                         edgecolor=colors['primary'], linewidth=2)
    ax.add_patch(header_bg)
    
    # Title
    ax.text(4, 145, '❤️ CardioInsight AI', fontsize=16, fontweight='bold', 
            color=colors['primary'], va='center')
    ax.text(4, 142, 'Professional Cardiovascular Risk Assessment Report', fontsize=11, 
            color=colors['text'], va='center')
    ax.text(96, 142, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
            fontsize=9, color=colors['secondary'], va='center', ha='right')
    
    # Divider line
    ax.plot([2, 98], [139, 139], color=colors['primary'], linewidth=2)


def draw_patient_info(ax, patient_data, colors, y_start=125):
    """Draw patient information section with clean formatting"""
    # Background
    patient_bg = Rectangle((2, y_start-20), 47, 18, facecolor='white', 
                          edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(patient_bg)
    
    # Title
    ax.text(4, y_start-2, 'Patient Information', fontsize=12, fontweight='bold', 
            color=colors['primary'])
    
    # Patient data with proper formatting
    feature_mapping = {
        'age': ('Age', 'years'),
        'sex': ('Gender', 'Male' if patient_data.get('sex', 0) == 1 else 'Female'),
        'cp': ('Chest Pain', get_chest_pain_type(patient_data.get('cp', 0))),
        'trestbps': ('Resting BP', 'mmHg'),
        'chol': ('Cholesterol', 'mg/dL'),
        'thalach': ('Max HR', 'bpm'),
        'oldpeak': ('ST Depression', 'mm'),
        'ca': ('Major Vessels', ''),
    }
    
    y_pos = y_start - 5
    for i, (key, (label, unit)) in enumerate(feature_mapping.items()):
        if key in patient_data and i < 7:  # Limit to prevent overflow
            value = patient_data[key]
            if key == 'sex':
                display_text = f"{label}: {unit}"
            else:
                display_text = f"{label}: {value} {unit}".strip()
            
            ax.text(4, y_pos - (i * 2.2), display_text, fontsize=9, 
                   color=colors['text'], va='center')


def draw_risk_assessment(ax, prediction, probability, colors, y_start=125):
    """Draw risk assessment section"""
    risk_level, risk_color = get_risk_level_and_color(prediction, probability, colors)
    
    # Background with risk color border
    risk_bg = Rectangle((51, y_start-20), 47, 18, facecolor=colors['light'], 
                       edgecolor=risk_color, linewidth=2.5)
    ax.add_patch(risk_bg)
    
    # Title
    ax.text(53, y_start-2, 'Risk Assessment', fontsize=12, fontweight='bold', 
            color=colors['primary'])
    
    # Risk level - large and prominent
    ax.text(53, y_start-6, f'Risk Level: {risk_level}', fontsize=11, fontweight='bold', 
            color=risk_color)
    ax.text(53, y_start-9, f'Probability: {probability:.1%}', fontsize=10, 
            color=colors['text'])
    
    # Risk score bar - clean design
    bar_x, bar_y = 53, y_start-13
    bar_width, bar_height = 40, 1.5
    
    # Background bar
    bar_bg = Rectangle((bar_x, bar_y), bar_width, bar_height, facecolor=colors['border'])
    ax.add_patch(bar_bg)
    
    # Fill bar
    bar_fill_width = bar_width * probability
    bar_fill = Rectangle((bar_x, bar_y), bar_fill_width, bar_height, facecolor=risk_color)
    ax.add_patch(bar_fill)
    
    # Score text
    ax.text(53, y_start-16, f'Risk Score: {probability*100:.1f}/100', fontsize=9, 
            color=colors['text'])


def draw_charts_section(ax, radar_chart_img, shap_chart_img, colors, y_start=80):
    """Draw charts section with proper spacing and no overlaps"""
    # Section title
    ax.text(4, y_start+2, 'Analysis Charts', fontsize=12, fontweight='bold', 
            color=colors['primary'])
    
    # Chart placeholders with proper positioning
    chart_height = 25
    chart_width = 47
    
    # Left chart area
    chart1_bg = Rectangle((2, y_start-chart_height), chart_width, chart_height, 
                         facecolor=colors['light'], edgecolor=colors['border'], linewidth=1)
    ax.add_patch(chart1_bg)
    
    if radar_chart_img:
        ax.text(25, y_start-2, 'Patient Profile vs Population Average', 
               ha='center', fontsize=10, fontweight='bold', color=colors['primary'])
        ax.text(25, y_start-chart_height/2, 'Radar Chart Available\n(View in interactive mode)', 
               ha='center', va='center', fontsize=9, color=colors['secondary'])
    else:
        ax.text(25, y_start-chart_height/2, 'Radar Chart\nNot Available', 
               ha='center', va='center', fontsize=10, color=colors['secondary'])
    
    # Right chart area
    chart2_bg = Rectangle((51, y_start-chart_height), chart_width, chart_height, 
                         facecolor=colors['light'], edgecolor=colors['border'], linewidth=1)
    ax.add_patch(chart2_bg)
    
    if shap_chart_img:
        ax.text(74, y_start-2, 'Feature Importance Analysis', 
               ha='center', fontsize=10, fontweight='bold', color=colors['primary'])
        ax.text(74, y_start-chart_height/2, 'SHAP Analysis Available\n(View in interactive mode)', 
               ha='center', va='center', fontsize=9, color=colors['secondary'])
    else:
        ax.text(74, y_start-chart_height/2, 'SHAP Analysis\nNot Available', 
               ha='center', va='center', fontsize=10, color=colors['secondary'])
    
    return True


def draw_recommendations(ax, prediction, probability, colors, y_start=35):
    """Draw recommendations section with proper text wrapping"""
    risk_level, _ = get_risk_level_and_color(prediction, probability, colors)
    
    # Background
    rec_bg = Rectangle((2, y_start-25), 96, 23, facecolor='white', 
                      edgecolor=colors['primary'], linewidth=1.5)
    ax.add_patch(rec_bg)
    
    # Title
    ax.text(4, y_start-2, 'Clinical Recommendations', fontsize=12, fontweight='bold', 
            color=colors['primary'])
    
    # Get recommendations
    recommendations = get_recommendations(risk_level, probability)
    
    # Display recommendations with proper spacing
    for i, rec in enumerate(recommendations[:5]):  # Limit to 5 for space
        # Wrap long text
        wrapped_text = textwrap.fill(rec, width=80)
        lines = wrapped_text.split('\n')
        
        y_pos = y_start - 6 - (i * 4)
        if y_pos > y_start - 22:  # Ensure we don't exceed bounds
            ax.text(5, y_pos, f'• {lines[0]}', fontsize=9, color=colors['text'])
            # Handle wrapped lines
            for j, line in enumerate(lines[1:], 1):
                if y_pos - (j * 1.2) > y_start - 22:
                    ax.text(7, y_pos - (j * 1.2), line, fontsize=9, color=colors['text'])


def draw_footer(ax, colors):
    """Draw clean footer with disclaimer"""
    # Disclaimer box
    disclaimer_text = "⚠️ DISCLAIMER: This AI assessment is for informational purposes only. Always consult healthcare professionals."
    
    ax.text(50, 3, disclaimer_text, ha='center', va='center', fontsize=8, 
            color=colors['danger'], style='italic', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['light'], 
                     edgecolor=colors['danger'], alpha=0.8))


def get_chest_pain_type(cp_value):
    """Convert chest pain type number to readable string"""
    cp_types = {
        0: 'Typical Angina',
        1: 'Atypical Angina', 
        2: 'Non-anginal Pain',
        3: 'Asymptomatic'
    }
    return cp_types.get(cp_value, 'Unknown')


def get_risk_level_and_color(prediction, probability, colors):
    """Determine risk level and corresponding color."""
    if prediction == 1 or probability >= 0.7:
        return "HIGH RISK", colors['danger']
    elif probability >= 0.4:
        return "MODERATE RISK", colors['warning']
    else:
        return "LOW RISK", colors['success']


def get_recommendations(risk_level, probability):
    """Generate appropriate clinical recommendations."""
    base_recommendations = {
        "HIGH RISK": [
            "Immediate cardiology consultation recommended within 1-2 weeks",
            "Comprehensive cardiac evaluation including stress testing and imaging",
            "Aggressive lifestyle modifications: diet, exercise, smoking cessation",
            "Regular BP and cholesterol monitoring (monthly initially)",
            "Discuss preventive medications with healthcare provider",
            "Consider cardiac rehabilitation program enrollment"
        ],
        "MODERATE RISK": [
            "Follow-up with healthcare provider within 3-6 months",
            "Implement structured lifestyle modifications program",
            "Monitor cardiovascular risk factors every 3-6 months",
            "Consider additional cardiac screening if symptoms develop",
            "Maintain healthy weight through diet and exercise plan",
            "Stress management and regular sleep schedule"
        ],
        "LOW RISK": [
            "Continue current healthy lifestyle practices",
            "Annual comprehensive check-ups with healthcare provider",
            "Maintain balanced diet and regular exercise routine",
            "Monitor blood pressure and cholesterol annually",
            "Stay aware of cardiovascular warning symptoms",
            "Maintain healthy weight and avoid smoking"
        ]
    }
    
    return base_recommendations.get(risk_level, base_recommendations["MODERATE RISK"])


# Backup function using PIL for better reliability
def create_clean_pil_report(patient_data, prediction, probability, radar_chart_img=None, shap_chart_img=None):
    """
    Clean PIL-based report generator as backup - more reliable than matplotlib
    """
    
    # High-resolution canvas
    width, height = 2100, 2970  # A4 at 250 DPI
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Colors
    primary = (30, 64, 175)    # Deep blue
    text_dark = (31, 41, 55)   # Dark gray
    text_light = (107, 114, 128) # Light gray
    success = (5, 150, 105)    # Green
    warning = (217, 119, 6)    # Orange  
    danger = (220, 38, 38)     # Red
    light_bg = (248, 250, 252) # Very light gray
    border = (229, 231, 235)   # Light border
    
    # Font sizes (larger for high resolution)
    try:
        title_font = ImageFont.truetype("arial.ttf", 48)
        header_font = ImageFont.truetype("arial.ttf", 36) 
        body_font = ImageFont.truetype("arial.ttf", 28)
        small_font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Use default font if arial not available
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default() 
        small_font = ImageFont.load_default()
    
    # Layout parameters
    margin = 80
    col_width = (width - 3*margin) // 2
    
    # 1. HEADER
    header_height = 200
    draw.rectangle([(margin, margin), (width-margin, margin+header_height)], 
                  fill=light_bg, outline=primary, width=3)
    
    draw.text((margin+30, margin+30), "❤️ CardioInsight AI", font=title_font, fill=primary)
    draw.text((margin+30, margin+90), "Professional Cardiovascular Risk Assessment", 
              font=header_font, fill=text_dark)
    draw.text((margin+30, margin+140), f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
              font=small_font, fill=text_light)
    
    # 2. CONTENT SECTIONS
    content_y = margin + header_height + 50
    
    # Patient Info (Left)
    section_height = 400
    draw.rectangle([(margin, content_y), (margin+col_width, content_y+section_height)], 
                  fill='white', outline=border, width=2)
    draw.text((margin+30, content_y+30), "Patient Information", font=header_font, fill=primary)
    
    # Patient data
    y_pos = content_y + 100
    feature_data = [
        ('Age', f"{patient_data.get('age', 'N/A')} years"),
        ('Gender', 'Male' if patient_data.get('sex', 0) == 1 else 'Female'),
        ('Chest Pain Type', get_chest_pain_type(patient_data.get('cp', 0))),
        ('Resting BP', f"{patient_data.get('trestbps', 'N/A')} mmHg"),
        ('Cholesterol', f"{patient_data.get('chol', 'N/A')} mg/dL"),
        ('Max Heart Rate', f"{patient_data.get('thalach', 'N/A')} bpm"),
        ('ST Depression', f"{patient_data.get('oldpeak', 'N/A')} mm"),
    ]
    
    for label, value in feature_data[:7]:  # Limit to fit in space
        draw.text((margin+30, y_pos), f"{label}: {value}", font=body_font, fill=text_dark)
        y_pos += 45
    
    # Risk Assessment (Right)
    risk_x = margin + col_width + margin
    risk_level, risk_color = get_risk_level_color_pil(prediction, probability)
    
    draw.rectangle([(risk_x, content_y), (width-margin, content_y+section_height)], 
                  fill=light_bg, outline=risk_color, width=4)
    draw.text((risk_x+30, content_y+30), "Risk Assessment", font=header_font, fill=primary)
    
    draw.text((risk_x+30, content_y+100), f"Risk Level: {risk_level}", 
              font=header_font, fill=risk_color)
    draw.text((risk_x+30, content_y+150), f"Probability: {probability:.1%}", 
              font=body_font, fill=text_dark)
    
    # Risk bar
    bar_x, bar_y = risk_x+30, content_y+200
    bar_width, bar_height = 300, 30
    draw.rectangle([(bar_x, bar_y), (bar_x+bar_width, bar_y+bar_height)], 
                  fill='lightgray', outline='gray', width=2)
    fill_width = int(bar_width * probability)
    draw.rectangle([(bar_x, bar_y), (bar_x+fill_width, bar_y+bar_height)], fill=risk_color)
    
    draw.text((bar_x, bar_y+50), f"Risk Score: {probability*100:.1f}/100", 
              font=body_font, fill=text_dark)
    
    # 3. RECOMMENDATIONS
    rec_y = content_y + section_height + 80
    rec_height = 400
    draw.rectangle([(margin, rec_y), (width-margin, rec_y+rec_height)], 
                  fill='white', outline=primary, width=2)
    draw.text((margin+30, rec_y+30), "Clinical Recommendations", 
              font=header_font, fill=primary)
    
    recommendations = get_recommendations(risk_level, probability)
    y_pos = rec_y + 100
    for i, rec in enumerate(recommendations[:5]):
        # Wrap text properly
        wrapped = textwrap.fill(rec, width=70)
        lines = wrapped.split('\n')
        draw.text((margin+50, y_pos), f"• {lines[0]}", font=body_font, fill=text_dark)
        for j, line in enumerate(lines[1:], 1):
            draw.text((margin+70, y_pos + j*35), line, font=body_font, fill=text_dark)
        y_pos += len(lines) * 35 + 20
    
    # 4. FOOTER
    footer_y = height - 150
    draw.text((width//2, footer_y), 
              "⚠️ DISCLAIMER: This AI assessment is for informational purposes only.", 
              font=small_font, fill=danger, anchor="mm")
    draw.text((width//2, footer_y+40), 
              "Always consult with qualified healthcare professionals for medical decisions.", 
              font=small_font, fill=text_dark, anchor="mm")
    
    # Save to buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", quality=95, optimize=True)
    buffer.seek(0)
    return buffer.getvalue()


def get_risk_level_color_pil(prediction, probability):
    """Helper for PIL version"""
    danger = (220, 38, 38)
    warning = (217, 119, 6)  
    success = (5, 150, 105)
    
    if prediction == 1 or probability >= 0.7:
        return "HIGH RISK", danger
    elif probability >= 0.4:
        return "MODERATE RISK", warning
    else:
        return "LOW RISK", success