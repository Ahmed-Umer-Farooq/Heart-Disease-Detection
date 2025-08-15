{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f1aad6a3-5be3-4da9-add8-5af3ed8a1b20",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-14 16:43:53.972 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\PMLS\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-08-14 16:43:53.976 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# --- Load the Saved Model ---\n",
    "model = joblib.load('best_forest_model.pkl')\n",
    "\n",
    "# --- Page Title and Introduction ---\n",
    "st.title('Heart Disease Prediction App')\n",
    "st.write('This app uses a machine learning model to predict the likelihood of heart disease based on user input.')\n",
    "st.write('---')\n",
    "\n",
    "# --- Create Input Fields for All 13 Features ---\n",
    "st.header('Please enter patient details:')\n",
    "\n",
    "# Create columns for a cleaner layout\n",
    "col1, col2, col3 = st.columns(3)\n",
    "\n",
    "with col1:\n",
    "    age = st.number_input('Age', min_value=1, max_value=120, value=52)\n",
    "    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')\n",
    "    cp = st.selectbox('Chest Pain Type (CP)', [0, 1, 2, 3])\n",
    "    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=125)\n",
    "\n",
    "with col2:\n",
    "    chol = st.number_input('Serum Cholestoral (chol) in mg/dl', min_value=100, max_value=600, value=212)\n",
    "    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])\n",
    "    restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', [0, 1, 2])\n",
    "    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=168)\n",
    "\n",
    "with col3:\n",
    "    exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])\n",
    "    oldpeak = st.number_input('ST depression induced by exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)\n",
    "    slope = st.selectbox('Slope of the peak exercise ST segment', [0, 1, 2])\n",
    "    ca = st.selectbox('Number of major vessels colored by flourosopy (ca)', [0, 1, 2, 3, 4])\n",
    "    thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])\n",
    "\n",
    "\n",
    "# --- Create a Button to Trigger Prediction ---\n",
    "if st.button('Predict Heart Disease', key='predict_button'):\n",
    "    \n",
    "    input_data = pd.DataFrame([{\n",
    "        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,\n",
    "        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,\n",
    "        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal\n",
    "    }])\n",
    "\n",
    "    prediction = model.predict(input_data)\n",
    "    probability = model.predict_proba(input_data)\n",
    "\n",
    "    st.write('---')\n",
    "    st.header('Prediction Result')\n",
    "\n",
    "    if prediction[0] == 1:\n",
    "        st.error(f'The model predicts this person HAS heart disease (Probability: {probability[0][1]:.2%})')\n",
    "    else:\n",
    "        st.success(f'The model predicts this person DOES NOT have heart disease (Probability: {probability[0][0]:.2%})')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
