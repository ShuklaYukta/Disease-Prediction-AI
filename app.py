import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Custom CSS 
st.markdown("""
<style>
/* Medical hero background */
.main { 
    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 50%, #e8f5e8 100%);
    padding: 2rem;
}

/* Professional medical button */
.stButton > button {
    background: linear-gradient(45deg, #1976d2, #42a5f5);
    color: white; 
    border-radius: 12px;
    border: none;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(25,118,210,0.3);
    height: 50px;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(25,118,210,0.4);
}

/* Sidebar polish */
.section-header {
    color: #1976d2 !important;
    font-weight: bold;
    font-size: 1.2rem;
}

/* Clean symptom display */
.success { 
    background: linear-gradient(90deg, #d4edda, #c8e6c9);
    border-radius: 8px; 
    padding: 8px 12px;
    margin: 4px 0;
    border-left: 4px solid #4caf50;
}

/* Info boxes */
.info-box { 
    background: linear-gradient(90deg, #e3f2fd, #bbdefb);
    border-radius: 10px; 
    padding: 12px;
    border-left: 4px solid #2196f3;
}

/* Title polish */
h1 { 
    color: #1e3a8a !important; 
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4caf50, #81c784);
}
</style>
""", unsafe_allow_html=True)

# DISEASE DATABASE 
DISEASE_INFO = {
    "Fungal infection": {
        "desc": "Fungal infections affect skin, hair, or nails caused by fungi.",
        "precautions": "• Keep skin dry and clean\n• Use antifungal creams\n• Avoid sharing towels\n• Wear loose cotton clothes"
    },
    "Chicken pox": {
        "desc": "Viral infection causing itchy rash and blisters all over body.",
        "precautions": "• Calamine lotion for itching\n• Oatmeal baths\n• Stay hydrated\n• Rest and isolation"
    },
    "Impetigo": {
        "desc": "Contagious bacterial skin infection causing red sores.",
        "precautions": "• Keep sores clean and covered\n• Antibiotic cream\n• Wash hands frequently\n• Don't touch sores"
    },
    "Peptic ulcer disease": {
        "desc": "Stomach ulcers causing burning stomach pain after eating.",
        "precautions": "• Take antacids as prescribed\n• Avoid spicy/acidic foods\n• Small frequent meals\n• No smoking/alcohol"
    },
    "Pneumonia": {
        "desc": "Lung infection causing cough, fever, and breathing difficulty.",
        "precautions": "• Rest completely\n• Stay hydrated\n• Steam inhalation\n• Elevate head while sleeping"
    },
    "Cervical spondylosis": {
        "desc": "Age-related neck pain and stiffness from wear and tear.",
        "precautions": "• Maintain good posture\n• Neck exercises daily\n• Hot/cold therapy\n• Avoid heavy lifting"
    },
    "Common Cold": {
        "desc": "Viral upper respiratory infection with runny nose.",
        "precautions": "• Rest and fluids\n• Saline gargle\n• Steam inhalation\n• Vitamin C"
    },
    "Malaria": {
        "desc": "Mosquito-borne parasitic infection with cyclic fever.",
        "precautions": "• Mosquito repellent\n• Bed nets\n• Full course antimalarials\n• Hydration"
    },
    "Dengue": {
        "desc": "Mosquito-borne viral fever with severe joint pain.",
        "precautions": "• Oral rehydration\n• Paracetamol only\n• Bed rest\n• Monitor platelets"
    },
    "Jaundice": {
        "desc": "Liver dysfunction causing yellow skin and eyes.",
        "precautions": "• Avoid alcohol\n• Light diet\n• Complete bed rest\n• Hydration"
    },
    "Bronchial Asthma": {
        "desc": "Chronic airway inflammation causing wheezing.",
        "precautions": "• Use inhaler as prescribed\n• Avoid triggers\n• Breathing exercises\n• Regular checkups"
    },
    "Hypertension ": {
        "desc": "High blood pressure damaging arteries over time.",
        "precautions": "• Low salt diet\n• Regular exercise\n• Stress management\n• Regular BP check"
    },
    "Osteoarthrosis": {
        "desc": "Joint cartilage breakdown causing pain and stiffness.",
        "precautions": "• Weight control\n• Low impact exercise\n• Hot/cold packs\n• Joint support"
    },
    "Diabetes ": {
        "desc": "High blood sugar due to insulin deficiency.",
        "precautions": "• Regular blood sugar monitoring\n• Balanced diet\n• Exercise\n• Medication compliance"
    },
    "Allergy": {
        "desc": "Immune overreaction to harmless substances.",
        "precautions": "• Avoid known allergens\n• Antihistamines\n• Carry emergency medication\n• Allergy testing"
    }
}

# MODEL LOADING 
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    le = joblib.load('label_encoder.pkl')
    symptom_df = pd.read_csv('dataset/Training.csv')
    
    # ORIGINAL order (model expects this)
    model_cols = symptom_df.drop('prognosis', axis=1).columns.tolist()
    
    # Alphabetical for UI (sorted display)
    display_cols = sorted(model_cols)
    
    return model, le, model_cols, display_cols

model, le, model_cols, display_cols = load_model()

st.title("🏥 AI Disease Predictor")
st.markdown("**Professional medical diagnosis in seconds**")

# SINGLE alphabetical searchable multiselect 
st.sidebar.header("🔍 Select Symptoms")
selected_symptoms = st.sidebar.multiselect(
    "Type to search symptoms (max 10):",
    options=display_cols,  # Alphabetical!
    max_selections=10,
    placeholder="e.g., itching, fever, cough...",
    help="Start typing to filter. Select up to 10 symptoms."
)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Patient Symptoms")
    if selected_symptoms:
        for symptom in selected_symptoms:
            st.success(f"✅ {symptom.replace('_', ' ').title()}")
        st.info(f"**Total: {len(selected_symptoms)} symptoms**")
    else:
        st.info("👆 Type symptoms in sidebar")

with col2:
    st.subheader("🎯 AI Analysis")
    if st.button("🔬 Diagnose Now", type="primary", use_container_width=True):
        if len(selected_symptoms) == 0:
            st.error("❌ Please select at least 1 symptom!")
        else:
            my_bar = st.progress(0)
            
            #  Map display names → model positions
            patient_vector = np.zeros(len(model_cols))
            for symptom in selected_symptoms:
                model_idx = model_cols.index(symptom)  # Correct position!
                patient_vector[model_idx] = 1
            
            my_bar.progress(50)
            
            # Predict
            prediction = model.predict([patient_vector])[0]
            confidence = model.predict_proba([patient_vector]).max()
            disease = le.inverse_transform([prediction])[0]
            
            my_bar.progress(100)
            st.success("✅ Analysis Complete!")
            
            # Results
            st.markdown(f"### **🎖️ Diagnosis: {disease.title()}**")
            
            if confidence > 0.85:
                st.success(f"**Confidence: {confidence:.1%}** ✅")
            elif confidence > 0.70:
                st.warning(f"**Confidence: {confidence:.1%}** ⚠️")
            else:
                st.error(f"**Confidence: {confidence:.1%}** ❌ Limited data")
            
            st.info(f"**Symptoms analyzed:** {len(selected_symptoms)}")
            
            # Disease Information
            if disease in DISEASE_INFO:
                with st.expander("📖 Disease Information", expanded=True):
                    info = DISEASE_INFO[disease]
                    st.markdown(f"**About {disease}:**")
                    st.markdown(f"> {info['desc']}")
                    
                    st.markdown("**🛡️ Precautions:**")
                    st.markdown(info['precautions'])
            else:
                st.info("ℹ️ Detailed information for this disease coming soon...")

#  Statistics Section
with st.expander("📊 App Statistics"):
    st.metric("Total Symptoms Available", len(display_cols))
    st.metric("Model Predicts", "41 diseases")  # YOUR FULL CAPABILITY
    st.metric("Detailed Info", len(DISEASE_INFO))  # Only 15 have descriptions
    st.metric("Max Symptoms per Test", "10")


# Footer
st.markdown("---")
st.markdown("*For educational purposes. Always consult a medical professional.*")
