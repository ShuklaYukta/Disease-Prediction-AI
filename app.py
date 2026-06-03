"""
AI Healthcare Assistant - Dashboard UI v3.0
Redesigned to match modern healthcare dashboard style.

Run: streamlit run app.py
Requires: model.pkl + label_encoder.pkl (run train_model.py first)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings 
from datetime import datetime
import streamlit as st
import os
from openai import OpenAI

api_key = st.secrets.get("OPENAI_API_KEY", "")

if not api_key:
    st.error("❌ API key missing in Streamlit secrets")
    st.stop()

client = OpenAI(api_key=api_key)

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── TRANSLATIONS ────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        "app_title": "AI Healthcare Assistant",
        "dashboard": "Dashboard",
        "symptom_pred": "Symptom Prediction",
        "xray_analysis": "X-ray Analysis",
        "ai_chatbot": "AI Chatbot",
        "history": "History",
        "health_tips": "Health Tips",
        "about": "About",
        "welcome": "Welcome to your AI-based Disease Prediction System",
        "symptom_card_title": "Symptom-Based Prediction",
        "symptom_card_desc": "Enter your symptoms and get possible disease prediction using ML model.",
        "xray_card_title": "X-ray Analysis",
        "xray_card_desc": "Upload chest X-ray image and analyze for pneumonia using CNN model.",
        "chatbot_card_title": "AI Health Chatbot",
        "chatbot_card_desc": "Ask any health-related questions and get instant AI assistance.",
        "start_prediction": "Start Prediction →",
        "analyze_xray": "Analyze X-ray →",
        "start_chat": "Start Chat →",
        "quick_stats": "Quick Statistics",
        "recent_predictions": "Recent Predictions",
        "view_all": "View All History",
        "important_note": "Important Note",
        "disclaimer": "This application is for educational purposes only and does not replace professional medical advice. Consult a doctor for proper diagnosis.",
        "dark_mode": "Dark Mode",
        "type": "Type",
        "input": "Input",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "time": "Time",
        "select_symptoms": "Select Your Symptoms",
        "search_symptoms": "Search and select symptoms (max 15)...",
        "predict_btn": "🔬 Predict Disease",
        "clear_btn": "Clear All",
        "diagnosis": "Diagnosis",
        "about_disease": "About Disease",
        "precautions": "Precautions",
        "alternatives": "Alternative Possibilities",
        "specialist": "Recommended Specialist",
        "no_symptoms": "Please select at least 1 symptom",
        "high_conf": "High Confidence",
        "med_conf": "Medium Confidence",
        "low_conf": "Low Confidence",
        "chatbot_hello": "Hello! I'm your AI health assistant. How can I help you today?",
        "chatbot_placeholder": "Type your question here...",
        "send": "Send",
        "clear_chat": "Clear Chat",
        "quick_q": "Quick Questions:",
        "upload_xray": "Upload Chest X-ray Image",
        "xray_note": "Supported: JPG, PNG, JPEG",
        "analyze_btn": "🔍 Analyze X-ray",
        "health_tip_title": "Daily Health Tips",
        "no_history": "No predictions yet. Go to Symptom Prediction to get started!",
        "severity": "Severity",
    },
    "hi": {
        "app_title": "AI स्वास्थ्य सहायक",
        "dashboard": "डैशबोर्ड",
        "symptom_pred": "लक्षण भविष्यवाणी",
        "xray_analysis": "X-ray विश्लेषण",
        "ai_chatbot": "AI चैटबॉट",
        "history": "इतिहास",
        "health_tips": "स्वास्थ्य सुझाव",
        "about": "जानकारी",
        "welcome": "AI आधारित रोग भविष्यवाणी प्रणाली में आपका स्वागत है",
        "symptom_card_title": "लक्षण-आधारित भविष्यवाणी",
        "symptom_card_desc": "लक्षण दर्ज करें और ML मॉडल से रोग की भविष्यवाणी प्राप्त करें।",
        "xray_card_title": "X-ray विश्लेषण",
        "xray_card_desc": "छाती X-ray अपलोड करें और CNN मॉडल से निमोनिया का विश्लेषण करें।",
        "chatbot_card_title": "AI स्वास्थ्य चैटबॉट",
        "chatbot_card_desc": "स्वास्थ्य संबंधी प्रश्न पूछें और तुरंत AI सहायता प्राप्त करें।",
        "start_prediction": "भविष्यवाणी शुरू करें →",
        "analyze_xray": "X-ray विश्लेषण करें →",
        "start_chat": "चैट शुरू करें →",
        "quick_stats": "त्वरित आँकड़े",
        "recent_predictions": "हाल की भविष्यवाणियाँ",
        "view_all": "सभी इतिहास देखें",
        "important_note": "महत्वपूर्ण नोट",
        "disclaimer": "यह एप्लिकेशन केवल शैक्षिक उद्देश्यों के लिए है। सटीक निदान के लिए डॉक्टर से परामर्श लें।",
        "dark_mode": "डार्क मोड",
        "type": "प्रकार",
        "input": "इनपुट",
        "prediction": "भविष्यवाणी",
        "confidence": "विश्वसनीयता",
        "time": "समय",
        "select_symptoms": "अपने लक्षण चुनें",
        "search_symptoms": "लक्षण खोजें और चुनें (अधिकतम 15)...",
        "predict_btn": "🔬 रोग की भविष्यवाणी करें",
        "clear_btn": "सब साफ करें",
        "diagnosis": "निदान",
        "about_disease": "रोग के बारे में",
        "precautions": "सावधानियां",
        "alternatives": "अन्य संभावनाएं",
        "specialist": "अनुशंसित विशेषज्ञ",
        "no_symptoms": "कृपया कम से कम 1 लक्षण चुनें",
        "high_conf": "उच्च विश्वसनीयता",
        "med_conf": "मध्यम विश्वसनीयता",
        "low_conf": "कम विश्वसनीयता",
        "chatbot_hello": "नमस्ते! मैं आपका AI स्वास्थ्य सहायक हूं। आज मैं आपकी कैसे मदद कर सकता हूं?",
        "chatbot_placeholder": "यहाँ अपना प्रश्न टाइप करें...",
        "send": "भेजें",
        "clear_chat": "चैट साफ करें",
        "quick_q": "त्वरित प्रश्न:",
        "upload_xray": "छाती X-ray छवि अपलोड करें",
        "xray_note": "समर्थित: JPG, PNG, JPEG",
        "analyze_btn": "🔍 X-ray विश्लेषण करें",
        "health_tip_title": "दैनिक स्वास्थ्य सुझाव",
        "no_history": "अभी तक कोई भविष्यवाणी नहीं। शुरू करने के लिए लक्षण भविष्यवाणी पर जाएं!",
        "severity": "गंभीरता",
    },
}

# ─── DISEASE DATABASE ────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Fungal infection": {
        "desc_en": "A skin condition caused by fungi, affecting skin, hair, or nails.",
        "desc_hi": "कवक के कारण त्वचा, बाल या नाखूनों को प्रभावित करने वाली स्थिति।",
        "precautions_en": "• Keep skin dry and clean\n• Use antifungal creams\n• Avoid sharing towels\n• Wear loose cotton clothes",
        "precautions_hi": "• त्वचा को सूखा रखें\n• एंटीफंगल क्रीम लगाएं\n• तौलिया साझा न करें",
        "severity": "mild", "specialist": "Dermatologist", "specialist_hi": "त्वचा रोग विशेषज्ञ", "hindi_name": "कवक संक्रमण",
    },
    "Allergy": {
        "desc_en": "Immune system reaction to foreign substances affecting skin, airways, or digestion.",
        "desc_hi": "विदेशी पदार्थों के प्रति प्रतिरक्षा प्रणाली की प्रतिक्रिया।",
        "precautions_en": "• Avoid known allergens\n• Take antihistamines\n• Carry emergency medication\n• Keep environment clean",
        "precautions_hi": "• एलर्जन से बचें\n• एंटीहिस्टामाइन लें\n• आपातकालीन दवा रखें",
        "severity": "moderate", "specialist": "Allergist", "specialist_hi": "एलर्जी विशेषज्ञ", "hindi_name": "एलर्जी",
    },
    "GERD": {
        "desc_en": "Chronic acid reflux causing heartburn and regurgitation.",
        "desc_hi": "पुराना एसिड रिफ्लक्स जिससे सीने में जलन होती है।",
        "precautions_en": "• Eat smaller, frequent meals\n• Avoid spicy, fatty foods\n• Don't lie down after eating\n• Elevate head while sleeping",
        "precautions_hi": "• छोटे भोजन करें\n• मसालेदार खाना न खाएं\n• खाने के बाद न लेटें",
        "severity": "moderate", "specialist": "Gastroenterologist", "specialist_hi": "गैस्ट्रोएंटेरोलॉजिस्ट", "hindi_name": "एसिड रिफ्लक्स",
    },
    "Chronic cholestasis": {
        "desc_en": "Reduced bile flow from the liver causing jaundice and itching.",
        "desc_hi": "यकृत से पित्त प्रवाह में कमी।",
        "precautions_en": "• Follow low-fat diet\n• Avoid alcohol\n• Take prescribed medications\n• Regular liver tests",
        "precautions_hi": "• कम वसा वाला आहार\n• शराब से बचें\n• नियमित जिगर परीक्षण",
        "severity": "severe", "specialist": "Hepatologist", "specialist_hi": "यकृत रोग विशेषज्ञ", "hindi_name": "पित्त रुकावट",
    },
    "Drug Reaction": {
        "desc_en": "Adverse reaction to medication ranging from mild rash to severe anaphylaxis.",
        "desc_hi": "दवा के प्रति प्रतिकूल प्रतिक्रिया।",
        "precautions_en": "• Stop suspected drug immediately\n• Seek emergency care if severe\n• Document all drug allergies\n• Inform healthcare providers",
        "precautions_hi": "• संदिग्ध दवा बंद करें\n• गंभीर होने पर आपातकाल जाएं\n• सभी दवा एलर्जी दर्ज करें",
        "severity": "severe", "specialist": "Emergency/Allergist", "specialist_hi": "आपातकालीन विशेषज्ञ", "hindi_name": "दवा प्रतिक्रिया",
    },
    "Peptic ulcer disease": {
        "desc_en": "Open sores in the stomach lining or small intestine.",
        "desc_hi": "पेट की परत में खुले घाव।",
        "precautions_en": "• Take antacids as prescribed\n• Avoid spicy/acidic foods\n• Eat small frequent meals\n• Stop smoking and alcohol",
        "precautions_hi": "• एंटासिड लें\n• मसालेदार खाना न खाएं\n• धूम्रपान और शराब बंद करें",
        "severity": "moderate", "specialist": "Gastroenterologist", "specialist_hi": "गैस्ट्रोएंटेरोलॉजिस्ट", "hindi_name": "पेट का अल्सर",
    },
    "AIDS": {
        "desc_en": "Advanced stage of HIV infection affecting the immune system.",
        "desc_hi": "HIV संक्रमण की उन्नत अवस्था।",
        "precautions_en": "• Take ART medications regularly\n• Practice safe sex\n• Avoid sharing needles\n• Regular medical check-ups",
        "precautions_hi": "• ART दवाएं नियमित लें\n• सुरक्षित यौन संबंध रखें\n• सुइयां साझा न करें",
        "severity": "severe", "specialist": "Infectious Disease Specialist", "specialist_hi": "संक्रामक रोग विशेषज्ञ", "hindi_name": "एड्स",
    },
    "Diabetes": {
        "desc_en": "Metabolic disease causing high blood sugar due to insulin deficiency or resistance.",
        "desc_hi": "इंसुलिन की कमी से रक्त शर्करा में वृद्धि।",
        "precautions_en": "• Monitor blood sugar regularly\n• Follow low-sugar diet\n• Exercise 30 mins daily\n• Never skip medications",
        "precautions_hi": "• नियमित रक्त शर्करा जांचें\n• कम चीनी वाला आहार\n• रोजाना व्यायाम करें",
        "severity": "moderate", "specialist": "Endocrinologist", "specialist_hi": "एंडोक्रिनोलॉजिस्ट", "hindi_name": "मधुमेह",
    },
    "Gastroenteritis": {
        "desc_en": "Stomach and intestine inflammation causing vomiting and diarrhea.",
        "desc_hi": "पेट और आंतों की सूजन।",
        "precautions_en": "• Stay hydrated with ORS\n• Rest and avoid solid food\n• Avoid dairy products\n• Wash hands frequently",
        "precautions_hi": "• ORS से हाइड्रेटेड रहें\n• आराम करें\n• बार-बार हाथ धोएं",
        "severity": "moderate", "specialist": "Gastroenterologist", "specialist_hi": "गैस्ट्रोएंटेरोलॉजिस्ट", "hindi_name": "गैस्ट्रोएंटेराइटिस",
    },
    "Bronchial Asthma": {
        "desc_en": "Chronic airway inflammation causing wheezing and breathlessness.",
        "desc_hi": "वायुमार्ग की पुरानी सूजन।",
        "precautions_en": "• Use prescribed inhaler correctly\n• Avoid triggers\n• Practice breathing exercises\n• Keep rescue inhaler nearby",
        "precautions_hi": "• इनहेलर सही से उपयोग करें\n• ट्रिगर से बचें\n• श्वास व्यायाम करें",
        "severity": "moderate", "specialist": "Pulmonologist", "specialist_hi": "फेफड़े के रोग विशेषज्ञ", "hindi_name": "दमा",
    },
    "Hypertension ": {
        "desc_en": "High blood pressure that damages arteries and increases cardiovascular risk.",
        "desc_hi": "उच्च रक्तचाप जो धमनियों को नुकसान पहुंचाता है।",
        "precautions_en": "• Eat low-sodium diet\n• Exercise regularly\n• Manage stress\n• Monitor BP regularly\n• Take medications consistently",
        "precautions_hi": "• कम नमक वाला आहार\n• नियमित व्यायाम\n• BP नियमित जांचें",
        "severity": "moderate", "specialist": "Cardiologist", "specialist_hi": "हृदय रोग विशेषज्ञ", "hindi_name": "उच्च रक्तचाप",
    },
    "Migraine": {
        "desc_en": "Severe recurring headaches with nausea and sensitivity to light/sound.",
        "desc_hi": "गंभीर आवर्ती सिरदर्द।",
        "precautions_en": "• Identify and avoid triggers\n• Maintain regular sleep\n• Stay hydrated\n• Take medications at onset",
        "precautions_hi": "• ट्रिगर से बचें\n• नियमित नींद\n• हाइड्रेटेड रहें",
        "severity": "moderate", "specialist": "Neurologist", "specialist_hi": "न्यूरोलॉजिस्ट", "hindi_name": "माइग्रेन",
    },
    "Cervical spondylosis": {
        "desc_en": "Age-related wear of cervical spine causing neck pain and stiffness.",
        "desc_hi": "उम्र से संबंधित ग्रीवा रीढ़ की टूट-फूट।",
        "precautions_en": "• Maintain good posture\n• Do neck exercises daily\n• Apply hot/cold packs\n• Avoid heavy lifting",
        "precautions_hi": "• अच्छा आसन बनाए रखें\n• गर्दन का व्यायाम करें\n• भारी वजन न उठाएं",
        "severity": "moderate", "specialist": "Orthopedic", "specialist_hi": "हड्डी रोग विशेषज्ञ", "hindi_name": "सर्वाइकल स्पॉन्डिलोसिस",
    },
    "Paralysis (brain hemorrhage)": {
        "desc_en": "Loss of muscle function due to brain bleeding — medical emergency.",
        "desc_hi": "मस्तिष्क में रक्तस्राव के कारण मांसपेशी कार्य का नुकसान।",
        "precautions_en": "• EMERGENCY - Call 108 immediately\n• Don't move patient\n• Monitor breathing\n• Begin physical therapy early",
        "precautions_hi": "• आपातकाल - 108 कॉल करें\n• मरीज को न हिलाएं",
        "severity": "severe", "specialist": "Neurologist/Emergency", "specialist_hi": "न्यूरोलॉजिस्ट/आपातकाल", "hindi_name": "पक्षाघात",
    },
    "Jaundice": {
        "desc_en": "Yellowing of skin/eyes due to bilirubin buildup from liver dysfunction.",
        "desc_hi": "यकृत की खराबी से त्वचा और आंखों का पीला पड़ना।",
        "precautions_en": "• Avoid alcohol completely\n• Eat light digestible food\n• Complete bed rest\n• Stay hydrated",
        "precautions_hi": "• शराब से बिल्कुल बचें\n• हल्का खाना खाएं\n• पूर्ण बिस्तर आराम",
        "severity": "moderate", "specialist": "Hepatologist", "specialist_hi": "यकृत रोग विशेषज्ञ", "hindi_name": "पीलिया",
    },
    "Malaria": {
        "desc_en": "Mosquito-borne parasitic disease causing cyclic fever and chills.",
        "desc_hi": "मच्छर जनित परजीवी रोग।",
        "precautions_en": "• Use mosquito repellent\n• Sleep under bed nets\n• Complete antimalarial course\n• Stay hydrated",
        "precautions_hi": "• मच्छर भगाने वाली दवा उपयोग करें\n• मच्छरदानी में सोएं",
        "severity": "severe", "specialist": "Infectious Disease Specialist", "specialist_hi": "संक्रामक रोग विशेषज्ञ", "hindi_name": "मलेरिया",
    },
    "Chicken pox": {
        "desc_en": "Highly contagious viral infection causing itchy blisters all over the body.",
        "desc_hi": "अत्यधिक संक्रामक वायरल संक्रमण।",
        "precautions_en": "• Apply calamine lotion\n• Take oatmeal baths\n• Stay hydrated\n• Isolate to prevent spread",
        "precautions_hi": "• कैलामाइन लोशन लगाएं\n• हाइड्रेटेड रहें\n• अलग रहें",
        "severity": "moderate", "specialist": "General Physician", "specialist_hi": "सामान्य चिकित्सक", "hindi_name": "चिकनपॉक्स",
    },
    "Dengue": {
        "desc_en": "Mosquito-borne viral fever with severe joint pain and low platelet count.",
        "desc_hi": "मच्छर जनित वायरल बुखार।",
        "precautions_en": "• Oral rehydration therapy\n• Take Paracetamol ONLY (no Aspirin)\n• Complete bed rest\n• Monitor platelet count",
        "precautions_hi": "• ORS पियें\n• केवल पेरासिटामोल लें\n• पूर्ण बिस्तर आराम",
        "severity": "severe", "specialist": "General Physician", "specialist_hi": "सामान्य चिकित्सक", "hindi_name": "डेंगू",
    },
    "Typhoid": {
        "desc_en": "Bacterial infection from contaminated food/water causing prolonged fever.",
        "desc_hi": "दूषित भोजन/पानी से होने वाला बैक्टीरियल संक्रमण।",
        "precautions_en": "• Drink only safe/boiled water\n• Eat properly cooked food\n• Complete antibiotic course\n• Get vaccinated",
        "precautions_hi": "• उबला पानी पियें\n• ठीक से पका खाना खाएं\n• एंटीबायोटिक कोर्स पूरा करें",
        "severity": "moderate", "specialist": "General Physician", "specialist_hi": "सामान्य चिकित्सक", "hindi_name": "टाइफाइड",
    },
    "Hepatitis A": {
        "desc_en": "Liver infection caused by Hepatitis A virus through contaminated food/water.",
        "desc_hi": "हेपेटाइटिस A वायरस से यकृत संक्रमण।",
        "precautions_en": "• Rest completely\n• Avoid alcohol\n• Eat light nutritious food\n• Get vaccinated",
        "precautions_hi": "• पूरी तरह आराम करें\n• शराब से बचें\n• टीका लगवाएं",
        "severity": "moderate", "specialist": "Hepatologist", "specialist_hi": "यकृत रोग विशेषज्ञ", "hindi_name": "हेपेटाइटिस A",
    },
    "Hepatitis B": {
        "desc_en": "Serious liver infection spread through blood and body fluids.",
        "desc_hi": "रक्त और शरीर के तरल पदार्थों से फैलने वाला यकृत संक्रमण।",
        "precautions_en": "• Get vaccinated\n• Practice safe sex\n• Don't share needles\n• Regular monitoring",
        "precautions_hi": "• टीका लगवाएं\n• सुरक्षित यौन संबंध\n• सुइयां साझा न करें",
        "severity": "severe", "specialist": "Hepatologist", "specialist_hi": "यकृत रोग विशेषज्ञ", "hindi_name": "हेपेटाइटिस B",
    },
    "Hepatitis C": {
        "desc_en": "Viral liver infection spread through blood contact.",
        "desc_hi": "रक्त संपर्क से फैलने वाला वायरल यकृत संक्रमण।",
        "precautions_en": "• Don't share needles\n• Practice safe sex\n• Take antiviral medications\n• Avoid alcohol",
        "precautions_hi": "• सुइयां साझा न करें\n• एंटीवायरल दवाएं लें\n• शराब से बचें",
        "severity": "severe", "specialist": "Hepatologist", "specialist_hi": "यकृत रोग विशेषज्ञ", "hindi_name": "हेपेटाइटिस C",
    },
    "Hepatitis D": {
        "desc_en": "Liver infection occurring only in people with Hepatitis B.",
        "desc_hi": "केवल हेपेटाइटिस B रोगियों में होने वाला संक्रमण।",
        "precautions_en": "• Hepatitis B vaccine also prevents D\n• Practice safe sex\n• Avoid sharing needles",
        "precautions_hi": "• हेपेटाइटिस B का टीका D को भी रोकता है\n• सुइयां साझा न करें",
        "severity": "severe", "specialist": "Hepatologist", "specialist_hi": "यकृत रोग विशेषज्ञ", "hindi_name": "हेपेटाइटिस D",
    },
    "Hepatitis E": {
        "desc_en": "Waterborne liver disease caused by Hepatitis E virus.",
        "desc_hi": "हेपेटाइटिस E वायरस से जलजनित यकृत रोग।",
        "precautions_en": "• Drink safe treated water\n• Eat properly cooked food\n• Good hand hygiene",
        "precautions_hi": "• सुरक्षित पानी पियें\n• ठीक से पका खाना खाएं",
        "severity": "moderate", "specialist": "Hepatologist", "specialist_hi": "यकृत रोग विशेषज्ञ", "hindi_name": "हेपेटाइटिस E",
    },
    "Alcoholic hepatitis": {
        "desc_en": "Liver inflammation caused by excessive alcohol consumption.",
        "desc_hi": "अत्यधिक शराब सेवन से यकृत में सूजन।",
        "precautions_en": "• STOP alcohol completely\n• Eat nutritious food\n• Take prescribed medications\n• Regular liver tests",
        "precautions_hi": "• शराब पूरी तरह बंद करें\n• पौष्टिक भोजन करें",
        "severity": "severe", "specialist": "Hepatologist", "specialist_hi": "यकृत रोग विशेषज्ञ", "hindi_name": "अल्कोहलिक हेपेटाइटिस",
    },
    "Tuberculosis": {
        "desc_en": "Bacterial lung infection spread through air when infected persons cough.",
        "desc_hi": "फेफड़ों का बैक्टीरियल संक्रमण।",
        "precautions_en": "• Complete 6-month medication course (DOTS)\n• Cover mouth when coughing\n• Adequate ventilation\n• Nutritious diet",
        "precautions_hi": "• 6 महीने का कोर्स पूरा करें\n• खांसते समय मुंह ढकें",
        "severity": "severe", "specialist": "Pulmonologist", "specialist_hi": "फेफड़े के रोग विशेषज्ञ", "hindi_name": "तपेदिक",
    },
    "Common Cold": {
        "desc_en": "Viral infection of the upper respiratory tract causing runny nose.",
        "desc_hi": "ऊपरी श्वसन पथ का वायरल संक्रमण।",
        "precautions_en": "• Rest and stay hydrated\n• Saline gargling\n• Steam inhalation\n• Vitamin C",
        "precautions_hi": "• आराम करें और पानी पियें\n• नमक के पानी से गरारे\n• भाप लें",
        "severity": "mild", "specialist": "General Physician", "specialist_hi": "सामान्य चिकित्सक", "hindi_name": "सामान्य सर्दी",
    },
    "Pneumonia": {
        "desc_en": "Lung infection causing cough, fever, and breathing difficulty.",
        "desc_hi": "फेफड़ों का संक्रमण।",
        "precautions_en": "• Complete rest\n• Stay well hydrated\n• Steam inhalation\n• Complete antibiotic course",
        "precautions_hi": "• पूरी तरह आराम करें\n• खूब पानी पियें\n• एंटीबायोटिक कोर्स पूरा करें",
        "severity": "severe", "specialist": "Pulmonologist", "specialist_hi": "फेफड़े के रोग विशेषज्ञ", "hindi_name": "निमोनिया",
    },
    "Dimorphic hemmorhoids(piles)": {
        "desc_en": "Swollen veins in rectum/anus causing pain and bleeding.",
        "desc_hi": "मलाशय में सूजी हुई नसें।",
        "precautions_en": "• Eat high-fiber diet\n• Drink plenty of water\n• Avoid straining\n• Warm sitz baths",
        "precautions_hi": "• उच्च फाइबर आहार\n• खूब पानी पियें\n• जोर न लगाएं",
        "severity": "moderate", "specialist": "Proctologist", "specialist_hi": "प्रोक्टोलॉजिस्ट", "hindi_name": "बवासीर",
    },
    "Heart attack": {
        "desc_en": "Medical emergency where blood flow to heart is blocked.",
        "desc_hi": "हृदय में रक्त प्रवाह अवरुद्ध — चिकित्सा आपातकाल।",
        "precautions_en": "• EMERGENCY - Call 108 immediately\n• Chew aspirin if available\n• Rest in comfortable position",
        "precautions_hi": "• आपातकाल - 108 कॉल करें\n• एस्पिरिन चबाएं\n• आरामदायक स्थिति में आराम करें",
        "severity": "severe", "specialist": "Cardiologist/Emergency", "specialist_hi": "हृदय रोग विशेषज्ञ", "hindi_name": "दिल का दौरा",
    },
    "Variceal bleeding": {
        "desc_en": "Bleeding from enlarged veins in esophagus/stomach due to liver cirrhosis.",
        "desc_hi": "यकृत सिरोसिस के कारण रक्तस्राव।",
        "precautions_en": "• EMERGENCY - Seek immediate hospital care\n• Avoid alcohol completely\n• Regular monitoring",
        "precautions_hi": "• आपातकाल - तुरंत अस्पताल जाएं\n• शराब से बिल्कुल बचें",
        "severity": "severe", "specialist": "Hepatologist/Emergency", "specialist_hi": "यकृत रोग विशेषज्ञ", "hindi_name": "वैरिकल ब्लीडिंग",
    },
    "Hypothyroidism": {
        "desc_en": "Underactive thyroid producing insufficient hormones, slowing metabolism.",
        "desc_hi": "थायरॉयड ग्रंथि पर्याप्त हार्मोन नहीं बनाती।",
        "precautions_en": "• Take thyroid medication daily\n• Don't skip doses\n• Regular TSH tests\n• Exercise regularly",
        "precautions_hi": "• रोजाना थायरॉयड दवाई लें\n• नियमित TSH परीक्षण\n• व्यायाम करें",
        "severity": "moderate", "specialist": "Endocrinologist", "specialist_hi": "एंडोक्रिनोलॉजिस्ट", "hindi_name": "हाइपोथायरायडिज्म",
    },
    "Hyperthyroidism": {
        "desc_en": "Overactive thyroid producing excess hormones, speeding metabolism.",
        "desc_hi": "अतिसक्रिय थायरॉयड अधिक हार्मोन बनाता है।",
        "precautions_en": "• Take antithyroid medications\n• Regular thyroid monitoring\n• Manage stress\n• Avoid excess iodine",
        "precautions_hi": "• एंटीथायरॉयड दवाएं लें\n• नियमित निगरानी\n• तनाव प्रबंधित करें",
        "severity": "moderate", "specialist": "Endocrinologist", "specialist_hi": "एंडोक्रिनोलॉजिस्ट", "hindi_name": "हाइपरथायरायडिज्म",
    },
    "Hypoglycemia": {
        "desc_en": "Abnormally low blood sugar levels.",
        "desc_hi": "असामान्य रूप से कम रक्त शर्करा।",
        "precautions_en": "• Carry glucose tablets always\n• Eat regular meals\n• Monitor blood sugar frequently\n• Don't skip meals",
        "precautions_hi": "• हमेशा ग्लूकोज टैबलेट रखें\n• नियमित भोजन करें",
        "severity": "severe", "specialist": "Endocrinologist", "specialist_hi": "एंडोक्रिनोलॉजिस्ट", "hindi_name": "हाइपोग्लाइसीमिया",
    },
    "Osteoarthrosis": {
        "desc_en": "Degenerative joint disease causing cartilage breakdown and stiffness.",
        "desc_hi": "अपक्षयी जोड़ रोग।",
        "precautions_en": "• Maintain healthy weight\n• Low-impact exercise\n• Hot/cold packs\n• Physical therapy",
        "precautions_hi": "• स्वस्थ वजन बनाए रखें\n• कम प्रभाव वाला व्यायाम",
        "severity": "moderate", "specialist": "Rheumatologist", "specialist_hi": "रुमेटोलॉजिस्ट", "hindi_name": "ऑस्टियोआर्थराइटिस",
    },
    "Arthritis": {
        "desc_en": "Joint inflammation causing pain, swelling, and reduced range of motion.",
        "desc_hi": "जोड़ों में सूजन।",
        "precautions_en": "• Take anti-inflammatory medications\n• Gentle exercise\n• Heat/cold therapy\n• Maintain healthy weight",
        "precautions_hi": "• सूजन-रोधी दवाएं लें\n• हल्का व्यायाम\n• गर्म/ठंडा उपचार",
        "severity": "moderate", "specialist": "Rheumatologist", "specialist_hi": "रुमेटोलॉजिस्ट", "hindi_name": "गठिया",
    },
    "(Vertigo) Paroxysmal positional vertigo": {
        "desc_en": "Brief episodes of dizziness caused by head position changes.",
        "desc_hi": "सिर की स्थिति बदलने पर चक्कर आना।",
        "precautions_en": "• Perform Epley maneuver (with doctor guidance)\n• Move slowly\n• Avoid sudden head movements\n• Sleep with head elevated",
        "precautions_hi": "• Epley maneuver करें\n• धीरे-धीरे चलें\n• अचानक सिर न हिलाएं",
        "severity": "moderate", "specialist": "ENT Specialist/Neurologist", "specialist_hi": "ENT विशेषज्ञ", "hindi_name": "वर्टिगो",
    },
    "Acne": {
        "desc_en": "Skin condition causing pimples, blackheads due to clogged pores.",
        "desc_hi": "बंद रोमछिद्रों के कारण मुंहासे।",
        "precautions_en": "• Wash face twice daily\n• Don't pop pimples\n• Use oil-free products\n• Stay hydrated",
        "precautions_hi": "• दिन में दो बार मुंह धोएं\n• मुंहासे न फोड़ें\n• ऑयल-फ्री उत्पाद उपयोग करें",
        "severity": "mild", "specialist": "Dermatologist", "specialist_hi": "त्वचा रोग विशेषज्ञ", "hindi_name": "मुंहासे",
    },
    "Urinary tract infection": {
        "desc_en": "Bacterial infection causing painful, frequent urination.",
        "desc_hi": "मूत्र प्रणाली का बैक्टीरियल संक्रमण।",
        "precautions_en": "• Drink plenty of water\n• Complete antibiotic course\n• Don't hold urine\n• Wipe front to back",
        "precautions_hi": "• खूब पानी पियें\n• एंटीबायोटिक कोर्स पूरा करें\n• पेशाब न रोकें",
        "severity": "moderate", "specialist": "Urologist", "specialist_hi": "मूत्र रोग विशेषज्ञ", "hindi_name": "मूत्र पथ संक्रमण",
    },
    "Psoriasis": {
        "desc_en": "Chronic autoimmune skin condition causing scales and red patches.",
        "desc_hi": "पुरानी स्वतः-प्रतिरक्षा त्वचा स्थिति।",
        "precautions_en": "• Moisturize skin regularly\n• Use medicated creams\n• Avoid stress triggers\n• Get moderate sun exposure",
        "precautions_hi": "• नियमित मॉइस्चराइज करें\n• औषधीय क्रीम उपयोग करें\n• तनाव से बचें",
        "severity": "moderate", "specialist": "Dermatologist", "specialist_hi": "त्वचा रोग विशेषज्ञ", "hindi_name": "सोरायसिस",
    },
    "Impetigo": {
        "desc_en": "Contagious bacterial skin infection causing red sores and honey-colored crusts.",
        "desc_hi": "संक्रामक बैक्टीरियल त्वचा संक्रमण।",
        "precautions_en": "• Keep sores clean and covered\n• Apply antibiotic cream\n• Wash hands frequently\n• Don't touch sores",
        "precautions_hi": "• घावों को साफ रखें\n• एंटीबायोटिक क्रीम लगाएं\n• बार-बार हाथ धोएं",
        "severity": "mild", "specialist": "Dermatologist", "specialist_hi": "त्वचा रोग विशेषज्ञ", "hindi_name": "इम्पेटिगो",
    },
    "Diabetes ": {
        "desc_en": "Metabolic disease causing high blood sugar due to insulin deficiency.",
        "desc_hi": "इंसुलिन की कमी से रक्त शर्करा में वृद्धि।",
        "precautions_en": "• Monitor blood sugar regularly\n• Follow balanced diet\n• Exercise daily\n• Take medications as prescribed",
        "precautions_hi": "• नियमित रक्त शर्करा जांचें\n• संतुलित आहार\n• रोजाना व्यायाम",
        "severity": "moderate", "specialist": "Endocrinologist", "specialist_hi": "एंडोक्रिनोलॉजिस्ट", "hindi_name": "मधुमेह",
    },
    "Hypertension": {
        "desc_en": "High blood pressure causing damage to arteries over time.",
        "desc_hi": "उच्च रक्तचाप जो समय के साथ धमनियों को नुकसान पहुंचाता है।",
        "precautions_en": "• Eat low-sodium diet\n• Exercise regularly\n• Manage stress\n• Monitor BP regularly",
        "precautions_hi": "• कम नमक वाला आहार\n• नियमित व्यायाम\n• BP जांचें",
        "severity": "moderate", "specialist": "Cardiologist", "specialist_hi": "हृदय रोग विशेषज्ञ", "hindi_name": "उच्च रक्तचाप",
    },
}

# ─── CHATBOT RESPONSES ───────────────────────────────────────────────────────
CHATBOT_KB = [
    {"kw": ["fever", "bukhar", "temperature"], "en": "🌡️ **Fever Tips**:\n• Take Paracetamol\n• Stay hydrated\n• Rest\n• See doctor if > 3 days or very high", "hi": "🌡️ **बुखार के सुझाव**:\n• पेरासिटामोल लें\n• पानी पियें\n• आराम करें\n• 3+ दिन रहे तो डॉक्टर से मिलें"},
    {"kw": ["headache", "sir dard", "migraine"], "en": "🤕 **Headache Tips**:\n• Rest in dark quiet room\n• Cold/warm compress\n• Stay hydrated\n• Avoid screens", "hi": "🤕 **सिरदर्द के सुझाव**:\n• अंधेरे कमरे में आराम करें\n• ठंडी/गर्म सिकाई\n• पानी पियें"},
    {"kw": ["cold", "cough", "khansi", "runny nose"], "en": "🤧 **Cold/Cough Home Remedies**:\n• Warm salt water gargling\n• Ginger-honey-lemon tea\n• Steam inhalation\n• Rest and hydration", "hi": "🤧 **सर्दी/खांसी के घरेलू उपाय**:\n• नमक पानी से गरारे\n• अदरक-शहद-नींबू की चाय\n• भाप लें"},
    {"kw": ["diabetes", "sugar", "blood sugar"], "en": "🩺 **Diabetes Management**:\n• Monitor blood sugar daily\n• Exercise 30 min/day\n• Low glycemic diet\n• Never skip medications\n• HbA1c every 3 months", "hi": "🩺 **मधुमेह प्रबंधन**:\n• रोजाना रक्त शर्करा जांचें\n• 30 मिनट व्यायाम\n• दवाएं कभी न छोड़ें"},
    {"kw": ["blood pressure", "bp", "hypertension"], "en": "❤️ **BP Management**:\n• Reduce salt (<5g/day)\n• Exercise daily\n• Limit alcohol\n• Take medications regularly\n• Manage stress with yoga", "hi": "❤️ **रक्तचाप प्रबंधन**:\n• नमक कम करें\n• नियमित व्यायाम\n• दवाएं नियमित लें"},
    {"kw": ["heart attack", "chest pain", "emergency"], "en": "🚨 **EMERGENCY! Call 108!**\n• Keep person calm\n• Loosen tight clothing\n• Give Aspirin if available\n• Don't leave them alone", "hi": "🚨 **आपातकाल! 108 कॉल करें!**\n• व्यक्ति को शांत रखें\n• तंग कपड़े ढीले करें"},
    {"kw": ["diet", "food", "healthy eating", "khaana"], "en": "🥗 **Healthy Eating**:\n• Half plate = vegetables\n• Choose whole grains\n• Limit sugar & processed food\n• 8+ glasses water daily", "hi": "🥗 **स्वस्थ खाना**:\n• आधी थाली सब्जियों से\n• साबुत अनाज चुनें\n• चीनी कम करें"},
    {"kw": ["exercise", "workout", "fit"], "en": "💪 **Exercise Tips**:\n• 150 min/week moderate activity\n• Brisk walking is excellent!\n• Strength training 2x/week\n• Yoga for flexibility", "hi": "💪 **व्यायाम सुझाव**:\n• 150 मिनट/सप्ताह\n• तेज चलना बहुत अच्छा है\n• योग करें"},
    {"kw": ["sleep", "insomnia", "neend"], "en": "😴 **Better Sleep**:\n• Same sleep/wake time daily\n• No screens 1hr before bed\n• Cool dark room\n• No caffeine after 3 PM\n• Adults need 7-9 hours", "hi": "😴 **बेहतर नींद**:\n• एक ही समय पर सोएं\n• सोने से पहले स्क्रीन बंद करें\n• 7-9 घंटे जरूरी"},
    {"kw": ["stress", "anxiety", "mental health", "depression"], "en": "🧘 **Stress Management**:\n• Deep breathing (4-7-8)\n• Meditation 10 min/day\n• Exercise regularly\n• Talk to someone\n• Helpline: 1800-599-0019", "hi": "🧘 **तनाव प्रबंधन**:\n• गहरी सांस लें\n• 10 मिनट ध्यान\n• हेल्पलाइन: 1800-599-0019"},
    {"kw": ["dengue", "mosquito", "malaria", "machar"], "en": "🦟 **Mosquito Prevention**:\n• Use mosquito repellent\n• Sleep under nets\n• Wear full-sleeve clothes\n• Eliminate standing water", "hi": "🦟 **मच्छर से बचाव**:\n• मच्छर भगाने वाली दवा उपयोग करें\n• मच्छरदानी में सोएं\n• खड़ा पानी हटाएं"},
    {"kw": ["pneumonia", "symptoms", "what are symptoms"], "en": "🫁 **Pneumonia Symptoms**:\n• Fever and chills\n• Cough (may produce phlegm)\n• Shortness of breath\n• Chest pain when breathing\n• Fatigue\n• Loss of appetite", "hi": "🫁 **निमोनिया के लक्षण**:\n• बुखार और ठंड\n• खांसी\n• सांस की तकलीफ\n• थकान"},
]

HEALTH_TIPS = [
    {"icon": "🌅", "title": "Morning Routine", "title_hi": "सुबह की दिनचर्या", "tip": "Start your day with a glass of warm water and 10 minutes of stretching.", "tip_hi": "गुनगुने पानी के एक गिलास और 10 मिनट स्ट्रेचिंग से दिन शुरू करें।"},
    {"icon": "💧", "title": "Stay Hydrated", "title_hi": "हाइड्रेटेड रहें", "tip": "Drink 8-10 glasses of water daily. Dehydration can cause headaches and fatigue.", "tip_hi": "रोजाना 8-10 गिलास पानी पियें। निर्जलीकरण से सिरदर्द और थकान हो सकती है।"},
    {"icon": "🥗", "title": "Balanced Diet", "title_hi": "संतुलित आहार", "tip": "Include fruits, vegetables, whole grains, and lean proteins in every meal.", "tip_hi": "हर भोजन में फल, सब्जियां, साबुत अनाज और प्रोटीन शामिल करें।"},
    {"icon": "🏃", "title": "Daily Exercise", "title_hi": "दैनिक व्यायाम", "tip": "30 minutes of moderate exercise daily reduces risk of heart disease by 30%.", "tip_hi": "30 मिनट व्यायाम से हृदय रोग का खतरा 30% कम होता है।"},
    {"icon": "😴", "title": "Quality Sleep", "title_hi": "गुणवत्तापूर्ण नींद", "tip": "7-9 hours of sleep strengthens immunity and improves mental health.", "tip_hi": "7-9 घंटे की नींद से रोग प्रतिरोधक क्षमता और मानसिक स्वास्थ्य बेहतर होता है।"},
    {"icon": "🧘", "title": "Stress Management", "title_hi": "तनाव प्रबंधन", "tip": "10 minutes of daily meditation can reduce anxiety and improve focus.", "tip_hi": "10 मिनट ध्यान से चिंता कम होती है और ध्यान बेहतर होता है।"},
    {"icon": "🦷", "title": "Oral Hygiene", "title_hi": "मौखिक स्वच्छता", "tip": "Brush twice daily and floss. Poor oral health is linked to heart disease.", "tip_hi": "दिन में दो बार ब्रश करें। खराब मौखिक स्वास्थ्य हृदय रोग से जुड़ा है।"},
    {"icon": "☀️", "title": "Vitamin D", "title_hi": "विटामिन D", "tip": "15-20 minutes of morning sunlight helps your body produce Vitamin D naturally.", "tip_hi": "सुबह की धूप में 15-20 मिनट से विटामिन D स्वाभाविक रूप से बनता है।"},
    {"icon": "🚭", "title": "No Smoking", "title_hi": "धूम्रपान न करें", "tip": "Quitting smoking reduces heart disease risk by 50% within 1 year.", "tip_hi": "धूम्रपान बंद करने से 1 साल में हृदय रोग का खतरा 50% कम होता है।"},
    {"icon": "🩺", "title": "Regular Checkups", "title_hi": "नियमित जांच", "tip": "Annual health checkups can detect diseases early when they're most treatable.", "tip_hi": "वार्षिक जांच से रोग जल्दी पकड़े जाते हैं जब इलाज आसान होता है।"},
]


# ─── CSS ──────────────────────────────────────────────────────────────────────
def inject_css(dark_mode: bool = False):
    if dark_mode:
        main_bg = "#0f1117"
        card_bg = "#1a1f2e"
        card_bg2 = "#141824"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        border_col = "#2d3748"
        input_bg = "#1e2536"
        table_header = "#1e2536"
        hover_bg = "#252d40"
    else:
        main_bg = "#f0f4f8"
        card_bg = "#ffffff"
        card_bg2 = "#f8fafc"
        text_primary = "#0f172a"
        text_secondary = "#64748b"
        border_col = "#e2e8f0"
        input_bg = "#f8fafc"
        table_header = "#f1f5f9"
        hover_bg = "#f0f7ff"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif !important;
    }}
    .main {{
        background: {main_bg} !important;
    }}
    #MainMenu, footer, header {{visibility: hidden;}}


    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: #0f1829 !important;
        border-right: 1px solid #1e293b !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #cbd5e1 !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        border-radius: 10px !important;
        text-align: left !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        width: 100% !important;
        transition: all 0.2s !important;
        justify-content: flex-start !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(99,102,241,0.15) !important;
        color: #fff !important;
    }}
    .sidebar-logo {{
        background: linear-gradient(135deg, #6366f1, #0ea5e9);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        color: white !important;
    }}
    .sidebar-logo-icon {{ font-size: 2.2rem; }}
    .sidebar-logo-title {{ font-family: 'Sora', sans-serif; font-size: 1rem; font-weight: 700; color: white !important; margin-top: 0.3rem; }}
    .sidebar-logo-sub {{ font-size: 0.72rem; opacity: 0.75; color: white !important; }}
    .nav-section-label {{
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #475569 !important;
        padding: 0.5rem 0 0.25rem 0 !important;
    }}
    .nav-active [data-testid="stSidebar"] .stButton > button {{
        background: rgba(99,102,241,0.25) !important;
        color: white !important;
    }}
    .sidebar-note {{
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.25);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        font-size: 0.78rem;
        color: #fca5a5 !important;
        margin-top: 1rem;
    }}
    .sidebar-note-title {{ font-weight: 700; color: #f87171 !important; margin-bottom: 0.4rem; }}
  
    /* ── Main area ── */
    .block-container {{
        padding: 1.5rem 2rem 2rem !important;
        max-width: 100% !important;
    }}

    /* ── Page Header ── */
    .page-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid {border_col};
    }}
    .page-title {{
        font-family: 'Sora', sans-serif;
        font-size: 1.7rem;
        font-weight: 800;
        color: {text_primary};
        margin: 0;
    }}
    .page-subtitle {{ color: {text_secondary}; font-size: 0.88rem; margin-top: 0.2rem; }}
    .page-datetime {{ color: {text_secondary}; font-size: 0.82rem; text-align: right; }}

    /* ── Feature Cards ── */
    .feature-card {{
        background: {card_bg};
        border-radius: 18px;
        padding: 1.6rem;
        border: 1px solid {border_col};
        height: 100%;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }}
    .feature-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    }}
    .feature-card-icon {{
        width: 52px; height: 52px;
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }}
    .feature-card-title {{
        font-family: 'Sora', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 0.5rem;
    }}
    .feature-card-desc {{ color: {text_secondary}; font-size: 0.85rem; line-height: 1.5; margin-bottom: 1.2rem; }}
    .feature-card-btn {{
        display: inline-block;
        padding: 0.5rem 1.1rem;
        border-radius: 10px;
        font-size: 0.83rem;
        font-weight: 600;
        cursor: pointer;
        border: none;
        text-decoration: none;
        transition: all 0.2s ease;
    }}
    .btn-green {{ background: #10b981; color: white; }}
    .btn-blue {{ background: #0ea5e9; color: white; }}
    .btn-purple {{ background: #6366f1; color: white; }}

    /* ── Stats Cards ── */
    .stat-card {{
        background: {card_bg};
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        border: 1px solid {border_col};
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .stat-value {{
        font-family: 'Sora', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1;
    }}
    .stat-label {{ color: {text_secondary}; font-size: 0.78rem; margin-top: 0.3rem; font-weight: 500; }}
    .stat-icon {{
        font-size: 2rem;
        width: 50px; height: 50px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
    }}

    /* ── Section Headers ── */
    .section-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }}
    .section-title {{
        font-family: 'Sora', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: {text_primary};
    }}

    /* ── History Table ── */
    .pred-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }}
    .pred-table th {{
        background: {table_header};
        padding: 0.7rem 1rem;
        text-align: left;
        font-weight: 600;
        color: {text_secondary};
        font-size: 0.78rem;
        border-bottom: 1px solid {border_col};
    }}
    .pred-table td {{
        padding: 0.75rem 1rem;
        border-bottom: 1px solid {border_col};
        color: {text_primary};
    }}
    .pred-table tr:hover td {{ background: {hover_bg}; }}
    .tag-symptoms {{ background: #dcfce7; color: #15803d; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
    .tag-xray {{ background: #dbeafe; color: #1d4ed8; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
    .tag-chatbot {{ background: #ede9fe; color: #5b21b6; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
    .conf-green {{ color: #16a34a; font-weight: 700; }}
    .conf-amber {{ color: #d97706; font-weight: 700; }}
    .conf-red {{ color: #dc2626; font-weight: 700; }}

    /* ── Chatbot ── */
    .chat-wrap {{
        background: {card_bg};
        border-radius: 16px;
        border: 1px solid {border_col};
        overflow: hidden;
    }}
    .chat-header {{
        background: linear-gradient(135deg,#6366f1,#0ea5e9);
        padding: 1rem 1.2rem;
        color: white;
        font-family: 'Sora', sans-serif;
        font-weight: 700;
    }}
    .chat-messages {{
        height: 340px;
        overflow-y: auto;
        padding: 1rem;
        background: {card_bg2};
    }}
    .msg-bot {{
        background: {card_bg};
        border: 1px solid {border_col};
        border-radius: 14px 14px 14px 4px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.75rem;
        margin-right: 20%;
        font-size: 0.85rem;
        color: {text_primary};
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .msg-user {{
        background: linear-gradient(135deg,#6366f1,#0ea5e9);
        color: white;
        border-radius: 14px 14px 4px 14px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.75rem;
        margin-left: 20%;
        font-size: 0.85rem;
    }}
    .msg-avatar {{ font-size: 1rem; margin-right: 0.4rem; }}

    /* ── Prediction Result ── */
    .result-card {{
        background: linear-gradient(135deg,#f0fdf4,#dcfce7);
        border: 2px solid #86efac;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }}
    .result-disease {{
        font-family: 'Sora', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #14532d;
    }}
    .severity-mild {{ background: #dcfce7; color: #15803d; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; display: inline-block; }}
    .severity-moderate {{ background: #fef9c3; color: #854d0e; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; display: inline-block; }}
    .severity-severe {{ background: #fee2e2; color: #991b1b; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; display: inline-block; }}

    /* ── Conf bar ── */
    .cbar-wrap {{ background: #e2e8f0; border-radius: 8px; height: 10px; overflow: hidden; margin: 0.4rem 0; }}
    .cbar-fill {{ height: 100%; border-radius: 8px; }}

    /* ── Health Tip Card ── */
    .tip-card {{
        background: {card_bg};
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid {border_col};
        margin-bottom: 0.75rem;
        display: flex;
        gap: 1rem;
        align-items: flex-start;
    }}
    .tip-icon {{ font-size: 1.8rem; }}
    .tip-title {{ font-family: 'Sora', sans-serif; font-weight: 700; font-size: 0.9rem; color: {text_primary}; }}
    .tip-text {{ color: {text_secondary}; font-size: 0.83rem; margin-top: 0.25rem; line-height: 1.5; }}

    /* ── General card ── */
    .gen-card {{
        background: {card_bg};
        border-radius: 16px;
        padding: 1.4rem;
        border: 1px solid {border_col};
        margin-bottom: 1rem;
    }}
    .gen-card-title {{
        font-family: 'Sora', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 0.75rem;
    }}

    /* ── Symptom tags ── */
    .stag {{
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 3px 12px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    /* ── Button overrides ── */
    .stButton > button {{
        background: linear-gradient(135deg,#6366f1,#0ea5e9) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
        transition: all 0.2s !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(99,102,241,0.4) !important;
    }}

    /* ── Disclaimer ── */
    .disclaimer {{
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #9a3412;
        margin-top: 1.5rem;
        text-align: center;
    }}

    /* scrollbar */
    .chat-messages::-webkit-scrollbar {{ width: 4px; }}
    .chat-messages::-webkit-scrollbar-track {{ background: transparent; }}
    .chat-messages::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 4px; }}

    [data-testid="collapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    background: #6366f1 !important;
    border-radius: 0 8px 8px 0 !important;
    width: 24px !important;
    color: white !important;
    opacity: 1 !important;
}}
    </style>
    """, unsafe_allow_html=True)


# ─── MODEL ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        le = joblib.load("label_encoder.pkl")
        df = pd.read_csv("dataset/Training.csv")
        model_cols = df.drop("prognosis", axis=1).columns.tolist()
        display_cols = sorted(model_cols)
        return model, le, model_cols, display_cols
    except FileNotFoundError as e:
        st.error(f"Model files not found. Run: python train_model.py\n{e}")
        st.stop()

@st.cache_resource
def load_xray_model():
    try:
        import tensorflow as tf
        import joblib
        xray_model = tf.keras.models.load_model("xray_model.h5")
        info = joblib.load("xray_model_info.pkl")
        return xray_model, info
    except Exception:
        return None, None

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, key))


def get_info(disease: str, lang: str = "en") -> dict:
    info = DISEASE_INFO.get(disease)
    if not info:
        for k in DISEASE_INFO:
            if disease.lower() in k.lower() or k.lower() in disease.lower():
                info = DISEASE_INFO[k]
                break
    if not info:
        return {"desc": "Information coming soon.", "precautions": "• Consult a doctor", "severity": "moderate", "specialist": "General Physician", "specialist_hi": "सामान्य चिकित्सक", "hindi_name": disease}
    return {
        "desc": info.get(f"desc_{lang}", info.get("desc_en", "")),
        "precautions": info.get(f"precautions_{lang}", info.get("precautions_en", "")),
        "severity": info.get("severity", "moderate"),
        "specialist": info.get("specialist", "General Physician"),
        "specialist_hi": info.get("specialist_hi", info.get("specialist", "General Physician")),
        "hindi_name": info.get("hindi_name", disease),
    }


def chat_response(msg: str, lang: str = "en") -> str:
    ml = msg.lower()
    for item in CHATBOT_KB:
        for kw in item["kw"]:
            if kw in ml:
                return item.get(f"{lang}", item["en"])
    for dname in DISEASE_INFO:
        if dname.lower() in ml:
            d = DISEASE_INFO[dname]
            if lang == "hi":
                return f"🏥 **{d.get('hindi_name', dname)}**\n\n{d.get('desc_hi', d.get('desc_en', ''))}\n\n**सावधानियां:**\n{d.get('precautions_hi', d.get('precautions_en', ''))}"
            return f"🏥 **{dname}**\n\n{d.get('desc_en', '')}\n\n**Precautions:**\n{d.get('precautions_en', '')}"
    return ("🤖 I can help with symptoms, diseases, diet, and health tips!\nTry asking about fever, headache, diabetes, or a specific disease."
            if lang == "en" else
            "🤖 मैं लक्षणों, रोगों और स्वास्थ्य सुझावों में मदद कर सकता हूं!\nबुखार, सिरदर्द, मधुमेह या किसी रोग के बारे में पूछें।")

def ai_chat_response(user_input, lang="en"):
    """Try real AI first, gracefully fall back to keyword responses."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    system_prompt = """You are a professional health assistant.
Give simple, safe, and informative health advice.
Do not provide medical diagnosis.
Always suggest consulting a doctor when necessary.
Keep answers clear and structured with bullet points when helpful.
Keep responses concise (under 150 words)."""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.chat_history[-8:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        return reply, True          # True  = real AI used
    except Exception:
        reply = chat_response(user_input, lang)
        return reply, False         # False = fallback used

def conf_color(c: float) -> str:
    return "#16a34a" if c >= 0.75 else "#d97706" if c >= 0.50 else "#dc2626"


def conf_class(c: float) -> str:
    return "conf-green" if c >= 0.75 else "conf-amber" if c >= 0.50 else "conf-red"


def add_history(type_: str, input_: str, prediction: str, confidence: float):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {
        "type": type_,
        "input": input_,
        "prediction": prediction,
        "confidence": confidence,
        "time": datetime.now().strftime("%I:%M %p"),
    })
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        lang = st.session_state.get("lang", "en")

        # Logo
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🏥</div>
            <div class="sidebar-logo-title">AI Healthcare</div>
            <div class="sidebar-logo-sub">v3.0 · Disease Predictor</div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)
        pages = [
            ("dashboard", "🏠 " + t("dashboard")),
            ("predict", "🔬 " + t("symptom_pred")),
            ("xray", "📷 " + t("xray_analysis")),
            ("chatbot", "💬 " + t("ai_chatbot")),
            ("history", "🕐 " + t("history")),
            ("tips", "💚 " + t("health_tips")),
            ("about", "ℹ️ " + t("about")),
        ]
        for key, label in pages:
            is_active = st.session_state.get("page", "dashboard") == key
            btn_style = "background: rgba(99,102,241,0.22) !important; color: white !important; border-left: 3px solid #6366f1 !important;" if is_active else ""
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
            if is_active:
                st.markdown(f"<style>[data-testid='stSidebar'] div:has(button[kind='secondary'][id^='nav_{key}']) button {{ {btn_style} }}</style>", unsafe_allow_html=True)

        st.markdown("---")

        # Language
        st.markdown('<div class="nav-section-label">🌐 Language</div>', unsafe_allow_html=True)
        lc1, lc2 = st.columns(2)
        with lc1:
            if st.button("🇮🇳 EN", use_container_width=True, key="lang_en"):
                st.session_state.lang = "en"
                st.rerun()
        with lc2:
            if st.button("🇮🇳 हि", use_container_width=True, key="lang_hi"):
                st.session_state.lang = "hi"
                st.rerun()

        # Dark mode
        st.markdown('<div class="nav-section-label">🎨 ' + t("dark_mode") + '</div>', unsafe_allow_html=True)
        dark = st.toggle(t("dark_mode"), value=st.session_state.get("dark_mode", False), key="dm_toggle", label_visibility="collapsed")
        if dark != st.session_state.get("dark_mode", False):
            st.session_state.dark_mode = dark
            st.rerun()

        st.markdown("---")

        # Important note
        st.markdown(f"""
        <div class="sidebar-note">
            <div class="sidebar-note-title">ℹ️ {t("important_note")}</div>
            {t("disclaimer")}
            <br><br><b>🆘 Emergency: 108</b>
        </div>
        """, unsafe_allow_html=True)


# ─── PAGES ───────────────────────────────────────────────────────────────────

def render_page_header(title: str, subtitle: str):
    now = datetime.now()
    st.markdown(f"""
    <div class="page-header">
        <div>
            <div class="page-title">{title} 👋</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        <div class="page-datetime">
            📅 {now.strftime("%B %d, %Y")}<br>
            ⏰ {now.strftime("%I:%M %p")}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard(model, le, model_cols, display_cols):
    lang = st.session_state.get("lang", "en")
    render_page_header(t("dashboard"), t("welcome"))

    # ── Feature cards ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-card-icon" style="background:#d1fae5">🩺</div>
            <div class="feature-card-title">{t("symptom_card_title")}</div>
            <div class="feature-card-desc">{t("symptom_card_desc")}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t("start_prediction"), key="dash_predict", use_container_width=True):
            st.session_state.page = "predict"
            st.rerun()

    with c2:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-card-icon" style="background:#dbeafe">🫁</div>
            <div class="feature-card-title">{t("xray_card_title")}</div>
            <div class="feature-card-desc">{t("xray_card_desc")}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t("analyze_xray"), key="dash_xray", use_container_width=True):
            st.session_state.page = "xray"
            st.rerun()

    with c3:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-card-icon" style="background:#ede9fe">🤖</div>
            <div class="feature-card-title">{t("chatbot_card_title")}</div>
            <div class="feature-card-desc">{t("chatbot_card_desc")}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t("start_chat"), key="dash_chat", use_container_width=True):
            st.session_state.page = "chatbot"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stats + Recent + Chatbot ────────────────────────────────────────
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # Stats row
        st.markdown(f'<div class="section-title">📊 {t("quick_stats")}</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        stats = [
            (s1, "95.3%", "ML Model Accuracy" if lang=="en" else "ML सटीकता", "#10b981", "#d1fae5", "🎯"),
            (s2, "96.7%", "CNN Model Accuracy" if lang=="en" else "CNN सटीकता", "#0ea5e9", "#dbeafe", "🧠"),
            (s3, "41+", "Diseases Covered" if lang=="en" else "रोग कवर", "#6366f1", "#ede9fe", "🏥"),
            (s4, "24/7", "AI Assistant" if lang=="en" else "AI सहायक", "#f59e0b", "#fef9c3", "⏰"),
        ]
        for col, val, lbl, color, bg, icon in stats:
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div>
                        <div class="stat-value" style="color:{color}">{val}</div>
                        <div class="stat-label">{lbl}</div>
                    </div>
                    <div class="stat-icon" style="background:{bg}">{icon}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Recent predictions table
        st.markdown(f"""
        <div class="section-header">
            <div class="section-title">📋 {t("recent_predictions")}</div>
        </div>
        """, unsafe_allow_html=True)

        history = st.session_state.get("history", [])
        if history:
            rows_html = ""
            for item in history[:5]:
                tag_cls = {"Symptoms": "tag-symptoms", "X-ray": "tag-xray", "Chatbot": "tag-chatbot"}.get(item["type"], "tag-symptoms")
                c_cls = conf_class(item["confidence"])
                inp_short = item["input"][:30] + "..." if len(item["input"]) > 30 else item["input"]
                rows_html += f"""<tr>
                    <td><span class="{tag_cls}">{item["type"]}</span></td>
                    <td>{inp_short}</td>
                    <td><b>{item["prediction"]}</b></td>
                    <td><span class="{c_cls}">{item["confidence"]:.1%}</span></td>
                    <td>{item["time"]}</td>
                </tr>"""
            st.markdown(f"""
            <div class="gen-card" style="padding:0;overflow:hidden">
                <table class="pred-table">
                    <thead><tr>
                        <th>{t("type")}</th><th>{t("input")}</th>
                        <th>{t("prediction")}</th><th>{t("confidence")}</th><th>{t("time")}</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="gen-card" style="text-align:center;padding:2rem;color:#94a3b8">
                🔬 {t("no_history")}
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        # Mini chatbot
        st.markdown(f'<div class="section-title">💬 {t("ai_chatbot")}</div>', unsafe_allow_html=True)

        if "dash_chat_hist" not in st.session_state:
            hello = t("chatbot_hello")
            st.session_state.dash_chat_hist = [{"role": "bot", "content": hello}]

        msgs_html = ""
        for m in st.session_state.dash_chat_hist[-6:]:
            if m["role"] == "bot":
                msgs_html += f'<div class="msg-bot"><span class="msg-avatar">🤖</span>{m["content"].replace(chr(10),"<br>")}</div>'
            else:
                msgs_html += f'<div class="msg-user"><span class="msg-avatar">👤</span>{m["content"]}</div>'

        st.markdown(f"""
        <div class="chat-wrap">
            <div class="chat-header">🤖 {t("ai_chatbot")} <span style="opacity:.75;font-size:0.75rem;font-weight:400;margin-left:8px">Always available</span></div>
            <div class="chat-messages">{msgs_html}</div>
        </div>
        """, unsafe_allow_html=True)

        q_input = st.text_input("", placeholder=t("chatbot_placeholder"), key="dash_chat_input", label_visibility="collapsed")
        dc1, dc2 = st.columns([4, 1])
        with dc1:
            if st.button(t("send"), key="dash_send", use_container_width=True):
                if q_input.strip():
                    resp = chat_response(q_input, lang)
                    st.session_state.dash_chat_hist.append({"role": "user", "content": q_input})
                    st.session_state.dash_chat_hist.append({"role": "bot", "content": resp})
                    st.rerun()
        with dc2:
            if st.button("🗑️", key="dash_clr"):
                st.session_state.dash_chat_hist = [{"role": "bot", "content": t("chatbot_hello")}]
                st.rerun()


def render_predict(model, le, model_cols, display_cols):
    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.rerun()

    lang = st.session_state.get("lang", "en")
    render_page_header("🔬 " + t("symptom_pred"), t("symptom_card_desc"))

    def _clear():
        st.session_state.syms = []

    st.markdown(f'<div class="gen-card-title">{t("select_symptoms")}</div>', unsafe_allow_html=True)

    col_sym, col_clr = st.columns([5, 1])
    with col_sym:
        selected = st.multiselect(
            label="Symptoms",
            options=display_cols,
            max_selections=15,
            placeholder=t("search_symptoms"),
            key="syms",
            format_func=lambda x: x.replace("_", " ").title(),
            label_visibility="collapsed",
        )
    with col_clr:
        st.write("")
        st.button(t("clear_btn"), on_click=_clear, use_container_width=True)

    if selected:
        tags = "".join([f'<span class="stag">✓ {s.replace("_"," ").title()}</span>' for s in selected])
        st.markdown(f"<div style='margin:0.5rem 0 1rem'>{tags}</div>", unsafe_allow_html=True)

    predict_clicked = st.button(t("predict_btn"), type="primary", use_container_width=True)

    if predict_clicked:
        if not selected:
            st.error(t("no_symptoms"))
            return

        vec = np.zeros(len(model_cols))
        for s in selected:
            if s in model_cols:
                vec[model_cols.index(s)] = 1

        pred = model.predict([vec])[0]
        probs = model.predict_proba([vec])[0]
        disease = le.inverse_transform([pred])[0]
        conf = probs.max()

        top_idx = np.argsort(probs)[-4:][::-1]
        alts = [(le.inverse_transform([i])[0], probs[i]) for i in top_idx if le.inverse_transform([i])[0] != disease][:3]

        add_history("Symptoms", ", ".join(selected[:3]), disease, conf)

        st.markdown("---")

        info = get_info(disease, lang)
        sev = info["severity"]
        conf_lbl = t("high_conf") if conf >= 0.75 else t("med_conf") if conf >= 0.50 else t("low_conf")
        hindi_sub = f"<div style='color:#4ade80;font-size:0.95rem;margin-top:0.3rem'>{info['hindi_name']}</div>" if lang == "en" and info["hindi_name"] != disease else ""

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:2.5rem">🏥</div>
            <div class="result-disease">{disease}</div>
            {hindi_sub}
            <div style="margin-top:0.75rem"><span class="severity-{sev}">{t("severity")}: {sev.capitalize()}</span></div>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="gen-card">
                <div class="gen-card-title">📊 {t("confidence")}</div>
                <div style="font-size:1.4rem;font-weight:800;color:{conf_color(conf)}">{conf:.1%}</div>
                <div style="color:#64748b;font-size:0.82rem">{conf_lbl}</div>
                <div class="cbar-wrap" style="margin-top:0.75rem">
                    <div class="cbar-fill" style="width:{conf*100:.1f}%;background:{conf_color(conf)}"></div>
                </div>
                <div style="margin-top:0.75rem;color:#64748b;font-size:0.82rem">
                    🔬 {t("symptoms_analyzed") if lang=="en" else "विश्लेषित लक्षण"}: <b>{len(selected)}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            spec = info["specialist_hi"] if lang == "hi" else info["specialist"]
            st.markdown(f"""
            <div class="gen-card">
                <div class="gen-card-title">🩺 {t("specialist")}</div>
                <div style="font-size:1rem;font-weight:700;color:#0ea5e9;margin-bottom:0.5rem">{spec}</div>
                <div style="color:#64748b;font-size:0.82rem">👨‍⚕️ {"Consult a doctor for professional diagnosis." if lang=="en" else "पेशेवर निदान के लिए डॉक्टर से मिलें।"}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander(f"📖 {t('about_disease')}", expanded=True):
            st.markdown(info["desc"])

        with st.expander(f"🛡️ {t('precautions')}", expanded=True):
            st.markdown(info["precautions"])

        if alts:
            with st.expander(f"🔄 {t('alternatives')}"):
                for ad, ap in alts:
                    ac1, ac2 = st.columns([4, 1])
                    with ac1:
                        st.markdown(f"**{ad}**")
                        st.markdown(f'<div class="cbar-wrap"><div class="cbar-fill" style="width:{ap*100:.1f}%;background:{conf_color(ap)}"></div></div>', unsafe_allow_html=True)
                    with ac2:
                        st.markdown(f"**{ap:.1%}**")
                    st.markdown("---")

        st.markdown(f'<div class="disclaimer">⚠️ {t("disclaimer")}</div>', unsafe_allow_html=True)


def render_xray():
    from PIL import Image
    import numpy as np
    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.rerun()

    lang = st.session_state.get("lang", "en")
    render_page_header("📷 " + t("xray_analysis"), t("xray_card_desc"))

    st.markdown(f"""
    <div class="gen-card">
        <div class="gen-card-title">📚 {"Recommended Datasets" if lang=="en" else "अनुशंसित डेटासेट"}</div>
        <table style="width:100%;font-size:0.85rem;border-collapse:collapse">
            <tr style="background:#f1f5f9"><th style="padding:8px;text-align:left">Dataset</th><th style="padding:8px">Images</th><th style="padding:8px">Diseases</th></tr>
            <tr><td style="padding:8px">NIH ChestX-ray14 ⭐</td><td style="padding:8px;text-align:center">112,000+</td><td style="padding:8px;text-align:center">14</td></tr>
            <tr style="background:#f8fafc"><td style="padding:8px">Chest X-Ray Pneumonia 🏆</td><td style="padding:8px;text-align:center">5,863</td><td style="padding:8px;text-align:center">2 (Easiest!)</td></tr>
            <tr><td style="padding:8px">CheXpert (Stanford)</td><td style="padding:8px;text-align:center">224,316</td><td style="padding:8px;text-align:center">14</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    uploaded = st.file_uploader(t("upload_xray"), type=["jpg", "jpeg", "png"], help=t("xray_note"))

    if uploaded:
        xray_model, xray_info = load_xray_model()
        img = Image.open(uploaded).convert("RGB")

        col_img, col_res = st.columns(2)
        with col_img:
            st.image(img, caption="Uploaded X-Ray", use_container_width=True)

        with col_res:
            if st.button(t("analyze_btn"), type="primary"):
                if xray_model is None:
                    st.error("Model not found. Run train_xray.py first.")
                else:
                    with st.spinner("Analyzing..." if lang == "en" else "विश्लेषण हो रहा है..."):
                        img_resized = img.resize((150, 150))
                        arr = np.array(img_resized) / 255.0
                        arr = np.expand_dims(arr, axis=0)

                        pred = xray_model.predict(arr, verbose=0)[0][0]
                        pneumo_p = float(pred)
                        normal_p = 1 - pneumo_p

                        add_history("X-ray", "Chest X-Ray Image",
                                "Pneumonia" if pneumo_p > 0.5 else "Normal",
                                pneumo_p if pneumo_p > 0.5 else normal_p)

                    st.markdown("### 📊 Results")
                    st.markdown(f"**Normal**: {normal_p:.1%}")
                    st.progress(normal_p)
                    st.markdown(f"**Pneumonia**: {pneumo_p:.1%}")
                    st.progress(pneumo_p)

                    if pneumo_p > 0.5:
                        st.warning("⚠️ Pneumonia detected. Please consult a doctor urgently.")
                    else:
                        st.success("✅ Looks Normal. Still recommend a professional review.")
def render_chatbot():
    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.rerun()
    lang = st.session_state.get("lang", "en")
    render_page_header("💬 " + t("ai_chatbot"), t("chatbot_card_desc"))

    if "chat_hist" not in st.session_state:
        st.session_state.chat_hist = [{"role": "bot", "content": t("chatbot_hello")}]

    # Quick questions
    st.markdown(f"**{t('quick_q')}**")
    qqs_en = ["Pneumonia symptoms", "Diabetes diet", "Heart attack signs", "Fever tips", "Dengue prevention", "Stress management"]
    qqs_hi = ["निमोनिया के लक्षण", "मधुमेह आहार", "दिल का दौरा", "बुखार सुझाव", "डेंगू से बचाव", "तनाव प्रबंधन"]
    qqs = qqs_hi if lang == "hi" else qqs_en
    qcols = st.columns(6)
    for i, q in enumerate(qqs):
        with qcols[i]:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                resp = chat_response(q, lang)
                st.session_state.chat_hist.append({"role": "user", "content": q})
                st.session_state.chat_hist.append({"role": "bot", "content": resp})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)


    # Chat display
    msgs_html = ""
    for m in st.session_state.chat_hist:
        content = m["content"].replace("\n", "<br>")
        if m["role"] == "bot":
            msgs_html += f'<div class="msg-bot"><span class="msg-avatar">🤖</span> {content}</div>'
        else:
            msgs_html += f'<div class="msg-user"><span class="msg-avatar">👤</span> {content}</div>'

    st.markdown(f"""
    <div class="chat-wrap">
        <div class="chat-header">🤖 AI Health Assistant &nbsp; <span style="opacity:.7;font-size:0.75rem;font-weight:400">● Online</span></div>
        <div class="chat-messages" id="chat_msgs">{msgs_html}</div>
    </div>
    """, unsafe_allow_html=True)

    ic1, ic2, ic3 = st.columns([6, 1, 1])
    with ic1:
        user_in = st.text_input("", placeholder=t("chatbot_placeholder"), key="chat_in", label_visibility="collapsed")
    with ic2:
        if st.button(t("send"), key="chat_send", use_container_width=True):
            if user_in.strip():
                resp = chat_response(user_in, lang)
                st.session_state.chat_hist.append({"role": "user", "content": user_in})
                st.session_state.chat_hist.append({"role": "bot", "content": resp})
                st.rerun()
    with ic3:
        if st.button(t("clear_chat"), key="chat_clr", use_container_width=True):
            st.session_state.chat_hist = [{"role": "bot", "content": t("chatbot_hello")}]
            st.rerun()

    st.markdown(f'<div class="disclaimer">⚠️ {t("disclaimer")}</div>', unsafe_allow_html=True)

#New AI chatbot page with actual AI response and chat history
def render_chatbot_page():
    lang = st.session_state.get("lang", "en")

    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.rerun()

    render_page_header("💬 " + t("ai_chatbot"), t("chatbot_card_desc"))

    # ── init histories ──────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_hist" not in st.session_state:
        st.session_state.chat_hist = [{"role": "bot", "content": t("chatbot_hello")}]

    # ── quick suggestion buttons ────────────────────────────────────
    qqs_en = ["Pneumonia symptoms", "Diabetes diet", "Heart attack signs",
              "Fever tips", "Dengue prevention", "Stress management"]
    qqs_hi = ["निमोनिया के लक्षण", "मधुमेह आहार", "दिल का दौरा",
              "बुखार सुझाव", "डेंगू से बचाव", "तनाव प्रबंधन"]
    qqs = qqs_hi if lang == "hi" else qqs_en

    st.markdown(f"**{t('quick_q')}**")
    qcols = st.columns(6)
    for i, q in enumerate(qqs):
        with qcols[i]:
            if st.button(q, key=f"aiqq_{i}", use_container_width=True):
                reply, used_ai = ai_chat_response(q, lang)
                st.session_state.chat_history.append({"role": "user",    "content": q})
                st.session_state.chat_history.append({"role": "assistant","content": reply})
                st.session_state.chat_hist.append({"role": "user", "content": q})
                st.session_state.chat_hist.append({"role": "bot",  "content": reply})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── chat display ────────────────────────────────────────────────
    msgs_html = ""
    for m in st.session_state.chat_hist:
        content = m["content"].replace("\n", "<br>")
        if m["role"] == "bot":
            msgs_html += f'<div class="msg-bot"><span class="msg-avatar">🤖</span> {content}</div>'
        else:
            msgs_html += f'<div class="msg-user"><span class="msg-avatar">👤</span> {content}</div>'

    st.markdown(f"""
    <div class="chat-wrap">
        <div class="chat-header">🤖 AI Health Assistant &nbsp;
          <span style="opacity:.7;font-size:0.75rem;font-weight:400">● Online</span>
        </div>
        <div class="chat-messages">{msgs_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── input row ───────────────────────────────────────────────────
    ic1, ic2, ic3 = st.columns([6, 1, 1])
    with ic1:
        user_in = st.text_input("", placeholder=t("chatbot_placeholder"),
                                key="ai_chat_in", label_visibility="collapsed")
    with ic2:
        if st.button(t("send"), key="ai_chat_send", use_container_width=True):
            if user_in.strip():
                reply, used_ai = ai_chat_response(user_in, lang)
                st.session_state.chat_history.append({"role": "user",    "content": user_in})
                st.session_state.chat_history.append({"role": "assistant","content": reply})
                st.session_state.chat_hist.append({"role": "user", "content": user_in})
                st.session_state.chat_hist.append({"role": "bot",  "content": reply})
                if not used_ai:
                    st.caption("ℹ️ AI quota reached — using built-in responses")
                st.rerun()
    with ic3:
        if st.button(t("clear_chat"), key="ai_chat_clr", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_hist = [{"role": "bot", "content": t("chatbot_hello")}]
            st.rerun()

    st.markdown(f'<div class="disclaimer">⚠️ {t("disclaimer")}</div>',
                unsafe_allow_html=True)

def render_history():
    lang = st.session_state.get("lang", "en")
    render_page_header("🕐 " + t("history"), t("recent_predictions"))

    history = st.session_state.get("history", [])
    if not history:
        st.markdown(f"""
        <div class="gen-card" style="text-align:center;padding:3rem;color:#94a3b8">
            <div style="font-size:3rem">📋</div>
            <div style="font-size:1rem;margin-top:1rem">{t("no_history")}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    rows_html = ""
    for item in history:
        tag_cls = {"Symptoms": "tag-symptoms", "X-ray": "tag-xray", "Chatbot": "tag-chatbot"}.get(item["type"], "tag-symptoms")
        c_cls = conf_class(item["confidence"])
        rows_html += f"""<tr>
            <td><span class="{tag_cls}">{item["type"]}</span></td>
            <td>{item["input"]}</td>
            <td><b>{item["prediction"]}</b></td>
            <td><span class="{c_cls}">{item["confidence"]:.1%}</span></td>
            <td>{item["time"]}</td>
        </tr>"""

    st.markdown(f"""
    <div class="gen-card" style="padding:0;overflow:hidden">
        <table class="pred-table">
            <thead><tr>
                <th>{t("type")}</th><th>{t("input")}</th>
                <th>{t("prediction")}</th><th>{t("confidence")}</th><th>{t("time")}</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear History" if lang == "en" else "🗑️ इतिहास साफ करें"):
        st.session_state.history = []
        st.rerun()


def render_tips():
    lang = st.session_state.get("lang", "en")
    render_page_header("💚 " + t("health_tips"), t("health_tip_title"))

    for tip in HEALTH_TIPS:
        title = tip["title_hi"] if lang == "hi" else tip["title"]
        text = tip["tip_hi"] if lang == "hi" else tip["tip"]
        st.markdown(f"""
        <div class="tip-card">
            <div class="tip-icon">{tip["icon"]}</div>
            <div>
                <div class="tip-title">{title}</div>
                <div class="tip-text">{text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="gen-card" style="margin-top:1rem;background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-color:#86efac">
        <div style="font-weight:700;margin-bottom:0.5rem">🆘 {"Emergency Contacts (India)" if lang=="en" else "आपातकालीन नंबर (भारत)"}</div>
        <div style="font-size:0.88rem;line-height:2">
        🚑 <b>{"Ambulance" if lang=="en" else "एम्बुलेंस"}</b>: 108 &nbsp;&nbsp;
        🚔 <b>{"Police" if lang=="en" else "पुलिस"}</b>: 100 &nbsp;&nbsp;
        🧠 <b>{"Mental Health" if lang=="en" else "मानसिक स्वास्थ्य"}</b>: 1800-599-0019 &nbsp;&nbsp;
        🩺 <b>{"Health Helpline" if lang=="en" else "स्वास्थ्य हेल्पलाइन"}</b>: 1800-180-1104
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_about():
    lang = st.session_state.get("lang", "en")
    render_page_header("ℹ️ " + t("about"), "AI Healthcare Assistant v3.0")

    if lang == "hi":
        st.markdown("""
        ## 🏥 AI रोग पूर्वानुमान के बारे में

        यह एक **मशीन लर्निंग आधारित** वेब एप्लिकेशन है।

        ### 📊 तकनीकी विवरण
        | विशेषता | विवरण |
        |---------|-------|
        | **एल्गोरिदम** | Random Forest Classifier |
        | **कुल रोग** | 41 |
        | **कुल लक्षण** | 132 |
        | **प्रशिक्षण डेटा** | 4,920 रिकॉर्ड |
        | **भाषाएं** | हिंदी + अंग्रेजी |

        ### 🆕 v3.0 नई विशेषताएं
        1. ✅ डैशबोर्ड UI (छवि जैसा)
        2. ✅ हिंदी भाषा समर्थन
        3. ✅ सभी 41 रोगों की जानकारी
        4. ✅ AI चैटबॉट
        5. ✅ X-Ray विश्लेषण
        6. ✅ भविष्यवाणी इतिहास
        7. ✅ डार्क मोड
        8. ✅ स्वास्थ्य सुझाव

        ### ⚠️ अस्वीकरण
        यह केवल शैक्षिक उद्देश्यों के लिए है।
        """)
    else:
        st.markdown("""
        ## 🏥 About AI Healthcare Assistant

        An **ML-powered** multi-feature healthcare web application with modern dashboard UI.

        ### 📊 Technical Specifications
        | Feature | Detail |
        |---------|--------|
        | **Algorithm** | Random Forest Classifier |
        | **Diseases** | 41 unique conditions |
        | **Symptoms** | 132 input features |
        | **Training Data** | 4,920 patient records |
        | **Languages** | English + Hindi |
        | **Framework** | Streamlit |

        ### 🆕 What's New in v3.0
        1. ✅ **Modern Dashboard UI** (matching reference design)
        2. ✅ **Hindi language** full UI translation
        3. ✅ **All 41 diseases** have detailed info
        4. ✅ **AI Chatbot** with 12+ health topics
        5. ✅ **X-Ray Analysis** module
        6. ✅ **Prediction History** tracking
        7. ✅ **Dark Mode** support
        8. ✅ **Health Tips** section with emergency contacts
        9. ✅ **Alternative diagnoses** shown
        10. ✅ **Severity badges** for all diseases

        ### 📊 Recommended X-Ray Datasets
        | Dataset | Link |
        |---------|------|
        | Chest X-Ray Pneumonia (Easiest) | kaggle.com/paultimothymooney |
        | NIH ChestX-ray14 | kaggle.com/nih-chest-xrays |
        | CheXpert Stanford | aimi.stanford.edu |

        ### ⚠️ Disclaimer
        For educational purposes only. Always consult a licensed medical professional.
        """)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    # Init session state 
    defaults = {"lang": "en", "page": "dashboard", "dark_mode": False,
                "history": [], "syms": [], "chat_hist": [], "dash_chat_hist": []}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    dark = st.session_state.get("dark_mode", False)
    inject_css(dark)

    model, le, model_cols, display_cols = load_model()

    render_sidebar()

    page = st.session_state.get("page", "dashboard")

    if page == "dashboard":
        render_dashboard(model, le, model_cols, display_cols)
    elif page == "predict":
        render_predict(model, le, model_cols, display_cols)
    elif page == "xray":
        render_xray()
    elif page == "chatbot":
        render_chatbot_page()
    elif page == "history":
        render_history()
    elif page == "tips":
        render_tips()
    elif page == "about":
        render_about()


if __name__ == "__main__":
    main()