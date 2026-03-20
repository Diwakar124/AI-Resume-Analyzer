import streamlit as st
import pickle
import docx
import PyPDF2
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer Pro", page_icon="🚀", layout="wide")

# -------------------- CSS --------------------
st.markdown("""
<style>
.main {background-color:#0e1117;color:white;}
h1,h2,h3 {color:#00f2ff;text-align:center;}
.stButton>button {background:#00f2ff;color:black;border-radius:10px;padding:10px;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<h1>🚀 AI Resume Analyzer</h1>
<p style='text-align:center;'>Analyze Resume + ATS Score + Job Match</p>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    svc_model = pickle.load(open('clf.pkl','rb'))
    tfidf = pickle.load(open('tfidf.pkl','rb'))
    le = pickle.load(open('encoder.pkl','rb'))
    return svc_model, tfidf, le

svc_model, tfidf, le = load_models()

# -------------------- HELPER FUNCTIONS --------------------
def clean_resume(txt):
    txt = re.sub('http\\S+\\s',' ',txt)
    txt = re.sub('[%s]' % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), ' ', txt)
    txt = re.sub('\\s+', ' ', txt)
    return txt

def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    text = ""
    if ext == 'pdf':
        reader = PyPDF2.PdfReader(file)
        for p in reader.pages:
            text += p.extract_text() or ""
    elif ext == 'docx':
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + '\n'
    elif ext == 'txt':
        text = file.read().decode('utf-8', errors='ignore')
    return text

def predict_category(text):
    vec = tfidf.transform([clean_resume(text)]).toarray()
    return le.inverse_transform(svc_model.predict(vec))[0]

def extract_skills(text):
    skills_db = ["python","java","c++","sql","machine learning","deep learning",
                 "data analysis","pandas","numpy","tensorflow","keras",
                 "excel","power bi","tableau","aws","html","css","javascript"]
    text = text.lower()
    return [s for s in skills_db if s in text]

def ats_score(text):
    keywords = ["experience","project","skills","education","internship","certification","achievement"]
    score = sum(1 for k in keywords if k in text.lower())
    return int((score/len(keywords))*100)

def recommend_role(skills):
    if "machine learning" in skills:
        return "Machine Learning Engineer"
    elif "python" in skills and "sql" in skills:
        return "Data Analyst"
    elif "html" in skills and "css" in skills:
        return "Frontend Developer"
    return "General Software Role"

def match_resume_job(resume, job_desc):
    vec = TfidfVectorizer(stop_words='english').fit_transform([resume, job_desc])
    return round(cosine_similarity(vec[0:1], vec[1:2])[0][0]*100, 2)

def missing_keywords(resume, job_desc):
    return list(set(job_desc.lower().split()) - set(resume.lower().split()))[:20]

def improve_resume(resume_text, job_desc):
    missing = missing_keywords(resume_text, job_desc)
    suggestions = f"""
Add keywords: {', '.join(missing[:15])}
Include relevant projects related to job description
Highlight experience using these keywords
Add measurable achievements
"""
    return suggestions

def generate_pdf(category, skills, score, role, match):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    style = getSampleStyleSheet()
    content = [
        Paragraph(f"<b>Category:</b> {category}", style["Normal"]),
        Paragraph(f"<b>Skills:</b> {', '.join(skills)}", style["Normal"]),
        Paragraph(f"<b>ATS Score:</b> {score}%", style["Normal"]),
        Paragraph(f"<b>Recommended Role:</b> {role}", style["Normal"])
    ]
    if match is not None:
        content.append(Paragraph(f"<b>Match Score:</b> {match}%", style["Normal"]))
    doc.build(content)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# -------------------- INPUT --------------------
tab1, tab2 = st.tabs(["📂 Upload Resume", "📌 Job Description"])

with tab1:
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf","docx","txt"])

with tab2:
    job_desc = st.text_area("Paste Job Description", height=200)

# -------------------- OUTPUT --------------------
if uploaded_file:
    text = extract_text(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume Content")
        st.text_area("", text, height=400)

    with col2:
        st.subheader("📊 Analysis Results")

        category = predict_category(text)
        skills = extract_skills(text)
        score = ats_score(text)
        role = recommend_role(skills)

        st.success(f"🎯 Category: {category}")
        st.write("🛠 Skills:", skills)
        st.progress(score)
        st.write(f"ATS Score: {score}%")
        st.success(f"💼 Recommended Role: {role}")

        match_score = None
        if job_desc:
            match_score = match_resume_job(text, job_desc)
            st.progress(int(match_score))
            st.write(f"Match Score: {match_score}%")
            st.write("⚠️ Missing Keywords:", missing_keywords(text, job_desc))
            if match_score < 60:
                st.warning(improve_resume(text, job_desc))

        # -------------------- PDF DOWNLOAD --------------------
        pdf_bytes = generate_pdf(category, skills, score, role, match_score)
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name="resume_analysis.pdf",
            mime="application/pdf"
        )

import os
import joblib

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'clf_compressed.pkl')

svc_model = joblib.load(model_path)
# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<center>Made by Diwakar 🚀</center>", unsafe_allow_html=True)
