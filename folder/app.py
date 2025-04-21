
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load NLP and embedding model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = "".join([page.get_text() for page in doc])
    return text.strip()

def extract_skills(text):
    doc = nlp(text)
    skills = set()
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 2:
            skills.add(chunk.text.strip().lower())
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "SKILL", "WORK_OF_ART"]:
            skills.add(ent.text.strip().lower())
    return sorted(skills)

st.title("📄 Resume vs JD Matcher")
st.write("Upload a resume and a job description (PDF) to find semantic similarity and skill matches.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if resume_file and jd_file:
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, jd_text])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)
    bert_score = util.cos_sim(embeddings[0], embeddings[1]).item()

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    matched_skills = set(resume_skills).intersection(set(jd_skills))

    st.subheader("🔍 Similarity Scores")
    st.write(f"**TF-IDF Similarity:** {tfidf_score:.4f}")
    st.write(f"**BERT Similarity:** {bert_score:.4f}")

    st.subheader("🧠 Skill/Keyword Matching")
    st.write(f"**Resume Skills ({len(resume_skills)}):**", resume_skills)
    st.write(f"**JD Skills ({len(jd_skills)}):**", jd_skills)
    st.write(f"**Matched Skills ({len(matched_skills)}):**", sorted(matched_skills))

    st.success("✅ Analysis Complete!")
