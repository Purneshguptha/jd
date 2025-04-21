import streamlit as st
import os
import tempfile
import PyPDF2
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class ResumeJobMatcher:
    def __init__(self):
        # Initialize the BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        # Load spaCy model for NER
        self.nlp = spacy.load('en_core_web_sm')
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def preprocess_text(self, text):
        """Clean and preprocess the text."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def get_embeddings(self, text):
        """Generate embeddings for the given text using BERT."""
        # Tokenize and get model inputs
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings[0].numpy()
    
    def calculate_similarity(self, resume_embedding, jd_embedding):
        """Calculate cosine similarity between resume and job description embeddings."""
        similarity = np.dot(resume_embedding, jd_embedding) / (
            np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding)
        )
        return similarity
    
    def extract_key_skills(self, text):
        """Extract key skills and technical terms from text."""
        doc = self.nlp(text)
        # Extract named entities and noun phrases
        skills = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                skills.append(ent.text.lower())
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to phrases of 3 words or less
                skills.append(chunk.text.lower())
        
        return list(set(skills))
    
    def match_resume_with_jd(self, resume_path, jd_path):
        """Main function to match resume with job description."""
        # Extract text from PDFs
        resume_text = self.extract_text_from_pdf(resume_path)
        jd_text = self.extract_text_from_pdf(jd_path)
        
        # Preprocess texts
        processed_resume = self.preprocess_text(resume_text)
        processed_jd = self.preprocess_text(jd_text)
        
        # Generate embeddings
        resume_embedding = self.get_embeddings(processed_resume)
        jd_embedding = self.get_embeddings(processed_jd)
        
        # Calculate similarity score
        similarity_score = self.calculate_similarity(resume_embedding, jd_embedding)
        
        # Extract key skills
        resume_skills = self.extract_key_skills(resume_text)
        jd_skills = self.extract_key_skills(jd_text)
        
        # Find matching and missing skills
        matching_skills = set(resume_skills) & set(jd_skills)
        missing_skills = set(jd_skills) - set(resume_skills)
        
        return {
            'similarity_score': similarity_score,
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills)
        }

# Streamlit UI
st.set_page_config(
    page_title="Resume-JD Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("Resume and Job Description Matcher")
st.write("Upload your resume and job description to see how well they match!")

# Initialize the matcher
@st.cache_resource
def load_matcher():
    return ResumeJobMatcher()

matcher = load_matcher()

# Create two columns for file uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Choose your resume", type=['pdf'])

with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Choose the job description", type=['pdf'])

if resume_file and jd_file:
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_resume:
        temp_resume.write(resume_file.getvalue())
        resume_path = temp_resume.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_jd:
        temp_jd.write(jd_file.getvalue())
        jd_path = temp_jd.name

    if st.button("Analyze Match"):
        with st.spinner("Analyzing..."):
            # Get matching results
            result = matcher.match_resume_with_jd(resume_path, jd_path)
            
            # Display results
            st.header("Matching Results")
            
            # Display similarity score with a progress bar
            similarity_percentage = result['similarity_score'] * 100
            st.subheader("Overall Match Score")
            st.progress(result['similarity_score'])
            st.write(f"{similarity_percentage:.2f}% Match")

            # Create two columns for skills
            skills_col1, skills_col2 = st.columns(2)

            with skills_col1:
                st.subheader("Matching Skills")
                if result['matching_skills']:
                    for skill in result['matching_skills']:
                        st.write(f"✅ {skill}")
                else:
                    st.write("No matching skills found")

            with skills_col2:
                st.subheader("Missing Skills")
                if result['missing_skills']:
                    for skill in result['missing_skills']:
                        st.write(f"❌ {skill}")
                else:
                    st.write("No missing skills found")

        # Clean up temporary files
        os.unlink(resume_path)
        os.unlink(jd_path)

else:
    st.info("Please upload both resume and job description files to begin analysis.")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Transformers") 