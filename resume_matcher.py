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

def main():
    # Initialize the matcher
    matcher = ResumeJobMatcher()
    
    # Match resume with job description
    result = matcher.match_resume_with_jd('up1.pdf', 'jjd.pdf')
    
    # Print results
    print("\n=== Resume Matching Results ===")
    print(f"\nSimilarity Score: {result['similarity_score']*100:.2f}%")
    
    print("\nMatching Skills:")
    for skill in result['matching_skills']:
        print(f"- {skill}")
    
    print("\nMissing Skills (from JD):")
    for skill in result['missing_skills']:
        print(f"- {skill}")

if __name__ == "__main__":
    main() 