# Resume-JD Matcher

A Streamlit application that matches resumes with job descriptions using NLP and transformer models to calculate similarity scores and identify matching/missing skills.

## Features

- PDF file upload for both resume and job description
- Overall similarity score calculation
- Identification of matching skills
- List of missing skills from the job description
- User-friendly interface

## Setup

1. Clone this repository
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Running the Application

To run the Streamlit app:
```bash
python -m streamlit run app.py
```

The application will open in your default web browser. Upload your resume and job description PDFs to see the matching results.

## Usage

1. Upload your resume (PDF format)
2. Upload the job description (PDF format)
3. Click "Analyze Match" to see the results
4. Review the similarity score and skills analysis

## Note

- Only PDF files are supported
- The analysis may take a few moments depending on the file sizes
- Make sure both files are properly formatted and readable 