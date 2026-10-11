# AI Resume ATS Analyzer

A lightweight AI-powered resume screening app that compares a candidate resume against a job description and calculates an ATS compatibility score. The application uses a saved scikit-learn model and TF-IDF vectorizer to estimate keyword overlap, similarity, and predicted industry confidence.

## Features

- Paste resume text and job description text into a browser form
- Calculate ATS score and match percentage in real time
- Highlight matched and missing keywords
- Predict the likely industry category based on resume content
- Show model confidence for the prediction
- Serve a simple frontend with a FastAPI backend

## Technology stack

- Python
- FastAPI
- Uvicorn
- scikit-learn
- NumPy
- pandas
- HTML/CSS/JavaScript

## Project structure

```text
Ai_resume_analyzer/
├── app.py                  # FastAPI backend and scoring logic
├── static/
│   ├── app.js              # Frontend logic for API calls
│   ├── index.html          # Resume analysis UI
│   └── styles.css          # Styling for the dashboard
├── data/
│   └── Resume.csv          # Resume dataset used for model development
├── README.md               # Project overview and setup guide
├── requirements.txt        # Python dependencies
├── resume_vectorizer.pkl   # Serialized TF-IDF vectorizer
├── resume_classifier.pkl   # Serialized trained classifier
├── ai_resume_analyzer.ipynb # Notebook for experimentation / model work
└── Resume.csv              # Root-level CSV file for reference
```

## How the scoring works

The backend performs the following checks:

1. Cleans and normalizes the resume and job description text
2. Extracts relevant keyword tokens from both inputs
3. Compares keyword overlap to identify matched and missing terms
4. Uses TF-IDF similarity to estimate text similarity between resume and JD
5. Uses the saved classifier to estimate the likely industry and confidence
6. Combines those signals into an ATS score

## Local setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Ai_resume_analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open the app in your browser at:

```text
http://localhost:8000
```

## Notes

- The app expects the model files `resume_vectorizer.pkl` and `resume_classifier.pkl` to be present in the project root.
- If those files are missing, regenerate or restore them before running the service.
- This project is intended as a resume-to-job matching prototype and can be extended with a full ML training pipeline, database storage, or resume upload support.

## License

This project is provided for educational and demo purposes.
