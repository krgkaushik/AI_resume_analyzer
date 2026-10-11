# AI Resume ATS Analyzer

A lightweight AI-powered resume screening app that compares a candidate resume against a job description and calculates an ATS compatibility score. The application uses a saved scikit-learn model and TF-IDF vectorizer to estimate keyword overlap, similarity, and predicted industry confidence.

## Features

- Paste resume text and job description text into a browser form
- Calculate ATS score and match percentage in real time
- Highlight matched and missing keywords
- Predict the likely industry category based on resume content
- Show model confidence for the prediction
- Support PDF resume upload via the backend API
- Serve a simple frontend with a FastAPI backend

## Technology stack

- Python
- FastAPI
- Uvicorn
- scikit-learn
- NumPy
- pandas
- PyPDF2
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
├── Resume.csv              # Root-level CSV file for reference
└── .venv                   # Local virtual environment
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

## API endpoints

- `GET /` — serves the frontend page
- `POST /api/score` — accepts JSON with `resumeText` and `jdText`
- `POST /api/upload-pdf` — accepts a PDF file upload and extracts text
- `GET /health` — health check endpoint

## Notes

- The app expects the model files `resume_vectorizer.pkl` and `resume_classifier.pkl` to be present in the project root.
- If those files are missing, regenerate or restore them before running the service.
- This project is intended as a resume-to-job matching prototype and can be extended with a full ML training pipeline, database storage, or resume upload support.

## License

This project is provided for educational and demo purposes.


### Features Available in the App

* Paste resume text directly.
* Upload resume files.
* Predict job category instantly.
* View confidence score for predictions.

---

## 🔬 Machine Learning Pipeline

### 1. Data Loading

```python
df = pd.read_csv("Resume.csv")
```

### 2. Text Preprocessing

The resumes undergo several preprocessing steps:

* Convert text to lowercase.
* Remove URLs.
* Remove email addresses.
* Remove special characters and punctuation.
* Normalize whitespace.

### 3. Feature Engineering

```python
TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
```

TF-IDF converts textual resume data into numerical vectors suitable for machine learning.

### 4. Model Training

```python
LogisticRegression(max_iter=1000)
```

The Logistic Regression model is trained on the TF-IDF feature vectors for multi-class classification.

### 5. Model Evaluation

Performance is evaluated using:

* Accuracy Score
* Classification Report
* Precision
* Recall
* F1-Score

### 6. Model Saving

```python
pickle.dump(vectorizer, open("resume_vectorizer.pkl", "wb"))
pickle.dump(model, open("resume_classifier.pkl", "wb"))
```

### 7. Inference Pipeline

```text
Resume Text
      ↓
Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Classification Model
      ↓
Predicted Category + Confidence Score
```

---

## 📈 Model Performance

| Metric        | Value                             |
| ------------- | --------------------------------- |
| Accuracy      | Check notebook for latest results |
| Categories    | 25+                               |
| Algorithm     | Logistic Regression               |
| Vectorization | TF-IDF                            |

---

## 🧪 Technologies Used

* **Python**
* **Pandas** – Data Manipulation
* **NumPy** – Numerical Computing
* **Scikit-Learn** – Machine Learning
* **TF-IDF Vectorization**
* **Logistic Regression**
* **Streamlit** – Web Application
* **Jupyter Notebook** – Development & Experimentation
* **Pickle / Joblib** – Model Serialization

---

## 🔮 Future Enhancements

Potential improvements include:

* Implementing **XGBoost**, **Random Forest**, or **LightGBM**.
* Using transformer-based models such as **BERT**, **RoBERTa**, or **DistilBERT**.
* Adding resume ranking and ATS score generation.
* Supporting PDF and DOCX parsing.
* Deploying the application on Streamlit Cloud or AWS.
* Adding advanced visualization dashboards.
* Improving prediction confidence calibration.

---

## 🤝 Contributing

Contributions are welcome.

You can contribute by:

* Improving model performance.
* Enhancing UI/UX.
* Adding new resume categories.
* Supporting additional file formats.
* Implementing deep learning models.

### Steps

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit and push.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project for educational, personal, or commercial purposes.

---

## 👨‍💻 Author

**Kaushik Gunjkar**

Machine Learning & Data Science Enthusiast

If you found this project useful, consider giving it a ⭐ on GitHub.

