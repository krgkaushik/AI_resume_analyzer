AI Resume Analyzer
A complete Machine Learning pipeline + Streamlit web app that automatically classifies resumes into job categories using NLP, TF-IDF, and Logistic Regression.

✨ Features
Text Cleaning & Preprocessing: Removes URLs, emails, special characters, and normalizes text.

TF-IDF Vectorization: Converts resumes into numerical features (5,000 max features).

Multi-class Classification: Predicts one of 25+ job categories.

Confidence Scoring: Shows prediction probability.

Streamlit Web App: User-friendly interface for uploading and analyzing resumes.

Model Persistence: Pre-trained models saved as .pkl files.

Jupyter Notebook: Full exploratory analysis and training pipeline.

📊 Dataset
File: Resume.csv

Source: Kaggle Resume Dataset

Columns:

ID

Category (target: e.g., HR, Java Developer, Data Science, etc.)

Resume_str (raw text)

Resume_html (dropped during processing)

textAI-Resume-Analyzer/
├── app.py                    # Streamlit web application
├── ai_resume_analyzer.ipynb  # Full ML pipeline & EDA
├── Resume.csv                # Raw dataset
├── data/                     # Additional data (if any)
├── requirements.txt          # Python dependencies
├── resume_vectorizer.pkl     # Saved TF-IDF vectorizer
├── resume_classifier.pkl     # Trained Logistic Regression model
├── README.md
└── (optional) notebooks/ or scripts/
🚀 Installation
Clone the repository
Bash
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer
Create a virtual environment (recommended)
Bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
Install dependencies
Bash
pip install -r requirements.txt
📋 Requirements (requirements.txt)
Plaintext
streamlit
pandas
scikit-learn
numpy
joblib
💻 Usage
1. Run the Streamlit App (Recommended)
Bash
streamlit run app.py
Then open your browser at the URL shown (usually http://localhost:8501). In the app you can:

Paste resume text.

Upload resume file (.txt, .pdf, etc.).

Get instant category prediction + confidence score.

2. Train / Retrain the Model
Open ai_resume_analyzer.ipynb and run all cells, or use the training code snippets.

3. Test Model via Script
Python
import pickle
import re
# (paste the test function from your code)
🔬 How It Works
Pipeline Steps
Data Loading → pd.read_csv('Resume.csv')

Cleaning: Lowercase, remove URLs/emails, remove special characters, normalize whitespace.

Feature Engineering → TfidfVectorizer(max_features=5000, stop_words='english')

Model Training → LogisticRegression(max_iter=1000)

Evaluation → Accuracy + Classification Report.

Saving → Pickle both vectorizer and model.

Inference → Clean → Vectorize → Predict + Probability.

📈 Model Performance
Accuracy: ~XX.XX% (check notebook for latest)

Multi-class support across 25+ categories.

🧪 Technologies Used
Python

pandas — Data handling

scikit-learn — TF-IDF + Logistic Regression

Streamlit — Web interface

Jupyter Notebook — Experimentation

pickle — Model serialization

🤝 Contributing
Contributions are welcome! Feel free to:

Improve model accuracy (try XGBoost, BERT, etc.).

Enhance the Streamlit UI.

Add PDF parsing support.

Add more evaluation metrics.

📄 License
MIT License — feel free to use this project for learning or commercial purposes.
