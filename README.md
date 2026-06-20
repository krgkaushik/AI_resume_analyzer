# AI Resume Analyzer

A complete **Machine Learning pipeline** + **Streamlit web app** that automatically classifies resumes into job categories using NLP and TF-IDF + Logistic Regression.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

## ✨ Features

- **Text Cleaning & Preprocessing**: Removes URLs, emails, special characters, and normalizes text
- **TF-IDF Vectorization**: Converts resumes into numerical features (5,000 max features)
- **Multi-class Classification**: Predicts one of 25+ job categories
- **Confidence Scoring**: Shows prediction probability
- **Streamlit Web App**: User-friendly interface for uploading and analyzing resumes
- **Model Persistence**: Pre-trained models saved as `.pkl` files
- **Jupyter Notebook**: Full exploratory analysis and training pipeline

## 📊 Dataset

- **File**: `Resume.csv`
- **Source**: Kaggle Resume Dataset
- **Columns**:
  - `ID`
  - `Category` (target: e.g., HR, Java Developer, Data Science, etc.)
  - `Resume_str` (raw text)
  - `Resume_html` (dropped during processing)

**Category Distribution** (example from training):

```python
# See notebook for full distribution
🛠️ Project Structure
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

Clone the repositoryBashgit clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer
Create a virtual environment (recommended)Bashpython -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
Install dependenciesBashpip install -r requirements.txt

📋 Requirements (requirements.txt)
txtstreamlit
pandas
scikit-learn
numpy
joblib  # or pickle
💻 Usage
1. Run the Streamlit App (Recommended)
Bashstreamlit run app.py
Then open your browser at the URL shown (usually http://localhost:8501).
In the app you can:

Paste resume text
Upload resume file (.txt, .pdf, etc. — depending on app implementation)
Get instant category prediction + confidence score

2. Train / Retrain the Model
Open ai_resume_analyzer.ipynb and run all cells, or use the training code snippets.
3. Test Model via Script
Pythonpython -c "
import pickle
import re
# (paste the test function from your code)
"
Example Predictions
Input (IT Resume snippet):
Senior Full Stack Engineer with 5 years of experience...
Output:

Predicted Class: Information Technology
Confidence: 94.8%

Input (Finance Resume snippet):
Certified Public Accountant (CPA) managing financial ledgers...
Output:

Predicted Class: Finance
Confidence: 96.2%

🔬 How It Works
Pipeline Steps

Data Loading → pd.read_csv('Resume.csv')
Cleaning:
Lowercase
Remove URLs, emails
Remove special characters
Normalize whitespace

Feature Engineering → TfidfVectorizer(max_features=5000, stop_words='english')
Model Training → LogisticRegression(max_iter=1000)
Evaluation → Accuracy + Classification Report
Saving → Pickle both vectorizer and model
Inference → Clean → Vectorize → Predict + Probability

📈 Model Performance
(From training run)

Accuracy: ~XX.XX% (check notebook for latest)
Multi-class support across 25+ categories

🧪 Technologies Used

Python
pandas — Data handling
scikit-learn — TF-IDF + Logistic Regression
Streamlit — Web interface
Jupyter Notebook — Experimentation
pickle — Model serialization

🤝 Contributing
Contributions are welcome! Feel free to:

Improve model accuracy (try XGBoost, BERT, etc.)
Enhance the Streamlit UI
Add PDF parsing support
Add more evaluation metrics

📄 License
MIT License — feel free to use this project for learning or commercial purposes.
