# AI Resume Analyzer

A complete **Machine Learning** pipeline and **Streamlit** web application that automatically classifies resumes into appropriate job categories using NLP and TF-IDF.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## ✨ Features

- Text cleaning and preprocessing (removes URLs, emails, special characters)
- TF-IDF vectorization with 5000 features
- Multi-class classification using Logistic Regression
- Confidence score for each prediction
- Streamlit web interface for easy use
- Pre-trained model and vectorizer included
- Full Jupyter notebook for EDA and training

## 📊 Dataset

- **File**: `Resume.csv`
- Contains resumes from multiple domains (HR, IT, Finance, Data Science, etc.)
- Target variable: `Category`

## 🛠 Project Structure
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

##🚀 Installation

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
