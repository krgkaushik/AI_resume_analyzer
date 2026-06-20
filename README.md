# AI Resume Analyzer

A complete **Machine Learning pipeline** and **Streamlit web application** that automatically analyzes and classifies resumes into job categories using **Natural Language Processing (NLP)**, **TF-IDF Vectorization**, and **Logistic Regression**.

---

## ✨ Features

* **Text Cleaning & Preprocessing**

  * Removes URLs, email addresses, special characters, and extra whitespace.
  * Converts text to a normalized format for better model performance.

* **TF-IDF Vectorization**

  * Transforms resume text into numerical feature vectors.
  * Uses up to **5,000 features** for efficient representation.

* **Multi-Class Resume Classification**

  * Predicts one of **25+ job categories** such as:

    * Data Science
    * HR
    * Java Developer
    * Python Developer
    * Testing
    * Business Analyst
    * And more.

* **Confidence Score**

  * Displays prediction probability to indicate model confidence.

* **Interactive Streamlit Web App**

  * User-friendly interface for uploading or pasting resume content.
  * Provides instant category predictions.

* **Model Persistence**

  * Pre-trained TF-IDF vectorizer and Logistic Regression model are saved as `.pkl` files for quick deployment.

* **Complete ML Pipeline**

  * Includes data preprocessing, feature engineering, model training, evaluation, and deployment.

---

## 📊 Dataset

**Dataset:** `Resume.csv`

### Dataset Columns

| Column      | Description                                           |
| ----------- | ----------------------------------------------------- |
| ID          | Unique identifier                                     |
| Category    | Target job category                                   |
| Resume_str  | Raw resume text                                       |
| Resume_html | HTML version of resume (removed during preprocessing) |

### Target Categories

The dataset contains resumes from **25+ professional domains**, enabling robust multi-class classification.

---

## 📁 Project Structure

```text
AI-Resume-Analyzer/
│
├── app.py                     # Streamlit web application
├── ai_resume_analyzer.ipynb   # ML pipeline, EDA, and training
├── Resume.csv                 # Dataset
├── data/                      # Additional data files
├── requirements.txt           # Project dependencies
├── resume_vectorizer.pkl      # Saved TF-IDF vectorizer
├── resume_classifier.pkl      # Trained Logistic Regression model
├── README.md
└── notebooks/ (optional)
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

#### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📋 Requirements

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

Install manually if needed:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

---

## 💻 Usage

### Run the Streamlit Application

```bash
streamlit run app.py
```

Open the generated local URL (typically):

```text
http://localhost:8501
```

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
