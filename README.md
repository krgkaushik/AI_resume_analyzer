Markdown# 📄 AI Resume Industry Categorizer

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline built to analyze raw text from resume documents and automatically classify them into their respective industry domains (e.g., Information Technology, Finance, Sales, Advocate). This project features a clean, interactive user dashboard built with Streamlit.

## 🗂️ Project Directory Structure

```text
AI_resume_analyzer/
├── data/
│   └── Resume.csv             # Raw resume dataset (2,484 profiles)
├── models/
│   ├── resume_vectorizer.pkl  # Serialized TF-IDF text vectorizer
│   └── resume_classifier.pkl  # Serialized Logistic Regression model
├── notebooks/
│   └── exploration.ipynb      # Exploratory Data Analysis & prototyping
├── src/
│   ├── __init__.py
│   └── train.py               # Main model training & engineering pipeline
├── app.py                     # Streamlit frontend web application
├── requirements.txt           # Python application dependencies
├── .gitignore                 # Specially ignored untracked files
└── README.md                  # System documentation
⚙️ How the Pipeline WorksExploratory Data Analysis (EDA): Filters noise, removes layout components (Resume_html), and analyzes structural class balance across 24 distinct professional categories.NLP Preprocessing: Text optimization using Regular Expressions (re) to clean text streams by stripping URLs, emails, special punctuation symbols, and numerical digits, followed by uniform lowercasing.Feature Extraction: Implements a text-to-numerical TfidfVectorizer mapping the top 5,000 predictive word tokens while filtering standard English stop words.Machine Learning Inference: Trains a regularized LogisticRegression classifier on a stratified train-test split ($80/20$).Model Serialization: Exports operational components to static Pickle binary matrices (.pkl) for immediate server deployment.🚀 Local Installation & SetupFollow these setup steps to deploy the application on your computer:1. Clone the RepositoryBashgit clone [https://github.com/krgkaushik/AI_resume_analyzer.git](https://github.com/krgkaushik/AI_resume_analyzer.git)
cd AI_resume_analyzer
2. Install Project DependenciesEnsure you have Python installed, then run:Bashpip install -r requirements.txt
3. Run the Machine Learning Training PipelineIf you wish to rebuild the model and generate fresh .pkl files inside the models/ folder, run:Bashpython src/train.py
4. Launch the Streamlit Web DashboardBoot up the interactive browser-based UI wrapper using the following command:Bashstreamlit run app.py
Open http://localhost:8501 in your web browser to upload resumes and test classifications in real-time.📊 Requirements & Core PackagesThe core libraries utilized to construct this text analytical system include:streamlitscikit-learnpandasnumpy