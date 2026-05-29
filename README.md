# 📄 AI Resume Analyzer & Classification Pipeline

An end-to-end Machine Learning and NLP pipeline that reads raw resume text data, preprocesses it, and classifies the profile into specific industry categories (e.g., Information Technology, Finance, Sales) using Scikit-Learn.

## 🚀 Project Workflow
1. **Data Engineering & EDA:** Cleaning structural columns and filtering noise from a dataset of 2,484 resumes.
2. **NLP Text Preprocessing:** Using regular expressions to remove URLs, emails, and special characters; converting text to lowercase; and filtering out English stopwords.
3. **Feature Extraction:** Applying a `TfidfVectorizer` to extract the top 5,000 text features.
4. **Machine Learning Classification:** Training a `LogisticRegression` model to achieve a predictive accuracy baseline.
5. **Serialization:** Exporting the operational vectorizer and classifier to reusable `.pkl` (Pickle) files for instant downstream web app integration.

## 📦 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/ai-resume-analyzer.git](https://github.com/YOUR_USERNAME/ai-resume-analyzer.git)
cd ai-resume-analyzer
