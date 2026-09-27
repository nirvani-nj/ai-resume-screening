# 🤖 AI Resume Screening & Candidate Ranking System

An AI-powered application that ranks resumes against a job description using NLP, Machine Learning, and semantic similarity.

🌐 **Live Demo:** https://ai-resume-screening-nj.streamlit.app/

---

## ✨ Features

- 📄 Upload multiple resumes
- 💼 Paste a job description
- 🤖 Resume categorization using Machine Learning (SVM, Random Forest, Naive Bayes compared; best model auto-selected)
- 🎯 Candidate ranking based on similarity, skills, and experience
- 🧠 Optional Sentence-BERT semantic similarity
- 🔍 Explainable scores with matched and missing skills
- 🎛️ Adjustable scoring weights, required skills, minimum experience, and shortlist threshold
- 📊 Interactive Streamlit dashboard

---

## ⚙️ How Ranking Works

```
Similarity = 0.6 × BERT + 0.4 × TF-IDF   (TF-IDF only when BERT is off)

Score = 0.5 × Similarity + 0.3 × Skill Match + 0.2 × Experience
```

Weights are adjustable in the app.

---

## 🛠 Tech Stack

**Machine Learning**
- Scikit-learn
- TF-IDF
- Linear SVM
- Sentence-BERT (all-MiniLM-L6-v2)

**Libraries**
- Streamlit
- Pandas
- NumPy
- Plotly

---

## 🏗️ System Architecture

<p align="center">
  <img src="images/architecture.png" width="900">
</p>

---

## 📊 Model Performance

Resume categorization across 25 job categories (stratified 80:20 split):

| Model | Accuracy |
|-------|---------:|
| **Linear SVM** | **88.24%** |
| Random Forest | 82.35% |
| Naive Bayes | 41.18% |

---

## 📂 Dataset

The training dataset is **not included** in this repository.

Download it from:

https://www.kaggle.com/datasets/serkanp/resume-screening/data

Place `Resume.csv` inside:

```
data/raw/
```

to retrain the model.

---

## 🚀 Run Locally

```bash
git clone https://github.com/nirvani-nj/ai-resume-screening.git
cd ai-resume-screening
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 🔮 Future Work

- Improve skill extraction for skills with symbols (C++, C#, Node.js)
- Add PDF resume parsing, including scanned PDFs via OCR