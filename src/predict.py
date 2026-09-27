# src/predict.py
import pandas as pd
import numpy as np
import pickle
import os
import re
import sys
import json
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import TextPreprocessor, SkillDatabase

# ──────────────────────────────────────────────────────────────────────
# Optional BERT import
# ──────────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


class ResumePredictor:
    """
    Handles prediction/ranking for new resumes and job descriptions.

    Upgrades in this version:
      1. Loads ALL trained models (SVM, NaiveBayes, RandomForest) and lets
         the caller switch the active one at runtime.
      2. Defaults to the best model recorded in model_meta.json.
      3. Per-candidate score breakdown for explainability.
      4. Optional Sentence-BERT semantic similarity (falls back to TF-IDF
         cosine when sentence-transformers is not installed).
    """

    # Model name → pkl filename mapping
    MODEL_FILES = {
        'SVM':          'classifier_SVM.pkl',
        'NaiveBayes':   'classifier_NaiveBayes.pkl',
        'RandomForest': 'classifier_RandomForest.pkl',
    }

    def __init__(self, model_dir='models', use_bert=False):
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        self.skill_db = SkillDatabase()
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.categories = None

        # Multi-model storage
        self._models = {}          # name → classifier object
        self.model_meta = {}       # loaded from model_meta.json
        self.active_model_name = None
        self.classifier = None     # alias for the active classifier

        # BERT
        self.use_bert = use_bert and BERT_AVAILABLE
        self._bert_model = None

        self.load_models()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_models(self):
        print("Loading trained models...")

        required = [
            f'{self.model_dir}/tfidf_vectorizer.pkl',
            f'{self.model_dir}/categories.pkl',
            f'{self.model_dir}/skills_database.pkl',
        ]
        for f in required:
            if not os.path.exists(f):
                raise FileNotFoundError(
                    f"❌ Model file not found: {f}\n"
                    "Please run training first: python src/train.py"
                )

        with open(f'{self.model_dir}/tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        print(f"✅ TF-IDF vectorizer loaded "
              f"(vocab size: {len(self.tfidf_vectorizer.vocabulary_)})")

        with open(f'{self.model_dir}/categories.pkl', 'rb') as f:
            self.categories = pickle.load(f)
        print(f"✅ Categories loaded ({len(self.categories)} categories)")

        self.skill_db.load(f'{self.model_dir}/skills_database.pkl')
        print(f"✅ Skills database loaded ({len(self.skill_db.skills)} skills)")

        # Load model metadata (has best_model name + accuracy scores)
        meta_path = f'{self.model_dir}/model_meta.json'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.model_meta = json.load(f)
        else:
            self.model_meta = {'best_model': 'SVM', 'scores': {}, 'available_models': ['SVM']}

        # Load whichever individual classifier files exist
        for name, fname in self.MODEL_FILES.items():
            path = f'{self.model_dir}/{fname}'
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self._models[name] = pickle.load(f)
                print(f"✅ Loaded classifier: {name} "
                      f"(acc={self.model_meta['scores'].get(name, '?')})")

        # Fall back to default classifier.pkl if no named files found
        if not self._models:
            default_path = f'{self.model_dir}/classifier.pkl'
            if os.path.exists(default_path):
                with open(default_path, 'rb') as f:
                    clf = pickle.load(f)
                name = type(clf).__name__
                self._models[name] = clf
                print(f"✅ Loaded default classifier: {name}")
            else:
                raise FileNotFoundError("❌ No classifier files found.")

        # Set active model to best (or first available)
        best = self.model_meta.get('best_model', None)
        if best and best in self._models:
            self.set_active_model(best)
        else:
            self.set_active_model(list(self._models.keys())[0])

        # Optionally load BERT
        if self.use_bert:
            self._load_bert()

    def _load_bert(self):
        if not BERT_AVAILABLE:
            print("⚠️  sentence-transformers not installed. "
                  "Run: pip install sentence-transformers")
            self.use_bert = False
            return
        print("🔄 Loading Sentence-BERT (all-MiniLM-L6-v2)…")
        self._bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ BERT model loaded")

    # ------------------------------------------------------------------
    # Model switching
    # ------------------------------------------------------------------
    @property
    def available_models(self):
        """Returns list of loaded model names."""
        return list(self._models.keys())

    def set_active_model(self, name):
        """Switch the active classifier. Raises ValueError if unknown."""
        if name not in self._models:
            raise ValueError(
                f"Model '{name}' not loaded. Available: {list(self._models.keys())}"
            )
        self.active_model_name = name
        self.classifier = self._models[name]
        print(f"🔀 Active model set to: {name}")

    # ------------------------------------------------------------------
    # Category prediction
    # ------------------------------------------------------------------
    def predict_category(self, resume_text):
        cleaned = self.preprocessor.clean_text(resume_text)
        vectorized = self.tfidf_vectorizer.transform([cleaned])
        pred_idx = self.classifier.predict(vectorized)[0]
        return self.categories[pred_idx] if pred_idx < len(self.categories) else "Unknown"

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------
    def _tfidf_similarity(self, text_a, text_b):
        """Cosine similarity via TF-IDF vectors."""
        vecs = self.tfidf_vectorizer.transform([text_a, text_b])
        return float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])

    def _bert_similarity(self, text_a, text_b):
        """Semantic similarity via Sentence-BERT."""
        if self._bert_model is None:
            self._load_bert()
        if self._bert_model is None:
            return self._tfidf_similarity(text_a, text_b)
        embs = self._bert_model.encode([text_a, text_b], convert_to_numpy=True)
        norm_a = embs[0] / (np.linalg.norm(embs[0]) + 1e-10)
        norm_b = embs[1] / (np.linalg.norm(embs[1]) + 1e-10)
        return float(np.dot(norm_a, norm_b))

    def _combined_similarity(self, text_a, text_b, bert_weight=0.6):
        """Blend BERT + TF-IDF similarity scores."""
        bert_sim = self._bert_similarity(text_a, text_b)
        tfidf_sim = self._tfidf_similarity(text_a, text_b)
        return bert_weight * bert_sim + (1 - bert_weight) * tfidf_sim

    # ------------------------------------------------------------------
    # Name helpers
    # ------------------------------------------------------------------
    def extract_name_from_filename(self, filename):
        if not filename:
            return "Unknown Candidate"
        name = os.path.splitext(filename)[0]
        name = name.replace('_', ' ').replace('-', ' ')
        name_lower = name.lower()
        for prefix in ['resume', 'cv', 'applicant', 'candidate', 'profile']:
            if name_lower.startswith(prefix):
                name = name[len(prefix):].strip()
                break
        for suffix in ['resume', 'cv', 'application']:
            if name_lower.endswith(suffix):
                name = name[:-len(suffix)].strip()
                break
        name = ' '.join(name.split()).title()
        if not name or len(name) < 3:
            return f"Candidate_{os.path.splitext(filename)[0][:10]}"
        return name

    def extract_name_from_text(self, text):
        if not text or not isinstance(text, str):
            return None
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return None
        first = lines[0]
        if (len(first.split()) <= 4 and len(first) <= 40
                and '@' not in first and 'http' not in first
                and not any(c.isdigit() for c in first[:5])):
            return first.strip()
        return None

    # ------------------------------------------------------------------
    # Core ranking
    # ------------------------------------------------------------------
    def rank_resumes_for_job(self, job_description, resumes_list,
                              weights=None, filenames=None):
        """
        Rank a list of resumes against a job description.

        Returns a DataFrame with per-candidate scores AND a
        'score_breakdown' column (dict) for explainability.
        """
        if weights is None:
            weights = {'similarity': 0.5, 'skill': 0.3, 'experience': 0.2}

        print("🔍 Analyzing job description...")
        cleaned_jd = self.preprocessor.clean_text(job_description)
        jd_vector = self.tfidf_vectorizer.transform([cleaned_jd])
        jd_skills = self.preprocessor.extract_skills(cleaned_jd, self.skill_db.skills)
        print(f"   Found {len(jd_skills)} required skills in job description")

        required_exp = self.preprocessor.extract_experience(cleaned_jd)
        if required_exp == 0:
            required_exp = 3
            print(f"   No experience specified, using default: {required_exp} years")
        else:
            print(f"   Required experience: {required_exp} years")

        print(f"\n📄 Analyzing {len(resumes_list)} resumes "
              f"[model: {self.active_model_name}"
              f"{', +BERT' if self.use_bert else ''}]...")

        results = []
        for idx, resume_text in enumerate(resumes_list):
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(resumes_list)} resumes...")

            filename = (filenames[idx] if filenames and idx < len(filenames)
                        else f"resume_{idx+1}.txt")
            cleaned_resume = self.preprocessor.clean_text(resume_text)

            # ── Similarity ───────────────────────────────────────────
            if self.use_bert:
                similarity = self._combined_similarity(cleaned_jd, cleaned_resume)
                sim_method = 'BERT+TF-IDF'
            else:
                resume_vector = self.tfidf_vectorizer.transform([cleaned_resume])
                similarity = float(
                    cosine_similarity(resume_vector, jd_vector)[0][0]
                )
                sim_method = 'TF-IDF'

            # ── Skill matching ────────────────────────────────────────
            resume_skills = self.preprocessor.extract_skills(
                cleaned_resume, self.skill_db.skills
            )
            matched_skills = resume_skills.intersection(jd_skills)
            missing_skills = jd_skills - resume_skills
            skill_match_pct = len(matched_skills) / len(jd_skills) if jd_skills else 0

            # ── Experience ────────────────────────────────────────────
            experience = self.preprocessor.extract_experience(cleaned_resume)
            experience_score = (
                min(experience / required_exp, 1.0) if required_exp > 0 else 0
            )

            # ── Category ─────────────────────────────────────────────
            try:
                category = self.predict_category(cleaned_resume)
            except Exception:
                category = "Unknown"

            # ── Name ─────────────────────────────────────────────────
            candidate_name = self.extract_name_from_text(cleaned_resume)
            if not candidate_name:
                candidate_name = self.extract_name_from_filename(filename)

            # ── Final score ───────────────────────────────────────────
            w_sim = weights['similarity']
            w_sk  = weights['skill']
            w_exp = weights['experience']

            sim_contrib  = w_sim * similarity
            skill_contrib = w_sk * skill_match_pct
            exp_contrib  = w_exp * experience_score
            final_score  = sim_contrib + skill_contrib + exp_contrib

            # ── Explainability breakdown ──────────────────────────────
            score_breakdown = {
                'similarity': {
                    'raw':        round(similarity, 3),
                    'weight':     round(w_sim, 2),
                    'contribution': round(sim_contrib, 3),
                    'method':     sim_method,
                },
                'skill_match': {
                    'raw':        round(skill_match_pct, 3),
                    'weight':     round(w_sk, 2),
                    'contribution': round(skill_contrib, 3),
                    'matched_count': len(matched_skills),
                    'required_count': len(jd_skills),
                },
                'experience': {
                    'raw':        round(experience_score, 3),
                    'weight':     round(w_exp, 2),
                    'contribution': round(exp_contrib, 3),
                    'years_found': round(experience, 1),
                    'years_required': required_exp,
                },
                'active_model': self.active_model_name,
            }

            results.append({
                'resume_id':        idx + 1,
                'filename':         filename,
                'candidate_name':   candidate_name,
                'category':         category,
                'similarity_score': round(similarity, 3),
                'skill_match_pct':  round(skill_match_pct * 100, 1),
                'experience_years': round(experience, 1),
                'experience_score': round(experience_score, 3),
                'matched_skills':   list(matched_skills)[:15],
                'missing_skills':   list(missing_skills)[:15],
                'final_score':      round(final_score, 3),
                'score_breakdown':  score_breakdown,   # ← explainability
            })

        results_df = (pd.DataFrame(results)
                      .sort_values('final_score', ascending=False)
                      .reset_index(drop=True))

        print(f"\n✅ Analysis complete! Top score: {results_df['final_score'].iloc[0]:.3f}")
        print(f"🏆 Top candidate: {results_df['candidate_name'].iloc[0]}")
        return results_df

    def get_resume_insights(self, resume_text, filename=None):
        cleaned = self.preprocessor.clean_text(resume_text)
        skills = self.preprocessor.extract_skills(cleaned, self.skill_db.skills)
        experience = self.preprocessor.extract_experience(cleaned)
        category = self.predict_category(cleaned)
        candidate_name = self.extract_name_from_text(cleaned)
        if not candidate_name and filename:
            candidate_name = self.extract_name_from_filename(filename)
        return {
            'candidate_name':  candidate_name or "Unknown",
            'category':        category,
            'skills':          list(skills),
            'experience_years': experience,
            'word_count':      len(cleaned.split()),
            'cleaned_text':    (cleaned[:500] + '...'
                                if len(cleaned) > 500 else cleaned),
        }


# ──────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = ResumePredictor(model_dir='models')

    print(f"\nAvailable models: {predictor.available_models}")
    print(f"Active model:     {predictor.active_model_name}")
    print(f"Model scores:     {predictor.model_meta.get('scores', {})}")

    jd = """
    Looking for a Data Scientist with 3+ years experience in Python,
    Machine Learning, and SQL. Experience with TensorFlow or PyTorch is a plus.
    """
    resumes = [
        "John Smith\nData Scientist 4 years. Python, Machine Learning, SQL, TensorFlow.",
        "Michael Chen\nSoftware Engineer 5 years Java. Spring Boot, REST APIs. Some Python.",
        "Sarah Johnson\nRecent graduate MS Data Science. Python, ML, Statistics. Internship.",
    ]
    filenames = [
        "john_smith_data_scientist.pdf",
        "michael_chen_java_developer.pdf",
        "sarah_johnson_junior_ds.pdf",
    ]

    results = predictor.rank_resumes_for_job(jd, resumes, filenames=filenames)

    print("\n" + "=" * 60)
    print("🏆 RANKED RESULTS WITH EXPLAINABILITY")
    print("=" * 60)
    for idx, row in results.iterrows():
        bd = row['score_breakdown']
        print(f"\n#{idx+1}: {row['candidate_name']}")
        print(f"   Final Score : {row['final_score']:.3f}")
        print(f"   Similarity  : {bd['similarity']['raw']:.3f} "
              f"× {bd['similarity']['weight']} = {bd['similarity']['contribution']:.3f} "
              f"[{bd['similarity']['method']}]")
        print(f"   Skill Match : {bd['skill_match']['raw']:.3f} "
              f"× {bd['skill_match']['weight']} = {bd['skill_match']['contribution']:.3f} "
              f"({bd['skill_match']['matched_count']}/{bd['skill_match']['required_count']} skills)")
        print(f"   Experience  : {bd['experience']['raw']:.3f} "
              f"× {bd['experience']['weight']} = {bd['experience']['contribution']:.3f} "
              f"({bd['experience']['years_found']} / {bd['experience']['years_required']} yrs)")