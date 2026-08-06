# src/predict.py
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import TextPreprocessor, SkillDatabase

class ResumePredictor:
    """
    Handles prediction/ranking for new resumes and job descriptions
    Similarity, Skill Matching, Candidate Scoring
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        self.skill_db = SkillDatabase()
        self.tfidf_vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.categories = None
        
        # Load all models
        self.load_models()
    
    def load_models(self):
        """
        Load all trained models from disk
        """
        print("Loading trained models...")
        
        # Check if models exist
        required_files = [
            f'{self.model_dir}/tfidf_vectorizer.pkl',
            f'{self.model_dir}/classifier.pkl',
            f'{self.model_dir}/categories.pkl',
            f'{self.model_dir}/skills_database.pkl'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"❌ Model file not found: {file}\nPlease run training first: python src/train.py")
        
        # Load TF-IDF vectorizer
        with open(f'{self.model_dir}/tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        print(f"✅ TF-IDF vectorizer loaded (vocab size: {len(self.tfidf_vectorizer.vocabulary_)})")
        
        # Load classifier
        with open(f'{self.model_dir}/classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        print(f"✅ Classifier loaded ({type(self.classifier).__name__})")
        
        # Load categories
        with open(f'{self.model_dir}/categories.pkl', 'rb') as f:
            self.categories = pickle.load(f)
        print(f"✅ Categories loaded ({len(self.categories)} categories)")
        
        # Load skills database
        self.skill_db.load(f'{self.model_dir}/skills_database.pkl')
        print(f"✅ Skills database loaded ({len(self.skill_db.skills)} skills)")
    
    def predict_category(self, resume_text):
        """
        Predict the category of a single resume
        """
        # Clean text
        cleaned = self.preprocessor.clean_text(resume_text)
        
        # Vectorize
        vectorized = self.tfidf_vectorizer.transform([cleaned])
        
        # Predict
        pred_idx = self.classifier.predict(vectorized)[0]
        
        # Get category name
        category = self.categories[pred_idx] if pred_idx < len(self.categories) else "Unknown"
        
        return category
    
    def extract_name_from_filename(self, filename):
        """
        Extract a readable candidate name from filename
        """
        if not filename:
            return "Unknown Candidate"
        
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        # Replace underscores and hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Remove common prefixes (case insensitive)
        prefixes = ['resume', 'cv', 'applicant', 'candidate', 'profile']
        name_lower = name.lower()
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                name = name[len(prefix):].strip()
                break
        
        # Remove common suffixes
        suffixes = ['resume', 'cv', 'application']
        for suffix in suffixes:
            if name_lower.endswith(suffix):
                name = name[:-len(suffix)].strip()
                break
        
        # Clean up multiple spaces
        name = ' '.join(name.split())
        
        # Title case
        name = name.title()
        
        # If name is empty or too short, return a default
        if not name or len(name) < 3:
            return f"Candidate_{os.path.splitext(filename)[0][:10]}"
        
        return name
    
    def extract_name_from_text(self, text):
        """
        Try to extract candidate name from the first line of resume text
        """
        if not text or not isinstance(text, str):
            return None
        
        # Get first non-empty line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None
        
        first_line = lines[0]
        
        # Check if first line looks like a name (not an email, not too long, no special chars)
        if (len(first_line.split()) <= 4 and 
            len(first_line) <= 40 and
            '@' not in first_line and
            'http' not in first_line and
            not any(char.isdigit() for char in first_line[:5])):
            return first_line.strip()
        
        return None
    
    def rank_resumes_for_job(self, job_description, resumes_list, weights=None, filenames=None):
        """
        - Similarity Calculation (cosine similarity)
        - Skill Matching
        - Candidate Scoring with weighted formula
        
        Parameters:
        - job_description: Text of job description
        - resumes_list: List of resume texts
        - weights: Dictionary with weights
        - filenames: List of original filenames (optional)
        
        Returns DataFrame with ranked results
        """
        if weights is None:
            # Default weights from synopsis: 0.5 similarity, 0.3 skill, 0.2 experience
            weights = {'similarity': 0.5, 'skill': 0.3, 'experience': 0.2}
        
        print("🔍 Analyzing job description...")
        
        # Clean job description
        cleaned_jd = self.preprocessor.clean_text(job_description)
        
        # Vectorize job description
        jd_vector = self.tfidf_vectorizer.transform([cleaned_jd])
        
        # Extract skills from job description
        jd_skills = self.preprocessor.extract_skills(cleaned_jd, self.skill_db.skills)
        print(f"   Found {len(jd_skills)} required skills in job description")
        
        # Extract required experience from JD
        required_experience = self.preprocessor.extract_experience(cleaned_jd)
        if required_experience == 0:
            required_experience = 3  # Default if not specified
            print(f"   No experience specified, using default: {required_experience} years")
        else:
            print(f"   Required experience: {required_experience} years")
        
        # Process each resume
        print(f"\n📄 Analyzing {len(resumes_list)} resumes...")
        results = []
        
        for idx, resume_text in enumerate(resumes_list):
            # Show progress
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(resumes_list)} resumes...")
            
            # Get filename if provided
            filename = filenames[idx] if filenames and idx < len(filenames) else f"resume_{idx+1}.txt"
            
            # Clean resume
            cleaned_resume = self.preprocessor.clean_text(resume_text)
            
            # Vectorize resume
            resume_vector = self.tfidf_vectorizer.transform([cleaned_resume])
            
            # Similarity Calculation (cosine similarity)
            similarity = cosine_similarity(resume_vector, jd_vector)[0][0]
            
            # Skill Matching
            resume_skills = self.preprocessor.extract_skills(cleaned_resume, self.skill_db.skills)
            
            # Calculate skill match
            matched_skills = resume_skills.intersection(jd_skills)
            missing_skills = jd_skills - resume_skills
            skill_match_pct = len(matched_skills) / len(jd_skills) if jd_skills else 0
            
            # Extract experience
            experience = self.preprocessor.extract_experience(cleaned_resume)
            
            # Calculate experience score
            experience_score = min(experience / required_experience, 1.0) if required_experience > 0 else 0
            
            # Predict category (optional insight)
            try:
                category = self.predict_category(cleaned_resume)
            except:
                category = "Unknown"
            
            # Extract candidate name - try multiple methods
            candidate_name = self.extract_name_from_text(cleaned_resume)
            if not candidate_name:
                candidate_name = self.extract_name_from_filename(filename)
            
            # Candidate Scoring with weighted formula
            final_score = (weights['similarity'] * similarity +
                          weights['skill'] * skill_match_pct +
                          weights['experience'] * experience_score)
            
            results.append({
                'resume_id': idx + 1,
                'filename': filename,
                'candidate_name': candidate_name,
                'category': category,
                'similarity_score': round(similarity, 3),
                'skill_match_pct': round(skill_match_pct * 100, 1),
                'experience_years': round(experience, 1),
                'experience_score': round(experience_score, 3),
                'matched_skills': list(matched_skills)[:15],  # Top 15
                'missing_skills': list(missing_skills)[:15],  # Top 15
                'final_score': round(final_score, 3)
            })
        
        # Create DataFrame and sort by final score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        print(f"\n✅ Analysis complete! Top score: {results_df['final_score'].iloc[0]:.3f}")
        print(f"🏆 Top candidate: {results_df['candidate_name'].iloc[0]}")
        
        return results_df
    
    def get_resume_insights(self, resume_text, filename=None):
        """
        Get detailed insights for a single resume
        """
        cleaned = self.preprocessor.clean_text(resume_text)
        
        # Extract skills
        skills = self.preprocessor.extract_skills(cleaned, self.skill_db.skills)
        
        # Extract experience
        experience = self.preprocessor.extract_experience(cleaned)
        
        # Predict category
        category = self.predict_category(cleaned)
        
        # Get candidate name
        candidate_name = self.extract_name_from_text(cleaned)
        if not candidate_name and filename:
            candidate_name = self.extract_name_from_filename(filename)
        
        insights = {
            'candidate_name': candidate_name or "Unknown",
            'category': category,
            'skills': list(skills),
            'experience_years': experience,
            'word_count': len(cleaned.split()),
            'cleaned_text': cleaned[:500] + '...' if len(cleaned) > 500 else cleaned
        }
        
        return insights


# For testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = ResumePredictor(model_dir='models')
    
    # Example job description
    jd = """
    Looking for a Data Scientist with 3+ years experience in Python, 
    Machine Learning, and SQL. Experience with TensorFlow or PyTorch is a plus.
    """
    
    # Example resumes with realistic filenames
    resumes = [
        """
        John Smith
        Data Scientist with 4 years experience. Proficient in Python, 
        Machine Learning, SQL, and TensorFlow. Built multiple ML models.
        """,
        """
        Michael Chen
        Software Engineer with 5 years Java experience. Worked on Spring Boot,
        REST APIs, and Microservices. Some exposure to Python.
        """,
        """
        Sarah Johnson
        Recent graduate with MS in Data Science. Coursework in Python,
        Machine Learning, and Statistics. Completed internship in data analysis.
        """
    ]
    
    filenames = [
        "john_smith_data_scientist.pdf",
        "michael_chen_java_developer.pdf", 
        "sarah_johnson_junior_ds.pdf"
    ]
    
    # Rank resumes
    results = predictor.rank_resumes_for_job(jd, resumes, filenames=filenames)
    
    print("\n" + "="*60)
    print("🏆 RANKED RESULTS")
    print("="*60)
    
    # Display results in a nice format
    for idx, row in results.iterrows():
        print(f"\n#{idx+1}: {row['candidate_name']}")
        print(f"   Score: {row['final_score']:.3f} | Category: {row['category']}")
        print(f"   Similarity: {row['similarity_score']:.3f} | Skill Match: {row['skill_match_pct']:.1f}% | Experience: {row['experience_years']:.1f} yrs")
        print(f"   Matched Skills: {', '.join(row['matched_skills'][:5])}")
        if row['missing_skills']:
            print(f"   Missing Skills: {', '.join(row['missing_skills'][:5])}")