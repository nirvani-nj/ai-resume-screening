import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import TextPreprocessor, SkillDatabase

class ModelTrainer:
    """Handles model training pipeline - matches synopsis methodology"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.skill_db = SkillDatabase()
        self.tfidf_vectorizer = None
        self.model = None
        self.df = None
        self.categories = None
        self.label_encoder = None
        
    def load_and_preprocess_data(self, filepath):
        
        print("📂 Step 1: Loading data from Kaggle dataset...")
        self.df = pd.read_csv(filepath)
        
        print(f"   Initial shape: {self.df.shape}")
        print(f"   Columns: {self.df.columns.tolist()}")
        
        # Check for required columns
        if 'Resume' not in self.df.columns or 'Category' not in self.df.columns:
            print("❌ Dataset must have 'Resume' and 'Category' columns")
            return None
        
        # Drop duplicates and NaNs
        initial_rows = self.df.shape[0]
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        print(f"   Removed {initial_rows - self.df.shape[0]} duplicate/NaN rows")
        
        # Step 3: Text Preprocessing
        print("Applying text preprocessing...")
        self.df['cleaned_resume'] = self.df['Resume'].apply(self.preprocessor.clean_text)
        
        # Extract categories
        self.categories = self.df['Category'].unique()
        print(f"   Found {len(self.categories)} unique categories")
        
        print(f"✅ Preprocessing complete! Final shape: {self.df.shape}")
        return self.df
    
    def create_features(self, max_features=5000, ngram_range=(1, 2)):
        """
        Feature Extraction using TF-IDF
        """
        print("Creating TF-IDF features...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform
        X = self.tfidf_vectorizer.fit_transform(self.df['cleaned_resume'])
        
        print(f"   TF-IDF matrix shape: {X.shape}")
        print(f"   Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return X
    
    def prepare_labels(self):
        """
        Prepare target labels for training
        """
        from sklearn.preprocessing import LabelEncoder
        
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.df['Category'])
        
        print(f"   Encoded {len(self.label_encoder.classes_)} categories")
        
        return y
    
    def train_model(self, X, y, model_type='svm'):
        """
        Train the selected model
        """
        print(f"\n🤖 Training {model_type.upper()} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   Training set size: {X_train.shape[0]} samples")
        print(f"   Test set size: {X_test.shape[0]} samples")
        
        # Select model
        if model_type == 'svm':
            self.model = LinearSVC(random_state=42, max_iter=2000, dual='auto')
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        print("   Training in progress...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model trained successfully!")
        print(f"📊 Test Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def save_models(self, model_dir='models'):
        """
        Save all trained models and artifacts
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save TF-IDF vectorizer
        with open(f'{model_dir}/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        print(f"✅ TF-IDF vectorizer saved")
        
        # Save classifier
        with open(f'{model_dir}/classifier.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Classifier saved")
        
        # Save label encoder
        with open(f'{model_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✅ Label encoder saved")
        
        # Save category names
        with open(f'{model_dir}/categories.pkl', 'wb') as f:
            pickle.dump(self.label_encoder.classes_, f)
        print(f"✅ Categories saved")
        
        # Save skills database
        self.skill_db.save(f'{model_dir}/skills_database.pkl')
        print(f"✅ Skills database saved")
        
        print(f"\n✅ All models saved to {model_dir}/")
    
    def run_training_pipeline(self, data_path, model_type='svm', max_features=5000):
        """
        Run the complete training pipeline
        """
        print("="*60)
        print("🚀 TRAINING PIPELINE STARTED")
        print("="*60)
        
        # Load and preprocess
        self.load_and_preprocess_data(data_path)
        
        # Create features
        X = self.create_features(max_features=max_features)
        
        # Prepare labels
        y = self.prepare_labels()
        
        # Train model
        accuracy = self.train_model(X, y, model_type=model_type)
        
        # Save models
        self.save_models()
        
        print("="*60)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print(f"📊 Final Accuracy: {accuracy:.4f}")
        print("="*60)
        
        return accuracy


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train resume screening model')
    parser.add_argument('--data', type=str, default='data/raw/Resume.csv',
                       help='Path to training data CSV')
    parser.add_argument('--model', type=str, default='svm',
                       choices=['svm', 'naive_bayes', 'random_forest'],
                       help='Model type to train')
    parser.add_argument('--features', type=int, default=5000,
                       help='Max features for TF-IDF')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Run pipeline
    trainer.run_training_pipeline(
        data_path=args.data,
        model_type=args.model,
        max_features=args.features
    )


if __name__ == "__main__":
    main()