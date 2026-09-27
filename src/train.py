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
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import TextPreprocessor, SkillDatabase


class ModelTrainer:
    """
    Handles model training pipeline.
    Trains SVM, Naive Bayes, and Random Forest, compares them,
    and automatically selects the best model.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.skill_db = SkillDatabase()
        self.tfidf_vectorizer = None
        self.model = None          # best model after evaluation
        self.df = None
        self.categories = None
        self.label_encoder = None
        self.model_scores = {}     # accuracy for all trained models
        self.best_model_name = None
        self._all_trained_models = {}

    # ------------------------------------------------------------------
    # Step 1-3: Load, clean, preprocess
    # ------------------------------------------------------------------
    def load_and_preprocess_data(self, filepath):
        print("📂 Step 1: Loading data from Kaggle dataset...")
        self.df = pd.read_csv(filepath)

        print(f"   Initial shape: {self.df.shape}")
        print(f"   Columns: {self.df.columns.tolist()}")

        if 'Resume' not in self.df.columns or 'Category' not in self.df.columns:
            print("❌ Dataset must have 'Resume' and 'Category' columns")
            return None

        initial_rows = self.df.shape[0]
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        print(f"   Removed {initial_rows - self.df.shape[0]} duplicate/NaN rows")

        print("Applying text preprocessing...")
        self.df['cleaned_resume'] = self.df['Resume'].apply(self.preprocessor.clean_text)

        self.categories = self.df['Category'].unique()
        print(f"   Found {len(self.categories)} unique categories")
        print(f"✅ Preprocessing complete! Final shape: {self.df.shape}")
        return self.df

    # ------------------------------------------------------------------
    # Step 4: TF-IDF feature extraction
    # ------------------------------------------------------------------
    def create_features(self, max_features=5000, ngram_range=(1, 2)):
        print("Creating TF-IDF features...")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        X = self.tfidf_vectorizer.fit_transform(self.df['cleaned_resume'])
        print(f"   TF-IDF matrix shape: {X.shape}")
        print(f"   Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        return X

    def prepare_labels(self):
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.df['Category'])
        print(f"   Encoded {len(self.label_encoder.classes_)} categories")
        return y

    # ------------------------------------------------------------------
    # Step 5: Train ALL models and auto-select best
    # ------------------------------------------------------------------
    def train_all_models(self, X, y):
        """
        Train SVM, Naive Bayes, and Random Forest.
        Compare accuracy scores and auto-select the best model.
        Returns dict of {model_name: accuracy}.
        """
        print("\n🤖 Training all models for comparison...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]} samples")

        candidates = {
            'SVM': LinearSVC(random_state=42, max_iter=2000, dual='auto'),
            'NaiveBayes': MultinomialNB(),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
        }

        for name, clf in candidates.items():
            print(f"\n   Training {name}...")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.model_scores[name] = round(acc, 4)
            self._all_trained_models[name] = clf
            print(f"   ✅ {name} accuracy: {acc:.4f}")

        # Auto-select best
        self.best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.model = self._all_trained_models[self.best_model_name]

        print(f"\n🏆 Best model: {self.best_model_name} "
              f"(accuracy: {self.model_scores[self.best_model_name]:.4f})")

        return self.model_scores

    def train_model(self, X, y, model_type='svm'):
        """Single-model training kept for CLI backward compatibility."""
        model_map = {
            'svm': ('SVM', LinearSVC(random_state=42, max_iter=2000, dual='auto')),
            'naive_bayes': ('NaiveBayes', MultinomialNB()),
            'random_forest': ('RandomForest', RandomForestClassifier(
                n_estimators=100, random_state=42)),
        }
        name, clf = model_map.get(
            model_type.lower(),
            ('SVM', LinearSVC(random_state=42, max_iter=2000, dual='auto'))
        )

        print(f"\n🤖 Training {name}...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        self.model = clf
        self.best_model_name = name
        self.model_scores = {name: round(acc, 4)}
        self._all_trained_models = {name: clf}
        print(f"✅ {name} accuracy: {acc:.4f}")
        return acc

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_models(self, model_dir='models'):
        os.makedirs(model_dir, exist_ok=True)

        with open(f'{model_dir}/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        print("✅ TF-IDF vectorizer saved")

        # Save ALL trained models individually so the UI can switch between them
        for name, clf in self._all_trained_models.items():
            with open(f'{model_dir}/classifier_{name}.pkl', 'wb') as f:
                pickle.dump(clf, f)
            print(f"✅ Classifier saved: {name}")

        # Also save the best as the default 'classifier.pkl' for backward compat
        with open(f'{model_dir}/classifier.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Default classifier saved ({self.best_model_name})")

        with open(f'{model_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("✅ Label encoder saved")

        with open(f'{model_dir}/categories.pkl', 'wb') as f:
            pickle.dump(self.label_encoder.classes_, f)
        print("✅ Categories saved")

        self.skill_db.save(f'{model_dir}/skills_database.pkl')
        print("✅ Skills database saved")

        # Save model comparison scores + metadata
        meta = {
            'best_model': self.best_model_name,
            'scores': self.model_scores,
            'available_models': list(self._all_trained_models.keys()),
        }
        with open(f'{model_dir}/model_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"✅ Model metadata saved → best={self.best_model_name}")

        print(f"\n✅ All models saved to {model_dir}/")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_training_pipeline(self, data_path, model_type='all', max_features=5000):
        print("=" * 60)
        print("🚀 TRAINING PIPELINE STARTED")
        print("=" * 60)

        self.load_and_preprocess_data(data_path)
        X = self.create_features(max_features=max_features)
        y = self.prepare_labels()

        if model_type == 'all':
            scores = self.train_all_models(X, y)
            best_acc = scores[self.best_model_name]
        else:
            best_acc = self.train_model(X, y, model_type=model_type)

        self.save_models()

        print("=" * 60)
        print("✅ TRAINING PIPELINE COMPLETED")
        print(f"📊 Best Model: {self.best_model_name}  |  Accuracy: {best_acc:.4f}")
        if len(self.model_scores) > 1:
            for name, acc in self.model_scores.items():
                marker = " ← best" if name == self.best_model_name else ""
                print(f"   {name}: {acc:.4f}{marker}")
        print("=" * 60)

        return best_acc


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train resume screening model')
    parser.add_argument('--data', type=str, default='data/raw/Resume.csv')
    parser.add_argument(
        '--model', type=str, default='all',
        choices=['all', 'svm', 'naive_bayes', 'random_forest'],
        help='Train a single model or all (default: all → auto-select best)'
    )
    parser.add_argument('--features', type=int, default=5000)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return

    trainer = ModelTrainer()
    trainer.run_training_pipeline(
        data_path=args.data,
        model_type=args.model,
        max_features=args.features
    )


if __name__ == "__main__":
    main()