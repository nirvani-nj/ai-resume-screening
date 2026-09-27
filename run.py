# run.py
import os
import sys
import argparse
import subprocess


def test_mode():
    """Test the prediction pipeline"""
    print("="*60)
    print("🧪 Testing Mode - Quick Prediction Test")
    print("="*60)

    if not os.path.exists('models/classifier.pkl'):
        print("❌ Models not found! Please train first:")
        print("   python run.py --mode train --model all")
        return

    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from src.predict import ResumePredictor

        print("\n📂 Loading trained models...")
        predictor = ResumePredictor(model_dir='models')

        jd = """
        Looking for a Data Scientist with strong Python skills and experience in 
        Machine Learning. Must know SQL and data visualization. 3+ years experience required.
        """

        resumes = [
            "Data Scientist 4 years Python ML SQL TensorFlow",
            "Web Developer HTML CSS JS React Python",
            "MS Data Science Python ML Statistics Internship",
            "Java Developer Spring Boot Microservices SQL",
            "HR Specialist recruitment employee relations HRIS"
        ]

        print("\n📄 Testing resumes...")
        results = predictor.rank_resumes_for_job(jd, resumes)

        print("\n🏆 Results:")
        for idx, row in results.iterrows():
            print(f"{idx+1}. Score: {row['final_score']} | {row['category']}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Resume Screening System')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'app', 'test'],
        default='app',
        help='Mode: train, app, or test'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/Resume.csv',
        help='Path to training data'
    )

    # ✅ FIXED HERE (added 'all')
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all', 'svm', 'naive_bayes', 'random_forest'],
        help='Model type: all (recommended) or specific model'
    )

    args = parser.parse_args()

    if args.mode == 'train':
        print("="*60)
        print("🚀 Starting Training Pipeline")
        print("="*60)

        if not os.path.exists(args.data):
            print(f"❌ Data file not found: {args.data}")
            return

        # ✅ Pass model type correctly
        result = subprocess.run([
            sys.executable, "src/train.py",
            "--data", args.data,
            "--model", args.model
        ])

    elif args.mode == 'app':
        print("="*60)
        print("🚀 Starting Streamlit App")
        print("="*60)

        os.system("streamlit run app/streamlit_app.py")

    elif args.mode == 'test':
        test_mode()


if __name__ == "__main__":
    main()