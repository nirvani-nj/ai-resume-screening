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
    
    # Check if models exist
    if not os.path.exists('models/classifier.pkl'):
        print("❌ Models not found! Please train first:")
        print("   python run.py --mode train --data data/raw/Resume.csv")
        return
    
    try:
        # Add src to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import the predictor
        from src.predict import ResumePredictor
        
        # Initialize predictor
        print("\n📂 Loading trained models...")
        predictor = ResumePredictor(model_dir='models')
        
        # Test job description
        jd = """
        Looking for a Data Scientist with strong Python skills and experience in 
        Machine Learning. Must know SQL and data visualization. 3+ years experience required.
        """
        
        # Test resumes
        resumes = [
            """
            Data Scientist with 4 years experience. Proficient in Python, 
            Machine Learning, SQL, and TensorFlow. Built multiple ML models
            and created data visualizations with Tableau.
            """,
            
            """
            Web Developer with 3 years experience. Skilled in HTML, CSS, 
            JavaScript, and React. Some experience with Python scripts.
            """,
            
            """
            Recent graduate with MS in Data Science. Coursework in Python,
            Machine Learning, and Statistics. Completed internship in data analysis.
            """,
            
            """
            Java Developer with 5 years experience. Expert in Spring Boot,
            Hibernate, and Microservices. Basic knowledge of SQL.
            """,
            
            """
            HR Specialist with 5 years experience in recruitment and 
            employee relations. Proficient in HRIS and Microsoft Office.
            """
        ]
        
        print("\n📝 Job Description:")
        print("-" * 40)
        print(jd.strip())
        print("-" * 40)
        
        print(f"\n📄 Testing with {len(resumes)} sample resumes...")
        
        # Rank resumes
        results = predictor.rank_resumes_for_job(jd, resumes)
        
        print("\n" + "="*60)
        print("🏆 RANKED RESULTS")
        print("="*60)
        
        # Display results in a nice table format
        print(f"\n{'Rank':<5} {'Final Score':<12} {'Category':<20} {'Similarity':<12} {'Skills':<10} {'Experience':<10}")
        print("-" * 80)
        
        for idx, row in results.iterrows():
            rank = idx + 1
            print(f"{rank:<5} {row['final_score']:<12.3f} {row['category'][:20]:<20} "
                  f"{row['similarity_score']:<12.3f} {row['skill_match_pct']:<10.1f}% "
                  f"{row['experience_years']:<10.1f}")
        
        # Show detailed view of top candidate
        print("\n" + "="*60)
        print("🔍 TOP CANDIDATE DETAILS")
        print("="*60)
        
        top = results.iloc[0]
        print(f"\n🥇 Rank #1 - Score: {top['final_score']:.3f}")
        print(f"   Category: {top['category']}")
        print(f"   Similarity Score: {top['similarity_score']:.3f}")
        print(f"   Skill Match: {top['skill_match_pct']:.1f}%")
        print(f"   Experience: {top['experience_years']:.1f} years")
        
        print("\n   ✅ Matched Skills:")
        if top['matched_skills']:
            for skill in top['matched_skills'][:10]:
                print(f"      • {skill}")
        else:
            print("      None")
        
        print("\n   ❌ Missing Skills:")
        if top['missing_skills']:
            for skill in top['missing_skills'][:10]:
                print(f"      • {skill}")
        else:
            print("      None")
        
        # Show statistics
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"   Total resumes tested: {len(results)}")
        print(f"   Average score: {results['final_score'].mean():.3f}")
        print(f"   Highest score: {results['final_score'].max():.3f}")
        print(f"   Lowest score: {results['final_score'].min():.3f}")
        print(f"   Categories found: {results['category'].nunique()}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Resume Screening System')
    parser.add_argument('--mode', type=str, choices=['train', 'app', 'test'],
                       default='app', help='Mode: train, app, or test')
    parser.add_argument('--data', type=str, default='data/raw/Resume.csv',
                       help='Path to training data (for train mode)')
    parser.add_argument('--model', type=str, default='svm',
                       choices=['svm', 'naive_bayes', 'random_forest'],
                       help='Model type for training')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("="*60)
        print("🚀 Starting Training Pipeline")
        print("="*60)
        
        # Check if data exists
        if not os.path.exists(args.data):
            print(f"❌ Data file not found: {args.data}")
            return
        
        # Run training and show output in real-time
        result = subprocess.run([
            sys.executable, "src/train.py", 
            "--data", args.data, 
            "--model", args.model
        ])
        
    elif args.mode == 'app':
        print("="*60)
        print("🚀 Starting Streamlit App")
        print("="*60)
        print("\n📱 The app will open in your browser at http://localhost:8501")
        print("   If it doesn't open automatically, copy and paste that URL")
        print("\n💡 Tips:")
        print("   • Click 'Load AI Models' in the sidebar first")
        print("   • Paste a job description")
        print("   • Upload resume files (.txt or .pdf)")
        print("   • Click 'Analyze Resumes'")
        print("-" * 60)
        
        # Run streamlit app
        os.system("streamlit run app/streamlit_app.py")
    
    elif args.mode == 'test':
        test_mode()

if __name__ == "__main__":
    main()