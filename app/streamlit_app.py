# app/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from io import StringIO
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import ResumePredictor
from src.preprocessing import TextPreprocessor

# Try to import PDF parser
try:
    import PyPDF2
    PDF_SUPPORT = True
except:
    PDF_SUPPORT = False
    st.warning("PyPDF2 not installed. PDF uploads will be disabled.")

# Page config
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.models_loaded = False

if 'results' not in st.session_state:
    st.session_state.results = None

# Title
st.title("📄 AI-Powered Resume Screening System")
st.markdown("""
This system uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically 
screen and rank resumes based on their relevance to a job description.
""")

# Sidebar for model loading and configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load models button
    if st.button("🔄 Load AI Models"):
        with st.spinner("Loading models... This may take a moment."):
            try:
                st.session_state.predictor = ResumePredictor(model_dir='models')
                st.session_state.models_loaded = True
                st.success("✅ Models loaded successfully!")
            except Exception as e:
                st.session_state.models_loaded = False
                st.error(f"❌ Error loading models: {str(e)}")
                st.info("Please run training first: `python src/train.py`")
    
    st.divider()
    
    # Only show configuration if models are loaded
    if st.session_state.models_loaded:
        st.subheader("🎯 Scoring Weights")
        st.markdown("*Basic Formula: 0.5×Similarity + 0.3×Skills + 0.2×Experience*")
        
        col1, col2 = st.columns(2)
        with col1:
            sim_weight = st.slider("Similarity", 0.0, 1.0, 0.5, 0.1)
            skill_weight = st.slider("Skills", 0.0, 1.0, 0.3, 0.1)
        with col2:
            exp_weight = st.slider("Experience", 0.0, 1.0, 0.2, 0.1)
        
        # Normalize weights
        total = sim_weight + skill_weight + exp_weight
        if abs(total - 1.0) > 0.01:
            st.caption(f"⚠️ Weights sum to {total:.1f} (will be normalized)")
            sim_weight /= total
            skill_weight /= total
            exp_weight /= total
        
        weights = {
            'similarity': sim_weight,
            'skill': skill_weight,
            'experience': exp_weight
        }
        st.session_state.weights = weights
        
        st.divider()
        
        # Model info
        st.subheader("🤖 Model Info")
        st.info(f"""
        **Classifier:** {type(st.session_state.predictor.classifier).__name__}
        **Categories:** {len(st.session_state.predictor.categories)}
        **Skills DB:** {len(st.session_state.predictor.skill_db.skills)} skills
        """)
    else:
        st.info("👈 Click 'Load AI Models' to start")
    
    st.divider()
    
    # About
    st.subheader("📌 About")
    st.markdown("""
    **How it works:**
    1. Enter job description
    2. Upload resumes (TXT/PDF)
    3. System analyzes:
       - Content similarity (cosine)
       - Skill matching
       - Experience relevance
    4. Get ranked candidates with names!
    """)

# Main area
if st.session_state.models_loaded:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Resume Ranking", "📈 Analytics", "ℹ️ Help"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Job Description")
            
            # Job description input
            job_description = st.text_area(
                "Paste the job description here:",
                height=250,
                placeholder="e.g., We are looking for a Data Scientist with 3+ years experience in Python, Machine Learning, and SQL..."
            )
            
            # Show character count
            if job_description:
                st.caption(f"Characters: {len(job_description)}")
        
        with col2:
            st.subheader("📎 Upload Resumes")
            
            # File upload for resumes
            uploaded_files = st.file_uploader(
                "Upload resume files (TXT or PDF)",
                type=['txt', 'pdf'] if PDF_SUPPORT else ['txt'],
                accept_multiple_files=True,
                key="resume_uploader"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} resumes uploaded")
                
                # Show file names
                with st.expander("View uploaded files"):
                    for file in uploaded_files:
                        file_type = "📄 PDF" if file.name.endswith('.pdf') else "📝 TXT"
                        st.write(f"{file_type} - {file.name}")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button(
                "🚀 Analyze Resumes", 
                type="primary", 
                use_container_width=True,
                disabled=not (job_description and uploaded_files)
            )
        
        if analyze_button:
            with st.spinner("🔍 Analyzing resumes... This may take a moment."):
                # Read resumes and collect filenames
                resumes_list = []
                filenames = []
                
                for file in uploaded_files:
                    try:
                        if file.name.endswith('.pdf') and PDF_SUPPORT:
                            # Handle PDF
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Extract text from PDF
                            with open(tmp_path, 'rb') as f:
                                pdf_reader = PyPDF2.PdfReader(f)
                                text = ""
                                for page in pdf_reader.pages:
                                    text += page.extract_text()
                            resumes_list.append(text)
                            filenames.append(file.name)
                            
                            # Clean up
                            os.unlink(tmp_path)
                        else:
                            # Handle TXT
                            stringio = StringIO(file.getvalue().decode("utf-8"))
                            resumes_list.append(stringio.read())
                            filenames.append(file.name)
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {str(e)}")
                
                if resumes_list:
                    # Rank resumes WITH FILENAMES
                    st.session_state.results = st.session_state.predictor.rank_resumes_for_job(
                        job_description=job_description,
                        resumes_list=resumes_list,
                        filenames=filenames,
                        weights=st.session_state.weights
                    )
        
        # Display results
        if st.session_state.results is not None:
            st.divider()
            st.subheader("🏆 Ranked Candidates")
            
            results = st.session_state.results
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Candidates", len(results))
            with col2:
                st.metric("Top Score", f"{results['final_score'].iloc[0]:.3f}")
            with col3:
                st.metric("Avg Score", f"{results['final_score'].mean():.3f}")
            with col4:
                st.metric("Categories", results['category'].nunique())
            
            # Display top candidates with NAMES
            for idx, row in results.head(10).iterrows():
                # Get candidate name (with fallback)
                candidate_name = row.get('candidate_name', row.get('filename', f"Candidate {idx+1}"))
                
                # Create expander with name and score
                with st.expander(f"#{idx+1}: {candidate_name} - Score: {row['final_score']:.3f} ({row['category']})"):
                    
                    # Show metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Similarity", f"{row['similarity_score']:.2f}")
                    with col2:
                        st.metric("Skill Match", f"{row['skill_match_pct']:.1f}%")
                    with col3:
                        st.metric("Experience", f"{row['experience_years']:.1f} yrs")
                    
                    # Show filename (if available)
                    if 'filename' in row and row['filename']:
                        st.caption(f"📎 File: {row['filename']}")
                    
                    # Matched and Missing Skills
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("✅ **Matched Skills:**")
                        if row['matched_skills']:
                            # Format skills nicely with commas
                            skills_text = ", ".join(row['matched_skills'][:10])
                            if len(row['matched_skills']) > 10:
                                skills_text += f" and {len(row['matched_skills'])-10} more"
                            st.write(skills_text)
                        else:
                            st.write("None")
                    
                    with col2:
                        st.write("⚠️ **Missing Skills:**")
                        if row['missing_skills']:
                            skills_text = ", ".join(row['missing_skills'][:10])
                            if len(row['missing_skills']) > 10:
                                skills_text += f" and {len(row['missing_skills'])-10} more"
                            st.write(skills_text)
                        else:
                            st.write("None")
            
            # Download button
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="ranking_results.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.subheader("📊 Analytics")
        
        # Check if results exist
        if st.session_state.results is not None:
            results = st.session_state.results
            
            # Show success message
            st.success(f"✅ Showing analytics for {len(results)} candidates")
            
            # Create 2 simple columns
            col1, col2 = st.columns(2)
            
            with col1:
                # 1. Simple score distribution
                st.markdown("### 📈 Score Distribution")
                
                # Create bins for scores
                score_ranges = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
                score_counts = [
                    len(results[results['final_score'] < 0.2]),
                    len(results[(results['final_score'] >= 0.2) & (results['final_score'] < 0.4)]),
                    len(results[(results['final_score'] >= 0.4) & (results['final_score'] < 0.6)]),
                    len(results[(results['final_score'] >= 0.6) & (results['final_score'] < 0.8)]),
                    len(results[results['final_score'] >= 0.8])
                ]
                
                # Simple bar chart
                score_df = pd.DataFrame({
                    'Score Range': score_ranges,
                    'Number of Candidates': score_counts
                })
                
                fig1 = px.bar(score_df, 
                             x='Score Range', 
                             y='Number of Candidates',
                             title='How many candidates in each score range?',
                             color='Number of Candidates',
                             color_continuous_scale='blues')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Simple explanation
                st.info("""
                **What this means:** 
                - Higher scores (0.6-1.0) = Good match for the job
                - Lower scores (0-0.4) = Poor match
                """)
            
            with col2:
                # 2. Category breakdown
                st.markdown("### 🏷️ Job Categories Found")
                
                # Get top categories
                cat_counts = results['category'].value_counts().reset_index()
                cat_counts.columns = ['Category', 'Count']
                
                if len(cat_counts) > 0:
                    # Simple pie chart
                    fig2 = px.pie(cat_counts.head(5),  # Show top 5
                                 values='Count', 
                                 names='Category',
                                 title='What types of professionals applied?')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.write("No category data available")
            
            # 3. Skills analysis - Full width section
            st.markdown("---")
            st.markdown("### 🔥 Most Common Skills in Top Candidates")
            
            # Get skills from top 10 candidates
            all_matched = []
            for skills in results['matched_skills'].head(10):
                if skills:  # Check if skills exist
                    all_matched.extend(skills)
            
            if all_matched:
                # Count skills
                skill_counts = pd.Series(all_matched).value_counts().head(10)
                
                # Simple horizontal bar chart
                skill_df = pd.DataFrame({
                    'Skill': skill_counts.index,
                    'Frequency': skill_counts.values
                })
                
                fig3 = px.bar(skill_df, 
                             y='Skill', 
                             x='Frequency',
                             title='Top 10 Skills Found in Best Candidates',
                             orientation='h',
                             color='Frequency',
                             color_continuous_scale='greens')
                st.plotly_chart(fig3, use_container_width=True)
                
                # Simple explanation
                st.success("""
                **💡 Insight:** These are the skills that appear most often in your top-ranked candidates. 
                If you're hiring, these skills are common among the best applicants!
                """)
            else:
                st.write("No skills data available from the current analysis")
            
            # 4. Simple statistics at the bottom
            st.markdown("---")
            st.markdown("### 📋 Quick Stats")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Candidates", len(results))
            with col2:
                st.metric("Average Score", f"{results['final_score'].mean():.2f}")
            with col3:
                st.metric("Best Score", f"{results['final_score'].max():.2f}")
            with col4:
                st.metric("Categories Found", results['category'].nunique())
            
        else:
            # Show message when no results exist
            st.warning("⚠️ No data to display!")
            st.info("""
            **Go to the 'Resume Ranking' tab and:**
            1. Enter a job description
            2. Upload resume files
            3. Click 'Analyze Resumes'
            
            Then come back here to see analytics!
            """)
            
            # Show a preview image or example
            st.markdown("### 👆 What you'll see:")
            st.markdown("""
            - **Score Distribution:** How many candidates are strong vs weak matches
            - **Category Breakdown:** What job titles people applied with
            - **Top Skills:** The most common skills among your best candidates
            - **Quick Stats:** Total candidates, average score, etc.
            """)
    
    with tab3:
        st.subheader("ℹ️ Help & Documentation")
        
        st.markdown("""
        ### How to Use This System
        
        **Step 1: Train the Model** (One-time setup)
        ```bash
        python src/train.py --data data/raw/Resume.csv

            """)