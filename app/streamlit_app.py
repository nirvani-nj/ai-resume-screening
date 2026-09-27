# app/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from io import StringIO
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import ResumePredictor, BERT_AVAILABLE
from src.preprocessing import TextPreprocessor

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────
for key, default in [
    ('predictor', None),
    ('models_loaded', False),
    ('results', None),
    ('model_meta', {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Title ──────────────────────────────────────────────────────────────
st.title("📄 AI-Powered Resume Screening System")
st.markdown(
    "Uses **NLP + ML** to screen and rank resumes against a job description. "
    "Supports multiple classifiers, explainable scoring, and optional BERT similarity."
)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Load / reload models ───────────────────────────────────────────
    use_bert = st.checkbox(
        "🧠 Use Sentence-BERT similarity",
        value=False,
        help=(
            "Combines BERT semantic embeddings with TF-IDF for richer similarity.\n"
            "Requires: pip install sentence-transformers\n\n"
            + ("✅ sentence-transformers installed" if BERT_AVAILABLE
               else "⚠️  sentence-transformers NOT installed")
        ),
        disabled=not BERT_AVAILABLE,
    )

    if st.button("🔄 Load AI Models"):
        with st.spinner("Loading models…"):
            try:
                st.session_state.predictor = ResumePredictor(
                    model_dir='models', use_bert=use_bert
                )
                st.session_state.models_loaded = True
                st.session_state.model_meta = (
                    st.session_state.predictor.model_meta
                )
                st.success("✅ Models loaded!")
            except Exception as e:
                st.session_state.models_loaded = False
                st.error(f"❌ {e}")
                st.info("Run training first: `python src/train.py`")

    st.divider()

    if st.session_state.models_loaded:
        pred = st.session_state.predictor
        meta = st.session_state.model_meta

        # ── Model selector ─────────────────────────────────────────────
        st.subheader("🤖 Classifier")

        available = pred.available_models
        best = meta.get('best_model', available[0])
        scores = meta.get('scores', {})

        # Build display labels like "SVM (acc 0.9712) ★"
        def model_label(name):
            acc = scores.get(name)
            acc_str = f" — acc {acc:.4f}" if acc else ""
            star = " ⭐ best" if name == best else ""
            return f"{name}{acc_str}{star}"

        label_to_name = {model_label(n): n for n in available}
        default_label = model_label(pred.active_model_name)

        chosen_label = st.selectbox(
            "Select model",
            options=list(label_to_name.keys()),
            index=list(label_to_name.keys()).index(default_label),
            help="All models were trained on the same data. "
                 "The ⭐ best model is auto-selected by accuracy.",
        )
        chosen_name = label_to_name[chosen_label]

        if chosen_name != pred.active_model_name:
            pred.set_active_model(chosen_name)
            st.toast(f"Switched to {chosen_name}", icon="🔀")

        # Model comparison table
        if len(scores) > 1:
            with st.expander("📊 Model comparison"):
                cmp_df = pd.DataFrame(
                    [(n, f"{a:.4f}", "⭐" if n == best else "")
                     for n, a in sorted(scores.items(),
                                        key=lambda x: x[1], reverse=True)],
                    columns=["Model", "Accuracy", ""],
                )
                st.dataframe(cmp_df, hide_index=True, use_container_width=True)

        st.divider()

        # ── Scoring weights ────────────────────────────────────────────
        st.subheader("🎯 Scoring Weights")
        st.caption("Formula: w₁×Similarity + w₂×Skills + w₃×Experience")

        sim_w  = st.slider("Similarity",  0.0, 1.0, 0.5, 0.05)
        sk_w   = st.slider("Skills",      0.0, 1.0, 0.3, 0.05)
        exp_w  = st.slider("Experience",  0.0, 1.0, 0.2, 0.05)

        total = sim_w + sk_w + exp_w
        if abs(total - 1.0) > 0.01:
            st.caption(f"⚠️ Weights sum to {total:.2f} (will be normalised)")
            sim_w /= total; sk_w /= total; exp_w /= total

        st.session_state.weights = {
            'similarity': sim_w, 'skill': sk_w, 'experience': exp_w
        }

        st.divider()

        # ── Model info ─────────────────────────────────────────────────
        st.subheader("ℹ️ Model Info")
        st.info(
            f"**Active:** {pred.active_model_name}\n\n"
            f"**BERT:** {'on' if pred.use_bert else 'off'}\n\n"
            f"**Categories:** {len(pred.categories)}\n\n"
            f"**Skills DB:** {len(pred.skill_db.skills)} skills"
        )

    else:
        st.info("👈 Click 'Load AI Models' to start")

    st.divider()
    st.subheader("📌 How it works")
    st.markdown(
        "1. Enter a job description\n"
        "2. Upload resumes (TXT / PDF)\n"
        "3. Choose a classifier + weights\n"
        "4. Click **Analyse** → ranked results with score breakdown"
    )


# ══════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════
if not st.session_state.models_loaded:
    st.warning("Please load the AI models using the sidebar button to begin.")
    st.stop()

tab_rank, tab_analytics, tab_help = st.tabs(
    ["📊 Resume Ranking", "📈 Analytics", "ℹ️ Help"]
)

# ──────────────────────────────────────────────────────────────────────
# TAB 1 – Resume Ranking
# ──────────────────────────────────────────────────────────────────────
with tab_rank:
    col_jd, col_up = st.columns(2)

    with col_jd:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=250,
            placeholder=(
                "e.g., We are looking for a Data Scientist with 3+ years "
                "experience in Python, Machine Learning, and SQL…"
            ),
        )
        if job_description:
            st.caption(f"Characters: {len(job_description)}")

    with col_up:
        st.subheader("📎 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload resume files (TXT or PDF)",
            type=['txt', 'pdf'] if PDF_SUPPORT else ['txt'],
            accept_multiple_files=True,
            key="resume_uploader",
        )
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} resumes uploaded")
            with st.expander("View uploaded files"):
                for f in uploaded_files:
                    icon = "📄 PDF" if f.name.endswith('.pdf') else "📝 TXT"
                    st.write(f"{icon} – {f.name}")

    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        analyse = st.button(
            "🚀 Analyse Resumes",
            type="primary",
            use_container_width=True,
            disabled=not (job_description and uploaded_files),
        )

    if analyse:
        with st.spinner("🔍 Analysing resumes…"):
            resumes_list, filenames = [], []

            for file in uploaded_files:
                try:
                    if file.name.endswith('.pdf') and PDF_SUPPORT:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix='.pdf'
                        ) as tmp:
                            tmp.write(file.getvalue())
                            tmp_path = tmp.name
                        with open(tmp_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            text = "".join(p.extract_text() for p in reader.pages)
                        os.unlink(tmp_path)
                    else:
                        text = StringIO(file.getvalue().decode("utf-8")).read()

                    resumes_list.append(text)
                    filenames.append(file.name)
                except Exception as e:
                    st.warning(f"⚠️ Could not read {file.name}: {e}")

            if resumes_list:
                pred = st.session_state.predictor
                results = pred.rank_resumes_for_job(
                    job_description,
                    resumes_list,
                    weights=st.session_state.get(
                        'weights',
                        {'similarity': 0.5, 'skill': 0.3, 'experience': 0.2}
                    ),
                    filenames=filenames,
                )
                st.session_state.results = results

    # ── Results display ────────────────────────────────────────────────
    if st.session_state.results is not None:
        results = st.session_state.results

        st.markdown("---")
        st.subheader(f"🏆 Ranked Candidates ({len(results)} total)")

        # Quick summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", len(results))
        m2.metric("Avg Score", f"{results['final_score'].mean():.3f}")
        m3.metric("Top Score", f"{results['final_score'].max():.3f}")
        m4.metric("Classifier",
                  st.session_state.predictor.active_model_name)

        for idx, row in results.iterrows():
            rank = idx + 1
            name = row['candidate_name']
            score = row['final_score']
            cat = row['category']
            bd = row['score_breakdown']

            with st.expander(
                f"#{rank}  {name}  —  Score: **{score:.3f}**  [{cat}]"
            ):
                # ── Core metrics ───────────────────────────────────────
                c1, c2, c3 = st.columns(3)
                c1.metric("Similarity",
                          f"{row['similarity_score']:.3f}",
                          help="Cosine similarity between resume and JD vectors")
                c2.metric("Skill Match",
                          f"{row['skill_match_pct']:.1f}%",
                          help="% of required skills found in resume")
                c3.metric("Experience",
                          f"{row['experience_years']:.1f} yrs")

                # ── Explainability breakdown ───────────────────────────
                st.markdown("##### 🔍 Score Breakdown")

                sim_bd  = bd['similarity']
                sk_bd   = bd['skill_match']
                exp_bd  = bd['experience']

                expl_data = {
                    'Component':     ['Similarity', 'Skill Match', 'Experience'],
                    'Raw Value':     [sim_bd['raw'], sk_bd['raw'], exp_bd['raw']],
                    'Weight':        [sim_bd['weight'], sk_bd['weight'], exp_bd['weight']],
                    'Contribution':  [sim_bd['contribution'],
                                      sk_bd['contribution'],
                                      exp_bd['contribution']],
                }
                expl_df = pd.DataFrame(expl_data)

                # Horizontal stacked bar
                fig_expl = go.Figure()
                colors = ['#4C8BF5', '#34A853', '#FBBC04']
                for i, comp in enumerate(expl_data['Component']):
                    fig_expl.add_trace(go.Bar(
                        name=comp,
                        x=[expl_data['Contribution'][i]],
                        y=['Score'],
                        orientation='h',
                        marker_color=colors[i],
                        text=f"{comp}: {expl_data['Contribution'][i]:.3f}",
                        textposition='inside',
                        hovertemplate=(
                            f"<b>{comp}</b><br>"
                            f"Raw: {expl_data['Raw Value'][i]:.3f}<br>"
                            f"Weight: {expl_data['Weight'][i]:.2f}<br>"
                            f"Contribution: {expl_data['Contribution'][i]:.3f}<br>"
                            "<extra></extra>"
                        ),
                    ))
                fig_expl.update_layout(
                    barmode='stack',
                    height=110,
                    margin=dict(l=0, r=0, t=10, b=10),
                    showlegend=True,
                    legend=dict(orientation='h', y=-0.4),
                    xaxis=dict(range=[0, 1], title='Score contribution'),
                )
                st.plotly_chart(fig_expl, use_container_width=True)

                # Plain-language explanation
                top_factor = max(
                    [('similarity', sim_bd['contribution']),
                     ('skill match', sk_bd['contribution']),
                     ('experience', exp_bd['contribution'])],
                    key=lambda x: x[1]
                )
                bottom_factor = min(
                    [('similarity', sim_bd['contribution']),
                     ('skill match', sk_bd['contribution']),
                     ('experience', exp_bd['contribution'])],
                    key=lambda x: x[1]
                )

                method = sim_bd.get('method', 'TF-IDF')
                st.info(
                    f"**Why this rank?** "
                    f"This candidate's strongest factor is **{top_factor[0]}** "
                    f"(contributing {top_factor[1]:.3f} to the final score). "
                    f"Their weakest area is **{bottom_factor[0]}** "
                    f"({bottom_factor[1]:.3f}). "
                    f"Similarity was measured using **{method}**. "
                    f"Classifier used: **{bd['active_model']}**."
                )

                # Detail stats
                st.caption(
                    f"Skill match: {sk_bd['matched_count']} / "
                    f"{sk_bd['required_count']} required skills  |  "
                    f"Experience: {exp_bd['years_found']} yrs found, "
                    f"{exp_bd['years_required']} yrs required"
                )

                # ── Skills ─────────────────────────────────────────────
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.write("✅ **Matched Skills:**")
                    if row['matched_skills']:
                        skills_str = ", ".join(row['matched_skills'][:10])
                        extra = len(row['matched_skills']) - 10
                        st.write(skills_str + (f" +{extra} more" if extra > 0 else ""))
                    else:
                        st.write("None")
                with sc2:
                    st.write("⚠️ **Missing Skills:**")
                    if row['missing_skills']:
                        skills_str = ", ".join(row['missing_skills'][:10])
                        extra = len(row['missing_skills']) - 10
                        st.write(skills_str + (f" +{extra} more" if extra > 0 else ""))
                    else:
                        st.write("None – great fit!")

                if 'filename' in row and row['filename']:
                    st.caption(f"📎 {row['filename']}")

        # ── Download ───────────────────────────────────────────────────
        export_df = results.drop(columns=['score_breakdown'], errors='ignore')
        st.download_button(
            "📥 Download Results as CSV",
            data=export_df.to_csv(index=False),
            file_name="ranking_results.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────────────────────────────
# TAB 2 – Analytics
# ──────────────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("📊 Analytics")

    if st.session_state.results is None:
        st.warning("⚠️ No data yet — run an analysis in the Ranking tab first.")
        st.info(
            "**What you'll see here:**\n"
            "- Score distribution across candidates\n"
            "- Category breakdown (job types)\n"
            "- Top skills in leading candidates\n"
            "- Per-component contribution chart\n"
            "- Quick stats"
        )
        st.stop()

    results = st.session_state.results
    st.success(f"Showing analytics for {len(results)} candidates")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📈 Score Distribution")
        bins = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
        counts = [
            len(results[results['final_score'] <  0.2]),
            len(results[(results['final_score'] >= 0.2) & (results['final_score'] < 0.4)]),
            len(results[(results['final_score'] >= 0.4) & (results['final_score'] < 0.6)]),
            len(results[(results['final_score'] >= 0.6) & (results['final_score'] < 0.8)]),
            len(results[results['final_score'] >= 0.8]),
        ]
        fig1 = px.bar(
            pd.DataFrame({'Range': bins, 'Candidates': counts}),
            x='Range', y='Candidates',
            title='Candidates per score range',
            color='Candidates', color_continuous_scale='Blues',
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.markdown("### 🏷️ Category Distribution")
        cat_df = (results['category'].value_counts()
                  .reset_index()
                  .rename(columns={'index': 'Category', 'category': 'Count'}))
        if len(cat_df) > 0:
            fig2 = px.pie(
                cat_df.head(6),
                values=cat_df.columns[1], names=cat_df.columns[0],
                title='Job categories in applicant pool',
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Component contribution radar for top 5
    st.markdown("---")
    st.markdown("### 🎯 Score Component Comparison (Top 5 Candidates)")

    top5 = results.head(5)
    fig3 = go.Figure()
    for _, row in top5.iterrows():
        bd = row['score_breakdown']
        fig3.add_trace(go.Scatterpolar(
            r=[
                bd['similarity']['contribution'],
                bd['skill_match']['contribution'],
                bd['experience']['contribution'],
            ],
            theta=['Similarity', 'Skill Match', 'Experience'],
            fill='toself',
            name=row['candidate_name'],
        ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 0.6])),
        title='Score component radar – top 5 candidates',
        height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Top skills
    st.markdown("---")
    st.markdown("### 🔥 Most Common Skills (Top 10 Candidates)")
    all_matched = [s for skills in results['matched_skills'].head(10)
                   for s in (skills or [])]
    if all_matched:
        skill_df = (pd.Series(all_matched)
                    .value_counts()
                    .head(10)
                    .reset_index()
                    .rename(columns={'index': 'Skill', 0: 'Frequency'}))
        fig4 = px.bar(
            skill_df,
            y=skill_df.columns[0], x=skill_df.columns[1],
            orientation='h',
            title='Top matched skills in leading candidates',
            color=skill_df.columns[1],
            color_continuous_scale='Greens',
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Quick stats
    st.markdown("---")
    st.markdown("### 📋 Quick Stats")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Total", len(results))
    q2.metric("Avg Score", f"{results['final_score'].mean():.3f}")
    q3.metric("Top Score", f"{results['final_score'].max():.3f}")
    q4.metric("Categories", results['category'].nunique())
    q5.metric("Classifier",
              st.session_state.predictor.active_model_name)


# ──────────────────────────────────────────────────────────────────────
# TAB 3 – Help
# ──────────────────────────────────────────────────────────────────────
with tab_help:
    st.subheader("ℹ️ Help & Documentation")

    st.markdown("""
### Quick-start

**Step 1 – Train (one-time)**
```bash
python src/train.py --data data/raw/Resume.csv
```
This trains **SVM, Naive Bayes, and Random Forest** simultaneously, compares
accuracy, and saves all three models plus a `model_meta.json` with the winner.

**Step 2 – Run the app**
```bash
streamlit run app/streamlit_app.py
```

---

### Upgrades in this version

| Feature | Details |
|---|---|
| **Multi-model training** | SVM + Naive Bayes + Random Forest trained every run |
| **Auto-selection** | Best model (by accuracy) loaded as default |
| **Model dropdown** | Switch classifier anytime via sidebar |
| **Score breakdown** | Stacked bar + plain-language explanation per candidate |
| **BERT similarity** | Enable in sidebar if `sentence-transformers` is installed |

### Install BERT support (optional)
```bash
pip install sentence-transformers
```
Then reload models with the **Use Sentence-BERT** checkbox enabled.

---

### Scoring formula
```
Final Score = w₁ × Similarity + w₂ × Skill Match% + w₃ × Experience Score
```
Default weights: **0.5 / 0.3 / 0.2** — adjustable in the sidebar.
""")