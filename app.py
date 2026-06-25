import streamlit as st
import json
import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Import the core logic directly
from rank import calculate_score, generate_reasoning

st.set_page_config(page_title="Redrob Candidate Ranker Sandbox", layout="wide")

st.title("🎯 Redrob Candidate Ranker Sandbox")
st.write("Upload a candidate sample (`.jsonl`) to rank the candidates against the Senior AI Engineer Job Description.")

# Job Description Viewer
with st.expander("📄 View Job Description (Target Role)"):
    try:
        with open("job_description.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except:
        st.info("job_description.md not found in the current directory.")

# File uploader
uploaded_file = st.file_uploader("Upload candidates file (.jsonl)", type=["jsonl"])

if uploaded_file is not None:
    # Read candidates
    candidates = []
    candidate_texts = []
    
    file_contents = uploaded_file.getvalue().decode("utf-8").splitlines()
    for line in file_contents:
        if not line.strip():
            continue
        try:
            c = json.loads(line)
            candidates.append(c)
            
            # Text profiles for TF-IDF
            profile = c.get("profile", {})
            summary = profile.get("summary", "")
            headline = profile.get("headline", "")
            current_title = profile.get("current_title", "")
            
            career = c.get("career_history", [])
            career_desc = " ".join([j.get("description", "") for j in career if j.get("description")])
            career_titles = " ".join([j.get("title", "") for j in career if j.get("title")])
            
            skills = c.get("skills", [])
            skills_str = " ".join([s.get("name", "") for s in skills if s.get("name")])
            
            text = f"{current_title} {headline} {summary} {career_titles} {career_desc} {skills_str}"
            candidate_texts.append(text)
        except Exception as e:
            st.error(f"Error parsing JSON line: {e}")
            
    if candidates:
        st.success(f"Successfully loaded {len(candidates)} candidates.")
        
        # Load Job Description
        jd_text = ""
        try:
            with open("job_description.md", "r", encoding="utf-8") as f:
                jd_text = f.read()
        except:
            jd_text = "Senior AI Engineer. Retrieval and ranking, vector databases, RAG, PyTorch, evaluation metrics."
            
        # Run Ranking
        with st.spinner("Ranking candidates..."):
            # Compute TF-IDF similarities
            vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
            candidate_matrices = vectorizer.fit_transform(candidate_texts)
            jd_vector = vectorizer.transform([jd_text])
            similarities = cosine_similarity(candidate_matrices, jd_vector).flatten()
            
            scored_candidates = []
            for i, c in enumerate(candidates):
                cid = c["candidate_id"]
                
                # Check for honeypot
                # We can't import is_honeypot since it uses helper files, but it's in rank.py
                from rank import is_honeypot, calculate_non_semantic_score
                
                honeypot_flag, reason = is_honeypot(c)
                if honeypot_flag:
                    scored_candidates.append((cid, -999.0, c))
                    continue
                    
                tfidf_score = similarities[i]
                title_score, yoe_score, loc_score, behavior_bonus, behavior_multiplier, research_penalty = calculate_non_semantic_score(c)
                
                if title_score < 0:
                    scored_candidates.append((cid, title_score, c))
                    continue
                    
                base_score = (0.50 * tfidf_score + 
                              0.20 * title_score + 
                              0.20 * yoe_score + 
                              0.10 * loc_score)
                              
                final_score = base_score * research_penalty * behavior_multiplier + behavior_bonus
                scored_candidates.append((cid, round(final_score, 4), c))
                
            # Sort
            scored_candidates.sort(key=lambda x: (-x[1], x[0]))
            
            # Write ranked list
            results = []
            for rank, (cid, score, c) in enumerate(scored_candidates, 1):
                reasoning = generate_reasoning(c)
                results.append({
                    "rank": rank,
                    "candidate_id": cid,
                    "score": score,
                    "reasoning": reasoning
                })
                
            df_results = pd.DataFrame(results)
            
            # Show download button
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Ranked Results CSV",
                data=csv_data,
                file_name="ranked_candidates.csv",
                mime="text/csv"
            )
            
            # Display results
            st.subheader("🏆 Top Ranked Candidates")
            st.dataframe(df_results.head(50), use_container_width=True)
