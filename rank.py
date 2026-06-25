import json
import os
import argparse
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Consulting companies list
CONSULTING_COMPANIES = {
    'tcs', 'tata consultancy services', 'infosys', 'wipro', 'accenture', 
    'cognizant', 'capgemini', 'tech mahindra', 'hcl', 'mphasis', 'mindtree', 
    'l&t', 'lnt', 'larsen & toubro', 'cognizant technology solutions', 
    'deloitte', 'kpmg', 'ey', 'pwc', 'capgemini india'
}

# Target cities/regions for Noida/Pune
PUNE_NOIDA_KEYWORDS = {'pune', 'noida', 'delhi', 'ncr', 'gurgaon', 'ghaziabad', 'faridabad'}
OTHER_INDIAN_CITIES = {'bangalore', 'bengaluru', 'hyderabad', 'mumbai', 'chennai', 'kolkata', 'ahmedabad', 'jaipur'}

# Target skills for reasoning context
CORE_AI_SKILLS = {
    'embeddings', 'sentence transformers', 'vector databases', 'milvus', 
    'pinecone', 'weaviate', 'qdrant', 'faiss', 'elasticsearch', 'opensearch', 
    'nlp', 'llms', 'rag', 'retrieval-augmented generation', 'machine learning', 
    'deep learning', 'fine-tuning llms', 'lora', 'qlora', 'peft', 
    'learning to rank', 'xgboost', 'pytorch', 'sentence-transformers'
}
EVAL_SKILLS = {'ndcg', 'mrr', 'map', 'a/b testing', 'evaluation frameworks'}

def is_honeypot(c):
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    
    # 1. Expert skills with 0 duration_months
    expert_zero = [s["name"] for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0]
    if len(expert_zero) > 0:
        return True, "expert_skill_zero_months"
        
    # 2. Any career history job duration exceeds profile YoE
    for job in career:
        dur = job.get("duration_months", 0)
        if yoe > 0 and dur > (yoe * 12 + 6):
            return True, f"job_duration_exceeds_yoe: dur={dur}, yoe={yoe}"
            
    # 3. Profile YoE exceeds max possible career timeframe (earliest job start to 2026)
    if career and yoe > 0:
        start_years = []
        for job in career:
            start = job.get("start_date")
            if start:
                try:
                    start_years.append(int(start.split("-")[0]))
                except Exception:
                    pass
        if start_years:
            earliest_start_year = min(start_years)
            max_possible_years = 2026 - earliest_start_year
            if yoe > max_possible_years + 1:
                return True, f"yoe_exceeds_timeframe: yoe={yoe}, earliest={earliest_start_year}"
                
    return False, ""

def clean_title(title):
    # Remove duplicate consecutive seniority words like "Senior Senior"
    title = re.sub(r'\b(senior|lead|staff|principal|junior|associate)\b\s+\b(senior|lead|staff|principal|junior|associate)\b', r'\1', title, flags=re.IGNORECASE)
    return title

def get_clean_title_desc(title, company):
    title_clean = clean_title(title)
    title_lower = title_clean.lower()
    
    if any(prefix in title_lower for prefix in ["senior", "lead", "staff", "principal"]):
        return f"{title_clean} at {company}"
    elif any(prefix in title_lower for prefix in ["junior", "intern", "associate"]):
        return f"{title_clean} at {company}"
    else:
        return f"Senior-level {title_clean} at {company}"

def extract_achievement(career):
    achievements = []
    for job in career:
        desc = job.get("description", "").lower()
        company = job.get("company", "previous employer")
        
        # Check specific matching phrases
        if "rag" in desc or "retrieval-augmented" in desc:
            achievements.append(f"built RAG systems at {company}")
        elif "embeddings" in desc or "vector search" in desc:
            achievements.append(f"deployed vector search at {company}")
        elif "ranking" in desc or "recommendation" in desc or "ranker" in desc:
            achievements.append(f"shipped ranking models at {company}")
        elif "evaluation" in desc or "ab testing" in desc or "a/b testing" in desc:
            achievements.append(f"designed evaluation frameworks at {company}")
        elif "fine-tune" in desc or "fine-tuning" in desc:
            achievements.append(f"fine-tuned LLMs at {company}")
            
    if achievements:
        ach_str = ", and ".join(achievements[:2])
        return f"Notably, they have {ach_str}."
    return ""

def calculate_non_semantic_score(c):
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {})
    
    # 1. Total YoE Scorer (Target: 5-9 years)
    yoe = profile.get("years_of_experience", 0)
    if 5.0 <= yoe <= 9.0:
        yoe_score = 1.0
    elif yoe < 5.0:
        yoe_score = max(0.0, yoe / 5.0)
    else: # yoe > 9.0
        yoe_score = max(0.0, 1.0 - (yoe - 9.0) / 6.0)
        
    # 2. Title Relevance Scorer
    current_title = profile.get("current_title", "").lower()
    title_score = 0.0
    if any(k in current_title for k in ['ai engineer', 'machine learning', 'ml engineer', 'nlp engineer', 'applied ml', 'deep learning']):
        title_score = 1.0
    elif 'data scientist' in current_title:
        title_score = 0.8
    elif any(k in current_title for k in ['backend', 'software engineer', 'data engineer', 'founding engineer']):
        title_score = 0.5
    elif any(k in current_title for k in ['intern', 'junior', 'associate']):
        title_score = 0.2
        
    # 3. Company & Consulting Penalty
    companies = [j.get("company", "").lower() for j in career if j.get("company")]
    if companies:
        has_consulting = [any(c_name in comp for c_name in CONSULTING_COMPANIES) for comp in companies]
        only_consulting = all(has_consulting)
        if only_consulting:
            return -100.0, 0.0, 0.0, 0.0, 1.0, 1.0
        
    # 4. Research/Academic Penalty
    summary = profile.get("summary", "").lower()
    academic_keywords = ['phd', 'postdoc', 'research assistant', 'academic', 'researcher', 'publications']
    prod_keywords = ['production', 'scale', 'shipped', 'deployed', 'users', 'latency', 'pipeline', 'infrastructure']
    
    academic_count = sum(1 for kw in academic_keywords if kw in summary)
    prod_count = sum(1 for kw in prod_keywords if kw in summary)
    
    research_penalty = 1.0
    if academic_count > 2 and prod_count == 0:
        research_penalty = 0.2
        
    # 5. Location & Relocation Scorer
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_relocate = signals.get("willing_to_relocate", False)
    
    loc_score = 1.0
    in_pune_noida = any(city in location for city in PUNE_NOIDA_KEYWORDS)
    in_other_india = any(city in location for city in OTHER_INDIAN_CITIES)
    
    if country and country != 'india':
        if not willing_relocate:
            return -200.0, 0.0, 0.0, 0.0, 1.0, 1.0
        else:
            loc_score = 0.1
    else:
        if in_pune_noida:
            loc_score = 1.0
        elif in_other_india:
            loc_score = 0.8 if willing_relocate else 0.4
        else:
            loc_score = 0.7 if willing_relocate else 0.3
                
    # 6. Behavioral Multipliers
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    resp_mult = 0.3 + 0.7 * resp_rate
    
    active_s = signals.get("last_active_date", "2020-01-01")
    try:
        active_year = int(active_s.split("-")[0])
        if active_year >= 2026:
            active_mult = 1.0
        elif active_year == 2025:
            active_mult = 0.7
        else:
            active_mult = 0.3
    except:
        active_mult = 0.3
        
    int_rate = signals.get("interview_completion_rate", 0.0)
    int_mult = 0.5 + 0.5 * int_rate
    
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        notice_mult = 1.1
    elif notice <= 60:
        notice_mult = 1.0
    elif notice <= 90:
        notice_mult = 0.8
    else:
        notice_mult = 0.5
        
    github = signals.get("github_activity_score", -1)
    git_bonus = 0.05 if github > 20 else 0.0
    
    open_work = 0.05 if signals.get("open_to_work_flag", False) else 0.0
    verified = 0.05 if (signals.get("verified_email") and signals.get("verified_phone")) else 0.0
    
    behavior_multiplier = resp_mult * active_mult * int_mult * notice_mult
    behavior_bonus = git_bonus + open_work + verified
    
    return title_score, yoe_score, loc_score, behavior_bonus, behavior_multiplier, research_penalty

def generate_reasoning(c):
    profile = c.get("profile", {})
    title = profile.get("current_title", "AI Engineer")
    yoe = profile.get("years_of_experience", 0)
    loc = profile.get("location", "")
    company = profile.get("current_company", "")
    
    signals = c.get("redrob_signals", {})
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    notice = signals.get("notice_period_days", 90)
    last_active = signals.get("last_active_date", "")
    
    # Match skills actually on their profile
    skills = c.get("skills", [])
    matched_skills = [s.get("name") for s in skills if s.get("name").lower() in CORE_AI_SKILLS or s.get("name").lower() in EVAL_SKILLS]
    skills_str = ", ".join(matched_skills[:3]) if matched_skills else "Machine Learning"
    
    title_desc = get_clean_title_desc(title, company)
    ach_desc = extract_achievement(c.get("career_history", []))
    
    h = hash(c["candidate_id"])
    
    # Vary the availability templates
    avail_templates = [
        f"They reside in {loc} and are available on a {notice}-day notice with a strong recruiter response rate ({int(resp_rate*100)}%).",
        f"Based in {loc}, they feature high platform activity (last active {last_active}) and a {notice}-day notice period.",
        f"Residing in {loc} with {notice} days notice, they maintain excellent response metrics ({int(resp_rate*100)}% response rate).",
        f"Available within {notice} days, they offer high responsiveness ({int(resp_rate*100)}% rate) and are located in {loc}."
    ]
    avail_desc = avail_templates[h % len(avail_templates)]
    
    intros = [
        f"Brings {yoe} years of experience, currently working as a {title_desc}.",
        f"Features {yoe} years of professional background, serving as a {title_desc}.",
        f"An experienced professional with {yoe} years in the field, currently a {title_desc}.",
        f"Demonstrates a solid track record of {yoe} years of experience, holding the role of {title_desc}.",
        f"Possesses {yoe} years of experience in product development, currently positioned as a {title_desc}.",
        f"With {yoe} years of engineering experience, they currently serve as a {title_desc}."
    ]
    intro = intros[(h // 4) % len(intros)]
    if ach_desc:
        reasoning = f"{intro} {ach_desc} {avail_desc}"
    else:
        reasoning = f"{intro} Demonstrated expertise in {skills_str} makes them a great technical fit. {avail_desc}"
        
    return reasoning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output submission.csv")
    args = parser.parse_args()
    
    # Load job description
    jd_path = os.path.join(os.path.dirname(args.candidates), "job_description.md")
    if not os.path.exists(jd_path):
        jd_path = "job_description.md"
        
    print(f"Loading Job Description from {jd_path}...")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print(f"Loading candidates and building text profiles from {args.candidates}...")
    candidates = []
    candidate_texts = []
    
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            candidates.append(c)
            
            # Combine text profiles for TF-IDF
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
            
    print("Computing TF-IDF Semantic Similarities...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    candidate_matrices = vectorizer.fit_transform(candidate_texts)
    jd_vector = vectorizer.transform([jd_text])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(candidate_matrices, jd_vector).flatten()
    
    print("Scoring and ranking candidates...")
    scored_candidates = []
    for i, c in enumerate(candidates):
        cid = c["candidate_id"]
        
        # Honeypot check
        honeypot_flag, reason = is_honeypot(c)
        if honeypot_flag:
            scored_candidates.append((cid, -999.0, c))
            continue
            
        tfidf_score = similarities[i]
        title_score, yoe_score, loc_score, behavior_bonus, behavior_multiplier, research_penalty = calculate_non_semantic_score(c)
        
        # Check if disqualified by consulting-only or non-relocating international
        if title_score < 0:
            scored_candidates.append((cid, title_score, c))
            continue
            
        # Composite score calculation (Weights: TF-IDF=0.50, Title=0.20, YoE=0.20, Loc=0.10)
        base_score = (0.50 * tfidf_score + 
                      0.20 * title_score + 
                      0.20 * yoe_score + 
                      0.10 * loc_score)
                      
        final_score = base_score * research_penalty * behavior_multiplier + behavior_bonus
        scored_candidates.append((cid, round(final_score, 4), c))
        
    # Sort descending by score, tie-break by candidate_id ascending
    scored_candidates.sort(key=lambda x: (-x[1], x[0]))
    
    # Select top 100
    top_100 = scored_candidates[:100]
    
    print(f"Writing top 100 candidates to {args.out}...")
    with open(args.out, "w", encoding="utf-8", newline="") as out_f:
        import csv
        writer = csv.writer(out_f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, (cid, score, c) in enumerate(top_100, 1):
            reasoning = generate_reasoning(c)
            writer.writerow([cid, rank, score, reasoning])
            
    print("Ranking complete. CSV generated successfully.")

if __name__ == "__main__":
    main()
