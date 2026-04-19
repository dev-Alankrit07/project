from extractor import (
    extract_skills,
    extract_degree,
    extract_experience_years,
    extract_project_score,
    extract_certification_score,
    is_fresher
)

def skill_match_score(candidate_skills, required_skills):
    if not required_skills:
        return 0.0

    matched = set(candidate_skills).intersection(set(required_skills))
    return len(matched) / len(required_skills)

def degree_match_score(candidate_degree, required_degree):
    if required_degree == "not_found":
        return 1.0

    candidate_degree = candidate_degree.lower()
    required_degree = required_degree.lower()

    if required_degree in candidate_degree or candidate_degree in required_degree:
        return 1.0
    return 0.0

def experience_match_score(candidate_exp, min_required_exp, max_required_exp):
    # Fresher-friendly role
    if min_required_exp == 0 and candidate_exp == 0:
        return 1.0

    # Candidate below minimum experience
    if candidate_exp < min_required_exp:
        if candidate_exp == 0:
            return 0.4
        return max(candidate_exp / max(min_required_exp, 1), 0.4)

    # Candidate within desired range
    if min_required_exp <= candidate_exp <= max_required_exp:
        return 1.0

    # Candidate above max range - still acceptable but slightly reduced
    if candidate_exp > max_required_exp:
        return 0.9

    return 0.0

def shortlist_status(score):
    if score >= 80:
        return "Shortlisted"
    elif score >= 60:
        return "Review"
    return "Rejected"

def calculate_resume_score(resume_text, role_data):
    candidate_skills = extract_skills(resume_text)
    candidate_degree = extract_degree(resume_text)
    candidate_exp = extract_experience_years(resume_text)
    project_score = extract_project_score(resume_text)
    certification_score = extract_certification_score(resume_text)
    fresher_flag = is_fresher(candidate_exp)

    required_skills = role_data["skills"]
    required_degree = role_data["degree"]
    min_required_exp = role_data["min_experience"]
    max_required_exp = role_data["max_experience"]

    skill_score = skill_match_score(candidate_skills, required_skills)
    degree_score = degree_match_score(candidate_degree, required_degree)
    exp_score = experience_match_score(candidate_exp, min_required_exp, max_required_exp)

    # Fresher-specific weighting
    if fresher_flag:
        final_score = (
            0.45 * skill_score +
            0.20 * degree_score +
            0.10 * exp_score +
            0.15 * project_score +
            0.10 * certification_score
        ) * 100
    else:
        final_score = (
            0.40 * skill_score +
            0.15 * degree_score +
            0.25 * exp_score +
            0.10 * project_score +
            0.10 * certification_score
        ) * 100

    matched_skills = list(set(candidate_skills).intersection(set(required_skills)))
    missing_skills = list(set(required_skills) - set(candidate_skills))

    return {
        "candidate_skills": candidate_skills,
        "candidate_degree": candidate_degree,
        "candidate_experience": candidate_exp,
        "required_skills": required_skills,
        "required_degree": required_degree,
        "min_required_experience": min_required_exp,
        "max_required_experience": max_required_exp,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "project_score": round(project_score * 100, 2),
        "certification_score": round(certification_score * 100, 2),
        "skill_score": round(skill_score * 100, 2),
        "degree_score": round(degree_score * 100, 2),
        "experience_score": round(exp_score * 100, 2),
        "final_score": round(final_score, 2),
        "status": shortlist_status(final_score),
        "candidate_type": "Fresher" if fresher_flag else "Experienced"
    }