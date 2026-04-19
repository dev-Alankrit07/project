import re

SKILL_DB = [
    "python", "java", "c", "c++", "sql", "mysql", "postgresql", "mongodb",
    "excel", "power bi", "tableau", "statistics", "data analysis",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "opencv", "scikit-learn",
    "html", "css", "javascript", "react", "node.js", "django", "flask",
    "spring boot", "git", "github",
    "aws", "azure", "gcp", "docker", "kubernetes", "linux", "ci/cd",
    "hadoop", "spark", "scala", "etl",
    "flutter", "dart", "firebase", "android", "kotlin", "swift", "ios",
    "networking", "cybersecurity", "testing", "manual testing",
    "automation testing", "selenium",
    "microcontrollers", "embedded systems", "iot",
    "database management", "database design", "problem solving",
    "communication", "cloud", "troubleshooting", "algorithms", "data structures",
    "opencv", "transformers", "llm", "generative ai"
]

DEGREE_DB = [
    "b.tech", "b.e", "bsc", "bca", "mca", "msc", "m.tech", "mba",
    "bachelor", "master"
]

CERTIFICATION_KEYWORDS = [
    "certification", "certified", "certificate", "aws certified",
    "google certified", "microsoft certified", "coursera", "udemy", "nptel"
]

PROJECT_KEYWORDS = [
    "project", "projects", "developed", "built", "created", "implemented",
    "designed", "prototype", "application", "system"
]

def extract_skills(text):
    text_lower = text.lower()
    found_skills = []

    for skill in SKILL_DB:
        if skill in text_lower:
            found_skills.append(skill)

    return list(set(found_skills))

def extract_degree(text):
    text_lower = text.lower()
    for degree in DEGREE_DB:
        if degree in text_lower:
            return degree
    return "not_found"

def extract_experience_years(text):
    text_lower = text.lower()

    patterns = [
        r'(\d+)\+?\s+years',
        r'(\d+)\+?\s+yrs',
        r'(\d+)\s+year',
        r'(\d+)\s+yr',
        r'experience\s*[:\-]?\s*(\d+)\+?\s*(?:years|yrs|year|yr)',
        r'(\d+)\+?\s*(?:years|yrs|year|yr)\s+of\s+experience'
    ]

    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        years.extend([int(x) for x in matches if str(x).isdigit()])

    return max(years) if years else 0

def extract_project_score(text):
    text_lower = text.lower()
    count = sum(text_lower.count(keyword) for keyword in PROJECT_KEYWORDS)

    if count >= 8:
        return 1.0
    elif count >= 5:
        return 0.8
    elif count >= 3:
        return 0.6
    elif count >= 1:
        return 0.4
    return 0.0

def extract_certification_score(text):
    text_lower = text.lower()
    count = sum(text_lower.count(keyword) for keyword in CERTIFICATION_KEYWORDS)

    if count >= 3:
        return 1.0
    elif count >= 2:
        return 0.7
    elif count >= 1:
        return 0.4
    return 0.0

def is_fresher(candidate_exp):
    return candidate_exp == 0