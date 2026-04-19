import os
import joblib
import pandas as pd

from parser import extract_resume_text
from scorer import calculate_resume_score
from roles import JOB_ROLES, ROLE_ALIASES


def normalize_role(job_role):
    job_role = job_role.strip().lower()
    return ROLE_ALIASES.get(job_role, job_role)


def load_ml_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def generate_reason(result):
    reasons = []

    if len(result["matched_skills"]) >= 3:
        reasons.append("strong skill match")
    elif len(result["matched_skills"]) >= 1:
        reasons.append("partial skill match")
    else:
        reasons.append("low skill match")

    if result["candidate_type"] == "Fresher":
        if result["project_score"] >= 60:
            reasons.append("good project background")
        if result["certification_score"] >= 40:
            reasons.append("useful certifications")
    else:
        if result["experience_score"] >= 80:
            reasons.append("relevant experience")
        elif result["experience_score"] >= 50:
            reasons.append("moderate experience")

    if result["missing_skills"]:
        reasons.append(f"missing skills: {', '.join(result['missing_skills'])}")

    return "; ".join(reasons)


def main():
    model = load_ml_model()

    job_role = input("Enter job role: ").strip().lower()
    job_role = normalize_role(job_role)

    resumes_folder = os.path.join("data", "resumes")

    if job_role not in JOB_ROLES:
        print("\nJob role not found in system.")
        print("\nAvailable roles are:")
        for role in sorted(JOB_ROLES.keys()):
            print("-", role.title())
        return

    role_data = JOB_ROLES[job_role]
    results = []

    if not os.path.exists(resumes_folder):
        print("The folder data/resumes does not exist.")
        return

    for filename in os.listdir(resumes_folder):
        file_path = os.path.join(resumes_folder, filename)

        if os.path.isfile(file_path):
            try:
                resume_text = extract_resume_text(file_path)
                result = calculate_resume_score(resume_text, role_data)
                result["resume_name"] = filename

                # ML prediction if model exists
                if model is not None:
                    features = pd.DataFrame([{
                        "skill_score": result["skill_score"],
                        "degree_score": result["degree_score"],
                        "experience_score": result["experience_score"],
                        "project_score": result["project_score"],
                        "certification_score": result["certification_score"],
                        "semantic_score": result["final_score"]  # temporary proxy
                    }])

                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0][1]

                    result["ml_prediction"] = "Accepted" if prediction == 1 else "Rejected"
                    result["confidence"] = round(probability * 100, 2)
                else:
                    result["ml_prediction"] = "Model Not Loaded"
                    result["confidence"] = 0.0

                result["reason"] = generate_reason(result)
                results.append(result)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if not results:
        print("No resumes found in data/resumes")
        return

    results.sort(key=lambda x: x["final_score"], reverse=True)

    print(f"\n===== Ranked Candidates for {job_role.title()} =====")
    for i, r in enumerate(results, start=1):
        print(f"\n{i}. {r['resume_name']}")
        print(f"   Candidate Type   : {r['candidate_type']}")
        print(f"   Final Score      : {r['final_score']}")
        print(f"   Status           : {r['status']}")
        print(f"   ML Decision      : {r['ml_prediction']}")
        print(f"   Confidence       : {r['confidence']}%")
        print(f"   Matched Skills   : {r['matched_skills']}")
        print(f"   Missing Skills   : {r['missing_skills']}")
        print(f"   Experience       : {r['candidate_experience']} years")
        print(f"   Degree           : {r['candidate_degree']}")
        print(f"   Project Score    : {r['project_score']}")
        print(f"   Certification    : {r['certification_score']}")
        print(f"   Reason           : {r['reason']}")

    best = results[0]
    print("\n===== Best Pick =====")
    print(
        f"{best['resume_name']} is the best match for {job_role.title()} "
        f"with score {best['final_score']} ({best['status']})"
    )

    if model is None:
        print("\nNote: model.pkl was not found, so only rule-based scoring was used.")
        print("Run: python src\\train_model.py")


if __name__ == "__main__":
    main()