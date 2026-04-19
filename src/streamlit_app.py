import os
import tempfile
import joblib
import pandas as pd
import streamlit as st

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
        reasons.append("missing skills: " + ", ".join(result["missing_skills"]))

    return "; ".join(reasons)


st.set_page_config(page_title="AI Resume Shortlisting System", layout="wide")
st.title("AI-Based Resume Shortlisting System")

model = load_ml_model()

role_options = sorted(JOB_ROLES.keys())
selected_role = st.selectbox("Select Job Role", role_options)

uploaded_files = st.file_uploader(
    "Upload Resume Files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if st.button("Analyze Resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        role_key = normalize_role(selected_role)
        role_data = JOB_ROLES[role_key]
        results = []

        for uploaded_file in uploaded_files:
            suffix = "." + uploaded_file.name.split(".")[-1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            try:
                resume_text = extract_resume_text(temp_path)
                result = calculate_resume_score(resume_text, role_data)
                result["resume_name"] = uploaded_file.name

                if model is not None:
                    features = pd.DataFrame([{
                        "skill_score": result["skill_score"],
                        "degree_score": result["degree_score"],
                        "experience_score": result["experience_score"],
                        "project_score": result["project_score"],
                        "certification_score": result["certification_score"],
                        "semantic_score": result["final_score"]
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
                st.error(f"Error processing {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if results:
            results.sort(key=lambda x: x["final_score"], reverse=True)

            st.subheader("Ranked Candidates")

            table_data = []
            for r in results:
                table_data.append({
                    "Resume": r["resume_name"],
                    "Type": r["candidate_type"],
                    "Final Score": r["final_score"],
                    "Status": r["status"],
                    "ML Decision": r["ml_prediction"],
                    "Confidence": r["confidence"]
                })

            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            best = results[0]
            st.success(
                f"Best Pick: {best['resume_name']} | "
                f"Score: {best['final_score']} | "
                f"Status: {best['status']}"
            )

            for r in results:
                with st.expander(r["resume_name"]):
                    st.write(f"**Candidate Type:** {r['candidate_type']}")
                    st.write(f"**Final Score:** {r['final_score']}")
                    st.write(f"**Status:** {r['status']}")
                    st.write(f"**ML Decision:** {r['ml_prediction']}")
                    st.write(f"**Confidence:** {r['confidence']}%")
                    st.write(f"**Matched Skills:** {r['matched_skills']}")
                    st.write(f"**Missing Skills:** {r['missing_skills']}")
                    st.write(f"**Experience:** {r['candidate_experience']} years")
                    st.write(f"**Degree:** {r['candidate_degree']}")
                    st.write(f"**Project Score:** {r['project_score']}")
                    st.write(f"**Certification Score:** {r['certification_score']}")
                    st.write(f"**Reason:** {r['reason']}")

if model is None:
    st.info("model.pkl not found. Run: python src\\train_model.py")