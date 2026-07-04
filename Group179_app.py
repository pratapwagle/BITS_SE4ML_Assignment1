import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


AUDIT_FILE = Path("resume_screening_audit.csv")


def get_streamlit():
    import streamlit as st

    return st


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_training_data() -> pd.DataFrame:
    data = [
        ("python machine learning pandas sklearn model training classification regression statistics nlp", "Data Scientist"),
        ("tensorflow pytorch feature engineering experimentation evaluation computer vision deep learning", "Data Scientist"),
        ("predictive modeling feature selection hypothesis testing data analysis notebooks python", "Data Scientist"),
        ("nlp transformers sentiment analysis model deployment experimentation metrics", "Data Scientist"),
        ("sql spark airflow kafka warehouse etl pipelines data lake orchestration", "Data Engineer"),
        ("bigquery snowflake ingestion batch streaming quality checks schema evolution", "Data Engineer"),
        ("data pipelines orchestration warehousing airflow spark dbt sql batch processing", "Data Engineer"),
        ("kafka streaming ingestion schema registry cloud warehouse lakehouse data engineering", "Data Engineer"),
        ("java spring boot rest api microservices backend docker kubernetes", "Backend Developer"),
        ("nodejs express graphql authentication mongodb scalable backend development", "Backend Developer"),
        ("backend api caching redis authentication service integration java spring", "Backend Developer"),
        ("rest endpoints message queues backend architecture nodejs express postgres", "Backend Developer"),
        ("selenium playwright test automation regression api testing jira quality", "QA Engineer"),
        ("test cases defect tracking automation framework performance testing quality assurance", "QA Engineer"),
        ("qa automation api validation selenium test plans bug triage regression suite", "QA Engineer"),
        ("playwright ui testing smoke testing exploratory testing defect logging", "QA Engineer"),
        ("react angular html css javascript ui ux responsive frontend typescript", "Frontend Developer"),
        ("frontend accessibility component library figma design system web performance", "Frontend Developer"),
        ("responsive ui frontend state management react typescript accessibility css", "Frontend Developer"),
        ("component design system figma handoff frontend performance optimization", "Frontend Developer"),
        ("aws terraform kubernetes docker ci cd observability linux grafana", "DevOps Engineer"),
        ("jenkins github actions infrastructure as code monitoring incident response cloud", "DevOps Engineer"),
        ("devops release automation infrastructure monitoring kubernetes terraform cloud reliability", "DevOps Engineer"),
        ("container orchestration ci cd observability logging linux automation aws", "DevOps Engineer"),
    ]
    return pd.DataFrame(data, columns=["resume_text", "job_role"])


def extract_resume_text(uploaded_file) -> str:
    from docx import Document
    from pypdf import PdfReader

    suffix = Path(uploaded_file.name).suffix.lower()
    payload = uploaded_file.getvalue()

    if suffix == ".txt":
        return payload.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(payload))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if suffix == ".docx":
        document = Document(io.BytesIO(payload))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n".join(paragraphs).strip()

    raise ValueError("Unsupported file type. Please upload a TXT, PDF, or DOCX resume.")


class ResumeClassifier:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    def train(self, X: List[str], y: List[str]):
        self.pipeline.fit([clean_text(text) for text in X], y)
        return self

    def predict(self, resume_text: str) -> Dict:
        cleaned = clean_text(resume_text)
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        predicted_role = str(self.pipeline.predict([cleaned])[0])
        ranking = sorted(
            zip(self.pipeline.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        return {
            "predicted_role": predicted_role,
            "confidence": round(float(max(probabilities)), 3),
            "role_ranking": [(str(role), round(float(score), 3)) for role, score in ranking],
        }


@dataclass
class ResumeSubmission:
    candidate_name: str
    resume_text: str
    source: str


class ResumeRepository:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path

    def save_result(self, submission: ResumeSubmission, result: Dict):
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_name": submission.candidate_name,
            "source": submission.source,
            "resume_characters": len(submission.resume_text),
            "predicted_role": result["predicted_role"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
        }
        history = self.list_results()
        updated = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
        updated.to_csv(self.storage_path, index=False)

    def list_results(self) -> pd.DataFrame:
        if self.storage_path.exists():
            return pd.read_csv(self.storage_path)
        return pd.DataFrame(
            columns=[
                "timestamp",
                "candidate_name",
                "source",
                "resume_characters",
                "predicted_role",
                "confidence",
                "explanation",
            ]
        )


class ResumeScoringService:
    KEYWORDS = {
        "Data Scientist": ["machine learning", "python", "statistics", "nlp", "tensorflow", "pytorch"],
        "Data Engineer": ["etl", "spark", "kafka", "warehouse", "airflow", "data lake"],
        "Backend Developer": ["java", "spring", "api", "microservices", "nodejs", "mongodb"],
        "QA Engineer": ["testing", "selenium", "playwright", "automation", "regression", "quality"],
        "Frontend Developer": ["react", "angular", "css", "frontend", "figma", "typescript"],
        "DevOps Engineer": ["docker", "kubernetes", "ci", "terraform", "aws", "grafana"],
    }

    def __init__(self, classifier: ResumeClassifier):
        self.classifier = classifier

    def score_resume(self, submission: ResumeSubmission) -> Dict:
        result = self.classifier.predict(submission.resume_text)
        predicted_role = result["predicted_role"]
        text = clean_text(submission.resume_text)
        matched_keywords = [
            keyword for keyword in self.KEYWORDS.get(predicted_role, []) if keyword in text
        ]
        result["explanation"] = "Matched keywords: " + (
            ", ".join(matched_keywords) if matched_keywords else "limited direct keyword match"
        )
        result["decision_note"] = (
            "Recommendation supports recruiters, but final shortlisting remains a human decision."
        )
        return result


class ResumeScreeningApplication:
    def __init__(self, service: ResumeScoringService, repository: ResumeRepository):
        self.service = service
        self.repository = repository

    def submit_resume(self, submission: ResumeSubmission) -> Dict:
        result = self.service.score_resume(submission)
        self.repository.save_result(submission, result)
        return result

    def audit_history(self) -> pd.DataFrame:
        return self.repository.list_results()


def build_model_summary() -> Dict:
    dataset = build_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["resume_text"],
        dataset["job_role"],
        test_size=0.5,
        random_state=42,
        stratify=dataset["job_role"],
    )
    model = ResumeClassifier().train(X_train.tolist(), y_train.tolist())
    predictions = [model.predict(text)["predicted_role"] for text in X_test]
    return {
        "classifier": ResumeClassifier().train(dataset["resume_text"].tolist(), dataset["job_role"].tolist()),
        "validation_accuracy": round(accuracy_score(y_test, predictions), 3),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
        "training_rows": len(dataset),
        "classes": sorted(dataset["job_role"].unique().tolist()),
    }


def load_application() -> Dict:
    st = get_streamlit()

    @st.cache_resource
    def _load_application() -> Dict:
        summary = build_model_summary()
        repository = ResumeRepository(AUDIT_FILE)
        service = ResumeScoringService(summary["classifier"])
        app = ResumeScreeningApplication(service, repository)
        return {"app": app, **summary}

    return _load_application()


def render_sidebar(summary: Dict, history: pd.DataFrame):
    st = get_streamlit()
    st.sidebar.header("Assignment Alignment")
    st.sidebar.write("Architectural patterns implemented:")
    st.sidebar.write("1. Layered Architecture")
    st.sidebar.write("2. Pipeline Pattern")

    st.sidebar.header("Model Snapshot")
    st.sidebar.metric("Validation Accuracy", summary["validation_accuracy"])
    st.sidebar.metric("Training Samples", summary["training_rows"])
    st.sidebar.write("Supported roles:")
    st.sidebar.write(", ".join(summary["classes"]))

    if not history.empty:
        st.sidebar.header("Audit Summary")
        st.sidebar.metric("Analyses Run", int(len(history)))
        st.sidebar.metric("Average Confidence", round(float(history["confidence"].mean()), 3))
        st.sidebar.metric("Latest Source", history.iloc[-1]["source"])


def render_history(history: pd.DataFrame):
    st = get_streamlit()
    st.subheader("Audit Log")
    if history.empty:
        st.info("No analyses have been stored yet. Run the app once to populate the audit log.")
        return

    st.dataframe(history.sort_values("timestamp", ascending=False), width="stretch")

    distribution = history["predicted_role"].value_counts().rename_axis("role").reset_index(name="count")
    st.subheader("Role Distribution")
    st.bar_chart(distribution.set_index("role"))


def main():
    st = get_streamlit()
    st.set_page_config(page_title="AI Resume Screening - Group 179", layout="wide")
    summary = load_application()
    application = summary["app"]
    history = application.audit_history()

    render_sidebar(summary, history)

    st.title("AI Resume Screening System")
    st.write(
        "Decision-support application for resume classification. The workflow supports "
        "resume upload or pasted text, predicts a suitable job role, explains the result, "
        "and stores each decision in an audit log."
    )

    with st.expander("Validation Report", expanded=False):
        st.code(summary["classification_report"])

    left, right = st.columns([1.2, 1])
    with left:
        candidate_name = st.text_input("Candidate Name", placeholder="Enter candidate name")
        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["txt", "pdf", "docx"],
            help="Supported formats: TXT, PDF, DOCX",
        )
        pasted_text = st.text_area("Paste Resume Text", height=220)

        if uploaded_file is not None:
            try:
                extracted_text = extract_resume_text(uploaded_file)
                st.success(f"Loaded resume from {uploaded_file.name}")
                st.caption(f"Extracted {len(extracted_text)} characters from the uploaded file.")
            except ValueError as error:
                extracted_text = ""
                st.error(str(error))
        else:
            extracted_text = ""

        if st.button("Analyze Resume", type="primary", width="stretch"):
            resume_text = pasted_text.strip() or extracted_text.strip()
            if not resume_text:
                st.error("Please upload a resume or paste resume text before analysis.")
            else:
                submission = ResumeSubmission(
                    candidate_name=candidate_name.strip() or "Candidate",
                    resume_text=resume_text,
                    source="upload" if extracted_text.strip() and not pasted_text.strip() else "text input",
                )
                result = application.submit_resume(submission)
                st.session_state["latest_result"] = result
                st.session_state["latest_submission"] = submission
                st.rerun()

    with right:
        st.subheader("Implementation Summary")
        st.write("Presentation Layer: Streamlit form and recruiter dashboard")
        st.write("Application Layer: submission orchestration and audit persistence")
        st.write("Business Layer: scoring rules, explanation, and human oversight")
        st.write("ML Layer: TF-IDF vectorizer and logistic regression classifier")
        st.write("Data Layer: CSV audit log for traceable predictions")

    latest_result: Optional[Dict] = st.session_state.get("latest_result")
    latest_submission: Optional[ResumeSubmission] = st.session_state.get("latest_submission")
    if latest_result and latest_submission:
        st.subheader("Prediction Result")
        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Predicted Role", latest_result["predicted_role"])
        metric_two.metric("Confidence", latest_result["confidence"])
        metric_three.metric("Source", latest_submission.source)
        st.write(latest_result["explanation"])
        st.info(latest_result["decision_note"])
        ranking_frame = pd.DataFrame(latest_result["role_ranking"], columns=["Role", "Probability"])
        st.table(ranking_frame)

    render_history(application.audit_history())


if __name__ == "__main__":
    main()
