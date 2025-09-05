import streamlit as st
import requests
import json
import os
import PyPDF2
import asyncio
import sys
from langchain_community.document_loaders import WebBaseLoader

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

def safe_json_parse(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            raise

# ---------------- Setup ---------------- #

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

RESUME_MAP = {
    "SDE": "resume_SDE.pdf",
    "ML/AI/DS": "resume_ML.pdf"
}

# ---------------- Custom Embeddings ---------------- #

# class LocalHFEmbeddings(Embeddings):
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name, device="cpu")  # Force CPU

#     def embed_documents(self, texts):
#         return self.model.encode(texts, convert_to_numpy=True).tolist()

#     def embed_query(self, text):
#         return self.model.encode([text], convert_to_numpy=True)[0].tolist()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
# ---------------- Helpers ---------------- #

def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def build_resume_db(role, save_dir="resume_db"):
    path = RESUME_MAP[role]
    if not os.path.exists(path):
        return None, f"Error: Resume file {path} not found."

    os.makedirs(save_dir, exist_ok=True)
    db_path = f"{save_dir}/{role}"

    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        return db, None

    text = extract_text_from_pdf(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(db_path)
    return db, None

def build_repos_db(role, save_dir="repos_db"):
    os.makedirs(save_dir, exist_ok=True)
    db_path = f"{save_dir}/{role}"

    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        return db, None

    with open("repos.json", "r", encoding="utf-8") as f:
        repo_map = json.load(f)

    repos = repo_map.get(role, [])
    docs = []
    for url in repos:
        name = url.split("/")[-1]
        docs.append(Document(page_content=f"{name} project: {url}"))

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(db_path)
    return db, None

def scrape_job_description(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

        # Combine text from docs
        text = "\n".join([doc.page_content for doc in docs])
        print(text)
        # Trim to 4000 chars (to stay safe for prompt size)
        return text if text else "Error: Empty job description."
    except Exception as e:
        return f"Error: Could not fetch job description ({e}). Please paste it manually."

def extract_job_info(raw_text):
    prompt = f"""
### SCRAPED TEXT FROM WEBSITE:
{raw_text}

### INSTRUCTION:
The scraped text is from the career's page of a website.
Your job is to extract the job postings from scraped text and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
Only return the valid JSON.
### Just provide extracted JSON without any explanation or code or PREAMBLE:
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600,
    )
    raw = response.choices[0].message.content
    print(safe_json_parse(raw))
    return safe_json_parse(raw)

def retrieve_relevant_chunks(resume_db, repos_db, job_info, k=3):
    query_text = job_info["description"] + " " + " ".join(job_info["skills"])
    resume_hits = resume_db.similarity_search(query_text, k=k)
    repo_hits = repos_db.similarity_search(query_text, k=k)
    resume_text = "\n".join([doc.page_content for doc in resume_hits])
    repo_text = "\n".join([doc.page_content for doc in repo_hits])
    return resume_text, repo_text

def generate_email(resume_db, repos_db, job_info):
    resume_text, repo_text = retrieve_relevant_chunks(resume_db, repos_db, job_info, k=3)

    prompt = f"""
I am a student applying for a {job_info['role']} position at your company.
Please draft a short and professional cold email (≤170 words) from my perspective.

Inputs:
- Relevant resume highlights:
{resume_text}

- Relevant GitHub projects:
{repo_text}

- Job description:
{job_info['description']}

Guidelines:
- Write in first person, as if I am introducing myself.
- Mention my projects and skills naturally in sentences, not bullet points.
- Align my background with the job's required skills: {job_info['skills']}.
- Keep tone polite, confident, and concise.
- If possible, acknowledge a recent company achievement from the description.
- Do not exceed 170 words.

Now write the final email:
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()

# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="Cold Email Generator", layout="centered")
st.title("📧 Cold Email Generator (Vectorstore + Structured JD)")

job_url = st.text_input("Enter job posting URL:")
job_desc_manual = st.text_area("Or paste the job description manually (if scraping fails):")
role = st.radio("Choose role type:", ["SDE", "ML/AI/DS"])

if st.button("Check Job Description"):
    if job_desc_manual.strip():
        raw_text = job_desc_manual
    else:
        with st.spinner("Fetching job description..."):
            raw_text = scrape_job_description(job_url)

    if raw_text and not raw_text.startswith("Error"):
        with st.spinner("Extracting job details..."):
            job_info = extract_job_info(raw_text)
        st.json(job_info)
        st.success("Job details extracted successfully!")
    else:
        st.error(raw_text)

if st.button("Generate Email"):
    if job_desc_manual.strip():
        raw_text = job_desc_manual
    else:
        with st.spinner("Fetching job description..."):
            raw_text = scrape_job_description(job_url)

    if raw_text and not raw_text.startswith("Error"):
        with st.spinner("Extracting job details..."):
            job_info = extract_job_info(raw_text)

        with st.spinner("Building/Loading vectorstores..."):
            resume_db, err = build_resume_db(role)
            repos_db, _ = build_repos_db(role)

        if err:
            st.error(err)
        else:
            with st.spinner("Generating personalized email..."):
                email = generate_email(resume_db, repos_db, job_info)

            st.subheader("✉️ Generated Cold Email")
            st.text_area("", email, height=400)
            st.success("Email generated! Copy and customize as needed.")
    else:
        st.error("Error: Job description is required to generate email.")
