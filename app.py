import streamlit as st
import requests
import json
import os
import PyPDF2
from groq import Groq
import asyncio
import sys
from bs4 import BeautifulSoup

# Windows fix for Playwright subprocess issues
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Connect to Ollama server (make sure `ollama serve` is running)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"

# Resumes (PDFs)
RESUME_MAP = {
    "SDE": "resume_SDE.pdf",
    "ML/AI/DS": "resume_ML.pdf"
}

def load_resume(resume_type):
    """Extract text from the PDF resume."""
    path = RESUME_MAP[resume_type]
    if not os.path.exists(path):
        return f"Error: Resume file {path} not found."

    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Load GitHub repos info for the given role
def load_repos(role):
    with open("repos.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    urls = data.get(role, [])
    repos = []
    for url in urls:
        name = url.split("/")[-1]
        repos.append({
            "name": name,
            "description": "",  # leave blank, LLM will infer relevance
            "url": url
        })
    return repos

# Combined scraper
def scrape_job_description(url):
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        text = "\n".join([p.get_text() for p in soup.find_all(["p", "li", "div"])])
        return text[:4000] if text else "Error: Could not extract job description."
    except Exception as e:
        return f"Error: {e}"

# Generate email using Ollama
def generate_email(resume, job_desc, repos, role):
    repo_text = "\n".join([f"- {r['name']} (link: {r['url']})" for r in repos])

    prompt = f"""
You are a professional career assistant.
Write a concise and personalized cold email (STRICTLY under 130 words) to a hiring manager for a {role} role. 
Use 2-3 projects from my resume and GitHub to demonstrate my skills.

Make sure the email:
- Address the recipient by name and briefly mention any mutual connections or specific aspects of the company that caught your attention.
- Clearly state your interest in working for the company and highlight one or two of your most relevant skills and accomplishments that align with their goals.
- Give a specific example of how you could positively impact the company or contribute to their success.
- Avoids exaggeration of my skills or experiences.
- References the job description for alignment.
- Keeps the tone professional, confident, and clear. 
- Does not mention my portfolio website and GitHub repos.
- Ends with a polite call-to-action.

Here is my resume:
{resume}

Here is the job description:
{job_desc}

Here are my GitHub projects:
{repo_text}

Now draft the final cold email (≤170 words):
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()

# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="Cold Email Generator", layout="centered")

st.title("📧 Cold Email Generator (with Robust Scraper)")

job_url = st.text_input("Enter job posting URL:")
job_desc_manual = st.text_area("Or paste the job description manually (if scraping fails):")
role = st.radio("Choose role type:", ["SDE", "ML/AI/DS"])

if st.button("Check Job Description"):
    if job_desc_manual.strip():
        job_desc = job_desc_manual
    else:
        with st.spinner("Fetching job description..."):
            job_desc = scrape_job_description(job_url)

    if "Error" in job_desc:
        st.error(job_desc)
    else:
        st.subheader("📄 Extracted Job Description (first 1000 chars)")
        st.text_area("Job Description:", job_desc[:1000], height=300)
        st.success("Job description fetched successfully!")

if st.button("Generate Email"):
    if job_desc_manual.strip():
        job_desc = job_desc_manual
    else:
        with st.spinner("Fetching job description..."):
            job_desc = scrape_job_description(job_url)

    if "Error" in job_desc:
        st.error(job_desc)
    else:
        with st.spinner("Generating personalized email..."):
            resume = load_resume(role)
            repos = load_repos(role)   # <-- fixed here
            email = generate_email(resume, job_desc, repos, role)

        st.subheader("✉️ Generated Cold Email")
        st.text_area("", email, height=400)
        st.success("Email generated! Copy and customize as needed.")
