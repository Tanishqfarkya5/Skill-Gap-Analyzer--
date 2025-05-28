from flask import Flask, render_template, request
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


job_role_profiles = {
    "Data Analyst": ["SQL", "Python", "Excel", "Tableau", "Power BI", "Data Visualization", "Statistics", "Data Cleaning", "R"],
    "Software Engineer": ["Python", "Java", "C++", "Git", "OOP", "Data Structures", "Algorithms", "APIs", "Testing", "Docker", "Kubernetes"],
    "Digital Marketer": ["SEO", "Google Analytics", "Email Marketing", "Content Writing", "Social Media", "Copywriting", "Canva", "Google Ads"],
    "Financial Analyst": ["Excel", "Financial Modeling", "Accounting", "SQL", "Power BI", "Valuation", "Statistics", "Risk Management"],
    "HR Specialist": ["Recruiting", "Onboarding", "Communication", "HRIS Systems", "Excel", "Conflict Resolution", "Payroll"],
    "Commerce Graduate": ["Tally", "Accounting", "Business Communication", "Excel", "Marketing Basics", "Taxation", "Banking Concepts"],
    "Graphic Designer": ["Adobe Photoshop", "Illustrator", "Figma", "Canva", "Color Theory", "Typography", "Creativity"],
    "UX/UI Designer": ["Figma", "Wireframing", "User Research", "Prototyping", "Design Thinking", "HTML", "CSS"]
}

course_recommendations = {
    "SQL": ("SQL for Data Science – Coursera", "https://www.coursera.org/learn/sql-for-data-science"),
    "Excel": ("Mastering Excel – Udemy", "https://www.udemy.com/course/microsoft-excel-2013-from-beginner-to-advanced-and-beyond/"),
    "Tableau": ("Tableau A-Z – Udemy", "https://www.udemy.com/course/tableau10/"),
    "Power BI": ("Power BI Essentials – LinkedIn Learning", "https://www.linkedin.com/learning/power-bi-essential-training"),
    "Statistics": ("Statistics for Data Science – edX", "https://www.edx.org/course/statistics-and-r"),
    "Python": ("Python for Everybody – Coursera", "https://www.coursera.org/specializations/python"),
    "Java": ("Java Programming Masterclass – Udemy", "https://www.udemy.com/course/java-the-complete-java-developer-course/"),
    "C++": ("C++ for Beginners – Codecademy", "https://www.codecademy.com/learn/learn-c-plus-plus"),
    "Git": ("Git & GitHub Crash Course – Udemy", "https://www.udemy.com/course/git-and-github-crash-course/"),
    "SEO": ("SEO Specialization – Coursera", "https://www.coursera.org/specializations/seo"),
    "Google Analytics": ("Google Analytics for Beginners – Google Academy", "https://analytics.google.com/analytics/academy/course/6"),
    "Email Marketing": ("Email Marketing Basics – HubSpot Academy", "https://academy.hubspot.com/courses/email-marketing"),
    "Figma": ("Figma UX Design – Coursera", "https://www.coursera.org/learn/figma-design"),
    "Illustrator": ("Adobe Illustrator for Beginners – Udemy", "https://www.udemy.com/course/adobe-illustrator-cc-for-beginners/"),
    "Tally": ("Tally ERP9 Training – Udemy", "https://www.udemy.com/course/tally-erp9/"),
    "Accounting": ("Financial Accounting Fundamentals – Coursera", "https://www.coursera.org/learn/wharton-accounting"),
    "Canva": ("Graphic Design with Canva – Skillshare", "https://www.skillshare.com/classes/Graphic-Design-Basics-The-Complete-Guide/1681639975"),
    "Financial Modeling": ("Financial Modeling & Valuation – CFI", "https://courses.corporatefinanceinstitute.com/courses/financial-modeling-valuation-analyst-fmva-certification-program"),
    "Communication": ("Business Communication Skills – Coursera", "https://www.coursera.org/learn/business-communication"),
    "Docker": ("Docker for Beginners – Udemy", "https://www.udemy.com/course/docker-tutorial-for-beginners/"),
    "Kubernetes": ("Kubernetes Essentials – Coursera", "https://www.coursera.org/learn/google-kubernetes-engine"),
    "Google Ads": ("Google Ads Certification – Google Academy", "https://skillshop.withgoogle.com/partner/home")
}

def extract_text_from_pdf(file_stream):
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text, skill_list):
    found_skills = set()
    for skill in skill_list:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.add(skill)
    return list(found_skills)

def compute_similarity(student_skills, role_skills):
    if not student_skills or not role_skills:
        return 0.0
    student_embeds = model.encode(student_skills, convert_to_tensor=True)
    role_embeds = model.encode(role_skills, convert_to_tensor=True)
    cosine_scores = util.cos_sim(student_embeds, role_embeds)
    max_scores = cosine_scores.max(dim=0).values.cpu().numpy()
    avg_similarity = np.mean(max_scores)
    return float(avg_similarity) * 100

def analyze_skill_gap(student_skills, role_skills):
    matched_skills = list(set(student_skills).intersection(role_skills))
    missing_skills = list(set(role_skills) - set(student_skills))
    match_score = compute_similarity(student_skills, role_skills)
    return matched_skills, missing_skills, round(match_score, 2)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        uploaded_file = request.files.get("resume")
        selected_role = request.form.get("role")
        if uploaded_file and selected_role:
            resume_text = extract_text_from_pdf(uploaded_file)
            role_skills = job_role_profiles.get(selected_role, [])
            all_skills = set(skill for skills in job_role_profiles.values() for skill in skills)
            extracted_skills = extract_skills(resume_text, all_skills)
            matched, missing, score = analyze_skill_gap(extracted_skills, role_skills)

            recommendations = []
            for skill in missing:
                if skill in course_recommendations:
                    name, url = course_recommendations[skill]
                    recommendations.append((name, url))
                else:
                    recommendations.append((skill + " (No course found)", "#"))

            result = {
                "selected_role": selected_role,
                "match_score": score,
                "matched_skills": matched,
                "missing_skills": missing,
                "recommendations": recommendations
            }
    return render_template("index.html", result=result, roles=job_role_profiles.keys())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))



