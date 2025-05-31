from flask import Flask, render_template, request
import fitz  # PyMuPDF
import os
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates')
)


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
    # Data Analyst & Financial Analyst
    "SQL": ("SQL for Data Science – Coursera", "https://www.coursera.org/learn/sql-for-data-science"),
    "Excel": ("Excel Skills for Business – Coursera", "https://www.coursera.org/specializations/excel"),
    "Tableau": ("Tableau Data Visualization – Coursera", "https://www.coursera.org/learn/data-visualization-tableau"),
    "Power BI": ("Power BI Essentials – LinkedIn Learning", "https://www.linkedin.com/learning/power-bi-essential-training"),
    "Data Visualization": ("Fundamentals of Data Visualization – Coursera", "https://www.coursera.org/learn/fundamentals-of-data-visualization"),
    "Statistics": ("Statistics with Python – Coursera", "https://www.coursera.org/specializations/statistics-with-python"),
    "Data Cleaning": ("Data Cleaning in Python – DataCamp", "https://www.datacamp.com/courses/cleaning-data-in-python"),
    "R": ("R Programming – Coursera", "https://www.coursera.org/learn/r-programming"),
    "Financial Modeling": ("Financial Modeling & Valuation – CFI", "https://courses.corporatefinanceinstitute.com/courses/financial-modeling-valuation-analyst-fmva-certification-program"),
    "Accounting": ("Financial Accounting Fundamentals – Coursera", "https://www.coursera.org/learn/wharton-accounting"),
    "Valuation": ("Valuation and Financial Analysis – Coursera", "https://www.coursera.org/learn/valuation-and-financial-analysis"),
    "Risk Management": ("Introduction to Risk Management – Coursera", "https://www.coursera.org/learn/risk-management"),

    # Software Engineer
    "Python": ("Python for Everybody – Coursera", "https://www.coursera.org/specializations/python"),
    "Java": ("Java Programming and Software Engineering – Coursera", "https://www.coursera.org/specializations/java-programming"),
    "C++": ("Learn C++ – Codecademy", "https://www.codecademy.com/learn/learn-c-plus-plus"),
    "Git": ("Git & GitHub Crash Course – freeCodeCamp", "https://www.freecodecamp.org/news/git-and-github-crash-course/"),
    "OOP": ("Object-Oriented Programming in Java – Coursera", "https://www.coursera.org/learn/object-oriented-java"),
    "Data Structures": ("Data Structures and Algorithms – Coursera", "https://www.coursera.org/specializations/data-structures-algorithms"),
    "Algorithms": ("Algorithms Specialization – Coursera", "https://www.coursera.org/specializations/algorithms"),
    "APIs": ("API Development – Udacity", "https://www.udacity.com/course/api-development--nd803"),
    "Testing": ("Software Testing – Udacity", "https://www.udacity.com/course/software-testing--cs258"),
    "Docker": ("Docker for Beginners – Udemy", "https://www.udemy.com/course/docker-tutorial-for-beginners/"),
    "Kubernetes": ("Architecting with Kubernetes – Coursera", "https://www.coursera.org/learn/gcp-architecture-kubernetes"),

    # Digital Marketer
    "SEO": ("SEO Fundamentals – Coursera", "https://www.coursera.org/learn/seo-fundamentals"),
    "Google Analytics": ("Google Analytics for Beginners – Google", "https://analytics.google.com/analytics/academy/course/6"),
    "Email Marketing": ("Email Marketing – HubSpot Academy", "https://academy.hubspot.com/courses/email-marketing"),
    "Content Writing": ("Content Strategy – Coursera", "https://www.coursera.org/learn/content-strategy"),
    "Social Media": ("Social Media Marketing – Coursera", "https://www.coursera.org/specializations/social-media-marketing"),
    "Copywriting": ("Copywriting for Beginners – Udemy", "https://www.udemy.com/course/copywriting-secrets/"),
    "Canva": ("Graphic Design with Canva – Skillshare", "https://www.skillshare.com/en/classes/Canva-Masterclass-Beginner-to-Advanced/785490614"),
    "Google Ads": ("Google Ads Certification – Skillshop", "https://skillshop.withgoogle.com/"),

    # HR Specialist
    "Recruiting": ("Technical Recruiting – LinkedIn Learning", "https://www.linkedin.com/learning/technical-recruiting"),
    "Onboarding": ("Employee Onboarding – Coursera", "https://www.coursera.org/learn/onboarding-employees"),
    "Communication": ("Business Communication – Coursera", "https://www.coursera.org/learn/business-communication"),
    "HRIS Systems": ("HRIS Fundamentals – Udemy", "https://www.udemy.com/course/hris/"),
    "Conflict Resolution": ("Conflict Management – Coursera", "https://www.coursera.org/learn/conflict-resolution-skills"),
    "Payroll": ("Payroll Management – Udemy", "https://www.udemy.com/course/payroll-management-system/"),

    # Commerce Graduate
    "Tally": ("Tally ERP9 Training – Udemy", "https://www.udemy.com/course/tally-erp9/"),
    "Business Communication": ("Business English Communication – Coursera", "https://www.coursera.org/learn/business-english-communication"),
    "Marketing Basics": ("Introduction to Marketing – Coursera", "https://www.coursera.org/learn/wharton-marketing"),
    "Taxation": ("Introduction to Taxation – Coursera", "https://www.coursera.org/learn/uva-darden-federal-income-taxation"),
    "Banking Concepts": ("Introduction to Banking – edX", "https://www.edx.org/course/banking-fundamentals"),

    # Graphic Designer
    "Adobe Photoshop": ("Photoshop Fundamentals – Coursera", "https://www.coursera.org/learn/photoshop-fundamentals"),
    "Illustrator": ("Adobe Illustrator CC – Udemy", "https://www.udemy.com/course/adobe-illustrator-cc-for-beginners/"),
    "Figma": ("Figma UX Design – Coursera", "https://www.coursera.org/learn/figma-design"),
    "Color Theory": ("Color Theory Basics – Udemy", "https://www.udemy.com/course/graphic-design-color-theory/"),
    "Typography": ("Typography and Design – Coursera", "https://www.coursera.org/learn/typography"),
    "Creativity": ("Creative Thinking – LinkedIn Learning", "https://www.linkedin.com/learning/creativity-for-all"),

    # UX/UI Designer
    "Wireframing": ("Wireframes for Web Design – Coursera", "https://www.coursera.org/lecture/web-design-strategy/wireframes-TzDZW"),
    "User Research": ("UX Research at Scale – Coursera", "https://www.coursera.org/learn/ux-research-at-scale"),
    "Prototyping": ("Prototyping and Design – Interaction Design Foundation", "https://www.interaction-design.org/courses/how-to-create-your-first-wireframe-and-prototype"),
    "Design Thinking": ("Design Thinking for Innovation – Coursera", "https://www.coursera.org/learn/uva-darden-design-thinking-innovation"),
    "HTML": ("HTML Basics – freeCodeCamp", "https://www.freecodecamp.org/learn/responsive-web-design/#basic-html-and-html5"),
    "CSS": ("CSS Basics – freeCodeCamp", "https://www.freecodecamp.org/learn/responsive-web-design/#basic-css")
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
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))




