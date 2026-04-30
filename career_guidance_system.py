"""
Career Guidance System - AI-Powered Professional Development Platform
=====================================================================
Author: Aryan Sharma
Version: 2.4.1
Last Updated: March 2025

A comprehensive career guidance platform that leverages machine learning
to provide personalized career path recommendations, skill gap analysis,
resume scoring, and job market insights.

Tech Stack:
    - Python 3.11
    - FastAPI (REST API backend)
    - SQLAlchemy (ORM)
    - scikit-learn (ML models)
    - OpenAI API (LLM integration)
    - Redis (caching)
    - PostgreSQL (primary database)
"""

import os
import json
import logging
import hashlib
import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

# ─── Logging Configuration ────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("career_guidance")


# ─── Constants & Configuration ────────────────────────────────────────────────

MODEL_VERSION = "cgs-v2.4.1"
MAX_CAREER_PATHS = 5
SKILL_SIMILARITY_THRESHOLD = 0.72
RESUME_SCORE_WEIGHTS = {
    "keywords": 0.35,
    "experience_relevance": 0.30,
    "education_match": 0.20,
    "formatting": 0.15,
}
SUPPORTED_INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Education",
    "Marketing", "Engineering", "Law", "Design",
    "Data Science", "Product Management", "Consulting",
]
EXPERIENCE_LEVELS = ["Entry", "Junior", "Mid", "Senior", "Lead", "Principal", "Executive"]


# ─── Enums ────────────────────────────────────────────────────────────────────

class CareerStage(Enum):
    STUDENT        = "student"
    EARLY_CAREER   = "early_career"
    MID_CAREER     = "mid_career"
    SENIOR         = "senior"
    EXECUTIVE      = "executive"
    CAREER_CHANGER = "career_changer"


class SkillCategory(Enum):
    TECHNICAL    = "technical"
    SOFT         = "soft"
    DOMAIN       = "domain"
    LEADERSHIP   = "leadership"
    CERTIFICATION= "certification"


class JobMarketTrend(Enum):
    GROWING      = "growing"
    STABLE       = "stable"
    DECLINING    = "declining"
    EMERGING     = "emerging"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Skill:
    name: str
    category: SkillCategory
    proficiency: float          # 0.0 – 1.0
    years_of_experience: float
    is_certified: bool = False
    last_used: Optional[datetime.date] = None

    def decay_factor(self) -> float:
        """Reduces skill weight if not recently used."""
        if self.last_used is None:
            return 0.85
        delta_days = (datetime.date.today() - self.last_used).days
        return max(0.5, 1.0 - (delta_days / 1825))  # 5-year full decay

    def effective_proficiency(self) -> float:
        return round(self.proficiency * self.decay_factor(), 4)


@dataclass
class UserProfile:
    user_id: str
    name: str
    email: str
    career_stage: CareerStage
    current_role: str
    target_roles: List[str]
    industry: str
    skills: List[Skill] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    work_experience: List[Dict[str, Any]] = field(default_factory=list)
    location: str = "Remote"
    salary_expectation: Optional[int] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    def skill_vector(self) -> np.ndarray:
        """Returns a normalized skill proficiency vector for ML computations."""
        return np.array([s.effective_proficiency() for s in self.skills])

    def total_experience_years(self) -> float:
        total = 0.0
        for exp in self.work_experience:
            start = datetime.datetime.strptime(exp.get("start_date", "2020-01"), "%Y-%m")
            end_str = exp.get("end_date", "present")
            end = datetime.datetime.utcnow() if end_str == "present" else \
                  datetime.datetime.strptime(end_str, "%Y-%m")
            total += (end - start).days / 365.25
        return round(total, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "career_stage": self.career_stage.value,
            "current_role": self.current_role,
            "target_roles": self.target_roles,
            "industry": self.industry,
            "total_experience_years": self.total_experience_years(),
            "skill_count": len(self.skills),
            "location": self.location,
        }


@dataclass
class CareerPath:
    path_id: str
    title: str
    description: str
    industry: str
    required_skills: List[str]
    average_salary: Dict[str, int]      # {"entry": 45000, "senior": 120000}
    growth_rate: float                   # annual % growth
    market_trend: JobMarketTrend
    time_to_transition: int             # estimated months
    match_score: float = 0.0
    skill_gap: List[str] = field(default_factory=list)
    recommended_resources: List[str] = field(default_factory=list)


@dataclass
class ResumeAnalysis:
    resume_id: str
    overall_score: float
    keyword_score: float
    experience_score: float
    education_score: float
    formatting_score: float
    missing_keywords: List[str]
    suggestions: List[str]
    ats_compatibility: bool
    estimated_interview_rate: float     # probability 0-1
    analyzed_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


# ─── Skill Gap Analyzer ───────────────────────────────────────────────────────

class SkillGapAnalyzer:
    """
    Computes the gap between a user's current skills and
    the skills required for their target career paths.
    Uses cosine similarity on skill embeddings.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._skill_embeddings: Dict[str, np.ndarray] = {}
        logger.info("SkillGapAnalyzer initialized.")

    def _get_skill_embedding(self, skill_name: str) -> np.ndarray:
        """Lazy-loads or generates a mock skill embedding."""
        if skill_name not in self._skill_embeddings:
            # In production: query vector DB (Pinecone / pgvector)
            np.random.seed(int(hashlib.md5(skill_name.encode()).hexdigest(), 16) % (2**32))
            self._skill_embeddings[skill_name] = np.random.randn(128)
        return self._skill_embeddings[skill_name]

    def compute_similarity(self, skill_a: str, skill_b: str) -> float:
        emb_a = self._get_skill_embedding(skill_a).reshape(1, -1)
        emb_b = self._get_skill_embedding(skill_b).reshape(1, -1)
        return float(cosine_similarity(emb_a, emb_b)[0][0])

    def analyze_gap(
        self,
        user_skills: List[Skill],
        required_skills: List[str],
    ) -> Tuple[List[str], List[str], float]:
        """
        Returns:
            missing_skills  - skills not present in user profile
            partial_skills  - skills present but below threshold
            readiness_score - overall readiness (0.0 – 1.0)
        """
        user_skill_names = {s.name.lower() for s in user_skills}
        missing, partial = [], []
        matched_count = 0

        for req in required_skills:
            req_lower = req.lower()
            # Exact match
            if req_lower in user_skill_names:
                matched_count += 1
                continue
            # Semantic similarity match
            best_sim = max(
                (self.compute_similarity(req, us) for us in user_skill_names),
                default=0.0,
            )
            if best_sim >= SKILL_SIMILARITY_THRESHOLD:
                matched_count += 0.75  # partial credit for semantic match
                partial.append(req)
            else:
                missing.append(req)

        readiness = matched_count / max(len(required_skills), 1)
        return missing, partial, round(readiness, 4)


# ─── Career Path Recommender ──────────────────────────────────────────────────

class CareerPathRecommender:
    """
    ML-based career path recommendation engine.
    Uses a gradient boosting model trained on anonymized career trajectory data.
    """

    CAREER_PATHS_DB: List[Dict[str, Any]] = [
        {
            "path_id": "cp_001",
            "title": "Machine Learning Engineer",
            "industry": "Technology",
            "required_skills": ["Python", "TensorFlow", "PyTorch", "SQL", "Statistics", "Git"],
            "average_salary": {"entry": 85000, "mid": 130000, "senior": 185000},
            "growth_rate": 22.5,
            "market_trend": JobMarketTrend.GROWING,
            "time_to_transition": 12,
            "recommended_resources": [
                "Deep Learning Specialization – Coursera",
                "Hands-On ML with Scikit-Learn – O'Reilly",
                "fast.ai Practical Deep Learning",
            ],
        },
        {
            "path_id": "cp_002",
            "title": "Product Manager",
            "industry": "Technology",
            "required_skills": ["Product Strategy", "Agile", "Data Analysis", "Roadmapping", "Stakeholder Communication"],
            "average_salary": {"entry": 75000, "mid": 115000, "senior": 160000},
            "growth_rate": 10.8,
            "market_trend": JobMarketTrend.STABLE,
            "time_to_transition": 8,
            "recommended_resources": [
                "Inspired by Marty Cagan",
                "Product School PM Certification",
                "Reforge Product Strategy Program",
            ],
        },
        {
            "path_id": "cp_003",
            "title": "Data Scientist",
            "industry": "Technology",
            "required_skills": ["Python", "R", "Machine Learning", "SQL", "Statistics", "Data Visualization"],
            "average_salary": {"entry": 78000, "mid": 120000, "senior": 165000},
            "growth_rate": 17.2,
            "market_trend": JobMarketTrend.GROWING,
            "time_to_transition": 10,
            "recommended_resources": [
                "IBM Data Science Professional Certificate",
                "Kaggle – Competitions & Courses",
                "Python for Data Analysis – Wes McKinney",
            ],
        },
        {
            "path_id": "cp_004",
            "title": "DevOps / Cloud Engineer",
            "industry": "Technology",
            "required_skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Linux", "Terraform"],
            "average_salary": {"entry": 80000, "mid": 125000, "senior": 170000},
            "growth_rate": 19.4,
            "market_trend": JobMarketTrend.GROWING,
            "time_to_transition": 14,
            "recommended_resources": [
                "AWS Solutions Architect Associate",
                "Linux Foundation Kubernetes (CKA)",
                "HashiCorp Terraform Associate",
            ],
        },
        {
            "path_id": "cp_005",
            "title": "UX/UI Designer",
            "industry": "Design",
            "required_skills": ["Figma", "User Research", "Prototyping", "Wireframing", "Accessibility"],
            "average_salary": {"entry": 55000, "mid": 90000, "senior": 130000},
            "growth_rate": 8.6,
            "market_trend": JobMarketTrend.STABLE,
            "time_to_transition": 6,
            "recommended_resources": [
                "Google UX Design Certificate",
                "Don't Make Me Think – Steve Krug",
                "Interaction Design Foundation Courses",
            ],
        },
    ]

    def __init__(self):
        self.gap_analyzer = SkillGapAnalyzer()
        self.model: Optional[GradientBoostingRegressor] = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        model_path = "models/career_recommender.pkl"
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.info("Pre-trained model not found. Using heuristic scoring.")

    def _heuristic_match_score(
        self,
        user: UserProfile,
        path: Dict[str, Any],
        readiness: float,
    ) -> float:
        """
        Computes a weighted match score when ML model is unavailable.
        Factors: skill readiness, industry alignment, experience level, salary fit.
        """
        score = readiness * 0.55

        # Industry alignment bonus
        if user.industry.lower() == path["industry"].lower():
            score += 0.15

        # Experience-level adjustment
        exp = user.total_experience_years()
        if exp < 2 and path["average_salary"]["entry"] > 0:
            score += 0.10
        elif 2 <= exp < 6:
            score += 0.15
        else:
            score += 0.10

        # Target role alignment
        target_keywords = " ".join(user.target_roles).lower()
        if any(kw in path["title"].lower() for kw in target_keywords.split()):
            score += 0.20

        return round(min(score, 1.0), 4)

    def recommend(self, user: UserProfile) -> List[CareerPath]:
        """
        Returns a ranked list of career path recommendations
        tailored to the user's profile.
        """
        recommendations: List[CareerPath] = []

        for path_data in self.CAREER_PATHS_DB:
            missing, partial, readiness = self.gap_analyzer.analyze_gap(
                user.skills,
                path_data["required_skills"],
            )
            match_score = self._heuristic_match_score(user, path_data, readiness)

            career_path = CareerPath(
                path_id=path_data["path_id"],
                title=path_data["title"],
                description=f"Transition into {path_data['title']} within {path_data['time_to_transition']} months.",
                industry=path_data["industry"],
                required_skills=path_data["required_skills"],
                average_salary=path_data["average_salary"],
                growth_rate=path_data["growth_rate"],
                market_trend=path_data["market_trend"],
                time_to_transition=path_data["time_to_transition"],
                match_score=match_score,
                skill_gap=missing + partial,
                recommended_resources=path_data["recommended_resources"],
            )
            recommendations.append(career_path)

        recommendations.sort(key=lambda p: p.match_score, reverse=True)
        return recommendations[:MAX_CAREER_PATHS]


# ─── Resume Analyzer ──────────────────────────────────────────────────────────

class ResumeAnalyzer:
    """
    Analyzes resumes for ATS compatibility, keyword density,
    experience relevance, and overall presentation quality.
    """

    ATS_BLOCKLIST = ["table", "text box", "header", "footer", "image", "graph"]
    HIGH_IMPACT_KEYWORDS = [
        "led", "built", "scaled", "optimized", "reduced", "increased",
        "delivered", "designed", "architected", "launched", "shipped",
    ]

    def __init__(self):
        self.classifier = self._init_classifier()

    def _init_classifier(self) -> Optional[RandomForestClassifier]:
        model_path = "models/resume_classifier.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        logger.warning("Resume classifier model not found. Using rule-based scoring.")
        return None

    def _extract_keywords(self, text: str) -> List[str]:
        """Naive keyword extraction — in production, use spaCy NER."""
        words = text.lower().split()
        stopwords = {"the", "and", "of", "to", "in", "a", "is", "for", "on", "with"}
        return [w.strip(".,;:()") for w in words if w not in stopwords and len(w) > 3]

    def _score_formatting(self, resume_text: str) -> float:
        """
        Checks for ATS-unfriendly elements and structure issues.
        Returns a 0.0-1.0 formatting score.
        """
        score = 1.0
        for bad_element in self.ATS_BLOCKLIST:
            if bad_element in resume_text.lower():
                score -= 0.12
        if len(resume_text) < 400:
            score -= 0.20   # Too short
        if len(resume_text) > 6000:
            score -= 0.10   # Too long (> ~2 pages)
        return max(0.0, round(score, 4))

    def _score_keywords(self, resume_text: str, job_description: str) -> Tuple[float, List[str]]:
        resume_kws = set(self._extract_keywords(resume_text))
        job_kws    = set(self._extract_keywords(job_description))
        if not job_kws:
            return 0.5, []
        matched   = resume_kws & job_kws
        missing   = list(job_kws - resume_kws)[:10]
        score     = len(matched) / len(job_kws)
        # Bonus for high-impact action verbs
        impact_bonus = sum(0.02 for kw in self.HIGH_IMPACT_KEYWORDS if kw in resume_kws)
        return round(min(score + impact_bonus, 1.0), 4), missing

    def analyze(
        self,
        resume_text: str,
        job_description: str,
        user: Optional[UserProfile] = None,
    ) -> ResumeAnalysis:
        resume_id = hashlib.sha256(resume_text.encode()).hexdigest()[:12]

        kw_score, missing_kws = self._score_keywords(resume_text, job_description)
        fmt_score             = self._score_formatting(resume_text)

        # Heuristic experience & education scores
        exp_score = 0.75 if user and user.total_experience_years() > 2 else 0.50
        edu_score = 0.80 if "bachelor" in resume_text.lower() or \
                            "master"   in resume_text.lower() else 0.60

        overall = (
            kw_score  * RESUME_SCORE_WEIGHTS["keywords"] +
            exp_score * RESUME_SCORE_WEIGHTS["experience_relevance"] +
            edu_score * RESUME_SCORE_WEIGHTS["education_match"] +
            fmt_score * RESUME_SCORE_WEIGHTS["formatting"]
        )

        suggestions = []
        if kw_score  < 0.50: suggestions.append("Add more job-specific keywords from the JD.")
        if fmt_score < 0.70: suggestions.append("Remove tables/images for better ATS compatibility.")
        if exp_score < 0.60: suggestions.append("Quantify achievements with metrics (%, $, time).")
        if edu_score < 0.70: suggestions.append("Include relevant certifications or coursework.")

        ats_ok = fmt_score >= 0.70 and len(self.ATS_BLOCKLIST) == 0

        return ResumeAnalysis(
            resume_id=resume_id,
            overall_score=round(overall, 4),
            keyword_score=kw_score,
            experience_score=exp_score,
            education_score=edu_score,
            formatting_score=fmt_score,
            missing_keywords=missing_kws,
            suggestions=suggestions,
            ats_compatibility=ats_ok,
            estimated_interview_rate=round(overall * 0.65, 4),
        )


# ─── Career Guidance Engine (Facade) ─────────────────────────────────────────

class CareerGuidanceEngine:
    """
    Main entry point for the Career Guidance System.
    Orchestrates profile analysis, path recommendations, and resume scoring.
    """

    def __init__(self):
        self.recommender    = CareerPathRecommender()
        self.resume_analyzer = ResumeAnalyzer()
        logger.info(f"CareerGuidanceEngine ready | model_version={MODEL_VERSION}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_recommendations(self, user: UserProfile) -> Dict[str, Any]:
        """Returns top career path recommendations for a user."""
        paths = self.recommender.recommend(user)
        return {
            "user_id": user.user_id,
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "model_version": MODEL_VERSION,
            "recommendations": [
                {
                    "rank": i + 1,
                    "path_id": p.path_id,
                    "title": p.title,
                    "match_score": p.match_score,
                    "market_trend": p.market_trend.value,
                    "growth_rate_pct": p.growth_rate,
                    "avg_salary_senior": p.average_salary.get("senior"),
                    "months_to_transition": p.time_to_transition,
                    "skill_gap": p.skill_gap,
                    "resources": p.recommended_resources,
                }
                for i, p in enumerate(paths)
            ],
        }

    def analyze_resume(
        self,
        resume_text: str,
        job_description: str,
        user: Optional[UserProfile] = None,
    ) -> Dict[str, Any]:
        """Returns a detailed resume analysis report."""
        analysis = self.resume_analyzer.analyze(resume_text, job_description, user)
        return {
            "resume_id": analysis.resume_id,
            "scores": {
                "overall": analysis.overall_score,
                "keywords": analysis.keyword_score,
                "experience": analysis.experience_score,
                "education": analysis.education_score,
                "formatting": analysis.formatting_score,
            },
            "ats_compatible": analysis.ats_compatibility,
            "estimated_interview_rate": f"{analysis.estimated_interview_rate * 100:.1f}%",
            "missing_keywords": analysis.missing_keywords,
            "suggestions": analysis.suggestions,
            "analyzed_at": analysis.analyzed_at.isoformat(),
        }

    def generate_roadmap(
        self,
        user: UserProfile,
        target_path: CareerPath,
    ) -> Dict[str, Any]:
        """
        Generates a month-by-month learning roadmap to bridge the skill gap
        between the user's current state and target career path.
        """
        months = target_path.time_to_transition
        gap    = target_path.skill_gap
        milestones = []

        chunk_size = max(1, len(gap) // max(months // 3, 1))
        for i in range(0, len(gap), chunk_size):
            month_num = (i // chunk_size + 1) * (months // max(len(gap) // chunk_size, 1))
            milestones.append({
                "month": min(month_num, months),
                "focus_skills": gap[i: i + chunk_size],
                "action_items": [
                    f"Complete online course for {skill}" for skill in gap[i: i + chunk_size]
                ] + ["Build a portfolio project incorporating these skills."],
            })

        return {
            "user_id": user.user_id,
            "target_role": target_path.title,
            "total_months": months,
            "roadmap": milestones,
            "resources": target_path.recommended_resources,
            "estimated_salary_on_completion": target_path.average_salary.get("entry"),
        }


# ─── Demo / Quick Test ────────────────────────────────────────────────────────

def _demo():
    engine = CareerGuidanceEngine()

    # Build a sample user
    user = UserProfile(
        user_id="usr_demo_001",
        name="Priya Mehta",
        email="priya.mehta@example.com",
        career_stage=CareerStage.EARLY_CAREER,
        current_role="Software Developer",
        target_roles=["Data Scientist", "Machine Learning Engineer"],
        industry="Technology",
        skills=[
            Skill("Python", SkillCategory.TECHNICAL, 0.80, 2.5, last_used=datetime.date(2025, 1, 15)),
            Skill("SQL",    SkillCategory.TECHNICAL, 0.70, 2.0, last_used=datetime.date(2025, 2, 1)),
            Skill("Git",    SkillCategory.TECHNICAL, 0.85, 3.0, last_used=datetime.date(2025, 3, 10)),
            Skill("Statistics", SkillCategory.DOMAIN, 0.55, 1.0),
        ],
        work_experience=[
            {"title": "Junior Developer", "company": "TechCorp", "start_date": "2022-07", "end_date": "present"},
        ],
        education=[{"degree": "B.Tech Computer Science", "institution": "NIT Delhi", "year": 2022}],
        location="Bangalore, India",
        salary_expectation=1_200_000,
    )

    # Get recommendations
    recs = engine.get_recommendations(user)
    print("\n🎯 Career Recommendations")
    print("=" * 60)
    for rec in recs["recommendations"]:
        print(f"  #{rec['rank']}  {rec['title']:30s}  Match: {rec['match_score']:.0%}  Trend: {rec['market_trend']}")

    # Analyze a sample resume
    sample_resume = """
    Priya Mehta | priya.mehta@example.com | Bangalore, India
    B.Tech Computer Science – NIT Delhi (2022)

    EXPERIENCE
    Junior Developer @ TechCorp (Jul 2022 – Present)
    - Built and maintained REST APIs using Python and FastAPI.
    - Optimized database queries resulting in 30% latency reduction.
    - Collaborated with cross-functional teams using Agile methodology.

    SKILLS
    Python, SQL, Git, FastAPI, PostgreSQL, Docker (basic)
    """
    sample_jd = "We are looking for a Data Scientist with Python, Machine Learning, SQL, TensorFlow, Statistics, and Data Visualization skills."

    resume_result = engine.analyze_resume(sample_resume, sample_jd, user)
    print("\n📄 Resume Analysis")
    print("=" * 60)
    for k, v in resume_result["scores"].items():
        print(f"  {k.capitalize():20s}: {v:.0%}")
    print(f"  ATS Compatible      : {resume_result['ats_compatible']}")
    print(f"  Interview Rate Est. : {resume_result['estimated_interview_rate']}")
    print(f"  Suggestions         :")
    for s in resume_result["suggestions"]:
        print(f"    • {s}")


if __name__ == "__main__":
    _demo()
