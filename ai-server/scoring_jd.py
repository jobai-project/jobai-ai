# scoring_jd.py

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

TECH_KEYWORDS = [
    "Python", "Java", "Kotlin", "Swift", "JavaScript", "TypeScript",
    "Spring", "Spring Boot", "FastAPI", "Django", "Flask", "Node.js",
    "React", "Vue", "Angular", "Docker", "Kubernetes", "AWS", "GCP", "Azure",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "Spark",
    "PyTorch", "TensorFlow", "LLM", "CUDA", "MLOps", "NLP", "Git", "Linux",
    "Terraform", "Jenkins", "CI/CD",
]

# 가중치 (cs_ts_equal 기준 - 실험 결과 최적)
W_TS = 0.30
W_CS = 0.35
W_QS = 0.20


def extract_tech_from_text(text: str) -> list[str]:
    return [t for t in TECH_KEYWORDS
            if re.search(r"\b" + re.escape(t) + r"\b", text, re.IGNORECASE)]


def parse_jd(jd_text: str) -> dict:
    preferred_text = ""
    main_text = jd_text
    m = re.search(r"[◆▶*#]?\s*우대\s*사항.*", jd_text, re.DOTALL)
    if m:
        preferred_text = jd_text[m.start():]
        main_text = jd_text[:m.start()]

    required_years = None
    ym = re.search(r"경력\s*(\d+)년\s*이상", jd_text)
    if ym:
        required_years = int(ym.group(1))

    preferred_keywords = [
        k.strip() for k in re.findall(r"-\s*([^\n]+)", preferred_text)
        if len(k.strip()) > 2
    ]

    return {
        "main_text": main_text,
        "required_years": required_years,
        "preferred_keywords": preferred_keywords,
        "jd_techs": extract_tech_from_text(jd_text),
    }


def parse_resume(resume_text: str, resume_skills: list[str], experience_years: int) -> dict:
    return {
        "resume_text": resume_text,
        "skills": resume_skills,
        "experience_years": experience_years,
    }


def calc_ts(jd_parsed: dict, resume_parsed: dict) -> tuple:
    jd_techs = jd_parsed["jd_techs"]
    skills_lower = [s.lower() for s in resume_parsed["skills"]]
    matched = [t for t in jd_techs if t.lower() in skills_lower]
    missing = [t for t in jd_techs if t.lower() not in skills_lower]
    ts = len(matched) / len(jd_techs) if jd_techs else 0.0
    return float(ts), matched, missing


def calc_cs(jd_parsed: dict, resume_parsed: dict, jd_vec: list, resume_vec: list) -> float:
    jd_arr = np.array(jd_vec).reshape(1, -1)
    resume_arr = np.array(resume_vec).reshape(1, -1)
    cs = float(cos_sim(jd_arr, resume_arr)[0][0])
    return (cs + 1) / 2


def calc_qs(jd_parsed: dict, resume_parsed: dict) -> float:
    required_years = jd_parsed.get("required_years")
    resume_years = resume_parsed.get("experience_years", 0) or 0

    if required_years is None:
        career_score = 1.0
    elif resume_years >= required_years:
        career_score = 1.0
    else:
        career_score = resume_years / required_years

    preferred_kws = jd_parsed.get("preferred_keywords", [])
    skills = resume_parsed.get("skills", [])
    if preferred_kws:
        preferred_matched = sum(
            1 for kw in preferred_kws
            if any(s.lower() in kw.lower() for s in skills)
        )
        preferred_score = preferred_matched / len(preferred_kws)
    else:
        preferred_score = 0.0

    return float(career_score * 0.7 + preferred_score * 0.3)


def apply_gp(base_score: float, missing_skills: list, jd_techs_count: int) -> tuple:
    score = float(base_score)
    penalties = []
    missing_count = len(missing_skills)
    missing_ratio = missing_count / jd_techs_count if jd_techs_count > 0 else 0

    if missing_count > 0:
        score -= 15
        penalties.append(f"필수 기술 {missing_count}개 누락 -15")
    if missing_count > 0:
        score = min(score, 60)
        penalties.append("score cap 60 적용")
    if missing_ratio > 0.5:
        score = min(score, 40)
        penalties.append("필수 기술 절반 이상 누락 score cap 40 적용")

    return float(max(0, score)), penalties


def score_private(
    jd_text: str,
    resume_text: str,
    jd_vec: list,
    resume_vec: list,
    resume_skills: list[str],
    experience_years: int,
) -> dict:
    jd_parsed = parse_jd(jd_text)
    resume_parsed = parse_resume(resume_text, resume_skills, experience_years)

    ts, matched_skills, missing_skills = calc_ts(jd_parsed, resume_parsed)
    cs = calc_cs(jd_parsed, resume_parsed, jd_vec, resume_vec)
    qs = calc_qs(jd_parsed, resume_parsed)

    base_score = 100 * (W_TS * ts + W_CS * cs + W_QS * qs)
    final_score, penalties = apply_gp(base_score, missing_skills, len(jd_parsed["jd_techs"]))

    required_year = jd_parsed.get("required_years") or 0
    skill_note = " (필수 기술 미보유로 점수 제한)" if missing_skills else ""
    score_reason = " ".join([
        f"필수 기술 {len(matched_skills)}/{len(jd_parsed['jd_techs'])}개 일치 ({', '.join(matched_skills[:3])}).",
        f"경력 {required_year}년 요구 중 {experience_years}년 보유.",
        f"필수 요구사항 충족{skill_note}.",
    ])

    return {
        "score": round(final_score, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills[:5],
        "career_met": (experience_years >= required_year) if required_year else True,
        "score_reason": score_reason,
        "penalties": penalties,
        "model_version": "kosimcse-final",
    }