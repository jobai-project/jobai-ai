# scoring_ncs.py
# 공기업 NCS 점수 산출 로직
# 원본: 현재_공고_matching_score_test.ipynb

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ALERT_THRESHOLD = 70

DEFAULT_WEIGHTS = {
    "Kw": 0.25,
    "Cs": 0.35,
    "Ws": 0.40,
}

# ── 유틸 ──────────────────────────────────────────

def compact_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def safe_join(parts):
    return " ".join(compact_text(p) for p in parts if compact_text(p))


def ratio(a, b):
    return float(a) / float(b) if b > 0 else 0.0


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


# ── 키워드 사전 ───────────────────────────────────

SKILL_KEYWORDS = [
    "Python", "Java", "JavaScript", "TypeScript", "Node.js", "React", "Vue",
    "Spring", "Spring Boot", "전자정부프레임워크", "Django", "FastAPI",
    "SQL", "MySQL", "PostgreSQL", "Oracle", "DB", "데이터베이스",
    "Linux", "Windows Server", "Docker", "Kubernetes", "Git",
    "API", "REST API", "서버", "백엔드", "프론트엔드", "웹",
    "정보시스템", "전산", "프로그래밍", "R", "Pandas", "Numpy",
    "라우터", "스위치", "VPN", "방화벽", "AMI"
]

KNOWLEDGE_KEYWORDS = [
    "정보통신", "정보기술", "시스템", "소프트웨어", "SW",
    "데이터", "빅데이터", "AI", "인공지능", "머신러닝",
    "분석", "보안", "정보보호", "개인정보", "취약점",
    "네트워크", "통신", "TCP/IP", "클라우드", "인프라",
    "운영", "유지보수", "장애 대응", "백업", "복구",
    "프로젝트관리", "IT프로젝트", "정보화사업", "요구사항",
    "품질관리", "감리", "사업관리", "RFP", "ISP", "ISMP",
    "데이터 품질", "데이터 표준화", "공공데이터"
]

PREFERRED_KEYWORDS = [
    "정보처리기사", "정보보안기사", "SQLD", "SQLP", "ADsP", "ADP",
    "빅데이터분석기사", "리눅스마스터", "네트워크관리사", "CCNA", "CCNP", "PMP"
]

ALIASES = {
    "spring boot": ["spring boot", "spring"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "postgresql": ["postgresql", "postgres"],
    "데이터베이스": ["데이터베이스", "database", "db"],
    "정보보호": ["정보보호", "보안", "security"],
    "네트워크": ["네트워크", "network", "tcp/ip"],
    "클라우드": ["클라우드", "aws", "azure", "gcp"],
}

CLUSTER_KEYWORDS = {
    "보안/정보보호": ["개인정보보호", "정보보호", "보안관제", "침해대응", "취약점", "ISMS", "암호화", "보안"],
    "DB/데이터관리": ["데이터베이스", "DB관리", "DBA", "SQL", "Oracle", "PostgreSQL"],
    "데이터/AI": ["빅데이터", "데이터분석", "머신러닝", "인공지능", "딥러닝", "자연어처리", "AI"],
    "네트워크/통신": ["정보통신설비", "통신설비", "네트워크구축", "네트워크운영", "무선통신", "네트워크"],
    "인프라/시스템운영": ["클라우드운영", "정보시스템운영", "시스템운영", "서버운영", "IT인프라", "클라우드", "서버"],
    "시스템개발/SW": ["소프트웨어개발", "정보시스템개발", "응용프로그램", "웹개발", "API개발", "소프트웨어", "웹"],
    "임베디드/HW": ["임베디드", "펌웨어", "IoT", "하드웨어", "회로설계"],
    "기획/PM": ["정보화기획", "IT서비스기획", "서비스기획", "사업기획", "PM", "PO"],
}

RESUME_CLUSTER_KEYWORDS = {
    "보안/정보보호": ["정보보호", "보안", "침해대응", "취약점"],
    "DB/데이터관리": ["데이터베이스", "DBA", "SQL"],
    "데이터/AI": ["데이터분석", "머신러닝", "인공지능", "AI", "딥러닝"],
    "네트워크/통신": ["네트워크", "통신", "TCP/IP"],
    "인프라/시스템운영": ["클라우드", "서버운영", "인프라", "DevOps"],
    "시스템개발/SW": ["백엔드", "프론트엔드", "웹개발", "앱개발", "소프트웨어"],
    "임베디드/HW": ["임베디드", "펌웨어", "하드웨어"],
    "기획/PM": ["기획", "PM", "PO", "서비스기획"],
}


# ── 텍스트 정규화 ─────────────────────────────────

def lower_text(text):
    return compact_text(text).lower()


def keyword_variants(keyword):
    k = str(keyword).lower()
    variants = {k}
    if k in ALIASES:
        variants.update([x.lower() for x in ALIASES[k]])
    return variants


def contains_keyword(text, keyword):
    t = lower_text(text)
    return any(v in t for v in keyword_variants(keyword))


def extract_keywords_by_dict(text, keyword_list):
    found = [kw for kw in keyword_list if contains_keyword(text, kw)]
    result, seen = [], set()
    for kw in found:
        key = kw.lower()
        if key not in seen:
            seen.add(key)
            result.append(kw)
    return result


def normalize_skills(skills):
    if skills is None:
        return []
    if isinstance(skills, dict):
        result = []
        for v in skills.values():
            if isinstance(v, list):
                result.extend(v)
            elif isinstance(v, str):
                result.append(v)
        return [compact_text(x) for x in result if compact_text(x)]
    if isinstance(skills, list):
        return [compact_text(x) for x in skills if compact_text(x)]
    if isinstance(skills, str):
        return [x.strip() for x in re.split(r"[,/|]", skills) if x.strip()]
    return []


def extract_required_years(text):
    patterns = [
        r"경력\s*(\d+)\s*년",
        r"(\d+)\s*년\s*이상",
        r"(\d+)\s*년\s*이상의\s*경력",
    ]
    for pat in patterns:
        m = re.search(pat, str(text or ""))
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def extract_required_certs(text):
    return [kw for kw in PREFERRED_KEYWORDS if contains_keyword(text, kw)]


# ── 클러스터 분류 ─────────────────────────────────

def get_job_cluster(text: str) -> str:
    scores = {}
    for cluster, keywords in CLUSTER_KEYWORDS.items():
        scores[cluster] = sum(1 for kw in keywords if kw in text)
    winner = max(scores, key=scores.get)
    return winner if scores[winner] > 0 else "UNKNOWN"


def get_resume_cluster(resume_text: str) -> str:
    scores = {}
    for cluster, keywords in RESUME_CLUSTER_KEYWORDS.items():
        scores[cluster] = sum(1 for kw in keywords if kw in resume_text)
    winner = max(scores, key=scores.get)
    return winner if scores[winner] > 0 else "UNKNOWN"


# ── 이력서 텍스트 변환 ────────────────────────────

def resume_to_text(resume: dict) -> str:
    skills = normalize_skills(resume.get("skills"))
    skill_text = "기술 스택: " + ", ".join(skills) if skills else ""

    certs = resume.get("certs") or resume.get("certificates") or []
    if isinstance(certs, str):
        certs = [certs]
    cert_text = "자격증: " + ", ".join(map(str, certs)) if certs else ""

    project_texts = []
    for p in resume.get("projects", []) or []:
        if isinstance(p, dict):
            tech = p.get("tech", [])
            if isinstance(tech, list):
                tech = " ".join(map(str, tech))
            project_texts.append(safe_join([
                p.get("title", ""),
                p.get("role", ""),
                p.get("description", ""),
                tech,
            ]))

    exp_texts = []
    for e in resume.get("experience", []) or []:
        if isinstance(e, dict):
            exp_texts.append(safe_join([
                e.get("company", ""),
                e.get("role", ""),
                e.get("description", ""),
            ]))

    career = f"경력 {resume.get('experience_years')}년" if resume.get("experience_years") is not None else ""

    return safe_join([
        career,
        skill_text,
        resume.get("summary", ""),
        " ".join(project_texts),
        " ".join(exp_texts),
        cert_text,
    ])


# ── 점수 계산 ─────────────────────────────────────

def keyword_match_score(resume: dict, jd_text: str) -> dict:
    resume_text = resume_to_text(resume)
    skills = normalize_skills(resume.get("skills"))
    resume_skill_text = " ".join(skills)

    skill_kws = extract_keywords_by_dict(jd_text, SKILL_KEYWORDS)
    knowledge_kws = extract_keywords_by_dict(jd_text, KNOWLEDGE_KEYWORDS)

    matched_skills = [kw for kw in skill_kws if contains_keyword(resume_skill_text, kw)]
    missing_skills = [kw for kw in skill_kws if kw not in matched_skills]
    matched_knowledge = [kw for kw in knowledge_kws if contains_keyword(resume_text, kw)]

    skill_score = ratio(len(matched_skills), len(skill_kws))
    knowledge_score = ratio(len(matched_knowledge), len(knowledge_kws))
    kw = 0.7 * skill_score + 0.3 * knowledge_score

    return {
        "Kw": float(kw),
        "skill_kws": skill_kws,
        "knowledge_kws": knowledge_kws,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_knowledge": matched_knowledge,
    }


def structural_score(resume: dict, jd_text: str, kw_info: dict) -> dict:
    resume_text = resume_to_text(resume)

    # 클러스터 매칭
    job_cluster = get_job_cluster(jd_text)
    resume_cluster = get_resume_cluster(resume_text)
    cluster_match = 1.0 if job_cluster == resume_cluster else (0.5 if resume_cluster != "UNKNOWN" else 0.0)

    # 기술 스킬
    skills = normalize_skills(resume.get("skills"))
    resume_skill_text = " ".join(skills)
    skill_kws = kw_info["skill_kws"]
    matched_skills = kw_info["matched_skills"]
    skill_part = ratio(len(matched_skills), len(skill_kws))

    # 지식
    knowledge_part = ratio(len(kw_info["matched_knowledge"]), len(kw_info["knowledge_kws"]))
    task_part = 0.60 * cluster_match + 0.40 * knowledge_part

    # 경력
    required_years = extract_required_years(jd_text)
    resume_years = _safe_float(resume.get("experience_years"), 0.0)
    if required_years is None:
        career_part = 1.0
        required_years_display = 0
    else:
        career_part = min(1.0, resume_years / max(required_years, 1))
        required_years_display = required_years

    # 자격
    required_certs = extract_required_certs(jd_text)
    resume_certs = resume.get("certs") or resume.get("certificates") or []
    if isinstance(resume_certs, str):
        resume_certs = [resume_certs]
    resume_cert_text = " ".join(map(str, resume_certs))
    matched_certs = [cert for cert in required_certs if contains_keyword(resume_cert_text, cert)]
    cert_part = ratio(len(matched_certs), len(required_certs))

    attitude_part = 1.0

    ws = (
        0.30 * skill_part
        + 0.25 * task_part
        + 0.20 * career_part
        + 0.15 * cert_part
        + 0.10 * attitude_part
    )

    return {
        "Ws": float(ws),
        "skill_part": float(skill_part),
        "task_part": float(task_part),
        "career_part": float(career_part),
        "cert_part": float(cert_part),
        "resume_cluster": resume_cluster,
        "job_cluster": job_cluster,
        "required_years": required_years_display,
        "resume_years": resume_years,
        "required_certs": required_certs,
        "matched_certs": matched_certs,
        "missing_certs": [c for c in required_certs if c not in matched_certs],
    }


def apply_gap_penalty(base_score: float, kw_info: dict, ws_info: dict) -> tuple:
    score = float(base_score)
    penalties = []

    total_skills = len(kw_info["skill_kws"])
    matched_skills = len(kw_info["matched_skills"])
    skill_ratio = ratio(matched_skills, total_skills)

    if total_skills >= 2 and skill_ratio < 0.25:
        score -= 10
        penalties.append("핵심 NCS 키워드 충족률 낮음 -10")

    required_years = ws_info["required_years"]
    resume_years = ws_info["resume_years"]
    if required_years and resume_years < required_years:
        if resume_years <= required_years * 0.5:
            score = min(score, 65)
            penalties.append("경력 요건 크게 미달: score cap 65")
        else:
            score -= 8
            penalties.append("경력 요건 일부 미달 -8")

    if ws_info["required_certs"] and not ws_info["matched_certs"]:
        score = min(score, 60)
        penalties.append("요구 자격/우대 자격 미보유: score cap 60")

    resume_cluster = ws_info["resume_cluster"]
    job_cluster = ws_info["job_cluster"]
    if resume_cluster == "UNKNOWN" and job_cluster != "UNKNOWN":
        score = min(score, 55)
        penalties.append("비IT/일반사무 이력서: IT 공고 score cap 55")

    return max(0.0, min(100.0, score)), penalties


def make_score_reason(kw_info: dict, ws_info: dict, penalties: list) -> str:
    total_skill_count = len(kw_info["skill_kws"])
    matched_skill_count = len(kw_info["matched_skills"])
    matched_skills = kw_info["matched_skills"]

    skill_preview = ", ".join(matched_skills[:3]) if matched_skills else "일치 키워드 없음"
    required_year = ws_info["required_years"]
    resume_year = ws_info["resume_years"]
    job_cluster = ws_info["job_cluster"]

    cert_note = " 요구 자격/우대 자격 미충족." if ws_info["missing_certs"] else ""
    penalty_note = " ".join(penalties) if penalties else "별도 감점 없음."

    return " ".join([
        f"NCS 필요기술 {matched_skill_count}/{total_skill_count}개 일치 ({skill_preview}).",
        f"경력 {required_year}년 요구 중 {resume_year}년 보유.",
        f"{job_cluster} 직무 기준으로 평가.",
        cert_note,
        f"penalty: {penalty_note}",
    ]).strip()


# ── 최종 점수 산출 ────────────────────────────────

def score_ncs(
    jd_text: str,
    resume: dict,
    jd_vec: list,
    resume_vec: list,
    weights: dict = None,
) -> dict:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Cs
    jd_arr = np.array(jd_vec).reshape(1, -1)
    resume_arr = np.array(resume_vec).reshape(1, -1)
    raw_cs = float(np.dot(jd_arr, resume_arr.T)[0][0])
    cs = max(0.0, min(1.0, raw_cs))

    # Kw, Ws
    kw_info = keyword_match_score(resume, jd_text)
    ws_info = structural_score(resume, jd_text, kw_info)

    kw = kw_info["Kw"]
    ws = ws_info["Ws"]

    base_score = 100.0 * (
        weights["Kw"] * kw
        + weights["Cs"] * cs
        + weights["Ws"] * ws
    )

    final_score, penalties = apply_gap_penalty(base_score, kw_info, ws_info)
    score_reason = make_score_reason(kw_info, ws_info, penalties)

    return {
        "score": round(final_score, 1),
        "base_score": round(base_score, 1),
        "above_threshold": final_score >= ALERT_THRESHOLD,
        "Kw": round(kw, 4),
        "Cs": round(cs, 4),
        "Ws": round(ws, 4),
        "matched_skills": kw_info["matched_skills"],
        "missing_skills": kw_info["missing_skills"][:5],
        "required_years": ws_info["required_years"],
        "resume_years": ws_info["resume_years"],
        "missing_certs": ws_info["missing_certs"],
        "job_cluster": ws_info["job_cluster"],
        "resume_cluster": ws_info["resume_cluster"],
        "score_reason": score_reason,
        "penalties": penalties,
        "model_version": "public_model_final",
    }