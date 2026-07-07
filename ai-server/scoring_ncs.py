# 공기업 NCS 점수 산출 로직
# 가중치: Kw 0.25 / Cs 0.35 / Ws 0.40

import re
import numpy as np

ALERT_THRESHOLD = 70

W_KW = 0.25
W_CS = 0.35
W_WS = 0.40

NCS_TECH_KEYWORDS = [
    "Python", "Java", "SQL", "R", "Linux", "서버", "네트워크", "방화벽",
    "정보보안", "보안", "개인정보", "ISMS", "DB", "데이터베이스",
    "데이터", "AI", "머신러닝", "통계", "분석", "공공데이터",
    "시스템", "정보시스템", "전산", "운영", "유지보수",
    "웹", "API", "Spring", "Spring Boot", "React",
    "클라우드", "AWS", "GCP", "Azure",
    "통신", "AMI", "DAS", "ICT", "정보화", "PM", "사업관리",
]

CERT_KEYWORDS = [
    "정보처리기사", "정보보안기사", "정보보안산업기사",
    "SQLD", "SQLP", "ADsP", "ADP", "빅데이터분석기사",
    "네트워크관리사", "리눅스마스터", "정보통신기사",
]

CLUSTER_KEYWORDS = {
    "보안/정보보호": [
        "보안", "정보보안", "개인정보", "침해", "취약점", "방화벽",
        "ISMS", "인증", "보안관제", "정보보호",
    ],
    "시스템개발/SW": [
        "개발", "SW", "소프트웨어", "프로그램", "웹", "앱", "API",
        "Java", "Spring", "Python", "React",
    ],
    "데이터/AI": [
        "데이터", "AI", "인공지능", "머신러닝", "통계", "분석",
        "공공데이터", "빅데이터",
    ],
    "DB/데이터관리": [
        "DB", "데이터베이스", "SQL", "DBA", "DB관리", "데이터 관리",
    ],
    "네트워크/통신": [
        "네트워크", "통신", "AMI", "DAS", "망", "회선", "라우터",
        "스위치", "정보통신",
    ],
    "인프라/시스템운영": [
        "서버", "인프라", "시스템 운영", "운영", "유지보수",
        "전산", "클라우드", "Linux", "장애",
    ],
    "기획/PM": [
        "기획", "PM", "사업관리", "프로젝트 관리", "정보화사업",
        "서비스 기획", "요구사항", "정책",
    ],
    "산업/설비·OT 운영": [
        "설비", "전기", "전자", "배전", "발전", "제어", "현장",
        "시설", "장비", "OT", "자동화",
    ],
}


# util
def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _to_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()

    if not text:
        return []

    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")

    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]

    return [text]


def _clean_items(items):
    seen = set()
    result = []

    for item in _to_list(items):
        item = re.sub(r"\s+", " ", str(item)).strip()

        if not item:
            continue

        key = item.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def _format_items(items, max_items=3):
    items = _clean_items(items)

    if not items:
        return "일치 키워드 없음"

    return ", ".join(items[:max_items])


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _get_resume_skills(resume: dict) -> list[str]:
    skills = resume.get("skills", {})

    if isinstance(skills, dict):
        return _clean_items(
            skills.get("core", [])
            + skills.get("sub", [])
        )

    return _clean_items(skills)


def _get_resume_certs(resume: dict) -> list[str]:
    return _clean_items(resume.get("certs", []))


def _get_resume_years(resume: dict) -> float:
    return _safe_float(resume.get("experience_years", 0), 0.0)


def _get_resume_role(resume: dict) -> str:
    return str(resume.get("job_role", "") or "")



# text 파싱
def build_ncs_text(
    jd_text: str,
    title: str | None = None,
    company_name: str | None = None,
    job_role: str | None = None,
    apply_qualification: str | None = None,
    html_content: str | None = None,
) -> str:
    parts = []

    if title:
        parts.append(title)

    if company_name:
        parts.append(company_name)

    if job_role:
        parts.append(job_role)

    if html_content:
        parts.append(html_content)

    if apply_qualification:
        parts.append(apply_qualification)

    if jd_text:
        parts.append(jd_text)

    return _normalize_text(" ".join(parts))


def extract_keywords(text: str, keywords: list[str]) -> list[str]:
    text = str(text or "")
    found = []

    for kw in keywords:
        pattern = r"(?<![A-Za-z0-9가-힣])" + re.escape(kw) + r"(?![A-Za-z0-9가-힣])"
        if re.search(pattern, text, re.IGNORECASE):
            found.append(kw)

    return _clean_items(found)


def extract_required_years(text: str):
    text = str(text or "")

    patterns = [
        r"경력\s*(\d+)\s*년",
        r"(\d+)\s*년\s*이상",
        r"(\d+)\s*년\s*이상의?\s*경력",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))

    return None


def classify_cluster(text: str) -> str:
    text = str(text or "")
    scores = {}

    for cluster, keywords in CLUSTER_KEYWORDS.items():
        score = 0

        for kw in keywords:
            if kw.lower() in text.lower():
                score += 1

        if score > 0:
            scores[cluster] = score

    if not scores:
        return "UNKNOWN"

    return max(scores.items(), key=lambda x: x[1])[0]


def classify_resume_cluster(resume: dict) -> str:
    text = " ".join([
        _get_resume_role(resume),
        " ".join(_get_resume_skills(resume)),
        str(resume.get("summary", "") or ""),
    ])

    return classify_cluster(text)



# 점수 계산
def calc_kw(job_keywords: list[str], resume_skills: list[str]) -> tuple[float, list[str], list[str]]:
    job_keywords = _clean_items(job_keywords)
    resume_skills = _clean_items(resume_skills)

    resume_skill_set = {s.lower() for s in resume_skills}

    matched = []
    missing = []

    for kw in job_keywords:
        if kw.lower() in resume_skill_set:
            matched.append(kw)
        else:
            missing.append(kw)

    kw_score = len(matched) / len(job_keywords) if job_keywords else 0.0

    return float(kw_score), matched, missing


def calc_cs(jd_vec: list, resume_vec: list) -> float:
    jd_arr = np.array(jd_vec, dtype=float)
    resume_arr = np.array(resume_vec, dtype=float)

    jd_norm = np.linalg.norm(jd_arr)
    resume_norm = np.linalg.norm(resume_arr)

    if jd_norm == 0 or resume_norm == 0:
        return 0.0

    raw_cs = float(np.dot(jd_arr, resume_arr) / (jd_norm * resume_norm))

    cs = (raw_cs + 1) / 2

    return float(max(0.0, min(1.0, cs)))


def calc_certs(job_text: str, resume_certs: list[str]) -> tuple[list[str], list[str]]:
    required_certs = extract_keywords(job_text, CERT_KEYWORDS)
    resume_cert_set = {c.lower() for c in _clean_items(resume_certs)}

    matched = []
    missing = []

    for cert in required_certs:
        if cert.lower() in resume_cert_set:
            matched.append(cert)
        else:
            missing.append(cert)

    return matched, missing


def calc_ws(
    job_cluster: str,
    resume_cluster: str,
    required_years,
    resume_years: float,
    matched_certs: list[str],
    missing_certs: list[str],
) -> float:
    # 1. 직무 클러스터 점수
    if job_cluster == resume_cluster:
        cluster_score = 1.0
    elif job_cluster != "UNKNOWN" and resume_cluster != "UNKNOWN":
        cluster_score = 0.65
    else:
        cluster_score = 0.4

    # 2. 경력 점수
    if required_years is None or required_years <= 0:
        career_score = 1.0
    elif resume_years >= required_years:
        career_score = 1.0
    else:
        career_score = resume_years / required_years

    # 3. 자격 점수
    total_certs = len(matched_certs) + len(missing_certs)

    if total_certs == 0:
        cert_score = 1.0
    else:
        cert_score = len(matched_certs) / total_certs

    ws = cluster_score * 0.45 + career_score * 0.35 + cert_score * 0.20

    return float(max(0.0, min(1.0, ws)))


def apply_gp(
    base_score: float,
    missing_certs: list[str],
    required_years,
    resume_years: float,
    resume_cluster: str,
) -> tuple[float, list[str]]:
    score = float(base_score)
    penalties = []

    if missing_certs:
        score = min(score, 65)
        penalties.append("필수 자격 미보유 score cap 65 적용")

    if required_years is not None and required_years > 0 and resume_years < required_years:
        score = min(score, 65)
        penalties.append("경력 요건 미달 score cap 65 적용")

    if resume_cluster == "UNKNOWN":
        score = min(score, 55)
        penalties.append("비IT 이력서의 IT 공고 과추천 방지 score cap 55 적용")

    return float(max(0.0, score)), penalties



# score_reason
def _make_skill_line(matched_skills: list, job_keywords: list) -> str:
    matched = _clean_items(matched_skills)
    total = len(_clean_items(job_keywords))
    matched_n = len(matched)

    if total <= 0:
        return "NCS 키워드 매칭률로 산출되었습니다."

    preview = _format_items(matched)

    return f"NCS 필요기술 {matched_n}/{total}개 일치 ({preview})."


def _make_career_line(required_years, resume_years: float) -> str:
    required_years = _safe_float(required_years, 0)

    if required_years <= 0:
        return "경력 요건 없음 또는 신입 지원 가능."

    return f"경력 {required_years:g}년 요구 중 {resume_years:g}년 보유."


def _make_role_line(job_cluster: str, resume_cluster: str) -> str:
    if job_cluster == resume_cluster:
        return f"{job_cluster} 직무 일치."

    if job_cluster == "UNKNOWN":
        return "직무 분류 정보가 부족하여 키워드 기반으로 산출되었습니다."

    if resume_cluster == "UNKNOWN":
        return f"{job_cluster} 직무와 직접 일치도는 낮음."

    return f"{job_cluster} 직무와 부분 일치."


def _make_final_line(
    penalties: list[str],
    missing_certs: list[str],
    required_years,
    resume_years: float,
    resume_cluster: str,
) -> str:
    if missing_certs:
        return "필수 자격 미보유로 최종 점수 제한."

    if required_years is not None and required_years > 0 and resume_years < required_years:
        return "경력 요건 미달로 최종 점수 제한."

    if resume_cluster == "UNKNOWN":
        return "비IT 이력서의 IT 공고 과추천 방지를 위해 최종 점수 제한."

    if penalties:
        return "필수 요건 미충족으로 최종 점수 제한."

    return "기술·경력·직무 적합도와 이력서-공고 유사도를 종합 반영함."


def make_score_reason(
    matched_skills: list,
    job_keywords: list,
    required_years,
    resume_years: float,
    job_cluster: str,
    resume_cluster: str,
    missing_certs: list[str],
    penalties: list[str],
) -> str:
    line1 = _make_skill_line(matched_skills, job_keywords)
    line2 = _make_career_line(required_years, resume_years)
    line3 = _make_role_line(job_cluster, resume_cluster)
    line4 = _make_final_line(
        penalties=penalties,
        missing_certs=missing_certs,
        required_years=required_years,
        resume_years=resume_years,
        resume_cluster=resume_cluster,
    )

    return "\n".join([line1, line2, line3, line4])



# 최종 점수 산출
def score_ncs(
    jd_text: str,
    resume: dict,
    jd_vec: list,
    resume_vec: list,
    title: str | None = None,
    company_name: str | None = None,
    job_role: str | None = None,
    apply_qualification: str | None = None,
    html_content: str | None = None,
) -> dict:
    ncs_text = build_ncs_text(
        jd_text=jd_text,
        title=title,
        company_name=company_name,
        job_role=job_role,
        apply_qualification=apply_qualification,
        html_content=html_content,
    )

    resume_skills = _get_resume_skills(resume)
    resume_certs = _get_resume_certs(resume)
    resume_years = _get_resume_years(resume)

    job_keywords = extract_keywords(ncs_text, NCS_TECH_KEYWORDS)
    kw, matched_skills, missing_skills = calc_kw(job_keywords, resume_skills)

    cs = calc_cs(jd_vec, resume_vec)

    job_cluster = classify_cluster(ncs_text)
    resume_cluster = classify_resume_cluster(resume)

    required_years = extract_required_years(ncs_text)

    matched_certs, missing_certs = calc_certs(ncs_text, resume_certs)

    ws = calc_ws(
        job_cluster=job_cluster,
        resume_cluster=resume_cluster,
        required_years=required_years,
        resume_years=resume_years,
        matched_certs=matched_certs,
        missing_certs=missing_certs,
    )

    base_score = 100 * (W_KW * kw + W_CS * cs + W_WS * ws)

    final_score, penalties = apply_gp(
        base_score=base_score,
        missing_certs=missing_certs,
        required_years=required_years,
        resume_years=resume_years,
        resume_cluster=resume_cluster,
    )

    gp = final_score - base_score

    score_reason = make_score_reason(
        matched_skills=matched_skills,
        job_keywords=job_keywords,
        required_years=required_years,
        resume_years=resume_years,
        job_cluster=job_cluster,
        resume_cluster=resume_cluster,
        missing_certs=missing_certs,
        penalties=penalties,
    )

    return {
        "score": round(final_score, 1),
        "above_threshold": final_score >= ALERT_THRESHOLD,

        "kw": round(kw, 4),
        "cs": round(cs, 4),
        "ws": round(ws, 4),
        "gp": round(gp, 4),

        "matched_skills": matched_skills,
        "missing_skills": missing_skills[:10],
        "matched_certs": matched_certs,
        "missing_certs": missing_certs[:10],

        "job_cluster": job_cluster,
        "resume_cluster": resume_cluster,

        "score_reason": score_reason,
        "penalties": penalties,
    }
