# 공기업 NCS 점수 산출 로직
# 가중치: Kw 0.25 / Cs 0.35 / Ws 0.40


import re
import numpy as np

from parser_ncs import enrich_public_job, classify_cluster


ALERT_THRESHOLD = 70

W_KW = 0.25
W_CS = 0.35
W_WS = 0.40


# 범용적인 한국어 일반 명사 제거
NCS_TECH_KEYWORDS = [
    "Python", "Java", "SQL", "R", "Linux", "방화벽",
    "정보보안", "보안", "개인정보", "ISMS", "DB", "데이터베이스",
    "AI", "머신러닝", "통계", "공공데이터",
    "정보시스템", "API", "Spring", "Spring Boot", "React",
    "클라우드", "AWS", "GCP", "Azure",
    "AMI", "DAS", "ICT", "PM",
]

CERT_KEYWORDS = [
    "정보처리기사", "정보보안기사", "정보보안산업기사",
    "SQLD", "SQLP", "ADsP", "ADP", "빅데이터분석기사",
    "네트워크관리사", "리눅스마스터", "정보통신기사",
]


CLUSTER_COMPATIBLE = {
    "DB/데이터관리": {
        "DB/데이터관리",
        "데이터/AI",
        "시스템개발/SW",
        "인프라/시스템운영",
    },
    "시스템개발/SW": {
        "시스템개발/SW",
        "DB/데이터관리",
        "데이터/AI",
        "기획/PM",
        "인프라/시스템운영",
    },
    "보안/정보보호": {
        "보안/정보보호",
        "인프라/시스템운영",
        "네트워크/통신",
    },
    "인프라/시스템운영": {
        "인프라/시스템운영",
        "네트워크/통신",
        "보안/정보보호",
        "산업/설비·OT 운영",
        "시스템개발/SW",
    },
    "네트워크/통신": {
        "네트워크/통신",
        "인프라/시스템운영",
        "보안/정보보호",
        "산업/설비·OT 운영",
    },
    "데이터/AI": {
        "데이터/AI",
        "DB/데이터관리",
        "시스템개발/SW",
        "기획/PM",
    },
    "기획/PM": {
        "기획/PM",
        "시스템개발/SW",
        "데이터/AI",
        "DB/데이터관리",
    },
    "산업/설비·OT 운영": {
        "산업/설비·OT 운영",
        "네트워크/통신",
        "인프라/시스템운영",
    },
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


# 부분 문자열 중복 추출 제거 함수 보강
def _tokenize_keyword(s: str) -> set[str]:
    return {t for t in re.split(r"[^A-Za-z0-9가-힣]+", s.lower()) if t}


def _dedup_subset_keywords(keywords: list[str]) -> list[str]:
    result = []
    tokens_by_kw = {kw: _tokenize_keyword(kw) for kw in keywords}
    for kw in keywords:
        kw_tokens = tokens_by_kw[kw]
        is_subset = any(
            other != kw
            and len(other) > len(kw)
            and kw_tokens
            and kw_tokens <= tokens_by_kw[other]
            for other in keywords
        )
        if not is_subset:
            result.append(kw)
    return result
    

# text 파싱
def extract_keywords(text: str, keywords: list[str]) -> list[str]:
    text = str(text or "")
    found = []

    for kw in keywords:
        pattern = r"(?<![A-Za-z0-9가-힣])" + re.escape(kw) + r"(?![A-Za-z0-9가-힣])"

        if re.search(pattern, text, re.IGNORECASE):
            found.append(kw)

    # 부분 문자열 중복 제거
    found = _dedup_subset_keywords(found)

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



# cluster / 직무 호환성 계산
def classify_resume_cluster(resume: dict) -> str:
    text = " ".join([
        _get_resume_role(resume),
        " ".join(_get_resume_skills(resume)),
        str(resume.get("summary", "") or ""),
    ])

    return classify_cluster(text, job_role=_get_resume_role(resume))


def get_job_cluster_set(job: dict):
    primary = (
        job.get("primary_cluster")
        or job.get("cluster")
        or job.get("job_cluster")
        or "UNKNOWN"
    )

    secondary = job.get("secondary_clusters") or []

    if isinstance(secondary, str):
        secondary = [x.strip() for x in secondary.split(",") if x.strip()]

    clusters = []

    for c in [primary] + list(secondary):
        if c and c not in clusters:
            clusters.append(c)

    return primary, clusters


def cluster_relation(resume_cluster: str, job: dict):
    primary, job_clusters = get_job_cluster_set(job)

    if resume_cluster in {"UNKNOWN", "비IT/일반사무"}:
        return "unknown_or_non_it", primary, job_clusters

    if resume_cluster == primary:
        return "primary_match", primary, job_clusters

    if resume_cluster in job_clusters:
        return "secondary_match", primary, job_clusters

    compatible = CLUSTER_COMPATIBLE.get(resume_cluster, {resume_cluster})

    if any(c in compatible for c in job_clusters):
        return "compatible_match", primary, job_clusters

    return "mismatch", primary, job_clusters


# .js 계열 키워드 정규화 추가
def _strip_js_suffix(s: str) -> str:
    s = s.lower()
    return s[:-3] if s.endswith(".js") else s


# 표기 일치 추가
_TECH_ALIASES = {
    "golang": "go",
    "k8s": "kubernetes",
    "tailwind": "tailwindcss",
    "rails": "ruby on rails",
}


def _normalize_tech_alias(s: str) -> str:
    s = _strip_js_suffix(s.lower())
    return _TECH_ALIASES.get(s, s)



# 점수 계산
def calc_kw(job_keywords: list[str], resume_skills: list[str]) -> tuple[float, list[str], list[str]]:
    job_keywords = _clean_items(job_keywords)
    resume_skills = _clean_items(resume_skills)

    resume_skill_set = {s.lower() for s in resume_skills}
    resume_skill_alias_set = {_normalize_tech_alias(s) for s in resume_skill_set}

    matched = []
    missing = []

    for kw in job_keywords:
        kw_l = kw.lower()
        if kw_l in resume_skill_set or _normalize_tech_alias(kw_l) in resume_skill_alias_set:
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


def calc_ws(
    job: dict,
    resume_cluster: str,
    required_years,
    resume_years: float,
    matched_certs: list[str],
    missing_certs: list[str],
    kw: float,
) -> tuple[float, dict]:
    relation, job_cluster, job_clusters = cluster_relation(resume_cluster, job)

    if relation == "primary_match":
        cluster_score = 1.0
    elif relation == "secondary_match":
        cluster_score = 0.9
    elif relation == "compatible_match":
        cluster_score = 0.75
    elif relation == "unknown_or_non_it":
        cluster_score = 0.35
    else:
        cluster_score = 0.0

    if required_years is None or required_years <= 0:
        career_score = 1.0
    elif resume_years >= required_years:
        career_score = 1.0
    else:
        career_score = max(0.0, min(1.0, resume_years / required_years))

    total_certs = len(matched_certs) + len(missing_certs)

    if total_certs == 0:
        cert_score = 1.0
    else:
        cert_score = len(matched_certs) / total_certs

    ws = (
        0.40 * cluster_score
        + 0.45 * career_score
        + 0.10 * cert_score
        + 0.05 * kw
    )

    ws_info = {
        "resume_cluster": resume_cluster,
        "job_cluster": job_cluster,
        "job_clusters": job_clusters,
        "cluster_relation": relation,
        "cluster_score": cluster_score,
        "career_score": career_score,
        "cert_score": cert_score,
    }

    return float(max(0.0, min(1.0, ws))), ws_info


def apply_gp(
    base_score: float,
    resume: dict,
    job: dict,
    missing_certs: list[str],
    required_years,
    resume_years: float,
    ws_info: dict,
) -> tuple[float, list[str]]:
    score = float(base_score)
    penalties = []

    resume_cluster = ws_info["resume_cluster"]
    job_cluster = ws_info["job_cluster"]
    job_clusters = ws_info.get("job_clusters", [job_cluster])
    relation = ws_info.get("cluster_relation", "mismatch")

    if missing_certs:
        score = min(score, 65)
        penalties.append("필수 자격 미보유 score cap 65 적용")

    if required_years is not None and required_years > 0 and resume_years < required_years:
        score = min(score, 65)
        penalties.append("경력 요건 미달 score cap 65 적용")

    if resume_cluster in {"UNKNOWN", "비IT/일반사무"} and job_cluster != "UNKNOWN":
        score = min(score, 55)
        penalties.append("비IT 이력서의 IT 공고 과추천 방지 score cap 55 적용")

    if (
        resume_cluster not in {"UNKNOWN", "비IT/일반사무"}
        and job_cluster != "UNKNOWN"
        and relation == "mismatch"
    ):
        score -= 5
        penalties.append(f"직무 클러스터 완전 불일치 -5 ({resume_cluster} vs {job_clusters})")

    if relation == "compatible_match":
        score -= 2
        penalties.append(f"직무 클러스터 부분 호환 -2 ({resume_cluster} vs {job_clusters})")

    is_ot_job = "산업/설비·OT 운영" in job_clusters

    if resume_cluster == "네트워크/통신" and is_ot_job and relation != "primary_match":
        score -= 2
        penalties.append("네트워크 이력서의 OT 공고 약한 감점 -2")

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


def _make_role_line(job_cluster: str, resume_cluster: str, relation: str) -> str:
    if relation == "primary_match":
        return f"{job_cluster} 직무 일치."

    if relation in {"secondary_match", "compatible_match"}:
        return f"{job_cluster} 직무와 부분 일치."

    if job_cluster == "UNKNOWN":
        return "직무 분류 정보가 부족하여 키워드 기반으로 산출되었습니다."

    return f"{job_cluster} 직무와 직접 일치도는 낮음."


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

    if resume_cluster in {"UNKNOWN", "비IT/일반사무"}:
        return "비IT 이력서의 IT 공고 과추천 방지를 위해 최종 점수 제한."

    if penalties:
        return "필수 요건 미충족으로 최종 점수 제한."

    return "기술·경력·직무 적합도와 이력서-공고 유사도를 종합 반영함."


def make_score_reason(
    matched_skills: list,
    job_keywords: list,
    required_years,
    resume_years: float,
    ws_info: dict,
    missing_certs: list[str],
    penalties: list[str],
) -> str:
    job_cluster = ws_info["job_cluster"]
    resume_cluster = ws_info["resume_cluster"]
    relation = ws_info.get("cluster_relation", "mismatch")

    line1 = _make_skill_line(matched_skills, job_keywords)
    line2 = _make_career_line(required_years, resume_years)
    line3 = _make_role_line(job_cluster, resume_cluster, relation)
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
    work_experience: str | None = None,
    recrut_type: str | None = None,
    application_method: str | None = None,
) -> dict:
    raw_job = {
        "jd_text": jd_text,
        "title": title,
        "company_name": company_name,
        "job_role": job_role,
        "apply_qualification": apply_qualification,
        "html_content": html_content,
        "work_experience": work_experience,
        "recrut_type": recrut_type,
        "application_method": application_method,
    }

    job = enrich_public_job(raw_job)

    ncs_text = _normalize_text(" ".join([
        str(job.get("title", "") or ""),
        str(job.get("company_name", "") or ""),
        str(job.get("job_role", "") or ""),
        str(job.get("anchor_text", "") or ""),
        str(job.get("apply_qualification", "") or ""),
        str(jd_text or ""),
    ]))

    resume_skills = _get_resume_skills(resume)

    # 이력서 원문에도 NCS 키워드 추출을 적용해서 resumeSkills에 병합
    resume_summary = str(resume.get("summary", "") or "")
    resume_keyword_hits = extract_keywords(resume_summary, NCS_TECH_KEYWORDS)
    resume_skills = list(dict.fromkeys(resume_skills + resume_keyword_hits))

    resume_certs = _get_resume_certs(resume)
    resume_years = _get_resume_years(resume)

    job_keywords = extract_keywords(ncs_text, NCS_TECH_KEYWORDS)
    kw, matched_skills, missing_skills = calc_kw(job_keywords, resume_skills)

    cs = calc_cs(jd_vec, resume_vec)

    resume_cluster = classify_resume_cluster(resume)

    required_years = extract_required_years(ncs_text)

    matched_certs, missing_certs = calc_certs(ncs_text, resume_certs)

    ws, ws_info = calc_ws(
        job=job,
        resume_cluster=resume_cluster,
        required_years=required_years,
        resume_years=resume_years,
        matched_certs=matched_certs,
        missing_certs=missing_certs,
        kw=kw,
    )

    base_score = 100 * (W_KW * kw + W_CS * cs + W_WS * ws)

    final_score, penalties = apply_gp(
        base_score=base_score,
        resume=resume,
        job=job,
        missing_certs=missing_certs,
        required_years=required_years,
        resume_years=resume_years,
        ws_info=ws_info,
    )

    gp = final_score - base_score

    score_reason = make_score_reason(
        matched_skills=matched_skills,
        job_keywords=job_keywords,
        required_years=required_years,
        resume_years=resume_years,
        ws_info=ws_info,
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

        "job_cluster": ws_info["job_cluster"],
        "resume_cluster": ws_info["resume_cluster"],

        "score_reason": score_reason,
        "penalties": penalties,
    }
