# 사기업 점수 산출 로직
# 가중치: Ts 0.30 / Cs 0.35 / Qs 0.20

import re
import numpy as np

ALERT_THRESHOLD = 70

W_TS = 0.30
W_CS = 0.35
W_QS = 0.20

# TECH_KEYWOTDS 추가 (TECH_DICTIONARY 참고)
TECH_KEYWORDS = [
    "Python", "Java", "Kotlin", "Swift", "JavaScript", "TypeScript",
    "Spring", "Spring Boot", "FastAPI", "Django", "Flask", "Node.js",
    "React", "Vue", "Vue.js", "Angular", "Next.js", "NestJS",
    "Docker", "Kubernetes", "AWS", "GCP", "Azure",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "Spark",
    "PyTorch", "TensorFlow", "LLM", "CUDA", "MLOps", "NLP", "AI",
    "Git", "Linux", "Terraform", "Jenkins", "CI/CD",
    "Figma", "Jira", "Notion", "SQL", "NoSQL",
    "보안", "정보보안", "취약점", "방화벽", "ISMS",
    "C++", "C#", "Golang", "Rust", "Scala", "Ruby", "PHP", "Dart", "Perl", "Lua", "Groovy",
    "Svelte", "Nuxt.js", "HTML", "CSS", "Sass", "Tailwind", "Bootstrap", "jQuery", "Webpack", "Vite",
    "Express", "Rails", "ASP.NET",
    "MariaDB", "Oracle", "MSSQL", "Elasticsearch", "Cassandra", "DynamoDB", "SQLite", "Neo4j", "InfluxDB",
    "K8s", "Ansible", "Nginx", "Apache",
    "RabbitMQ", "Hadoop", "Airflow", "Pandas", "NumPy", "scikit-learn",
    "GitHub", "GitLab", "Bitbucket", "Confluence", "GraphQL", "gRPC", "Swagger",
    "JUnit", "Jest", "Cypress", "Selenium",
]


# util
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


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


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
    

# JD 파싱
def extract_tech_from_text(text: str) -> list[str]:
    text = str(text or "")
    found = []

    for tech in TECH_KEYWORDS:
        pattern = r"(?<![A-Za-z0-9가-힣])" + re.escape(tech) + r"(?![A-Za-z0-9가-힣])"
        if re.search(pattern, text, re.IGNORECASE):
            found.append(tech)
            
    # 문자열 중복 제거
    found = _dedup_subset_keywords(found)

    return _clean_items(found)


def _extract_years_from_text(text: str):
    text = str(text or "")

    patterns = [
        r"경력\s*(\d+)\s*년",
        r"(\d+)\s*년\s*이상",
        r"(\d+)\s*년\s*이상의?\s*경력",
        r"(\d+)\+\s*years",
        r"(\d+)\s*years",
        r"over\s*(\d+)\s*years",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))

    return None


def _extract_preferred_text(jd_text: str) -> tuple[str, str]:
    text = str(jd_text or "")

    preferred_patterns = [
        r"우대사항",
        r"우대 조건",
        r"preferred",
        r"nice to have",
        r"우대",
    ]

    for pat in preferred_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return text[:m.start()], text[m.start():]

    return text, ""


def parse_jd(jd_text: str) -> dict:
    main_text, preferred_text = _extract_preferred_text(jd_text)

    preferred_keywords = [
        k.strip()
        for k in re.findall(r"[-•]\s*([^\n]+)", preferred_text)
        if len(k.strip()) > 2
    ]

    return {
        "main_text": main_text,
        "preferred_text": preferred_text,
        "required_years": _extract_years_from_text(jd_text),
        "preferred_keywords": preferred_keywords,
        "jd_techs": extract_tech_from_text(jd_text),
    }


# .js 계열 키워드 정규화
def _strip_js_suffix(s: str) -> str:
    s = s.lower()
    return s[:-3] if s.endswith(".js") else s


# 표기 일치
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
def calc_ts(jd_parsed: dict, resume_skills: list) -> tuple[float, list[str], list[str]]:
    jd_techs = _clean_items(jd_parsed.get("jd_techs", []))
    resume_skills = _clean_items(resume_skills)

    resume_skill_set = {s.lower() for s in resume_skills}
    resume_skill_alias_set = {_normalize_tech_alias(s) for s in resume_skill_set}

    matched = []
    missing = []

    for tech in jd_techs:
        tech_l = tech.lower()
        if tech_l in resume_skill_set or _normalize_tech_alias(tech_l) in resume_skill_alias_set:
            matched.append(tech)
        else:
            missing.append(tech)

    ts = len(matched) / len(jd_techs) if jd_techs else 0.0

    return float(ts), matched, missing


def calc_cs(jd_vec: list, resume_vec: list) -> float:
    jd_arr = np.array(jd_vec, dtype=float)
    resume_arr = np.array(resume_vec, dtype=float)

    jd_norm = np.linalg.norm(jd_arr)
    resume_norm = np.linalg.norm(resume_arr)

    if jd_norm == 0 or resume_norm == 0:
        return 0.0

    raw_cs = float(np.dot(jd_arr, resume_arr) / (jd_norm * resume_norm))

    # cosine -1~1 → 0~1
    cs = (raw_cs + 1) / 2

    return float(max(0.0, min(1.0, cs)))


def calc_qs(jd_parsed: dict, resume_skills: list, experience_years: float) -> float:
    required_years = jd_parsed.get("required_years")

    if required_years is None or required_years <= 0:
        career_score = 1.0
    elif experience_years >= required_years:
        career_score = 1.0
    else:
        career_score = experience_years / required_years

    preferred_kws = jd_parsed.get("preferred_keywords", [])
    resume_skills = _clean_items(resume_skills)

    if preferred_kws:
        preferred_matched = sum(
            1
            for kw in preferred_kws
            if any(skill.lower() in kw.lower() for skill in resume_skills)
        )
        preferred_score = preferred_matched / len(preferred_kws)
    else:
        preferred_score = 0.0

    qs = career_score * 0.7 + preferred_score * 0.3

    return float(max(0.0, min(1.0, qs)))


def apply_gp(base_score: float, missing_skills: list, jd_techs_count: int, career_met: bool) -> tuple[float, list[str]]:
    score = float(base_score)
    penalties = []

    missing_count = len(missing_skills)
    missing_ratio = missing_count / jd_techs_count if jd_techs_count > 0 else 0.0

    if missing_count > 0:
        score -= 15
        penalties.append(f"필수 기술 {missing_count}개 누락 -15")

    if missing_count > 0:
        score = min(score, 60)
        penalties.append("필수 기술 미보유 score cap 60 적용")

    if missing_ratio >= 0.5:
        score = min(score, 40)
        penalties.append("필수 기술 다수 누락 score cap 40 적용")

    if not career_met:
        score = min(score, 65)
        penalties.append("경력 요건 미달 score cap 65 적용")

    return float(max(0.0, score)), penalties



# score_reason
def _make_skill_line(matched_skills: list, jd_techs: list) -> str:
    matched = _clean_items(matched_skills)
    total = len(_clean_items(jd_techs))
    matched_n = len(matched)

    if total <= 0:
        return "기술스택 매칭률로 산출되었습니다."

    preview = _format_items(matched)

    return f"기술스택 {matched_n}/{total}개 일치 ({preview})."


def _make_career_line(required_years, resume_years: float) -> str:
    required_years = _safe_float(required_years, 0)

    if required_years <= 0:
        return "경력 요건 없음 또는 신입 지원 가능."

    return f"경력 {required_years:g}년 요구 중 {resume_years:g}년 보유."


def _make_role_line(job_category: str | None, resume_role: str | None = None) -> str:
    category = _normalize_text(job_category) or "해당 공고"

    if not resume_role:
        return f"{category} 직무와 부분 일치."

    resume_role = _normalize_text(resume_role)

    if category.lower() in resume_role.lower() or resume_role.lower() in category.lower():
        return f"{category} 직무 일치."

    return f"{category} 직무와 부분 일치."


def _make_final_line(penalties: list[str], missing_skills: list, jd_techs: list, career_met: bool) -> str:
    penalty_text = " ".join(map(str, penalties))

    missing_count = len(missing_skills)
    total_count = len(jd_techs)
    missing_ratio = missing_count / total_count if total_count > 0 else 0.0

    if missing_count > 0 and missing_ratio >= 0.5:
        return "필수 기술 다수 누락으로 최종 점수 제한."

    if missing_count > 0:
        return "필수 기술 미보유로 최종 점수 제한."

    if not career_met:
        return "경력 요건 미달로 최종 점수 제한."

    if "자격" in penalty_text or "qualification" in penalty_text.lower():
        return "요구 자격 미충족으로 최종 점수 제한."

    if "cap" in penalty_text.lower() or "제한" in penalty_text:
        return "필수 요건 미충족으로 최종 점수 제한."

    return "기술·경력·직무 적합도와 이력서-공고 유사도를 종합 반영함."


def _make_score_reason(
    matched_skills,
    missing_skills,
    jd_techs,
    required_years,
    resume_years,
    job_category,
    resume_role,
    career_met,
    penalties,
) -> str:
    line1 = _make_skill_line(matched_skills, jd_techs)
    line2 = _make_career_line(required_years, resume_years)
    line3 = _make_role_line(job_category, resume_role)
    line4 = _make_final_line(penalties, missing_skills, jd_techs, career_met)

    return "\n".join([line1, line2, line3, line4])



# 최종 점수 산출
def score_private(
    jd_text: str,
    jd_vec: list,
    resume_vec: list,
    resume_skills: list[str],
    experience_years: int,
    job_category: str | None = None,
    title: str | None = None,
    resume_role: str | None = None,
) -> dict:
    jd_parsed = parse_jd(jd_text)
    resume_years = _safe_float(experience_years, 0.0)

    ts, matched_skills, missing_skills = calc_ts(jd_parsed, resume_skills)
    cs = calc_cs(jd_vec, resume_vec)
    qs = calc_qs(jd_parsed, resume_skills, resume_years)

    required_years = jd_parsed.get("required_years")
    career_met = True

    if required_years is not None and required_years > 0:
        career_met = resume_years >= required_years

    base_score = 100 * (W_TS * ts + W_CS * cs + W_QS * qs)

    final_score, penalties = apply_gp(
        base_score=base_score,
        missing_skills=missing_skills,
        jd_techs_count=len(jd_parsed["jd_techs"]),
        career_met=career_met,
    )

    gp = final_score - base_score

    score_reason = _make_score_reason(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        jd_techs=jd_parsed["jd_techs"],
        required_years=required_years,
        resume_years=resume_years,
        job_category=job_category,
        resume_role=resume_role,
        career_met=career_met,
        penalties=penalties,
    )

    return {
        "score": round(final_score, 1),
        "above_threshold": final_score >= ALERT_THRESHOLD,

        "ts": round(ts, 4),
        "cs": round(cs, 4),
        "qs": round(qs, 4),
        "gp": round(gp, 4),

        "matched_skills": matched_skills,
        "missing_skills": missing_skills[:10],
        "career_met": bool(career_met),

        "score_reason": score_reason,
        "penalties": penalties,
    }
