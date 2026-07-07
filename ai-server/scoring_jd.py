# scoring_jd.py
# 사기업 점수 산출 로직
# 가중치: Ts 0.30 / Cs 0.35 / Qs 0.20 (cs_ts_equal 실험 결과 최적)

import re
import numpy as np

ALERT_THRESHOLD = 70

W_TS = 0.30
W_CS = 0.35
W_QS = 0.20

TECH_KEYWORDS = [
    "Python", "Java", "Kotlin", "Swift", "JavaScript", "TypeScript",
    "Spring", "Spring Boot", "FastAPI", "Django", "Flask", "Node.js",
    "React", "Vue", "Angular", "Docker", "Kubernetes", "AWS", "GCP", "Azure",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "Spark",
    "PyTorch", "TensorFlow", "LLM", "CUDA", "MLOps", "NLP", "Git", "Linux",
    "Terraform", "Jenkins", "CI/CD",
]


# ── 유틸 ──────────────────────────────────────────

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


def extract_tech_from_text(text: str) -> list[str]:
    return [t for t in TECH_KEYWORDS
            if re.search(r"\b" + re.escape(t) + r"\b", text, re.IGNORECASE)]


def _extract_years_from_text(text: str):
    patterns = [
        r"경력\s*(\d+)\s*년",
        r"(\d+)\s*년\s*이상",
        r"(\d+)\s*년\s*이상의\s*경력",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


# ── 문구 생성 ─────────────────────────────────────

def _make_penalty_note(penalties: list[str]) -> str:
    joined = " ".join(penalties)
    if "필수 기술" in joined or "기술 미보유" in joined:
        return "필수 기술 미보유로 최종 점수 제한."
    if "경력" in joined:
        return "경력 요건 미달로 최종 점수 제한."
    if penalties:
        return "필수 요건 미충족으로 최종 점수 제한."
    return ""


def _make_skill_line(matched_skills: list, jd_techs: list) -> str:
    matched = _clean_items(matched_skills)
    total = len(jd_techs)
    matched_n = len(matched)
    preview = _format_items(matched)
    if total > 0:
        return f"필요기술 {matched_n}/{total}개 일치 ({preview})."
    return "기술 키워드 매칭 기반으로 산출되었습니다."


def _make_career_line(required_years, resume_years: float) -> str:
    if required_years is None or _safe_float(required_years, 0) <= 0:
        return "경력 요건 없음 또는 신입 지원 가능."
    return f"경력 {required_years:g}년 요구 중 {resume_years:g}년 보유."


def _make_score_reason(
    matched_skills, jd_techs,
    required_years, resume_years,
    penalties
) -> str:
    line1 = _make_skill_line(matched_skills, jd_techs)
    line2 = _make_career_line(required_years, resume_years)
    line3 = "기술·경력·직무 적합도와 이력서-공고 유사도를 종합 반영함."
    penalty_note = _make_penalty_note(penalties)
    line4 = penalty_note if penalty_note else "기술·경력·직무 적합도와 이력서-공고 유사도를 종합 반영함."
    return "\n".join([line1, line2, line3, line4])


# ── 파싱 ──────────────────────────────────────────

def parse_jd(jd_text: str) -> dict:
    preferred_text = ""
    main_text = jd_text
    m = re.search(r"[◆▶*#]?\s*우대\s*사항.*", jd_text, re.DOTALL)
    if m:
        preferred_text = jd_text[m.start():]
        main_text = jd_text[:m.start()]

    required_years = _extract_years_from_text(jd_text)
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


# ── 점수 계산 ─────────────────────────────────────

def calc_ts(jd_parsed: dict, resume_skills: list) -> tuple:
    jd_techs = jd_parsed["jd_techs"]
    skills_lower = [s.lower() for s in resume_skills]
    matched = [t for t in jd_techs if t.lower() in skills_lower]
    missing = [t for t in jd_techs if t.lower() not in skills_lower]
    ts = len(matched) / len(jd_techs) if jd_techs else 0.0
    return float(ts), matched, missing


def calc_cs(jd_vec: list, resume_vec: list) -> float:
    jd_arr = np.array(jd_vec).reshape(1, -1)
    resume_arr = np.array(resume_vec).reshape(1, -1)
    raw_cs = float(np.dot(jd_arr, resume_arr.T)[0][0])
    return (raw_cs + 1) / 2


def calc_qs(jd_parsed: dict, resume_skills: list, experience_years: float) -> float:
    required_years = jd_parsed.get("required_years")
    if required_years is None:
        career_score = 1.0
    elif experience_years >= required_years:
        career_score = 1.0
    else:
        career_score = experience_years / required_years

    preferred_kws = jd_parsed.get("preferred_keywords", [])
    if preferred_kws:
        preferred_matched = sum(
            1 for kw in preferred_kws
            if any(s.lower() in kw.lower() for s in resume_skills)
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


# ── 최종 점수 산출 ────────────────────────────────

def score_private(
    jd_text: str,
    jd_vec: list,
    resume_vec: list,
    resume_skills: list[str],
    experience_years: int,
) -> dict:
    jd_parsed = parse_jd(jd_text)
    resume_years = _safe_float(experience_years, 0.0)

    ts, matched_skills, missing_skills = calc_ts(jd_parsed, resume_skills)
    cs = calc_cs(jd_vec, resume_vec)
    qs = calc_qs(jd_parsed, resume_skills, resume_years)

    base_score = 100 * (W_TS * ts + W_CS * cs + W_QS * qs)
    final_score, penalties = apply_gp(base_score, missing_skills, len(jd_parsed["jd_techs"]))

    score_reason = _make_score_reason(
        matched_skills=matched_skills,
        jd_techs=jd_parsed["jd_techs"],
        required_years=jd_parsed.get("required_years"),
        resume_years=resume_years,
        penalties=penalties,
    )

    required_year = jd_parsed.get("required_years") or 0

    return {
        "score": round(final_score, 1),
        "above_threshold": final_score >= ALERT_THRESHOLD,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills[:5],
        "career_met": resume_years >= required_year if required_year else True,
        "score_reason": score_reason,
        "penalties": penalties,
        "model_version": "kosimcse-final",
    }