# 공기업 NCS 공고 파싱 / anchor 생성 / cluster 분류


import re
from collections import Counter



# common util
def _ct(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def _has(text: str, patterns: list[str]) -> bool:
    text = str(text or "")
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _contains_any(text: str, keywords: list[str]) -> bool:
    text = str(text or "")
    return any(k in text for k in keywords)


def clean_html_text(text: str) -> str:
    text = str(text or "")

    text = re.sub(r"<[^>]+>", " ", text)

    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )

    text = re.sub(r"[□■○●◎◇◆▶▷▣▪▫·•※ㆍ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def join_anchor_parts(parts: list[str]) -> str:
    cleaned = []

    for part in parts:
        part = clean_html_text(part)
        if part:
            cleaned.append(part)

    return _ct(" ".join(cleaned))



# A~G legacy parser
def detect_pattern(text: str) -> str:
    text = str(text or "")

    if "필요지식" in text and "필요기술" in text:
        return "A"

    if "직무수행내용" in text and "필요기술" in text:
        return "B"

    if "수행내용" in text and "지식" in text and "기술" in text:
        return "C"

    if "능력단위" in text and "직무내용" in text:
        return "D"

    if "직무기술서" in text:
        return "E"

    if "담당업무" in text or "주요업무" in text or "수행업무" in text:
        return "F"

    return "G"


def _extract_between(
    text: str,
    start_keywords: list[str],
    end_keywords: list[str],
    max_chars: int = 1800,
) -> str:
    text = str(text or "")

    starts = []

    for kw in start_keywords:
        idx = text.find(kw)
        if idx >= 0:
            starts.append(idx)

    if not starts:
        return ""

    start = min(starts)

    ends = []

    for kw in end_keywords:
        idx = text.find(kw, start + 1)
        if idx >= 0:
            ends.append(idx)

    end = min(ends) if ends else min(len(text), start + max_chars)

    return text[start:end]


def extract_A(text: str) -> str:
    return _extract_between(
        text,
        ["직무수행내용", "필요지식", "필요기술"],
        ["직무수행태도", "직업기초능력", "자격요건", "지원자격", "우대사항"],
    )


def extract_B(text: str) -> str:
    return _extract_between(
        text,
        ["직무수행내용", "필요기술"],
        ["직무수행태도", "직업기초능력", "자격요건", "지원자격", "우대사항"],
    )


def extract_C(text: str) -> str:
    return _extract_between(
        text,
        ["수행내용", "지식", "기술"],
        ["태도", "직업기초능력", "자격요건", "지원자격", "우대사항"],
    )


def extract_D(text: str) -> str:
    return _extract_between(
        text,
        ["능력단위", "직무내용"],
        ["직업기초능력", "자격요건", "지원자격", "우대사항"],
    )


def extract_E(text: str) -> str:
    return _extract_between(
        text,
        ["직무기술서", "직무수행내용", "필요지식", "필요기술"],
        ["직무수행태도", "직업기초능력", "자격요건", "지원자격", "우대사항"],
    )


def extract_F(text: str) -> str:
    return _extract_between(
        text,
        ["담당업무", "주요업무", "수행업무", "직무내용"],
        ["자격요건", "지원자격", "우대사항", "근무조건", "전형절차"],
    )


def extract_G(text: str) -> str:
    return text[:1800]


_EXTRACTORS = {
    "A": extract_A,
    "B": extract_B,
    "C": extract_C,
    "D": extract_D,
    "E": extract_E,
    "F": extract_F,
    "G": extract_G,
}


def parse_job_anchor(job: dict) -> tuple[str, str]:
    text = join_anchor_parts([
        job.get("html_content", ""),
        job.get("apply_qualification", ""),
        job.get("job_role", ""),
        job.get("title", ""),
        job.get("jd_text", ""),
    ])

    pattern = detect_pattern(text)
    anchor = _EXTRACTORS.get(pattern, extract_G)(text)

    return pattern, clean_anchor(anchor)



# IT section parser
IT_SECTION_HINTS = [
    "정보통신",
    "20. 정보통신",
    "정보기술",
    "전산",
    "IT",
    "ICT",
    "소프트웨어",
    "데이터",
    "정보보안",
    "정보보호",
    "네트워크",
    "시스템",
    "DB",
    "AI",
]


FIELD_START_KEYWORDS = [
    "직무수행내용",
    "수행내용",
    "담당업무",
    "주요업무",
    "수행업무",
    "필요지식",
    "필요기술",
    "필요역량",
    "직무내용",
    "자격요건",
    "지원자격",
    "우대사항",
]


FIELD_END_KEYWORDS = [
    "직무수행태도",
    "직업기초능력",
    "의사소통능력",
    "대인관계능력",
    "조직이해능력",
    "윤리의식",
    "근무조건",
    "전형절차",
    "제출서류",
    "기타사항",
]


def _section_bounds(text: str) -> list[str]:
    text = clean_html_text(text)

    if not text:
        return []

    parts = re.split(
        r"(?=직무기술서|NCS\s*기반\s*채용|채용분야|모집분야|대분류|중분류|소분류|세분류)",
        text,
    )

    sections = []

    for part in parts:
        part = _ct(part)

        if len(part) >= 40:
            sections.append(part)

    if not sections and text:
        sections = [text]

    return sections


def extract_it_sections(html_content: str, job_role: str = "") -> list[str]:
    sections = _section_bounds(html_content)

    if not sections:
        return []

    job_role_text = str(job_role or "")
    it_sections = []

    for section in sections:
        combined = f"{job_role_text} {section}"

        if _contains_any(combined, IT_SECTION_HINTS):
            it_sections.append(section)

    return it_sections


def _extract_fields(section: str, max_chars_per_field: int = 500) -> str:
    section = clean_html_text(section)

    if not section:
        return ""

    collected = []

    for start_kw in FIELD_START_KEYWORDS:
        if start_kw not in section:
            continue

        start = section.find(start_kw)

        end_candidates = []

        for end_kw in FIELD_END_KEYWORDS:
            idx = section.find(end_kw, start + len(start_kw))

            if idx > start:
                end_candidates.append(idx)

        end = min(end_candidates) if end_candidates else min(
            len(section),
            start + max_chars_per_field,
        )

        snippet = section[start:end]
        snippet = clean_anchor(snippet)

        if snippet:
            collected.append(snippet[:max_chars_per_field])

    if not collected:
        return clean_anchor(section[:1200])

    deduped = []
    seen = set()

    for item in collected:
        key = _safe_compact(item)

        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    return clean_anchor(" ".join(deduped))


def parse_job_anchor_v2(job: dict) -> tuple[str, str, int]:
    html_content = job.get("html_content", "")
    job_role = job.get("job_role", "")

    it_sections = extract_it_sections(html_content, job_role)

    if it_sections:
        anchor = " ".join(
            _extract_fields(section)
            for section in it_sections
        ).strip()

        anchor = clean_anchor(anchor)

        if anchor:
            return "IT_SECTION", anchor, len(it_sections)

    return "", "", 0



# anchor cleaning / validation
NOISE_PHRASES = [
    "직무수행태도",
    "직업기초능력",
    "의사소통능력",
    "대인관계능력",
    "조직이해능력",
    "문제해결능력",
    "자원관리능력",
    "정보능력",
    "수리능력",
    "자기개발능력",
    "윤리의식",
    "공직윤리",
    "블라인드 채용",
    "국가직무능력표준",
    "NCS 기반 채용",
    "기관소개",
    "회사소개",
]


def clean_anchor(text: str) -> str:
    text = clean_html_text(text)

    for phrase in NOISE_PHRASES:
        text = text.replace(phrase, " ")

    text = re.sub(r"채용\s*공고문\s*참조", " ", text)
    text = re.sub(r"자세한\s*사항은?\s*.*?참조", " ", text)
    text = re.sub(r"※\s*.*?참조", " ", text)

    return _ct(text)


def is_short_but_it_anchor(text: str) -> bool:
    text = clean_anchor(text)

    if len(_safe_compact(text)) < 25:
        return False

    return _contains_any(text, IT_SECTION_HINTS)


def is_valid_anchor(text: str, min_len: int = 80) -> bool:
    text = clean_anchor(text)

    if len(_safe_compact(text)) < min_len:
        return is_short_but_it_anchor(text)

    intro_noise = ["설립", "비전", "미션", "기관", "공사", "공단", "재단"]

    if (
        sum(1 for k in intro_noise if k in text[:300]) >= 4
        and not _contains_any(text, FIELD_START_KEYWORDS)
    ):
        return False

    return True


def is_broken_prefix_anchor(text: str) -> bool:
    text = clean_anchor(text)

    broken_prefixes = [
        "및 ",
        "등 ",
        "또는 ",
        "관련 ",
        "운영 및 ",
    ]

    return any(text.startswith(prefix) for prefix in broken_prefixes)


# fallback
def current_job_fallback_anchor(job: dict, max_chars: int = 1800) -> str:
    text = join_anchor_parts([
        job.get("title", ""),
        job.get("company_name", ""),
        job.get("job_role", ""),
        job.get("work_experience", ""),
        job.get("recrut_type", ""),
        job.get("apply_qualification", ""),
        job.get("application_method", ""),
        job.get("html_content", ""),
        job.get("jd_text", ""),
    ])

    return clean_anchor(text)[:max_chars]


def parse_job_anchor_finetuning_base(job: dict) -> tuple[str, str, int]:
    pattern, anchor, it_section_count = parse_job_anchor_v2(job)

    if not _safe_compact(anchor):
        pattern, anchor = parse_job_anchor(job)
        it_section_count = 0

    if len(_safe_compact(anchor)) <= 30:
        fallback = current_job_fallback_anchor(job)

        if fallback:
            pattern = f"{pattern}_FALLBACK" if pattern else "FALLBACK"
            anchor = fallback

    return pattern, clean_anchor(anchor), it_section_count


def extract_anchor_current_matching(
    job: dict,
    max_chars: int = 1800,
) -> tuple[str, str, int]:
    if _safe_compact(job.get("anchor_text", "")):
        return clean_anchor(job.get("anchor_text", ""))[:max_chars], "EXISTING_ANCHOR", 0

    html_content = job.get("html_content", "")

    if _safe_compact(html_content):
        pattern, anchor, it_section_count = parse_job_anchor_finetuning_base(job)
        anchor = clean_anchor(anchor)[:max_chars]

        if is_valid_anchor(anchor) and not is_broken_prefix_anchor(anchor):
            final_anchor = clean_anchor(join_anchor_parts([
                job.get("title", ""),
                job.get("job_role", ""),
                anchor,
            ]))

            return final_anchor[:max_chars], pattern, it_section_count

        fallback = current_job_fallback_anchor(job, max_chars=max_chars)

        return fallback, "CURRENT_JOB_FALLBACK", it_section_count

    fallback = current_job_fallback_anchor(job, max_chars=max_chars)

    return fallback, "CURRENT_JOB_FALLBACK", 0



# cluster - notebook 기준 핵심 로직 반영
CLUSTER_PRIORITY = [
    "보안/정보보호",
    "시스템개발/SW",
    "데이터/AI",
    "DB/데이터관리",
    "인프라/시스템운영",
    "네트워크/통신",
    "기획/PM",
    "산업/설비·OT 운영",
]


def get_job_cluster_v3(anchor_text: str, job_role: str = "") -> str:
    text = _ct(anchor_text)
    role = _ct(job_role)
    full = f"{text} {role}"

    scores = {
        "보안/정보보호": 0,
        "시스템개발/SW": 0,
        "네트워크/통신": 0,
        "인프라/시스템운영": 0,
        "데이터/AI": 0,
        "DB/데이터관리": 0,
        "기획/PM": 0,
        "산업/설비·OT 운영": 0,
    }

    patterns = {
        "DB/데이터관리": [
            r"DBA", r"데이터베이스", r"\bDB\b", r"SQL", r"Oracle",
            r"MySQL", r"PostgreSQL", r"DB\s*모델링", r"데이터\s*모델링",
            r"데이터\s*품질", r"데이터\s*표준", r"데이터\s*거버넌스",
            r"데이터\s*아키텍처", r"메타데이터", r"데이터\s*관리",
        ],
        "데이터/AI": [
            r"데이터\s*분석", r"빅데이터", r"AI", r"인공지능",
            r"머신러닝", r"딥러닝", r"통계", r"분석", r"알고리즘",
            r"공공데이터", r"데이터\s*플랫폼", r"데이터\s*기반",
            r"Python", r"\bR\b", r"Pandas", r"Numpy",
        ],
        "보안/정보보호": [
            r"정보보안", r"정보보호", r"보안", r"개인정보",
            r"침해", r"취약점", r"방화벽", r"ISMS", r"보안관제",
            r"악성코드", r"암호화", r"접근통제", r"보안장비",
            r"보안점검", r"인증관리",
        ],
        "시스템개발/SW": [
            r"소프트웨어\s*개발", r"SW\s*개발", r"시스템\s*개발",
            r"정보시스템\s*개발", r"프로그램\s*개발", r"응용SW",
            r"웹\s*개발", r"앱\s*개발", r"서버\s*개발", r"API",
            r"Java", r"Spring", r"Python", r"React", r"프로그래밍",
            r"개발\s*및\s*운영",
        ],
        "네트워크/통신": [
            r"네트워크", r"통신", r"정보통신", r"통신망",
            r"AMI", r"DAS", r"망\s*관리", r"회선", r"라우터",
            r"스위치", r"LAN", r"WAN", r"VPN", r"모뎀",
            r"무선", r"유선", r"통신설비",
        ],
        "인프라/시스템운영": [
            r"서버", r"인프라", r"시스템\s*운영", r"정보시스템\s*운영",
            r"시스템\s*관리", r"전산", r"전산운영", r"유지보수",
            r"장애", r"장애처리", r"클라우드", r"Linux",
            r"Windows\s*Server", r"IDC", r"가상화", r"운영관리",
        ],
        "기획/PM": [
            r"기획", r"PM", r"PO", r"사업관리", r"프로젝트\s*관리",
            r"정보화사업", r"ISP", r"ISMP", r"요구사항",
            r"정책", r"성과관리", r"서비스\s*기획", r"사업\s*기획",
            r"관리계획",
        ],
        "산업/설비·OT 운영": [
            r"설비", r"전기", r"전자", r"배전", r"발전",
            r"제어", r"현장", r"시설", r"장비", r"OT",
            r"자동화", r"계측", r"전력", r"공사", r"배전자동화",
            r"발전제어", r"시설관리", r"현장관리",
        ],
    }

    for cluster, pats in patterns.items():
        for p in pats:
            if re.search(p, full, re.IGNORECASE):
                scores[cluster] += 4

    # 보정: 특정 키워드가 강하게 나오면 추가 가중
    if _has(full, [r"AMI", r"DAS", r"통신망", r"정보통신"]):
        scores["네트워크/통신"] += 5

    if _has(full, [r"배전", r"발전", r"제어", r"설비", r"현장"]):
        scores["산업/설비·OT 운영"] += 5

    if _has(full, [r"정보보안", r"정보보호", r"보안관제", r"침해대응"]):
        scores["보안/정보보호"] += 5

    if _has(full, [r"DBA", r"데이터베이스", r"\bSQL\b", r"DB관리"]):
        scores["DB/데이터관리"] += 4

    best_cluster = max(
        scores,
        key=lambda c: (
            scores[c],
            -CLUSTER_PRIORITY.index(c) if c in CLUSTER_PRIORITY else -999,
        ),
    )

    if scores[best_cluster] <= 0:
        return "UNKNOWN"

    return best_cluster


def cluster_score_table_finetuning_base(
    anchor_text: str,
    job_role: str = "",
    html_content: str = "",
) -> dict:
    text = _ct(anchor_text)
    role = _ct(job_role)
    html = _ct(html_content)
    full = f"{text} {role} {html}"

    scores = {
        "보안/정보보호": 0.0,
        "시스템개발/SW": 0.0,
        "네트워크/통신": 0.0,
        "인프라/시스템운영": 0.0,
        "데이터/AI": 0.0,
        "DB/데이터관리": 0.0,
        "기획/PM": 0.0,
        "산업/설비·OT 운영": 0.0,
    }

    patterns = {
        "DB/데이터관리": [
            r"DBA", r"데이터베이스", r"\bDB\b", r"SQL", r"Oracle",
            r"MySQL", r"PostgreSQL", r"DB\s*모델링", r"데이터\s*모델링",
            r"데이터\s*품질", r"데이터\s*표준", r"데이터\s*거버넌스",
            r"데이터\s*관리",
        ],
        "데이터/AI": [
            r"데이터\s*분석", r"빅데이터", r"AI", r"인공지능",
            r"머신러닝", r"통계", r"분석", r"알고리즘",
            r"공공데이터", r"데이터\s*플랫폼", r"Python", r"\bR\b",
        ],
        "보안/정보보호": [
            r"정보보안", r"정보보호", r"보안", r"개인정보",
            r"침해", r"취약점", r"방화벽", r"ISMS", r"보안관제",
            r"암호화", r"접근통제", r"보안점검",
        ],
        "시스템개발/SW": [
            r"소프트웨어\s*개발", r"SW\s*개발", r"시스템\s*개발",
            r"정보시스템\s*개발", r"프로그램\s*개발", r"응용SW",
            r"웹\s*개발", r"앱\s*개발", r"서버\s*개발", r"API",
            r"Java", r"Spring", r"Python", r"React", r"프로그래밍",
        ],
        "네트워크/통신": [
            r"네트워크", r"통신", r"정보통신", r"통신망",
            r"AMI", r"DAS", r"망\s*관리", r"회선", r"라우터",
            r"스위치", r"LAN", r"WAN", r"VPN", r"모뎀",
        ],
        "인프라/시스템운영": [
            r"서버", r"인프라", r"시스템\s*운영", r"정보시스템\s*운영",
            r"시스템\s*관리", r"전산", r"전산운영", r"유지보수",
            r"장애", r"클라우드", r"Linux", r"IDC", r"가상화",
        ],
        "기획/PM": [
            r"기획", r"PM", r"PO", r"사업관리", r"프로젝트\s*관리",
            r"정보화사업", r"ISP", r"ISMP", r"요구사항",
            r"정책", r"성과관리", r"서비스\s*기획",
        ],
        "산업/설비·OT 운영": [
            r"설비", r"전기", r"전자", r"배전", r"발전",
            r"제어", r"현장", r"시설", r"장비", r"OT",
            r"자동화", r"계측", r"전력", r"공사", r"배전자동화",
            r"발전제어", r"시설관리",
        ],
    }

    for cluster, pats in patterns.items():
        base_weight = 4.0

        if cluster in {"시스템개발/SW", "인프라/시스템운영", "네트워크/통신"}:
            base_weight = 4.5

        if cluster in {"기획/PM", "산업/설비·OT 운영"}:
            base_weight = 3.5

        for p in pats:
            if re.search(p, text, re.IGNORECASE):
                scores[cluster] += base_weight
            if re.search(p, role, re.IGNORECASE):
                scores[cluster] += 2.0
            if re.search(p, html, re.IGNORECASE):
                scores[cluster] += 1.0

    # fallback 강한 보정
    if _has(full, [r"AMI", r"DAS", r"통신망", r"정보통신"]):
        scores["네트워크/통신"] += 5.0

    if _has(full, [r"배전", r"발전", r"제어", r"설비", r"현장"]):
        scores["산업/설비·OT 운영"] += 5.0

    if _has(full, [r"정보보안", r"정보보호", r"보안관제", r"침해대응"]):
        scores["보안/정보보호"] += 5.0

    if _has(full, [r"DBA", r"데이터베이스", r"\bSQL\b", r"DB관리"]):
        scores["DB/데이터관리"] += 4.0

    return {k: float(v) for k, v in scores.items() if v > 0}


def classify_cluster_multi(
    anchor_text: str,
    job_role: str = "",
    html_content: str = "",
    top_n: int = 3,
    secondary_ratio: float = 0.45,
) -> dict:
    scores = cluster_score_table_finetuning_base(
        anchor_text,
        job_role,
        html_content,
    )

    if not scores:
        return {
            "primary_cluster": "UNKNOWN",
            "secondary_clusters": [],
            "cluster_scores": {},
            "cluster_candidates": [],
        }

    ordered = sorted(
        scores.items(),
        key=lambda kv: (
            -kv[1],
            CLUSTER_PRIORITY.index(kv[0]) if kv[0] in CLUSTER_PRIORITY else 999,
        ),
    )

    primary, best_score = ordered[0]

    secondary = []
    candidates = []

    for cluster, score in ordered:
        candidates.append({
            "cluster": cluster,
            "score": float(score),
        })

        if cluster == primary:
            continue

        if best_score > 0 and score / best_score >= secondary_ratio:
            secondary.append(cluster)

    return {
        "primary_cluster": primary,
        "secondary_clusters": secondary[: max(0, top_n - 1)],
        "cluster_scores": {k: float(v) for k, v in scores.items()},
        "cluster_candidates": candidates[:top_n],
    }


def classify_cluster(
    anchor_text: str,
    job_role: str = "",
    html_content: str = "",
) -> str:
    return classify_cluster_multi(
        anchor_text,
        job_role,
        html_content,
    )["primary_cluster"]


def get_job_cluster(
    anchor_text: str,
    html_content: str = "",
    job_role: str = "",
) -> str:
    return get_job_cluster_v3(
        join_anchor_parts([anchor_text, html_content]),
        job_role=job_role,
    )



# final public parser
def parse_public_job(job: dict, max_anchor_chars: int = 1800) -> dict:
    anchor, pattern, it_section_count = extract_anchor_current_matching(
        job,
        max_chars=max_anchor_chars,
    )

    cluster_info = classify_cluster_multi(
        anchor,
        job_role=job.get("job_role", ""),
        html_content=job.get("html_content", ""),
    )

    return {
        "anchor_text": anchor,
        "pattern": pattern,
        "it_section_count": it_section_count,
        "anchor_source": "finetuning_full_parser_with_current_job_fallback",
        "primary_cluster": cluster_info["primary_cluster"],
        "secondary_clusters": cluster_info["secondary_clusters"],
        "cluster_scores": cluster_info["cluster_scores"],
        "cluster_candidates": cluster_info["cluster_candidates"],
    }


def enrich_public_job(job: dict, max_anchor_chars: int = 1800) -> dict:
    parsed = parse_public_job(job, max_anchor_chars=max_anchor_chars)

    new_job = dict(job)
    new_job.update(parsed)

    # scoring_ncs.py 호환용
    new_job["cluster"] = parsed["primary_cluster"]

    return new_job


def audit_public_jobs(jobs: list[dict]) -> dict:
    enriched = [enrich_public_job(job) for job in jobs]

    cluster_counts = Counter(job.get("primary_cluster", "UNKNOWN") for job in enriched)
    pattern_counts = Counter(job.get("pattern", "") for job in enriched)

    return {
        "n_jobs": len(enriched),
        "cluster_counts": dict(cluster_counts),
        "pattern_counts": dict(pattern_counts),
        "unknown_count": cluster_counts.get("UNKNOWN", 0),
        "secondary_cluster_count": sum(bool(job.get("secondary_clusters")) for job in enriched),
    }
