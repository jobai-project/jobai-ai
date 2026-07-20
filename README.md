
## IT 채용공고 자동 수집 · AI 적합도 매칭 · 실시간 알림 서비스

<p>
  <a href="https://app.jobai.site/"><img src="https://img.shields.io/badge/🔗 서비스 바로가기-6C5CE7?style=for-the-badge&logoColor=white"></a>
  
</p>

<img width="1920" height="1080" alt="표지" src="https://github.com/user-attachments/assets/fa4c9941-82ce-4194-bf24-f2a444191e19" />

---

## 📌 Project Overview
> JobA!는 사용자가 채용공고를 직접 찾는 대신,
**IT 채용공고를 자동으로 수집**하고
**AI 모델이 이력서와 매칭해 적합도를 점수화**하며,
**사용자에게 적합한 공고를 실시간으로 알려 주는** 채용 준비 자동화 서비스입니다.

공고 수집부터 이력서 매칭, 알림 발송, 지원 현황 관리까지
채용 준비의 모든 과정을 하나로 연결했습니다.

<br>

---

## 🎯 Problem & Solution

### &ensp;[Problem]
- **플랫폼 분산** — 취준생 절반 이상이 사람인·잡코리아·원티드 등 2개 이상을 동시에 사용
- **수시채용 전환** — 공고를 놓칠 위험이 증가
- **직접 확인** — 공고를 하나씩 읽어 봐야 나와 맞는지 알 수 있음
- **반복 확인** — 새 공고가 올라왔는지 계속 확인해봐야 함
- **근거 부족** — 왜 이 공고가 나에게 맞는지 추천 근거를 알기 어려움
- **복잡한 조건** — 조건이 다양하고 복잡해 의도에 맞는 공고를 정확히 찾기 번거로움
- **흩어진 일정** — 채용 관련 일정(지원 현황·면접 등)을 한곳에서 관리하기 어려움

<br>

### &ensp;[Solution]
- **자동 통합 수집** — 사기업·공공기관 IT 공고를 한곳에 자동으로 수집해 제공
- **AI 적합도 분석** — 이력서를 분석해 공고별 적합도를 0~100점으로 산출
- **AI 요약 제공** — 공고의 핵심 내용을 요약해 빠르게 확인
- **신규 공고 알림** — 설정한 적합도 기준을 넘는 공고가 올라오면 Slack·이메일로 알림
- **매칭 근거 제시** — 적합도 점수의 근거를 함께 제공해 신뢰성을 높임
- **자연어 검색** — 키워드 한 단어든 긴 문장이든, 의도를 파악해 적합한 공고를 찾을 수 있음
- **일정 통합 관리** — 스크랩한 공고의 지원 현황·면접 일정을 한곳에서 관리

  <br>

---

## ✨ Key Features
- **자동 수집 & 실시간 알림** — 사기업·공공기관 IT 공고를 통합 수집하고, 적합도 기준을 넘는 공고를 알림
- **AI 적합도 매칭** — 이력서–공고 적합도를 0~100점으로 점수화하고 보유·부족 기술과 추천 근거 제공
- **자연어 공고 검색** — 키워드부터 문장까지, 검색 의도를 이해해 적합한 공고 탐색
- **AI 공고 요약** — 공고 핵심을 요약하고 이력서와 비교 분석
- **지원 현황 트래커** — 스크랩한 공고의 지원 단계·면접 일정을 자동 정리하고 다가오는 일정 요약
- **온보딩 기반 맞춤 설정** — 희망 직무·근무 지역·고용 형태·적합도 기준·알림 채널

  <br>

---

## 📈 Model Finetuning

**사기업 모델 파인튜닝 결과**

|  | Loss | avg pos sim | avg neg sim | gap | pairwise acc | ROC-AUC | n_pairs |
| --- | ---| --- | --- | --- | --- | --- | --- |
| base model | -| 0.5915 | 0.6261 | -0.0347 | 0.3300 | 0.4123 | 100 |
| finetuning model | MultipleNegativesRankingLoss| 0.9566 | 0.3012 | **0.6618** | **1.0000** | **1.0000** | 100 |


<br>


**공기업 모델 파인튜닝 결과**

|  | Loss | avg pos sim | avg neg sim | gap | pairwise acc | ROC-AUC | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base model | - | 0.682927 | 0.63123 | 0.051698 | 0.702182 | 0.649468 | 779 |
| finetuning model | MultipleNegativesRankingLoss | 0.534970 | -0.084504 | **0.629473** | **0.975610** | **0.977042** | 779 |

<br>

---

## ⚙️ AI Matching Workflow

<table>
<tr>
<td width="50%">



<img width="2720" height="6059" alt="image" src="https://github.com/user-attachments/assets/c5acd68a-e359-4a84-b7c7-f828643e19ab" />


</td>
<td width="50%">



<img width="2720" height="6059" alt="image" src="https://github.com/user-attachments/assets/354df053-d225-4080-948d-63ced62a8da1" />


</td>
</tr>
</table>

---


## ⚛️ AI Matching Engine


> 공기업 공고는 사기업과 달리 자유 서술형 JD가 아니라 NCS(국가직무능력표준) 기반 직무기술서 형태로 올라오는 경우가 많아, 이 구조를 반영한 별도의 적합도 계산 로직이 필요했고, 이에 따라 사기업·공기업 모델을 분리하여 진행했습니다.

<br>

### 사기업 매칭 로직

| 항목 | 설명 | 계산 방식 | 가중치 |
| --- | --- | --- | --- |
| **Ts** | 기술스택 매칭률 | 공고 텍스트에서 추출한 기술 키워드 ↔ 이력서 skills 교집합 비율 | 0.30 |
| **Cs** | 임베딩 유사도 | 파인튜닝 모델로 변환한 공고 벡터 ↔ 이력서 벡터 코사인 유사도 | 0.35 |
| **Qs** | 자격요건 충족도 | 경력 연차 충족 여부(70%) + 우대사항 매칭률(30%) | 0.20 |
| **Gp** | 누락 패널티 | 필수 기술 누락/경력 미달 시 감점 + score cap 별도 적용 | 별도 |

```
base_score = 100 × (0.30Ts + 0.35Cs + 0.20Qs)
final_score = Gp(base_score)
```
<br>

### 공기업 매칭 로직

| 항목 | 설명 | 계산 방식 | 가중치 |
| --- | --- | --- | --- |
| **Kw** | 키워드 매칭률 | NCS 텍스트에서 추출한 필요기술 키워드 ↔ 이력서 skills 교집합 비율 | 0.25 |
| **Cs** | 임베딩 유사도 | 파인튜닝 모델로 변환한 NCS anchor 벡터 ↔ 이력서 벡터 코사인 유사도 | 0.35 |
| **Ws** | 가중 구조 점수 | 직무 클러스터 매칭(0.4) + 경력 충족(0.45) + 자격 충족(0.10) + Kw(0.05)의 재가중합 | 0.40 |
| **Gp** | 누락 패널티 | 필수 자격 미보유·경력 미달·클러스터 불일치 시 감점 + score cap 별도 적용 | 별도 |

```
base_score = 100 × (0.25Kw + 0.35Cs + 0.40Ws)
final_score = Gp(base_score)
```

<br>


---
## 🔧 Tech Stack
<div>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Hugging Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Uvicorn-499848?style=flat-square&logo=uvicorn&logoColor=white">
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white">
</div>


<br>

---
## 📁 Directory Structure

```
jobai-ai/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── ai-server/
│   ├── .gitignore
│   ├── main.py
│   ├── parser_ncs.py
│   ├── requirements.txt
│   ├── scoring_jd.py
│   └── scoring_ncs.py
├── .dockerignore
├── Dockerfile
├── entrypoint.sh
└── README.md
```


<br>
