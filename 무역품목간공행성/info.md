
# 무역 품목 공행성 기반 무역량 예측 대회

## 1. 대회 목표 (Competition Goal)

원시 무역 수입 데이터(2022년 1월 ~ 2025년 7월)를 분석하여 품목 간 선후행 공행성 쌍 (A → B)을 예측하고, 공행성이 있다고 판단된 쌍에 대해 후행 품목(B)의 다음 달(2025년 8월) 총 무역량(`value`)을 예측하는 AI 모델을 개발하는 것

원시 무역 수입 데이터(2022년 1월 ~ 2025년 7월)를 기반으로, 품목 간 공행성(comovement)이 존재하는 선후행 쌍을 예측하고, 공행성이 있다고 판단된 경우에는 후행 품목의 다음 달(2025년 8월) 총 무역량(value)을 예측하는 AI 모델을 개발합니다.

참가자는 주어진 원시 무역 데이터를 분석하여 품목 간 선후행 관계가 존재하는 공행성 쌍(A → B)을 찾아야 하며, 이후에는 선행 품목(A)의 흐름을 활용해 후행 품목(B)의 다음달의 **총 무역량(value)**을 예측해야 합니다.

---

## 2. 데이터셋 정보 (Dataset Info)

### 2.1. train.csv (원시 무역 데이터)

| 컬럼명 | 설명 |
| :--- | :--- |
| **item\_id** | 무역품 식별 ID |
| **year** | 년도 (2022~2025) |
| **month** | 월 (1~12) |
| seq | 동일 년-월 내 일련번호 |
| type | 유형 구분 코드 |
| hs4 | HS4 코드 (품목 분류) |
| weight | 중량 |
| quantity | 수량 |
| **value** | **총 무역량** (예측 목표값) |

### 2.2. 데이터 기간

* 학습/분석 기간**: 2022년 1월 ~ 2025년 7월
* **예측 기간**: 2025년 8월 (다음 달)

---

## 3. 평가 산식 (Evaluation Metric)

최종 점수는 **공행성 쌍 예측 정확도 (F1 Score)**와 **무역량 예측 정확도 (NMAE)**를 결합하여 산출됩니다.

$$
\text{Score} = 0.6 \times \text{F1} + 0.4 \times (1 − \text{NMAE})
$$

### 3.1. F1 Score (가중치 60%)

선후행 공행성 쌍 예측의 정밀도(Precision)와 재현율(Recall)의 조화 평균입니다.

$$
\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

* **TP (True Positive)**: 정답과 예측 모두에 포함된 공행성 쌍
* **FP (False Positive)**: 예측에는 있으나 정답에는 없는 쌍
* **FN (False Negative)**: 정답에는 있으나 예측에 없는 쌍

### 3.2. NMAE (Normalized Mean Absolute Error, 가중치 40%)

정규화된 평균 절대 오차로, 예측 무역량의 정확도를 측정합니다. NMAE 값이 낮을수록 (0에 가까울수록) 좋은 점수입니다.

$$
\text{NMAE} = \frac{1}{|\text{U}|} \times \sum\left[\min\left(1, \frac{|y_{\text{true}} - y_{\text{pred}}|}{\left|y_{\text{true}}\right| + \varepsilon}\right)\right]
$$

* **$y_{\text{true}}$**: 정답의 다음 달 무역량 (2025년 8월 실제 값)
* **$y_{\text{pred}}$**: 예측 무역량 (정수 반올림된 값)
* **FN 또는 FP인 쌍**: 공행성 쌍 예측이 틀린 경우, 해당 쌍의 오차는 무조건 **1.0 (최하점)**으로 처리됩니다. **정확한 쌍을 찾는 것이 매우 중요**합니다.

---

## 4. 권장 파일 구조 및 코드 관리

대회 코드를 체계적으로 관리하기 위해 다음 폴더 구조를 추천합니다.

```
dacon_project/
├── data/
│   ├── train.csv
│   └── sample_submission.csv
│
├── submissions/
│   ├── baseline_submit.csv
│   └── lgbm_corr045_log_roll3_v1.csv  <-- 현재 개선된 모델의 결과 파일
│
└── Trade_Comovement.ipynb      <-- 주피터 노트북 파일 (핵심 분석 코드)
```

---

## 5. 제출 파일 형식 (Submission File Format)

최종 제출 파일은 `sample_submission.csv`와 완전히 동일한 헤더와 형식을 가져야 합니다.

| 컬럼명 | 설명 |
| :--- | :--- |
| **leading\_item\_id** | 선행 무역품 식별 ID (A) |
| **following\_item\_id** | 후행 무역품 식별 ID (B) |
| **value** | **[예측 값]** 후행 품목(B)의 2025년 8월 **총 무역량** (정수) |

### 제출 파일 예시

| leading\_item\_id | following\_item\_id | value |
| :--- | :--- | :--- |
| AANGBULD | APQGTRMF | 360075 |
| AANGBULD | DEWLVASR | 610115 |
| RJGPVEXX | AHMDUILJ | 125890 |
| ... | ... | ... |

**주의:** 제출 파일은 반드시 **CSV 형식**이어야 하며, 예측된 `value`는 **정수**로 변환되어야 합니다.