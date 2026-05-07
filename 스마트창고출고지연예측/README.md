# 스마트 창고 출고 지연 예측 AI 프로젝트

## 1. 프로젝트 개요

### 배경

AMR(자율이동로봇) 기반 스마트 물류창고가 빠르게 확산되고 있습니다. 하지만 로봇 대수, 주문량, 배터리 상태, 통로 혼잡도 등 다양한 운영 요인이 복합적으로 작용하면서, 출고 지연을 사전에 예측하고 대응하는 것은 여전히 어려운 과제입니다.

본 프로젝트는 가상의 스마트 물류창고 시뮬레이션에서 생성된 운영 스냅샷 데이터를 활용하여, **현재 시점으로부터 향후 30분간의 평균 출고 지연 시간을 예측하는 AI 모델을 개발**하는 것을 목표로 합니다.

### 대회 정보

- **기간**: 2026.04.01 ~ 2026.05.04 10:00
- **주제**: 스마트 물류창고 운영 데이터를 기반으로 향후 30분간의 평균 출고 지연 시간(분)을 예측하는 모델 개발
- **설명**: 참가자는 창고 운영 스냅샷 데이터(로봇 상태, 주문량, 배터리, 혼잡도 등 90개 피처)를 분석하여, 각 시점에서 향후 30분간 출고되는 주문들의 평균 지연 시간('avg_delay_minutes_next_30m')을 예측해야 합니다. 메인 테이블 외에 창고 레이아웃 정보를 담은 보조 테이블(`layout_info.csv`)이 제공되며, 이를 효과적으로 활용하는 것이 중요합니다.

### 평가 산식: MAE (Mean Absolute Error)

본 프로젝트의 모델 성능은 **MAE(평균 절대 오차)**로 평가됩니다. 실제값과 예측값의 차이의 절댓값에 대한 평균으로, 모델의 예측이 실제와 얼마나 차이 나는지를 직관적으로 보여주는 지표입니다.

$MAE = \frac{\sum |y - \hat{y}|}{n}$

- **Public Score**: 전체 테스트 데이터 중 샘플링된 30%로 채점
- **Private Score**: 전체 테스트 데이터 중 나머지 70%로 최종 순위 결정

---

## 2. 데이터 구조 및 설명

### 파일 구성

```
.
├── data/
│   ├── train.csv              # 학습 데이터 (250,000행 × 94컬럼)
│   ├── test.csv               # 평가 데이터 (50,000행 × 93컬럼)
│   ├── layout_info.csv        # 창고 레이아웃 보조 정보 (300행 × 15컬럼)
│   └── sample_submission.csv  # 제출 양식 파일
├── code/
│   ├── maincode.ipynb         # 메인 분석 및 모델링 코드
│   └── [Baseline]...ipynb     # 베이스라인 코드
├── features/
│   ├── train_feat.parquet     # 생성된 학습용 피처 데이터
│   └── test_feat.parquet      # 생성된 테스트용 피처 데이터
└── submit/
    └── (submission files)     # 생성된 제출 파일
```

### 주요 데이터 컬럼

- **기본 정보**: `ID`, `layout_id`, `scenario_id`, `avg_delay_minutes_next_30m` (타겟)
- **주문 및 작업 부하**: `order_inflow_15m`, `unique_sku_15m`, `urgent_order_ratio` 등
- **로봇 및 배터리 상태**: `robot_active`, `robot_utilization`, `battery_mean`, `charge_queue_length` 등
- **창고 내부 혼잡도**: `congestion_score`, `max_zone_density`, `pack_utilization` 등
- **환경 및 작업자 요인**: `warehouse_temp_avg`, `staff_on_floor`, `worker_avg_tenure_months` 등
- **IT 인프라**: `wms_response_time_ms`, `wifi_signal_db`, `path_optimization_score` 등

---

## 3. 개발 현황 및 코드 설명

현재 프로젝트는 **LightGBM 단일 모델**을 사용하여 빠르고 효율적으로 예측 결과를 도출하는 데 초점을 맞추고 있습니다.

### `code/maincode.ipynb` 핵심 로직

1.  **데이터 로드 및 전처리**:
    - `pandas`를 사용하여 `train.csv`, `test.csv`, `layout_info.csv` 데이터를 불러옵니다.
    - `ROOT` 경로를 동적으로 설정하여 어떤 환경에서도 데이터 경로를 올바르게 찾도록 구현했습니다.

2.  **피처 엔지니어링 (`build_features` 함수)**:
    - **시계열 피처 생성**: 각 `scenario_id` 내에서 시간 순서대로 데이터를 정렬한 후, 주요 수치형 데이터(`congestion_score`, `order_inflow_15m` 등)에 대해 Lag, Difference, Rolling(이동 평균) 피처를 생성합니다.
    - **파생 변수 생성**: 도메인 지식을 바탕으로 로봇 운영 효율, 충전 압박, 공간 혼잡도 등 새로운 의미를 갖는 40여 개의 상호작용 및 비율 피처를 생성합니다.
        - 예: `effective_robot_utilization` (실질 로봇 가동률), `task_density_per_robot` (로봇당 작업 밀도)
    - **결측치 처리**: 시계열 피처 생성 시 발생하는 `NaN` 값은 각 시나리오 내에서 `ffill`과 `bfill`을 사용하여 보간하고, 남은 결측은 0으로 채웁니다.
    - **피처 캐싱(Caching)**: 생성된 피처는 `features` 폴더에 `parquet` 형식으로 저장하여, 두 번째 실행부터는 피처 생성 과정을 건너뛰고 저장된 파일을 바로 불러와 분석 시간을 단축합니다. (`FORCE_UPDATE=True`로 설정 시 강제 재생성)

3.  **모델 학습 및 검증**:
    - **모델**: `LightGBM` (`LGBMRegressor`) 단일 모델을 사용합니다. 빠른 학습 속도와 높은 성능으로 초기 베이스라인 구축에 유리합니다.
    - **교차 검증 (Cross-Validation)**: `GroupKFold`를 사용하여 `scenario_id`를 그룹으로 묶어 교차 검증을 수행합니다. 이를 통해 동일 시나리오의 데이터가 학습 및 검증 데이터에 동시에 포함되는 것을 방지하여 데이터 누수(Leakage)를 막고, 보다 신뢰도 높은 OOF(Out-of-Fold) 성능을 측정합니다.
    - **하이퍼파라미터**: `n_estimators=4000`, `learning_rate=0.02` 등 성능 위주로 튜닝된 파라미터를 사용하며, `early_stopping`을 적용하여 과적합을 방지하고 최적의 학습 반복 횟수를 찾습니다.

4.  **예측 및 제출**:
    - 교차 검증 과정에서 각 Fold에서 생성된 테스트 데이터 예측값을 평균하여 최종 예측 결과를 만듭니다.
    - 예측값에 `np.clip(val, 0, None)`을 적용하여 음수 예측값을 0으로 보정합니다.
    - 최종 결과는 `submission_lgbm_only.csv` 파일로 저장됩니다.

### 현재 상태

- 복잡한 앙상블 모델(XGBoost, CatBoost)을 제외하고, LightGBM 단일 모델을 중심으로 코드를 리팩토링하여 빠른 실험과 결과 제출이 가능하도록 최적화되었습니다.
- 현재 OOF MAE 기준으로 **약 9.3점대**의 안정적인 성능을 보이고 있습니다.
- 다음 단계로는 피처 중요도를 분석하여 불필요한 피처를 제거하거나, 새로운 피처를 추가하여 성능을 개선하는 작업을 계획하고 있습니다.
