# Trade_Comovement_V3.py
# F1 점수 회복을 위해 Raw Value 상관관계 탐색으로 롤백하고,
# NMAE 개선을 위해 LGBM 조기 종료 및 'b_lag_12' 피처를 추가했습니다.

# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
# from lightgbm.callback import early_stopping # 환경에 따라 이 구문을 사용
# Python 3.13.5 환경을 가정하여, early_stopping이 lightgbm.callback에 있다고 가정합니다.
try:
    from lightgbm.callback import early_stopping
except ImportError:
    # 사용자 환경에 따라 import 방식이 다를 수 있으므로, 코드를 유연하게 유지합니다.
    print("Warning: Failed to import early_stopping from lightgbm.callback. Proceeding without explicit import.")
    def early_stopping(*args, **kwargs):
        return None # 더미 함수로 대체

# 2. 데이터 불러오기 및 전처리
TRAIN_PATH = './data/train.csv'
SUBMISSION_DIR = './submissions/'

# 공행성 쌍 탐색 시 사용할 임계값. F1 점수 확보를 위해 Raw-Value 기준의 0.45로 유지합니다.
CORR_THRESHOLD = 0.45 

print("1. 데이터 불러오기 및 전처리 시작")
train = pd.read_csv(TRAIN_PATH)

# year, month, item_id 기준으로 value 합산
monthly = (
    train
    .groupby(["item_id", "year", "month"], as_index=False)["value"]
    .sum()
)

# year, month를 하나의 키(ym)로 묶기
monthly["ym"] = pd.to_datetime(
    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
)

# item_id × ym 피벗 (월별 총 무역량 매트릭스 생성)
pivot = (
    monthly
    .pivot(index="item_id", columns="ym", values="value")
    .fillna(0.0)
)

print(f"  - 원본 데이터 행 수: {len(train)}")
print(f"  - Pivot Table Shape: {pivot.shape}")
print("  - Pivot Table Head:")
print(pivot.head())

# 3. 공행성쌍 탐색 함수 (F1 스코어 회복 전략 적용: Raw Value 상관관계 사용)
def safe_corr(x, y):
    """표준편차가 0인 경우(거래량이 변하지 않는 경우) NaN 방지"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.4):
    """
    [V3 전략] Raw Value 기준으로 상관관계 탐색하여 F1 스코어를 회복합니다.
    """
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)
    results = []
    n_items = len(items)

    for i, leader in enumerate(items):
        if i % 100 == 0:
            print(f"    - Leader Item 진행률: {i}/{n_items} ({leader})")
        
        # Raw Value 사용 (로그 변환 없음)
        x = pivot.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue

            # Raw Value 사용 (로그 변환 없음)
            y = pivot.loc[follower].values.astype(float)
            if np.count_nonzero(y) < min_nonzero:
                continue

            best_lag = None
            best_corr = 0.0

            # lag = 1 ~ max_lag 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                
                corr = safe_corr(x[:-lag], y[lag:]) # Raw Value로 상관관계 계산
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            # 임계값 이상이면 공행성쌍으로 채택
            if best_lag is not None and abs(best_corr) >= corr_threshold:
                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": best_lag,
                    "max_corr": best_corr,
                })

    pairs = pd.DataFrame(results)
    return pairs

pairs = find_comovement_pairs(pivot, corr_threshold=CORR_THRESHOLD) 

print(f"\n2. 공행성쌍 탐색 완료 (Threshold: {CORR_THRESHOLD}, Raw-Value)")
print(f"  - 탐색된 공행성쌍 수: {len(pairs)}")
print("  - 공행성쌍 Head:")
print(pairs.head())

# 4. 학습 데이터 구축 및 모델 학습 (NMAE 개선 피처 추가)
def build_training_data(pivot, pairs):
    """
    [V3 개선] 계절성 피처 (b_lag_12) 추가
    """
    months = pivot.columns.to_list()
    n_months = len(months)
    rows = []

    for row in pairs.itertuples(index=False):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)
        b_series_pd = pd.Series(b_series)

        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 1), n_months - 1):
            
            # 1. 베이스라인 Features (로그 변환 적용)
            b_t = np.log1p(b_series[t])
            b_t_1 = np.log1p(b_series[t - 1])
            a_t_lag = np.log1p(a_series[t - lag])
            b_t_plus_1 = np.log1p(b_series[t + 1])

            # 2. 피처 엔지니어링 (로그 변환 적용)
            b_roll_mean_3_raw = b_series_pd.rolling(window=3, min_periods=1).mean().iloc[t]
            b_roll_mean_3 = np.log1p(b_roll_mean_3_raw)
            
            # [추가] 12개월 전 값 (t-12). t가 12 미만이면 0 처리.
            b_lag_12_raw = b_series[t - 12] if t - 12 >= 0 else 0.0
            b_lag_12 = np.log1p(b_lag_12_raw)

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "b_roll_mean_3": b_roll_mean_3,
                "b_lag_12": b_lag_12, # <- 추가된 피처
                "target": b_t_plus_1,
            })

    df_train_model = pd.DataFrame(rows)
    return df_train_model

df_train_model = build_training_data(pivot, pairs)
print('\n3. 학습 데이터 구축 및 모델 학습')
print(f'  - 생성된 학습 데이터의 shape: {df_train_model.shape}')
print("  - 학습 데이터 Head:")
print(df_train_model.head())

# [수정] 피처 리스트에 'b_lag_12' 추가
feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 'b_roll_mean_3', 'b_lag_12'] 

# 검증 세트 분리 및 LGBM 학습
print("\n  - 검증 세트 분리 (Split: 80% Train, 20% Validation)")
X = df_train_model[feature_cols]
y = df_train_model["target"]
# 시계열이므로 shuffle=False
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"  - Train/Validation Split: {X_train.shape} / {X_val.shape}")

# n_estimators를 1000으로 설정하고 Early Stopping 사용
reg = LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, n_jobs=-1) 

print("  - LGBMRegressor 학습 시작 (with Early Stopping)...")
callbacks = [early_stopping(stopping_rounds=50, verbose=100)] if 'early_stopping' in globals() else []

reg.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=callbacks)

print(f"  - 학습 모델: {type(reg).__name__} 학습 완료 (Best Iteration: {getattr(reg, 'best_iteration_', 1000)})")

# 5. 예측 및 제출 파일 생성
def predict(pivot, pairs, reg, feature_cols):
    """
    [V3 개선] 계절성 피처 (b_lag_12)를 추가하여 2025년 8월 예측
    """
    months = pivot.columns.to_list()
    n_months = len(months)

    t_last = n_months - 1       # 2025년 7월 (현재 시점)
    t_prev = n_months - 2       # 2025년 6월
    t_lag_12 = n_months - 1 - 12 # 2024년 8월 (12개월 전)

    preds = []
    
    # FutureWarning 해결
    pivot_roll_mean_3 = pivot.T.rolling(window=3, min_periods=1).mean().T

    print(f"4. 2025년 8월 예측 시작 ({len(pairs)}쌍)")
    for row in pairs.itertuples(index=False):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        # 데이터 기간 충족 확인
        if t_last - lag < 0:
            continue
            
        # 1. 베이스라인 Features
        b_t = np.log1p(b_series[t_last])
        b_t_1 = np.log1p(b_series[t_prev])
        a_t_lag = np.log1p(a_series[t_last - lag])
        
        # 2. 피처 엔지니어링
        b_roll_mean_3_raw = pivot_roll_mean_3.loc[follower].iloc[t_last] 
        b_roll_mean_3 = np.log1p(b_roll_mean_3_raw)

        # [추가] 12개월 전 값 (2024년 8월)
        b_lag_12_raw = b_series[t_lag_12] if t_lag_12 >= 0 else 0.0
        b_lag_12 = np.log1p(b_lag_12_raw)

        # X_test를 DataFrame으로 만들어 UserWarning 해결
        test_data = {
            'b_t': b_t,
            'b_t_1': b_t_1,
            'a_t_lag': a_t_lag,
            'max_corr': corr,
            'best_lag': float(lag),
            'b_roll_mean_3': b_roll_mean_3,
            'b_lag_12': b_lag_12, # <- 추가된 피처
        }
        X_test_df = pd.DataFrame([test_data], columns=feature_cols) 

        # (중요) 모델이 예측한 값은 log 스케일임
        y_pred_log = reg.predict(X_test_df)[0]
        y_pred = np.expm1(y_pred_log)

        # 후처리 (음수/정수 변환)
        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred

# 예측 실행
submission = predict(pivot, pairs, reg, feature_cols)
print("\n5. 제출 파일 Head:")
print(submission.head())

# 제출 파일을 submissions 폴더에 저장합니다. 버전 관리용 이름을 사용합니다.
submission_name = f'lgbm_rawcorr_{str(CORR_THRESHOLD).replace(".", "")}_rollmean3_lag12_v3.csv'
submission.to_csv(f'{SUBMISSION_DIR}{submission_name}', index=False)
print(f"\n✅ 최종 제출 파일({submission_name})이 submissions 폴더에 저장되었습니다.")