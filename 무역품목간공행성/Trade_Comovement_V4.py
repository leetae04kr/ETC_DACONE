# Trade_Comovement_V4.py
# NMAE 개선을 위해 단기 시차 피처 (b_lag_2, a_lag_1)를 추가하고 LGBM 규제(regularization)를 강화했습니다.

# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
# early_stopping 콜백 임포트 (환경에 따라 조정될 수 있음)
try:
    from lightgbm.callback import early_stopping
except ImportError:
    print("Warning: Failed to import early_stopping from lightgbm.callback. Proceeding with dummy function.")
    def early_stopping(*args, **kwargs):
        return None 

# 2. 데이터 불러오기 및 전처리
TRAIN_PATH = './data/train.csv'
SUBMISSION_DIR = './submissions/'
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

# 3. 공행성쌍 탐색 함수 (V3와 동일하게 Raw Value 상관관계 사용)
def safe_corr(x, y):
    """표준편차가 0인 경우(거래량이 변하지 않는 경우) NaN 방지"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.4):
    """Raw Value 기준으로 상관관계 탐색"""
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)
    results = []
    n_items = len(items)

    for i, leader in enumerate(items):
        if i % 100 == 0:
            print(f"    - Leader Item 진행률: {i}/{n_items} ({leader})")
        
        x = pivot.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue

            y = pivot.loc[follower].values.astype(float)
            if np.count_nonzero(y) < min_nonzero:
                continue

            best_lag = None
            best_corr = 0.0

            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                
                corr = safe_corr(x[:-lag], y[lag:])
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

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

# 4. 학습 데이터 구축 및 모델 학습 (V4 개선: 단기 피처 추가)
def build_training_data(pivot, pairs):
    """
    [V4 개선] 단기 시차 피처 (b_lag_2, a_lag_1) 추가
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
        # b_lag_2를 위해 시작점을 max(lag, 2)로 변경
        for t in range(max(lag, 2), n_months - 1):
            
            # 1. 베이스라인 Features (로그 변환 적용)
            b_t = np.log1p(b_series[t])
            b_t_1 = np.log1p(b_series[t - 1])
            b_t_plus_1 = np.log1p(b_series[t + 1])
            
            # 2. 공행성 기반 피처
            a_t_lag = np.log1p(a_series[t - lag])
            b_roll_mean_3_raw = b_series_pd.rolling(window=3, min_periods=1).mean().iloc[t]
            b_roll_mean_3 = np.log1p(b_roll_mean_3_raw)
            
            # 3. 추가 피처 엔지니어링
            # V3: 12개월 전 값
            b_lag_12_raw = b_series[t - 12] if t - 12 >= 0 else 0.0
            b_lag_12 = np.log1p(b_lag_12_raw)
            
            # [V4 추가] 2개월 전 값
            b_lag_2 = np.log1p(b_series[t - 2])
            
            # [V4 추가] 선행 품목 직전월 값 (A의 t-1 시점)
            a_lag_1_raw = a_series[t - 1] if t - 1 >= 0 else 0.0
            a_lag_1 = np.log1p(a_lag_1_raw)
            
            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "b_roll_mean_3": b_roll_mean_3,
                "b_lag_12": b_lag_12,
                "b_lag_2": b_lag_2, # <- V4 추가
                "a_lag_1": a_lag_1, # <- V4 추가
                "target": b_t_plus_1,
            })

    df_train_model = pd.DataFrame(rows)
    return df_train_model

df_train_model = build_training_data(pivot, pairs)
print('\n3. 학습 데이터 구축 및 모델 학습')
print(f'  - 생성된 학습 데이터의 shape: {df_train_model.shape}')
print("  - 학습 데이터 Head:")
print(df_train_model.head())

# [V4 수정] 피처 리스트에 'b_lag_2', 'a_lag_1' 추가
feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 'b_roll_mean_3', 'b_lag_12', 'b_lag_2', 'a_lag_1'] 

# 검증 세트 분리 및 LGBM 학습
print("\n  - 검증 세트 분리 (Split: 80% Train, 20% Validation)")
X = df_train_model[feature_cols]
y = df_train_model["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"  - Train/Validation Split: {X_train.shape} / {X_val.shape}")

# [V4 수정] 규제(L1/L2)를 추가하여 모델 안정성 확보
reg = LGBMRegressor(
    random_state=42, 
    n_estimators=1000, 
    learning_rate=0.05, 
    n_jobs=-1,
    lambda_l1=0.1,    # L1 규제 추가
    lambda_l2=0.1     # L2 규제 추가
) 

print("  - LGBMRegressor 학습 시작 (with Early Stopping & Regularization)...")
callbacks = [early_stopping(stopping_rounds=50, verbose=100)] if 'early_stopping' in globals() else []

reg.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=callbacks)

print(f"  - 학습 모델: {type(reg).__name__} 학습 완료 (Best Iteration: {getattr(reg, 'best_iteration_', 1000)})")

# 5. 예측 및 제출 파일 생성 (V4 개선)
def predict(pivot, pairs, reg, feature_cols):
    """
    [V4 개선] 단기 시차 피처를 사용하여 2025년 8월 예측
    """
    months = pivot.columns.to_list()
    n_months = len(months)

    t_last = n_months - 1       # 2025년 7월 (현재 시점)
    t_prev = n_months - 2       # 2025년 6월
    t_lag_2 = n_months - 3      # 2025년 5월
    t_lag_12 = n_months - 1 - 12 # 2024년 8월

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

        # 예측에 필요한 최소 기간 (lag 또는 lag_2)을 충족하는지 확인
        if t_last - max(lag, 2) < 0: 
            continue
            
        # 1. 시계열/공행성 Features
        b_t = np.log1p(b_series[t_last])
        b_t_1 = np.log1p(b_series[t_prev])
        a_t_lag = np.log1p(a_series[t_last - lag])
        
        # 2. 롤링 평균
        b_roll_mean_3_raw = pivot_roll_mean_3.loc[follower].iloc[t_last] 
        b_roll_mean_3 = np.log1p(b_roll_mean_3_raw)

        # 3. 추가 피처 엔지니어링
        # 12개월 전 값
        b_lag_12_raw = b_series[t_lag_12] if t_lag_12 >= 0 else 0.0
        b_lag_12 = np.log1p(b_lag_12_raw)
        
        # [V4 추가] 2개월 전 값
        b_lag_2 = np.log1p(b_series[t_lag_2])
        
        # [V4 추가] 선행 품목 직전월 값 (A의 t_last - 1 시점 = 2025년 6월 값)
        a_lag_1_raw = a_series[t_prev] 
        a_lag_1 = np.log1p(a_lag_1_raw)
        
        # X_test를 DataFrame으로 만들어 UserWarning 해결
        test_data = {
            'b_t': b_t,
            'b_t_1': b_t_1,
            'a_t_lag': a_t_lag,
            'max_corr': corr,
            'best_lag': float(lag),
            'b_roll_mean_3': b_roll_mean_3,
            'b_lag_12': b_lag_12,
            'b_lag_2': b_lag_2,      # <- V4 추가
            'a_lag_1': a_lag_1,      # <- V4 추가
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
submission_name = f'lgbm_rawcorr_{str(CORR_THRESHOLD).replace(".", "")}_lag_enhanced_v4.csv'
submission.to_csv(f'{SUBMISSION_DIR}{submission_name}', index=False)
print(f"\n✅ 최종 제출 파일({submission_name})이 submissions 폴더에 저장되었습니다.")