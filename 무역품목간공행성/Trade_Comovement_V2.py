# Trade_Comovement_V2.py

# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
# 선형 회귀 모델 대신 더 강력한 트리 기반 모델을 사용하기 위해 LightGBM을 임포트합니다.
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression # 베이스라인 코드를 위해 남겨둡니다.

# [수정] 모델 검증 및 조기 종료를 위한 라이브러리 임포트
from sklearn.model_selection import train_test_split
from lightgbm.callback import early_stopping

# 2. 데이터 불러오기 및 전처리
TRAIN_PATH = './data/train.csv'
SUBMISSION_DIR = './submissions/' # 결과물을 저장할 폴더 경로

print("1. 데이터 불러오기 및 전처리 시작")
train = pd.read_csv(TRAIN_PATH)

# year, month, item_id 기준으로 value 합산 (seq만 다르다면 value 합산)
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

# 3. 공행성쌍 탐색 함수 (개선)
def safe_corr(x, y):
    """표준편차가 0인 경우(거래량이 변하지 않는 경우) NaN 방지"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.4):
    """
    [개선] pivot 테이블에서 (A -> B) 공행성쌍을 찾아내는 함수
           원본 값이 아닌, np.log1p로 변환된 시계열의 상관관계를 계산합니다.
    """
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []
    
    n_items = len(items)

    for i, leader in enumerate(items):
        if i % 100 == 0:
            print(f"    - Leader Item 진행률: {i}/{n_items} ({leader})")
        
        # [수정] 원본 값은 0 개수 체크에만 사용
        x_raw = pivot.loc[leader].values
        if np.count_nonzero(x_raw) < min_nonzero:
            continue
        # [수정] 로그 변환된 값을 상관관계 계산에 사용
        x_log = np.log1p(x_raw.astype(float))

        for follower in items:
            if follower == leader:
                continue

            # [수정] y도 동일하게 적용
            y_raw = pivot.loc[follower].values
            if np.count_nonzero(y_raw) < min_nonzero:
                continue
            y_log = np.log1p(y_raw.astype(float))

            best_lag = None
            best_corr = 0.0

            # lag = 1 ~ max_lag 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                    
                # [수정] 로그 변환된 시계열로 상관관계를 계산합니다.
                corr = safe_corr(x_log[:-lag], y_log[lag:])
                
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

# [튜닝 포인트] 이 임계값(0.4, 0.45, 0.5 등)을 변경하며 F1 스코어를 테스트하세요.
CORR_THRESHOLD = 0.45 
pairs = find_comovement_pairs(pivot, corr_threshold=CORR_THRESHOLD) 

print(f"\n2. 공행성쌍 탐색 완료 (Threshold: {CORR_THRESHOLD}, Log-Transformed)")
print(f"  - 탐색된 공행성쌍 수: {len(pairs)}")
print("  - 공행성쌍 Head:")
print(pairs.head())

# 4. 학습 데이터 구축 및 모델 학습 (개선)
def build_training_data(pivot, pairs):
    """
    공행성쌍 + 시계열을 이용해 (X, y) 학습 데이터를 만드는 함수
    (이 함수는 원본과 동일하게 유지)
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
        b_series_pd = pd.Series(b_series) # 이동 평균을 쉽게 계산하기 위해 Series로 변환

        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 1), n_months - 1):
            
            # 1. 베이스라인 Features (로그 변환 적용!)
            b_t = np.log1p(b_series[t])
            b_t_1 = np.log1p(b_series[t - 1])
            a_t_lag = np.log1p(a_series[t - lag])
            b_t_plus_1 = np.log1p(b_series[t + 1]) # <- Target에도 적용

            # 2. 피처 엔지니어링 (로그 변환 적용!)
            # 3개월 이동 평균을 계산 (이동 평균은 원본 값으로 계산)
            b_roll_mean_3_raw = b_series_pd.rolling(window=3, min_periods=1).mean().iloc[t]
            b_roll_mean_3 = np.log1p(b_roll_mean_3_raw)

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "b_roll_mean_3": b_roll_mean_3,
                "target": b_t_plus_1,  # <- log1p(Target)이 저장됨
            })

    df_train_model = pd.DataFrame(rows)
    return df_train_model

df_train_model = build_training_data(pivot, pairs)
print('\n3. 학습 데이터 구축 및 모델 학습')
print(f'  - 생성된 학습 데이터의 shape: {df_train_model.shape}')
print("  - 학습 데이터 Head:")
print(df_train_model.head())

# 모델 학습
feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 'b_roll_mean_3'] 

# --- [수정] 검증 세트 분리 (시계열이므로 shuffle=False) ---
print("\n  - 검증 세트 분리 (Split: 80% Train, 20% Validation)")
X = df_train_model[feature_cols]
y = df_train_model["target"]

# test_size=0.2는 마지막 20%의 데이터를 검증용으로 사용
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"  - Train/Validation Split: {X_train.shape} / {X_val.shape}")
# --- [수정 완료] ---

# LightGBM 모델로 학습을 수행합니다.
# [수정] n_estimators를 1000 정도로 늘리고 Early Stopping 사용
reg = LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, n_jobs=-1) 

# [수정] 조기 종료(Early Stopping) 설정
print("  - LGBMRegressor 학습 시작 (with Early Stopping)...")
reg.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse', # NMAE와 유사한 RMSE 또는 MAE 사용
        callbacks=[early_stopping(stopping_rounds=50, verbose=100)]) # 50 라운드 동안 성능 향상 없으면 중지

print(f"  - 학습 모델: {type(reg).__name__} 학습 완료 (Best Iteration: {reg.best_iteration_})")

# 5. 예측 및 제출 파일 생성 (개선)
def predict(pivot, pairs, reg, feature_cols):
    """
    [개선] 2025년 8월 총 무역량(value)을 예측하는 함수
           Warning을 수정합니다.
    """
    months = pivot.columns.to_list()
    n_months = len(months)

    # 가장 마지막 두 달 index (2025-7, 2025-6)
    t_last = n_months - 1 # 2025년 7월 (현재 시점)
    t_prev = n_months - 2 # 2025년 6월

    preds = []
    
    # [수정] .T.rolling(...).T 방식으로 FutureWarning 해결
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

        # t_last - lag 가 0 이상인 경우만 예측 (데이터 기간 충족 확인)
        if t_last - lag < 0:
            continue

        # 1. 베이스라인 Features (로그 변환 적용!)
        b_t = np.log1p(b_series[t_last])
        b_t_1 = np.log1p(b_series[t_prev])
        a_t_lag = np.log1p(a_series[t_last - lag])
        
        # 2. 피처 엔지니어링 (로그 변환 적용!)
        b_roll_mean_3_raw = pivot_roll_mean_3.loc[follower].iloc[t_last] 
        b_roll_mean_3 = np.log1p(b_roll_mean_3_raw)
        
        # [수정] X_test를 DataFrame으로 만들어 UserWarning 해결
        test_data = {
            'b_t': b_t,
            'b_t_1': b_t_1,
            'a_t_lag': a_t_lag,
            'max_corr': corr,
            'best_lag': float(lag),
            'b_roll_mean_3': b_roll_mean_3
        }
        X_test_df = pd.DataFrame([test_data], columns=feature_cols) 

        # (중요) 모델이 예측한 값은 log 스케일임
        y_pred_log = reg.predict(X_test_df)[0]
        
        # (중요) np.expm1을 사용해 원래 스케일로 복원
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
submission_name = f'lgbm_logcorr_{str(CORR_THRESHOLD).replace(".", "")}_rollmean3_v2.csv'
submission.to_csv(f'{SUBMISSION_DIR}{submission_name}', index=False)
print(f"\n✅ 최종 제출 파일({submission_name})이 submissions 폴더에 저장되었습니다.")