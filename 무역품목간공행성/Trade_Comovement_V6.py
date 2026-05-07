# Trade_Comovement_V7.py
# V7 개선: Optuna 에러 (TypeError: squared) 해결 및 안정화 버전

# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error # mse로 계산 후 np.sqrt() 적용
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
FEATURE_COLS = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 'b_roll_mean_3', 'b_lag_12', 'b_lag_2', 'a_lag_1', 'b_diff_log_2m']


def load_and_preprocess_data():
    """데이터를 불러와 pivot table을 생성하고, 월별 합계를 구합니다."""
    print("1. 데이터 불러오기 및 전처리 시작")
    train = pd.read_csv(TRAIN_PATH)
    
    monthly = (
        train
        .groupby(["item_id", "year", "month"], as_index=False)["value"]
        .sum()
    )

    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )

    pivot = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )
    print(f"  - Pivot Table Shape: {pivot.shape}")
    return pivot

# 3. 공행성쌍 탐색 함수 (F1 확보 전략: Raw Value 상관관계 사용)
def safe_corr(x, y):
    """표준편차가 0인 경우(거래량이 변하지 않는 경우) NaN 방지"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.4):
    """Raw Value 기준으로 상관관계 탐색"""
    items = pivot.index.to_list()
    n_months = len(pivot.columns)
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
    print(f"\n2. 공행성쌍 탐색 완료 (Threshold: {corr_threshold}, Raw-Value). 탐색된 쌍 수: {len(pairs)}")
    return pairs

# 4. 학습 데이터 구축 (V5 피처 세트 사용)
def build_training_data(pivot, pairs):
    """
    V5의 모든 피처를 사용하여 학습 데이터 생성
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

        # b_lag_2와 b_diff_log_2m를 위해 시작점을 max(lag, 2)로 변경
        for t in range(max(lag, 2), n_months - 1):
            
            # 1. 시계열/공행성 Features (로그 변환 적용)
            b_t = np.log1p(b_series[t])
            b_t_1 = np.log1p(b_series[t - 1])
            b_t_plus_1 = np.log1p(b_series[t + 1])
            a_t_lag = np.log1p(a_series[t - lag])

            # 2. 피처 엔지니어링
            b_roll_mean_3_raw = b_series_pd.rolling(window=3, min_periods=1).mean().iloc[t]
            b_roll_mean_3 = np.log1p(b_roll_mean_3_raw)
            
            b_lag_12_raw = b_series[t - 12] if t - 12 >= 0 else 0.0
            b_lag_12 = np.log1p(b_lag_12_raw)
            
            b_lag_2 = np.log1p(b_series[t - 2])
            
            a_lag_1_raw = a_series[t - 1] if t - 1 >= 0 else 0.0
            a_lag_1 = np.log1p(a_lag_1_raw)
            
            # V5: 2개월 모멘텀 (로그 값 차이)
            b_diff_log_2m = np.log1p(b_series[t]) - np.log1p(b_series[t - 2])

            rows.append({
                "b_t": b_t, "b_t_1": b_t_1, "a_t_lag": a_t_lag, "max_corr": corr, "best_lag": float(lag),
                "b_roll_mean_3": b_roll_mean_3, "b_lag_12": b_lag_12, "b_lag_2": b_lag_2,
                "a_lag_1": a_lag_1, "b_diff_log_2m": b_diff_log_2m,
                "target": b_t_plus_1,
            })

    df_train_model = pd.DataFrame(rows)
    print(f'  - 생성된 학습 데이터의 shape: {df_train_model.shape}')
    return df_train_model

# 5. Optuna Objective Function 정의
def objective(trial, X_train, X_val, y_train, y_val, feature_cols):
    """
    [수정] RMSE 계산 시 squared=False 인수를 제거하고 np.sqrt()를 적용하여 에러 해결
    """
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
    }

    model = LGBMRegressor(**param)
    
    callbacks = [early_stopping(stopping_rounds=50, verbose=False)] if 'early_stopping' in globals() else []

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=callbacks)

    # [에러 해결 부분] squared=False 인수를 제거하고 MSE에 np.sqrt()를 적용합니다.
    preds = model.predict(X_val[feature_cols])
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    
    return rmse


# 6. 예측 함수 (최적 파라미터를 사용)
def predict(pivot, pairs, reg, feature_cols):
    """
    최적의 LGBM 모델을 사용하여 2025년 8월 예측
    """
    months = pivot.columns.to_list()
    n_months = len(months)

    t_last = n_months - 1
    t_prev = n_months - 2
    t_lag_2 = n_months - 3
    t_lag_12 = n_months - 1 - 12

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

        if t_last - max(lag, 2) < 0: 
            continue
            
        # 피처 계산 (V5 피처 세트)
        b_t = np.log1p(b_series[t_last])
        b_t_1 = np.log1p(b_series[t_prev])
        a_t_lag = np.log1p(a_series[t_last - lag])
        b_roll_mean_3_raw = pivot_roll_mean_3.loc[follower].iloc[t_last] 
        b_roll_mean_3 = np.log1p(b_roll_mean_3_raw)
        b_lag_12_raw = b_series[t_lag_12] if t_lag_12 >= 0 else 0.0
        b_lag_12 = np.log1p(b_lag_12_raw)
        b_lag_2 = np.log1p(b_series[t_lag_2])
        a_lag_1_raw = a_series[t_prev] 
        a_lag_1 = np.log1p(a_lag_1_raw)
        b_diff_log_2m = np.log1p(b_series[t_last]) - np.log1p(b_series[t_lag_2])
        
        test_data = {
            'b_t': b_t, 'b_t_1': b_t_1, 'a_t_lag': a_t_lag, 'max_corr': corr, 'best_lag': float(lag),
            'b_roll_mean_3': b_roll_mean_3, 'b_lag_12': b_lag_12, 'b_lag_2': b_lag_2,
            'a_lag_1': a_lag_1, 'b_diff_log_2m': b_diff_log_2m,
        }
        X_test_df = pd.DataFrame([test_data], columns=feature_cols) 

        y_pred_log = reg.predict(X_test_df)[0]
        y_pred = np.expm1(y_pred_log)

        # 후처리
        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred

# ====== 메인 실행 블록 ======
if __name__ == '__main__':
    pivot_data = load_and_preprocess_data()
    
    # 3. 공행성쌍 탐색
    comovement_pairs = find_comovement_pairs(pivot_data, corr_threshold=CORR_THRESHOLD)

    # 4. 학습 데이터 구축
    df_model = build_training_data(pivot_data, comovement_pairs)
    
    # 학습/검증 세트 분리 (시계열이므로 shuffle=False)
    X = df_model[FEATURE_COLS]
    y = df_model["target"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"  - Train/Validation Split: {X_train.shape} / {X_val.shape}")

    # --- 5. Optuna 튜닝 실행 ---
    N_TRIALS = 50 # 테스트 실행을 위한 샘플 트라이얼 수. 실제 튜닝을 위해서는 이 값을 500 이상으로 늘리세요.
    print(f"\n3. Optuna 튜닝 시작 (Trials: {N_TRIALS})...")
    
    # Optuna Study 생성 및 실행
    study = optuna.create_study(direction='minimize', study_name='lgbm_trade_comovement')
    
    # objective 함수를 실행할 때, 학습 데이터셋을 인수로 전달
    func = lambda trial: objective(trial, X_train, X_val, y_train, y_val, FEATURE_COLS)
    study.optimize(func, n_trials=N_TRIALS, show_progress_bar=True)
    
    best_params = study.best_params
    best_value = study.best_value
    print(f"\n✅ Optuna 튜닝 완료 (Best RMSE: {best_value:.5f})")
    print(f"  - Best Parameters: {best_params}")

    # --- 6. 최종 모델 학습 (Best Parameters 사용) ---
    print("\n5. 최종 모델 학습 (Best Parameters 적용)...")
    
    # Best Parameters를 기반으로 최종 LGBM 모델 생성 및 학습
    final_params = {
        'n_estimators': 2000, 
        'n_jobs': -1, 
        'random_state': 42,
        'verbose': -1,
        **best_params # Optuna가 찾은 최적 파라미터 적용
    }
    # n_estimators를 2000으로 설정하고, early_stopping을 통해 최적의 트리를 사용합니다.
    final_regressor = LGBMRegressor(**final_params)
    
    # 전체 학습 데이터(Train+Validation)로 재학습
    final_callbacks = [early_stopping(stopping_rounds=50, verbose=False)] if 'early_stopping' in globals() else []

    final_regressor.fit(X, y,
                        eval_set=[(X_val, y_val)], # validation set으로 조기 종료 시점만 확인
                        eval_metric='rmse',
                        callbacks=final_callbacks)
    
    # --- 7. 예측 및 제출 파일 생성 ---
    submission_df = predict(pivot_data, comovement_pairs, final_regressor, FEATURE_COLS)
    print("\n6. 제출 파일 Head:")
    print(submission_df.head())

    submission_name = f'lgbm_optuna_v7.csv'
    submission_df.to_csv(f'{SUBMISSION_DIR}{submission_name}', index=False)
    print(f"\n✅ 최종 제출 파일({submission_name})이 submissions 폴더에 저장되었습니다.")