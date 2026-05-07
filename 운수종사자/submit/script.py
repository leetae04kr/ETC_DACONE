
import pandas as pd
import numpy as np
import re
import glob
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import gc
import os
import joblib
from tqdm import tqdm # tqdm 임포트 추가

# --- 1. 전처리 함수 정의 (노트북 1.3과 동일한 고급 버전) ---

def preprocess_age(age_str):
    """'40b', '60a' 같은 나이 문자열을 숫자로 변환합니다."""
    if pd.isna(age_str):
        return np.nan
    try:
        # 정규식 수정: ^\d+ (숫자로 시작하는 부분)
        base_age = int(re.findall(r'^\d+', age_str)[0])
        if 'a' in age_str:
            return base_age
        elif 'b' in age_str:
            return base_age + 5
    except:
        return np.nan

def get_string_list_stats(series, col_name):
    """
    [개선된 버전] 쉼표로 구분된 문자열 리스트를 받아
    통계 특성(평균, std, min, max, 중앙값, 개수, NaN개수, CV, 합계, 유효개수)을 계산합니다.
    """
    
    def calculate_stats_for_row(s):
        if pd.isna(s):
            # 빈 행에 대한 기본값 (10개 피처)
            return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan, np.nan, 0])
        try:
            num_list = [pd.to_numeric(x, errors='coerce') for x in str(s).split(',')]
        except:
            num_list = [pd.to_numeric(s, errors='coerce')]
            
        mean_val = np.nanmean(num_list)
        std_val = np.nanstd(num_list)
        min_val = np.nanmin(num_list)
        max_val = np.nanmax(num_list)
        median_val = np.nanmedian(num_list)
        sum_val = np.nansum(num_list)
        count_val = len(num_list)
        nan_count_val = pd.Series(num_list).isna().sum()
        non_nan_count_val = count_val - nan_count_val
        cv_val = std_val / (mean_val + 1e-6)
        
        return pd.Series([mean_val, std_val, min_val, max_val, median_val, count_val, nan_count_val, cv_val, sum_val, non_nan_count_val])

    stat_names = ['mean', 'std', 'min', 'max', 'median', 'count', 'nan_count', 'cv', 'sum', 'non_nan_count']
    col_names = [f'{col_name}_{stat}' for stat in stat_names]
    df_stats = series.apply(calculate_stats_for_row)
    df_stats.columns = col_names
    return df_stats.astype(np.float32) # 타입 지정

def parse_list_safe(s, dtype=float):
    """ 쉼표로 구분된 문자열을 안전하게 리스트로 변환 """
    if pd.isna(s):
        return []
    try:
        return [dtype(x) for x in str(s).split(',') if x.strip()]
    except Exception:
        try:
            return [dtype(s)]
        except Exception:
            return []

def create_cognitive_features(df_chunk):
    """
    PDF 명세를 기반으로 "오반응 비율", "인지 부하(Stroop)" 피처를 생성합니다.
    """
    results = []
    has_a4 = all(c in df_chunk.columns for c in ['A4-1', 'A4-3', 'A4-5'])
    has_b4 = all(c in df_chunk.columns for c in ['B4-1', 'B4-2'])

    for _, row in df_chunk.iterrows():
        features = {}
        if has_a4:
            conditions = parse_list_safe(row['A4-1'], dtype=float)
            responses = parse_list_safe(row['A4-3'], dtype=float)
            times = parse_list_safe(row['A4-5'], dtype=float)
            if len(conditions) == len(responses) == len(times):
                con_times = [t for c, t in zip(conditions, times) if c == 1]
                incon_times = [t for c, t in zip(conditions, times) if c == 2]
                con_correct = [r for c, r in zip(conditions, responses) if c == 1 and r == 1]
                incon_correct = [r for c, r in zip(conditions, responses) if c == 2 and r == 1]
                con_rt_mean = np.nanmean(con_times) if con_times else np.nan
                incon_rt_mean = np.nanmean(incon_times) if incon_times else np.nan
                features['A4_stroop_delta'] = incon_rt_mean - con_rt_mean
                features['A4_con_accuracy'] = len(con_correct) / (len(con_times) + 1e-6)
                features['A4_incon_accuracy'] = len(incon_correct) / (len(incon_times) + 1e-6)
                features['A4_total_accuracy'] = (len(con_correct) + len(incon_correct)) / (len(times) + 1e-6)
        if has_b4:
            responses = parse_list_safe(row['B4-1'], dtype=float)
            times = parse_list_safe(row['B4-2'], dtype=float)
            if len(responses) == len(times):
                con_times = [t for r, t in zip(responses, times) if r in [1, 2]]
                incon_times = [t for r, t in zip(responses, times) if r in [3, 4, 5, 6]]
                con_correct = [r for r in responses if r == 1]
                incon_correct = [r for r in responses if r in [3, 5]]
                con_rt_mean = np.nanmean(con_times) if con_times else np.nan
                incon_rt_mean = np.nanmean(incon_times) if incon_times else np.nan
                features['B4_stroop_delta'] = incon_rt_mean - con_rt_mean
                features['B4_con_accuracy'] = len(con_correct) / (len([r for r in responses if r in [1, 2]]) + 1e-6)
                features['B4_incon_accuracy'] = len(incon_correct) / (len([r for r in responses if r in [3, 4, 5, 6]]) + 1e-6)
                features['B4_total_accuracy'] = (len(con_correct) + len(incon_correct)) / (len(times) + 1e-6)
        code_cols_b = ['B1-1', 'B1-3', 'B2-1', 'B2-3', 'B3-1', 'B5-1', 'B6', 'B7', 'B8']
        for col in code_cols_b:
            if col in df_chunk.columns:
                codes = parse_list_safe(row[col], dtype=float)
                if codes:
                    incorrect_codes_simple = {2}
                    incorrect_codes_complex = {2, 4}
                    target_codes = incorrect_codes_complex if col in ['B1-3', 'B2-3'] else incorrect_codes_simple
                    incorrect_count = sum(1 for c in codes if c in target_codes)
                    features[f'{col}_incorrect_ratio'] = incorrect_count / (len(codes) + 1e-6)
        results.append(features)
    return pd.DataFrame(results, index=df_chunk.index)

def process_dataframe(df, file_type='A'):
    """
    DataFrame(청크)을 받아 전처리 및 특성 공학을 수행합니다. (고급 버전)
    """
    processed_cols = ['Test_id', 'PrimaryKey', 'TestDate']
    valid_cols = [col for col in processed_cols if col in df.columns]
    df_processed = df[valid_cols].copy()
    if 'Age' in df.columns: df_processed['Age'] = df['Age'].apply(preprocess_age)
    if 'TestDate' in df.columns:
        test_date_str = df['TestDate'].astype(str)
        df_processed['TestDate_Year'] = pd.to_numeric(test_date_str.str[:4], errors='coerce')
        df_processed['TestDate_Month'] = pd.to_numeric(test_date_str.str[4:], errors='coerce')
    
    if file_type == 'A':
        numeric_cols = [f'A6-1', 'A7-1', 'A8-1', 'A8-2'] + [f'A9-{i}' for i in range(1, 6)]
        time_string_cols = ['A1-4', 'A2-4', 'A3-7', 'A4-5', 'A5-3']
        code_string_cols = ['A1-1', 'A1-2', 'A1-3', 'A2-1', 'A2-2', 'A2-3', 
                            'A3-1', 'A3-2', 'A3-3', 'A3-4', 'A3-5', 'A3-6',
                            'A4-1', 'A4-2', 'A4-3', 'A4-4', 'A4-6', 'A4-7',
                            'A5-1', 'A5-2', 'A5-4', 'A5-5', 'A5-6', 'A5-7']
        string_cols = time_string_cols + code_string_cols
    elif file_type == 'B':
        numeric_cols = [f'B9-{i}' for i in range(1, 6)] + [f'B10-{i}' for i in range(1, 7)]
        time_string_cols = ['B1-2', 'B2-2', 'B3-2', 'B4-2', 'B5-2']
        code_string_cols = ['B1-1', 'B1-3', 'B2-1', 'B2-3', 'B3-1', 
                            'B4-1', 'B5-1', 'B6', 'B7', 'B8']
        string_cols = time_string_cols + code_string_cols
    else: 
        return pd.DataFrame()

    for col in numeric_cols:
        if col in df.columns:
            df_processed[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  Processing {len(string_cols)} string-list columns for basic stats (CV, etc.)...")
    # script.py에서는 tqdm_notebook 대신 tqdm 사용
    for col in tqdm(string_cols, desc=f"  Stats for {file_type}", leave=False):
        if col in df.columns:
            stats_df = get_string_list_stats(df[col], col)
            df_processed = pd.concat([df_processed, stats_df], axis=1)

    print(f"  Creating domain-specific cognitive features (Stroop, Accuracy)...")
    try:
        cognitive_features_df = create_cognitive_features(df)
        df_processed = pd.concat([df_processed, cognitive_features_df], axis=1)
    except Exception as e:
        print(f"    !! Warning: Cognitive feature creation failed. Error: {e}")
            
    return df_processed

# --- 2. 추론(Inference) 실행 ---
def main():
    print("[+] Inference script started.")
    DATA_DIR, MODEL_DIR, OUTPUT_DIR = './data', './model', './output'
    TEST_A_PATH = os.path.join(DATA_DIR, 'test', 'A.csv')
    TEST_B_PATH = os.path.join(DATA_DIR, 'test', 'B.csv')
    TEST_IDS_PATH = os.path.join(DATA_DIR, 'test.csv')
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, 'submission.csv')
    CHUNKSIZE = 100000

    # --- 모델 및 교정기 경로 정의 (3개 모델) ---
    LGB_CLF_MODEL_PATH = os.path.join(MODEL_DIR, 'model.txt')
    CAT_CLF_MODEL_PATH = os.path.join(MODEL_DIR, 'model_cat.cbm')
    LGB_REG_MODEL_PATH = os.path.join(MODEL_DIR, 'model_reg.txt') 
    LGB_CLF_CALIBRATOR_PATH = os.path.join(MODEL_DIR, 'calibrator_lgb.joblib')
    CAT_CLF_CALIBRATOR_PATH = os.path.join(MODEL_DIR, 'calibrator_cat.joblib')
    LGB_REG_CALIBRATOR_PATH = os.path.join(MODEL_DIR, 'calibrator_lgb_reg.joblib')
    
    # --- 모델, 교정기 로드 (3개 모델) ---
    try:
        lgb_clf_model = lgb.Booster(model_file=LGB_CLF_MODEL_PATH)
        lgb_features = lgb_clf_model.feature_name() 
        print(f"LGBM Classifier Model loaded from {LGB_CLF_MODEL_PATH}")
        
        cat_clf_model = CatBoostClassifier()
        cat_clf_model.load_model(CAT_CLF_MODEL_PATH)
        cat_features_model = cat_clf_model.feature_names_
        print(f"CatBoost Classifier Model loaded from {CAT_CLF_MODEL_PATH}")

        lgb_reg_model = lgb.Booster(model_file=LGB_REG_MODEL_PATH) 
        print(f"LGBM Regressor Model loaded from {LGB_REG_MODEL_PATH}")

        lgb_clf_calibrator = joblib.load(LGB_CLF_CALIBRATOR_PATH)
        print(f"LGBM Classifier Calibrator loaded from {LGB_CLF_CALIBRATOR_PATH}")
        cat_clf_calibrator = joblib.load(CAT_CLF_CALIBRATOR_PATH)
        print(f"CatBoost Classifier Calibrator loaded from {CAT_CLF_CALIBRATOR_PATH}")
        lgb_reg_calibrator = joblib.load(LGB_REG_CALIBRATOR_PATH) 
        print(f"LGBM Regressor Calibrator loaded from {LGB_REG_CALIBRATOR_PATH}")
    except Exception as e:
        print(f"!! Error loading models or calibrators: {e}")
        return

    # --- 테스트 데이터 전처리 A, B ---
    print("Processing Test A file...")
    processed_chunks_A = []
    try:
        with pd.read_csv(TEST_A_PATH, chunksize=CHUNKSIZE) as reader:
             # tqdm을 script.py에서 사용하도록 수정
             processed_chunks_A = [process_dataframe(chunk, file_type='A') for chunk in tqdm(reader, desc="Test A Chunks")]
    except FileNotFoundError:
        print("Test A file not found. Skipping.")
    gc.collect()

    print("Processing Test B file...")
    processed_chunks_B = []
    try:
        with pd.read_csv(TEST_B_PATH, chunksize=CHUNKSIZE) as reader:
             processed_chunks_B = [process_dataframe(chunk, file_type='B') for chunk in tqdm(reader, desc="Test B Chunks")]
    except FileNotFoundError:
        print("Test B file not found. Skipping.")
    gc.collect()

    df_A_processed = pd.concat(processed_chunks_A, ignore_index=True) if processed_chunks_A else pd.DataFrame()
    df_B_processed = pd.concat(processed_chunks_B, ignore_index=True) if processed_chunks_B else pd.DataFrame()
    
    # 예외 처리: A나 B 둘 중 하나만 있을 경우
    if df_A_processed.empty and df_B_processed.empty:
        print("!! Error: No test data found in A.csv or B.csv.")
        return
    
    df_all_features = pd.concat([df_A_processed, df_B_processed], ignore_index=True, sort=False)
    print(f"Total processed test features shape: {df_all_features.shape}")

    # --- 2차 피처 엔지니어링 (Lag, Delta, PK, Risk) ---
    try:
        df_test_meta = pd.read_csv(TEST_IDS_PATH)[['Test_id', 'Test']]
        df_test_final = pd.merge(df_test_meta, df_all_features, on='Test_id', how='left')
        
        df_test_final['TestDate_Numeric'] = pd.to_datetime(df_test_final['TestDate'], format='%Y%m', errors='coerce')
        df_test_final['TestDate_Numeric'] = df_test_final['TestDate_Numeric'].dt.year + (df_test_final['TestDate_Numeric'].dt.month / 12.0)
        df_test_final['TestDate_Numeric'] = df_test_final['TestDate_Numeric'].fillna(0)
        
        print("Creating Lag (Time-Series) features for test set...")
        df_test_final = df_test_final.sort_values(by=['PrimaryKey', 'TestDate_Numeric']).reset_index(drop=True)
        
        lag_cols = ['TestDate_Numeric', 'Age', 'A1-2_mean', 'A1-3_mean', 'A1-4_mean', 'A6-1', 'A7-1', 'B1-1_mean', 'B6_mean', 'B7_mean', 'B10-1', 'B10-2']
        
        lag_cols = [col for col in lag_cols if col in df_test_final.columns]
        for col in lag_cols:
            df_test_final[f'Prev_{col}'] = df_test_final.groupby('PrimaryKey')[col].shift(1)
            
        df_test_final['Test_Gap_Time'] = df_test_final.apply(lambda row: row['TestDate_Numeric'] - row['Prev_TestDate_Numeric'] if pd.notna(row['Prev_TestDate_Numeric']) and row['Prev_TestDate_Numeric'] > 0 else np.nan, axis=1)
        df_test_final['Age_Delta'] = df_test_final['Age'] - df_test_final['Prev_Age']
        if 'A1-2_mean' in lag_cols: df_test_final['A1-2_mean_Delta'] = df_test_final['A1-2_mean'] - df_test_final['Prev_A1-2_mean']
        if 'B10-1' in lag_cols: df_test_final['B10-1_Delta'] = df_test_final['B10-1'] - df_test_final['Prev_B10-1']
        print("Lag/Delta feature creation finished for test set.")
        
        print("Creating Hypothesis-Driven Features for test set...")
        if 'A9-2' in df_test_final.columns and 'A9-5' in df_test_final.columns: df_test_final['Risk_Impulse_x_RuleViolation'] = df_test_final['A9-2'] * df_test_final['A9-5']
        if 'B9-2' in df_test_final.columns and 'B10-2' in df_test_final.columns: df_test_final['Risk_Impulse_x_RiskTaking'] = df_test_final['B9-2'] * df_test_final['B10-2']
        if 'B9-3' in df_test_final.columns and 'B10-6' in df_test_final.columns: df_test_final['Risk_Antisocial_x_Aggressive'] = df_test_final['B9-3'] * df_test_final['B10-6']
        risk_cols = ['A9-1', 'A9-2', 'A9-3', 'A9-4', 'A9-5', 'B9-1', 'B9-2', 'B9-3', 'B9-4', 'B9-5', 'B10-1', 'B10-2', 'B10-3', 'B10-4', 'B10-5', 'B10-6']
        existing_risk_cols = [col for col in risk_cols if col in df_test_final.columns]
        if existing_risk_cols: 
            df_test_final['Risk_Index_Sum'] = df_test_final[existing_risk_cols].fillna(0).sum(axis=1)
            df_test_final['Risk_Index_Mean'] = df_test_final[existing_risk_cols].fillna(0).mean(axis=1)
        print("Hypothesis-Driven feature creation finished for test set.")
        
        print("Creating PrimaryKey-based features for test set...")
        df_test_final['TestDate_Numeric_Agg'] = df_test_final['TestDate_Numeric'].replace(0, np.nan)
        grouped_by_pk = df_test_final.groupby('PrimaryKey')
        pk_features = grouped_by_pk.agg(PK_Test_Count=('Test_id', 'count'), PK_First_TestDate=('TestDate_Numeric_Agg', 'min'), PK_Last_TestDate=('TestDate_Numeric_Agg', 'max'), PK_Mean_Age=('Age', 'mean'), PK_Min_Age=('Age', 'min'), PK_Max_Age=('Age', 'max')).reset_index()
        pk_features['PK_Test_Duration'] = pk_features['PK_Last_TestDate'] - pk_features['PK_First_TestDate']
        pk_features['PK_Age_Range'] = pk_features['PK_Max_Age'] - pk_features['PK_Min_Age']
        
        pk_test_type_counts = df_test_final.groupby(['PrimaryKey', df_test_final['Test'].astype(str)]).size().unstack(fill_value=0)
        pk_test_type_counts.columns = [f'PK_{col}_Test_Count' for col in pk_test_type_counts.columns]
        pk_test_type_counts = pk_test_type_counts.reset_index()
        
        df_test_final = pd.merge(df_test_final, pk_features, on='PrimaryKey', how='left')
        df_test_final = pd.merge(df_test_final, pk_test_type_counts, on='PrimaryKey', how='left')
        df_test_final = df_test_final.drop(columns=['TestDate_Numeric', 'TestDate_Numeric_Agg'])
        print("PrimaryKey features merged for test set.")
        
        df_test_final['Test'] = df_test_final['Test'].astype('category') # LGBM을 위해 category 타입 변환
        print("Merged with test.csv successfully.")
        
    except Exception as e:
        print(f"Error during feature engineering for test set: {e}")
        return

    # --- 예측을 위한 특성 준비 ---
    # (LGBM용: 학습 시 사용된 모든 피처 순서대로 reindex)
    X_test_lgb = df_test_final.reindex(columns=lgb_features, fill_value=np.nan)
    print(f"Final X_test_lgb shape for prediction: {X_test_lgb.shape}")

    # (CatBoost용: 학습 시 사용된 피처만 선택)
    # CatBoost는 피처 이름 기반으로 매핑하므로 순서가 달라도 되지만, 누락되면 안 됨
    final_cat_features = [col for col in cat_features_model if col in df_test_final.columns]
    X_test_cat_df = df_test_final[final_cat_features]
    
    # CatBoost Pool 생성 시 'Test' 컬럼을 범주형으로 지정
    categorical_features_indices = [i for i, col in enumerate(X_test_cat_df.columns) if col == 'Test']
    print(f"Categorical feature index for CatBoost Pool: {categorical_features_indices}")
    test_pool = Pool(data=X_test_cat_df, cat_features=categorical_features_indices)
    print(f"CatBoost Pool created successfully.")

    # --- 3개 모델 앙상블 예측 (최적 가중치 적용) ---
    print("Predicting probabilities (LGBM Classifier - Raw)...")
    preds_lgb_clf_raw = lgb_clf_model.predict(X_test_lgb)
    
    print("Predicting probabilities (CatBoost Classifier - Raw)...")
    preds_cat_clf_raw = cat_clf_model.predict_proba(test_pool)[:, 1]
    
    print("Predicting probabilities (LGBM Regressor - Raw)...") 
    preds_lgb_reg_raw = lgb_reg_model.predict(X_test_lgb) 
    
    print("Applying calibration (IsotonicRegression)...")
    preds_lgb_clf = lgb_clf_calibrator.predict(preds_lgb_clf_raw)
    preds_cat_clf = cat_clf_calibrator.predict(preds_cat_clf_raw)
    preds_lgb_reg = lgb_reg_calibrator.predict(np.clip(preds_lgb_reg_raw, 0, 1))

    # ★ 3개 모델 최적 가중치 (4.3 셀 Optuna 결과) ★
    WEIGHT_LGB_CLF = 0.5474
    WEIGHT_CAT_CLF = 0.3271
    WEIGHT_LGB_REG = 0.1256
    
    print(f"Applying final optimized 3-model weights (LGB_C: {WEIGHT_LGB_CLF:.4f}, Cat_C: {WEIGHT_CAT_CLF:.4f}, LGB_R: {WEIGHT_LGB_REG:.4f})...")
    final_predictions = (WEIGHT_LGB_CLF * preds_lgb_clf +
                           WEIGHT_CAT_CLF * preds_cat_clf +
                           WEIGHT_LGB_REG * preds_lgb_reg)

    final_predictions = np.clip(final_predictions, 0, 1)
    
    # --- 제출 파일 생성 ---
    df_submission = pd.DataFrame({'Test_id': df_test_final['Test_id'], 'Label': final_predictions})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"[+] Submission file saved to {SUBMISSION_PATH}")
    print("[+] Inference script finished.")

if __name__ == "__main__":
    main()
