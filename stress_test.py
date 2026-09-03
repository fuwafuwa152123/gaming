import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import time

# =================################================#
# 檔案路徑與基本設定
# =================################================#
mypath = r"C:\Users\Administrator\Desktop\專題\pjpy\專題過程\盲測"  # home

# 讀取三個 part 的檔案
csv_part1 = mypath + r"\gaming_part1_100k.csv"  # 10萬筆：跑迴圈找參數
csv_part2 = mypath + r"\gaming_part2_100k.csv"  # 10萬筆：第一道驗證(盲測)
csv_part3 = mypath + r"\gaming_part3_71807.csv"  # 7萬筆：進階分析(終極盲測/壓力測試)

dropcolumn = ['age','gender','income','weekly_sessions','years_gaming',
              'caffeine_intake','stress_level','anxiety_score','depression_score','social_interaction_score',
              'relationship_satisfaction','academic_performance','work_productivity','multiplayer_ratio',
              'toxic_exposure','violent_games_ratio','mobile_gaming_ratio','night_gaming_ratio','weekend_gaming_hours',
              'friends_gaming_count','online_friends','streaming_hours','esports_interest','headset_usage',
              'microtransactions_spending','parental_supervision','loneliness_score','aggression_score','happiness_score',
              'bmi','eye_strain_score','back_pain_score','competitive_rank','internet_quality']
proValue_cols = ['daily_gaming_hours', 'sleep_hours', 'exercise_hours', 'screen_time_total'] 

'''
# 移除無關與類別欄位
dropcolumn = [
    "gender", "income", "academic_performance", "work_productivity", "multiplayer_ratio",
    "toxic_exposure", "violent_games_ratio", "mobile_gaming_ratio", "night_gaming_ratio",
    "weekend_gaming_hours", "friends_gaming_count", "streaming_hours", "headset_usage",
    "parental_supervision", "bmi", "competitive_rank", "internet_quality", "aggression_score",
    "happiness_score", "relationship_satisfaction", "caffeine_intake", "depression_score",
    "weekly_sessions", "years_gaming"
]

# 自變數特徵欄位 X (14個)
proValue_cols = [
    "age", "daily_gaming_hours", "sleep_hours", "exercise_hours", "stress_level", "anxiety_score",
    "social_interaction_score", "online_friends", "esports_interest", "microtransactions_spending",
    "loneliness_score", "screen_time_total", "eye_strain_score", "back_pain_score"
]
'''
predictCol = "addiction_level"  # 目標變數 Y

# 預先定義固定的縮放器，確保所有 Part 的縮放基準完全統一
scaler_X = MinMaxScaler(feature_range=(0.2, 0.8))
scaler_y = MinMaxScaler(feature_range=(0.2, 0.8))


def process_dataset(filepath, fit_scaler=False):
    """資料讀取與工程的輔助函式"""
    df = pd.read_csv(filepath, header=0)
    # 保留真實的 Y，用於最後反轉比對
    raw_y = df[predictCol].values

    # 特徵工程
    df_processed = df.drop(dropcolumn, axis=1)
    df_processed = df_processed.astype(np.float32)

    # 進行縮放
    if fit_scaler:
        df_processed[proValue_cols] = scaler_X.fit_transform(df_processed[proValue_cols])
        df_processed[[predictCol]] = scaler_y.fit_transform(df_processed[[predictCol]])
    else:
        df_processed[proValue_cols] = scaler_X.transform(df_processed[proValue_cols])
        df_processed[[predictCol]] = scaler_y.transform(df_processed[[predictCol]])

    y_scaled = df_processed[predictCol]
    X_scaled = df_processed.drop([predictCol], axis=1)
    return X_scaled, y_scaled, raw_y


# =================################================#
# STEP 1: 讀取並處理所有資料集
# =================################================#
print("【資料準備】正在讀取與套用正規化...")
# Part1 當作建立 Scaler 的基準
X_part1, y_part1, raw_y_part1 = process_dataset(csv_part1, fit_scaler=True)
X_part2, y_part2, raw_y_part2 = process_dataset(csv_part2, fit_scaler=False)
X_part3, y_part3, raw_y_part3 = process_dataset(csv_part3, fit_scaler=False)

print(f"-> Part1 (參數優化) 結構: {X_part1.shape}")
print(f"-> Part2 (盲測驗證) 結構: {X_part2.shape}")
print(f"-> Part3 (終極分析) 結構: {X_part3.shape}\n")

# =================================================#
# STEP 2: 使用 Part1 進行參數優化與訓練 (模擬你的現有行為)
# =================================================#
# 在 Part1 內部切分訓練與 Early Stopping 驗證集
X_train_p1, X_val_p1, y_train_p1, y_val_p1 = train_test_split(
    X_part1, y_part1, test_size=0.33, random_state=101
)

# 使用你指定的最佳參數進行初始化
xgbr_best = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05,
    max_depth=3, min_child_weight=1,
    gamma=0, subsample=0.7,
    colsample_bytree=1.0, colsample_bylevel=1.0,
    colsample_bynode=1.0, reg_alpha=0,
    reg_lambda=0, scale_pos_weight=1,
    booster='gbtree', tree_method='auto',
    n_jobs=None, random_state=42,
    verbosity=1, objective='reg:squarederror',
    early_stopping_rounds=100  # 固定使用 100 輪
)

print("【階段一】正在使用 Part1 進行模型訓練與 Early Stopping...")
xgbr_best.fit(X_train_p1, y_train_p1, eval_set=[(X_val_p1, y_val_p1)], verbose=False)
print("-> Part1 基礎模型訓練完成。")

# =================================================#
# STEP 3: 使用 Part2 進行模型盲測驗證 (Model Validation)
# =================================================#
print("\n【階段二】正在使用 Part2 (10萬筆) 進行第一道全量盲測驗證...")
pred_y_p2_scaled = xgbr_best.predict(X_part2)
# 反轉回 0~10 的原始分數
pred_y_p2 = scaler_y.inverse_transform(pred_y_p2_scaled.reshape(-1, 1)).flatten()

mae_p2 = mean_absolute_error(raw_y_part2, pred_y_p2)
rmse_p2 = sqrt(mean_squared_error(raw_y_part2, pred_y_p2))
print(f"💥 Part2 盲測結果統計： MAE = {mae_p2:.4f} 分, RMSE = {rmse_p2:.4f} 分")

# =================================================#
# STEP 4: 進階亮點分析 —— 使用 Part3 (7萬筆)
# =================================================#
print("\n==================================================")
print("🚀 啟動 Part3 (7萬筆) 專題終極亮點分析")
print("==================================================")

try:
    # 🌟【亮點一】：建議一：合併 Part1 + Part2，用 20 萬筆訓練全量模型，拿 Part3 盲測
    print("【建議一】正在合併 Part1 + Part2 (共20萬筆) 建立全量終極模型...")
    X_combined = pd.concat([X_part1, X_part2], axis=0)
    y_combined = pd.concat([y_part1, y_part2], axis=0)

    # 由於沒有第三個獨立 part 做 early stopping，全量模型直接使用最佳樹棵數(例如100棵)進行訓練
    xgbr_final = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.01, max_depth=3, min_child_weight=1,
        random_state=42, objective='reg:squarederror'
    )
    xgbr_final.fit(X_combined, y_combined)

    # 對 Part3 進行終極外部盲測
    pred_y_p3_scaled = xgbr_final.predict(X_part3)
    pred_y_p3 = scaler_y.inverse_transform(pred_y_p3_scaled.reshape(-1, 1)).flatten()

    mae_p3 = mean_absolute_error(raw_y_part3, pred_y_p3)
    rmse_p3 = sqrt(mean_squared_error(raw_y_part3, pred_y_p3))
    print(f"🎯【建議一結果】Part3 全量終極盲測成效：")
    print(f"   -> 終極外部盲測 MAE  : {mae_p3:.4f} 分")
    print(f"   -> 終極外部盲測 RMSE : {rmse_p3:.4f} 分 (資料量增大通常會使分數更優)")

    # 🌟【亮點二】：建議二：高風險成癮者壓力測試 (真實分數 >= 7.5)
    print("\n【建議二】正在對 Part3 進行高風險重度成癮者之『壓力測試』...")
    # 建立一個 Part3 的對照 DataFrame
    p3_analysis_df = pd.DataFrame({
        'True_Level': raw_y_part3,
        'Predicted_Level': pred_y_p3
    })

    # 篩選出真實世界中，成癮度大於等於 7.5 分的高危個案
    high_risk_df = p3_analysis_df[p3_analysis_df['True_Level'] >= 7.5]

    if len(high_risk_df) > 0:
        mae_high_risk = mean_absolute_error(high_risk_df['True_Level'], high_risk_df['Predicted_Level'])
        rmse_high_risk = sqrt(mean_squared_error(high_risk_df['True_Level'], high_risk_df['Predicted_Level']))
        print(f"🔥【建議二結果】在 7 萬人中成功篩選出 {len(high_risk_df)} 筆重度成癮個案。")
        print(f"   -> 重度成癮群壓力測試 MAE  : {mae_high_risk:.4f} 分")
        print(f"   -> 重度成癮群壓力測試 RMSE : {rmse_high_risk:.4f} 分")
        print("   (此分數能強烈證明模型在面對『真正需要預警的危險個案』時是否有精準度)")
    else:
        print("⚠ 提示：Part3 中沒有真實分數 >= 7.5 的資料，無法進行壓力測試。")

    # 🌟【亮點三】：建議三：特徵重要性驗證
    print("\n【建議三】正在輸出 XGBoost 模型之『14個行為特徵重要性權重』...")
    # 取得特徵貢獻度
    importances = xgbr_final.feature_importances_
    # 排序特徵
    feature_importance_df = pd.DataFrame({
        '特徵欄位 (Feature)': proValue_cols,
        '重要性權重 (Importance)': importances
    }).sort_values(by='重要性權重 (Importance)', ascending=False)

    print("📈【建議三結果】AI 判定成癮度的關鍵特徵排名如下（可直接放入論文表格）：")
    print(feature_importance_df.to_string(index=False))

    # =================================================#
    # 自動將 Part3 的終極預測報告匯出成 CSV
    # =================================================#
    p3_export_df = pd.read_csv(csv_part3, header=0)[proValue_cols].copy()
    p3_export_df["True_Addiction_Level"] = raw_y_part3
    p3_export_df["AI_Predicted_Level"] = pred_y_p3

    export_path = mypath + r"\part3_final_evaluation_report.csv"
    p3_export_df.to_csv(export_path, index=False, encoding="utf-8-sig")
    print(f"\n💾【報告生成】Part3 的 7 萬筆完整預測對照表已儲存至：")
    print(f"-> {export_path}")

except Exception as e:
    print("進階分析錯誤原因:", e)