import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def create_advanced_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """建構高階互動效應特徵 (High-Order Interaction Features)"""
    df_fe = df.copy()

    # --- 1. 高階多重乘積特徵 (Multi-way Products) ---
    # 身體負擔總和 = 遊戲時數 * (眼睛酸痛 + 背痛)
    df_fe["body_strain_index"] = df_fe["daily_gaming_hours"] * (
        df_fe.get("eye_strain_score", 0) + df_fe.get("back_pain_score", 0)
    )

    # 金錢與時間雙重沉浸 = 遊戲時數 * 微交易金額
    df_fe["spending_gaming_impact"] = (
        df_fe["daily_gaming_hours"]
        * df_fe.get("microtransactions_spending", 0)
    )

    # 孤立暴躁指數 = 孤獨感 * 攻擊性
    df_fe["isolated_aggression"] = df_fe.get(
        "loneliness_score", 0
    ) * df_fe.get("aggression_score", 0)

    # 心理健康總崩潰分數
    df_fe["total_mental_stress"] = (
        df_fe.get("anxiety_score", 0)
        + df_fe.get("depression_score", 0)
        + df_fe.get("aggression_score", 0)
    ) * df_fe.get("loneliness_score", 0)

    # --- 2. 比率與相除特徵 (Non-linear Ratios) ---
    # 課業毀滅比 = 遊戲時數 / 學業表現
    df_fe["academic_damage_ratio"] = (df_fe["daily_gaming_hours"] + 0.1) / (
        df_fe.get("academic_performance", 1) + 0.1
    )

    # 咖啡因與睡眠失衡衝擊
    df_fe["caffeine_sleep_disruption"] = (
        (df_fe.get("caffeine_intake", 0) + 0.1)
        # 加上 0.1 避免除以零
        / (df_fe.get("sleep_hours", 1) + 0.1)
    ) * df_fe["daily_gaming_hours"]

    # 遊戲內容暴戾沉浸度
    df_fe["violent_immersion"] = (
        df_fe["daily_gaming_hours"]
        * df_fe.get("violent_games_ratio", 0)
        * (10 - df_fe.get("parental_supervision", 0))
    )

    # --- 3. 次方與門檻效應 (Polynomial Features) ---
    # 遊戲時間非線性次方 (捕捉超過一定時數後的指數型飆升)
    df_fe["gaming_hours_sq"] = df_fe["daily_gaming_hours"] ** 2

    # 極端高風險族群開關 (符合越多項分數越高)
    c1 = df_fe["daily_gaming_hours"] > 6
    c2 = df_fe.get("sleep_hours", 8) < 6
    c3 = df_fe.get("loneliness_score", 0) > 6
    c4 = df_fe.get("aggression_score", 0) > 6
    df_fe["extreme_risk_combo"] = (
        c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int)
    )

    # 清理非數值或無效值
    df_fe = df_fe.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_fe


def evaluate_model(df: pd.DataFrame, target_col: str = "addiction_level"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    print("==================================================")
    print(f" 模型評估結果 (R² Score): {r2:.4f}")
    print("==================================================")
    print(" 前 20 個重要特徵排名：")
    print(importances.head(20).to_string())

    return importances, r2


# ==========================================
# 執行測試 (請將 df 替換為你的原始資料集)
# ==========================================
if __name__ == "__main__":
    df = pd.read_csv("gaming_part1_100k.csv")
    df = df.drop(['gender', 'headset_usage'], axis=1)
    # 執行高階特徵工程
    df_advanced = create_advanced_interactions(df)

    # 評估模型
    importances, r2 = evaluate_model(
        df_advanced, target_col="addiction_level"
    )