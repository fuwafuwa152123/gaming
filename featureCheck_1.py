import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def create_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """替遊戲成癮資料集進行特徵工程組合"""
    df_fe = df.copy()

    # 1. 時間佔比與沉浸度
    # 遊戲時間占總螢幕時間比例
    df_fe["gaming_to_screen_ratio"] = df_fe["daily_gaming_hours"] / (
        df_fe["screen_time_total"] + 1e-5
    )

    # 深夜遊戲估算時數
    df_fe["night_gaming_hours"] = (
        df_fe["daily_gaming_hours"] * df_fe["night_gaming_ratio"]
    )

    # 週末遊戲突發指數 (週末平均每日 - 平日每日)
    df_fe["weekend_gaming_spike"] = (
        df_fe["weekend_gaming_hours"] - df_fe["daily_gaming_hours"]
    )

    # 2. 生活失衡指標
    # 遊戲與睡眠失衡比
    df_fe["game_to_sleep_ratio"] = df_fe["daily_gaming_hours"] / (
        df_fe["sleep_hours"] + 0.1
    )

    # 久坐健康風險指數 (遊戲時數 / 運動時數 * BMI)
    df_fe["sedentary_health_risk"] = (
        df_fe["daily_gaming_hours"] / (df_fe["exercise_hours"] + 0.1)
    ) * df_fe["bmi"]

    # 3. 心理與社交負擔指標
    # 心理困擾綜合分數 (負面情緒累加 - 正面情緒)
    df_fe["psychological_distress_score"] = (
        df_fe["loneliness_score"]
        + df_fe["anxiety_score"]
        + df_fe["depression_score"]
        - df_fe["happiness_score"]
    )

    # 生產力受損指數 (缺失生產力 * 遊戲時數)
    df_fe["productivity_loss_index"] = (
        10 - df_fe["work_productivity"]
    ) * df_fe["daily_gaming_hours"]

    # 社交失衡分數 (線上朋友數 / (社交分數 + 0.1))
    df_fe["online_social_imbalance"] = df_fe["online_friends"] / (
        df_fe["social_interaction_score"] + 0.1
    )

    # 4. 遊戲消費強度
    # 每小時微交易消費金額
    df_fe["spending_per_gaming_hour"] = df_fe["microtransactions_spending"] / (
        (df_fe["daily_gaming_hours"] * 30) + 0.1
    )
    #print(df_fe[['daily_gaming_hours','screen_time_total','gaming_to_screen_ratio']])
    #exit()
    # 5. 多重警訊風險指數 (簡單示範：符合極端條件的數量計數)
    high_game = df_fe["daily_gaming_hours"] > 6
    low_sleep = df_fe["sleep_hours"] < 6
    high_lonely = df_fe["loneliness_score"] > 7

    df_fe["risk_flag_count"] = (
        high_game.astype(int) + low_sleep.astype(int) + high_lonely.astype(int)
    )

    # 清理因相除可能產生的 inf 或 nan
    df_fe = df_fe.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_fe


def evaluate_feature_importance(df: pd.DataFrame, target_col: str):
    """計算並印出隨機森林特徵重要性"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    print(f"==================================================")
    print(f" 模型評估結果 (R² Score): {r2:.4f}")
    print(f"==================================================")
    print(" 前 15 個重要特徵排名：")
    print(importances.head(15).to_string())
    print("\n")

    return importances, r2


# ==========================================
# 示範使用方式 (請替換為你的 df 載入邏輯)
# ==========================================
print(__name__)
if __name__ == "__main__":
    # 假設 df 是你原本包含原始欄位的 Pandas DataFrame
    df = pd.read_csv("gaming_part1_100k.csv")
    # 1. 執行特徵工程
    df = df.drop(['gender','headset_usage'], axis=1)
    #print(df)
    #exit()
    df_new = create_feature_engineering(df)

    # 2. 評估特徵工程後的特徵重要性
    importances, r2 = evaluate_feature_importance(
        df_new, target_col="addiction_level"
    )