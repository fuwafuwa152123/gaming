import pandas as pd

# 1. 匯入資料
df = pd.read_csv("gaming_mental_health_10M_40features.csv")

# 2. 先進行隨機抽樣 25 萬筆
#df = df.sample(frac=0.25, random_state=42)

# 3. 定義要刪除的資料條件 (drop_conditions)

drop_conditions = (
    # 條件 1: 刪除未成年高收入
    ((df["age"] < 18) & (df["income"] > 15000))
    # 條件 2: 刪除成年人家長高監督
    | ((df["age"] >= 18) & (df["parental_supervision"] > 3))
    # 條件 3: 不合理時間分配
    | (df["screen_time_total"] > 20)  # 螢幕總時間 > 20小時
    | (df["daily_gaming_hours"] > 20)  # 每日遊戲時間 > 20小時
    | (df["exercise_hours"] > 20)  # 運動時間 > 20小時
    | (
        (
            df["daily_gaming_hours"]
            + df["sleep_hours"]
            + df["exercise_hours"]
        )
        > 24
    )  # 遊戲+睡眠+運動 > 24小時
    | (
        (
            df["screen_time_total"]
            + df["sleep_hours"]
            + df["exercise_hours"]
        )
        > 24
    )  # 螢幕+睡眠+運動 > 24小時
    | (df["weekend_gaming_hours"] > 40)  # 週末遊戲時間 > 40小時
    # 條件 5 : 刪除國小前開始打電動 (即開始打遊戲年齡 < 7 歲)
    | ((df["age"] - df["years_gaming"]) < 7)
    # 條件 6: 刪除遊戲年資大於年齡
    | (df["years_gaming"] > df["age"])
    # 條件 7: 刪除 BMI 離群值 (< 15 或 > 40)
    | (df["bmi"] < 15)
    | (df["bmi"] > 40)
)

# 4. 刪除符合上述條件的資料 (保留反向資料)
df_clean = df[~drop_conditions].copy()

# 5. 匯出為新的 CSV 檔案
df_clean.to_csv("gaming_cleaned.csv", index=False, encoding="utf-8-sig")

# 6. 印出資料刪除前後對比數據
#print(f"抽樣後原始筆數: {len(df)} 筆")
print(f"共剔除: {drop_conditions.sum()} 筆不合理資料")
print(f"最終清洗完成剩餘筆數: {len(df_clean)} 筆")
