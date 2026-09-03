import pandas as pd

# 讀取 CSV
df = pd.read_csv("gaming_cleaned.csv")

# 拆分
df_1 = df.iloc[:100000]
df_2 = df.iloc[100000:200000]
df_3 = df.iloc[200000:]

# 儲存
df_1.to_csv("gaming_part1_100k.csv", index=False, encoding="utf-8-sig")
df_2.to_csv("gaming_part2_100k.csv", index=False, encoding="utf-8-sig")
df_3.to_csv("gaming_part3_71807.csv", index=False, encoding="utf-8-sig")

# 確認筆數
print("第1份：", len(df_1), "筆")
print("第2份：", len(df_2), "筆")
print("第3份：", len(df_3), "筆")
print("總筆數：", len(df))