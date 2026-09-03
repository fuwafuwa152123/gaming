import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#######################################
# 基本設定 只要改這邊就好               #
#######################################
mypath = r'C:\Users\Administrator\Desktop\專題\pjpy'  #school
#mypath = r'C:\Users\user\PycharmProjects\PythonProject'  #home
df = pd.read_csv(mypath + r"\gh_1_1000.csv")  # 讀取資料
target = "addiction_level"  # Y 是目標欄位 "depression_score"


#######################################
# 關聯性：1. Pearson 相關係數            #
#######################################

print("\n檢查關聯性 1 : 自己取絕對值後, 數值越大越有關聯, 已排序看前面幾個就可以了")
correlation = df.corr(numeric_only=True)[target] # 計算所有 X 與 Y 的 Pearson 相關係數
correlation = correlation.drop(target) # 排除 Y 自己

# 按照「關聯程度絕對值」由大到小排序
result = correlation.reindex(
    correlation.abs().sort_values(ascending=False).index
)
print(result) #daily_gaming_hours(0.897161), screen_time_total(0.653820), depression_score(0.065430), multiplayer_ratio(-0.059245) streaming_hours(-0.057320)

#######################################
# 關聯性：2. 同時使用 Pearson + Spearman #
#######################################

print("\n\n檢查關聯性 2 : 看平均關聯程度, 數值越大越有關聯, 已排序看前面幾個就可以了")
df_numeric = df.select_dtypes(include="number")
pearson    = df_numeric.corr(method="pearson")[target]
spearman   = df_numeric.corr(method="spearman")[target]

result = pd.DataFrame({
    "Pearson": pearson,
    "Spearman": spearman
})

result = result.drop(target) # 排除 Y

# 使用兩者絕對值平均來排序
result["平均關聯程度"] = (
    result["Pearson"].abs() +
    result["Spearman"].abs()
) / 2

result = result.sort_values(
    "平均關聯程度",
    ascending=False
)

print(result) #平均關聯程度: daily_gaming_hours(0.876048), screen_time_total(0.628828), depression_score(0.072693), loneliness_score(0.054959), academic_performance(0.054746)

#######################################
# 關聯性：3. 使用 Random Forest         #
#######################################

print("\n\n檢查關聯性 3 : 數值越大越有關聯, 已排序看前面幾個就可以了")
X = df.drop(columns=[target,"gender"])
y = df[target]

# 分割訓練資料 / 測試資料
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 建立模型
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

# 訓練
model.fit(X_train, y_train)

# 計算每個 X 的重要程度
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

print(importance) # daily_gaming_hours(0.819151), academic_performance(0.008293), screen_time_total(0.007538), depression_score(0.006390), weekend_gaming_hours(0.006370)

####################################
# 重要性： 1. RandomForestRegressor #
####################################

print("\n\n檢查重要性 1 : 數值越大越有關聯, 已排序看前面幾個就可以了")
df = df.select_dtypes(include="number")
if target not in df.columns: #檢查 Y 是否存在
    raise ValueError(f"找不到目標欄位：{target}")

X = df.drop(columns=[target])
y = df[target]

X = X.fillna(X.mean())  # X 的缺失值使用該欄位平均值補上
valid = y.notna() # Y 有缺失值的資料列直接移除
X = X[valid]
y = y[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=" * 50)
print("AI 模型結果")
print("=" * 50)

print(f"R²：{r2:.4f}")
print(f"MSE：{mse:.4f}")


importance = pd.DataFrame({
    "欄位": X.columns,
    "AI重要性": model.feature_importances_
})

importance = importance.sort_values(
    by="AI重要性",
    ascending=False
)

# 重新編號
importance = importance.reset_index(drop=True)
importance["排名"] = importance.index + 1
importance = importance[
    ["排名", "欄位", "AI重要性"]
]

print("\n")
print("=" * 50)
print("每個 X 對 Y 的 AI 重要性")
print("=" * 50)

print(importance.to_string(index=False)) #daily_gaming_hours(0.819151), academic_performance(0.008293), screen_time_total(0.007538), depression_score(0.006390), weekend_gaming_hours(0.006370)

#importance.to_excel(
#    "X_AI重要性結果.xlsx",
#    index=False
#)
#
#print("\n結果已儲存：X_AI重要性結果.xlsx")

