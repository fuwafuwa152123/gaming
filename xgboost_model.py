import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from math import sqrt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 檔案路徑設定
mypath = r'C:\Users\Administrator\Desktop\專題\pjpy'  # school
#csvFile = mypath + r'\gh_1_1000.csv' 
csvFile = mypath + r'\gaming_part1_100k.csv'           # 資料檔

# 移除無關與類別欄位
'''
dropcolumn = [ 'gender', 'income',
    'academic_performance', 'work_productivity', 'multiplayer_ratio',	
    'toxic_exposure', 'violent_games_ratio', 'mobile_gaming_ratio',
    'night_gaming_ratio', 'weekend_gaming_hours', 'friends_gaming_count',
    'streaming_hours',  'headset_usage',	
    'parental_supervision', 'bmi',		
    'competitive_rank',
    'internet_quality', 'aggression_score', 'happiness_score',
     'relationship_satisfaction', 'caffeine_intake',
    'depression_score',   'weekly_sessions', 'years_gaming']

# 自變數特徵欄位 X (14個)
proValue_cols = ['age','daily_gaming_hours', 'sleep_hours',	'exercise_hours', 'stress_level', 'anxiety_score', 
                 'social_interaction_score', 'online_friends', 'esports_interest', 'microtransactions_spending',	
                 'loneliness_score',	'screen_time_total',	'eye_strain_score',	'back_pain_score']  
'''

dropcolumn = ['age','gender','income','weekly_sessions','years_gaming',
              'caffeine_intake','stress_level','anxiety_score','depression_score','social_interaction_score',
              'relationship_satisfaction','academic_performance','work_productivity','multiplayer_ratio',
              'toxic_exposure','violent_games_ratio','mobile_gaming_ratio','night_gaming_ratio','weekend_gaming_hours',
              'friends_gaming_count','online_friends','streaming_hours','esports_interest','headset_usage',
              'microtransactions_spending','parental_supervision','loneliness_score','aggression_score','happiness_score',
              'bmi','eye_strain_score','back_pain_score','competitive_rank','internet_quality']
proValue_cols = ['daily_gaming_hours', 'sleep_hours', 'exercise_hours', 'screen_time_total'] 

predictCol = 'addiction_level'          # 目標變數 Y

# 使用者輸入的全新資料 (保持原始真實數值)3.68
#UserInput = pd.DataFrame([[51, 24, 5.26, 0.18, 3, 4.06, 7.85, 186, 1, 1746.97, 2.87, 24, 10.71, 4.81]], columns=proValue_cols)
#UserInput = pd.DataFrame([[20, 4, 0, 20]], columns=proValue_cols)
#UserInput = pd.DataFrame([[24, 0, 0, 24]], columns=proValue_cols)
UserInput = pd.DataFrame([[24, 4, 0, 24]], columns=proValue_cols)

##################################################
# Step 2 : 讀取資料
##################################################
data = pd.read_csv(csvFile, header=0)
data = data.drop(dropcolumn, axis=1) 
data = data.astype(np.float32)

##################################################
# Step 3 : X 與 Y 共同正規化 (【核心修改 1】)
##################################################
# 建立 X 的獨立縮放器 (0.2 ~ 0.8)
scaler_X = MinMaxScaler(feature_range=(0.2, 0.8))
data[proValue_cols] = scaler_X.fit_transform(data[proValue_cols])

# 建立 Y 的獨立縮放器 (0.2 ~ 0.8)，符合您 X 與 Y 都要先經過正規化再 fit 的需求
scaler_y = MinMaxScaler(feature_range=(0.2, 0.8))
data[[predictCol]] = scaler_y.fit_transform(data[[predictCol]])

# 提取已經正規化後的 X 與 y
y = data[predictCol]                                                            
data = data.drop([predictCol], axis=1)

##################################################
# Step 4 : 切割資料集
##################################################
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.20, random_state=101)
print('Training:', X_train.shape, '  Testing:', X_test.shape)

# 定義模型
xgbr1 = XGBRegressor(n_estimators=500, learning_rate=0.05,
                    max_depth=3, min_child_weight=1,
                    gamma=0, subsample=0.7,
                    colsample_bytree=1, colsample_bylevel=1.0,
                    colsample_bynode=1.0, reg_alpha=0,
                    reg_lambda=0, scale_pos_weight=1,
                    booster='gbtree', tree_method='auto',
                    n_jobs=None, random_state=42,
                    verbosity=1, objective='reg:squarederror')

'''
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.33, random_state=101)
print('Training:', X_train.shape, '  Testing:', X_test.shape)

# 定義模型
xgbr1 = XGBRegressor(n_estimators=800, learning_rate=0.01,
                    max_depth=3, min_child_weight=2,
                    gamma=0, subsample=1.0,
                    colsample_bytree=1.0, colsample_bylevel=1.0,
                    colsample_bynode=1.0, reg_alpha=0.1,
                    reg_lambda=0.2, scale_pos_weight=1,
                    booster='gbtree', tree_method='auto',
                    n_jobs=None, random_state=42,
                    verbosity=1, objective='reg:squarederror')
'''
##################################################
# Step 5 : 訓練與互動預測百分比 (【核心修改 2】)
##################################################
try:
    # 進行模型 fit 訓練 (此時輸入的題目與答案皆為 0.2 ~ 0.8 的縮放數值)
    print("正在訓練模型...")
    xgbr1.fit(X_train, y_train)
    print("模型訓練完成！")
    
    # 1. 為了防範特徵順序錯亂，先自動重排使用者的輸入，對齊 X_train 的欄位順序
    aligned_user_input = UserInput[X_train.columns]
    
    # 2. 將使用者輸入的資料進行特徵縮放 (正規化後再預測)
    scaled_user_array = scaler_X.transform(aligned_user_input)
    final_user_X = pd.DataFrame(scaled_user_array, columns=X_train.columns)
    
    # 3. 進行預測 (此時得到的預測值會落在 0.2 ~ 0.8 區間)
    predicted_y_scaled = xgbr1.predict(final_user_X)
    
    # 4. 【關鍵】：將縮放後的預測 Y 值進行「反轉換」，還原回真實世界 0 ~ 10 分的尺度
    # 因為 predict 回傳的是 1D array，我們用 reshape(-1, 1) 配合 scaler_y 反轉，再取回單一數值
    raw_score = scaler_y.inverse_transform(predicted_y_scaled.reshape(-1, 1))[0][0]
    
    # 5. 根據原始分數上限（10分）換算為 0% ~ 100% 的百分比，並防呆限幅
    percentage_score = (raw_score / 10.0) * 100
    percentage_score = np.clip(percentage_score, 0.0, 100.0)
    
    # 6. 輸出結果
    print(f"\n🎯 AI 預測與百分比轉換成功！")
    print(f"-> 模型直接預測的縮放值 (0.2~0.8)：{predicted_y_scaled[0]:.4f}")
    print(f"-> 反轉換還原的原始分數 (0~10)  ：{raw_score:.2f} 分")
    print(f"-> 最終換算之成癮傾向百分比    ：{percentage_score:.1f}%")
    
except Exception as e:
    print('錯誤原因:', e)




