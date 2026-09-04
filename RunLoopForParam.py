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
#import warnings; warnings.filterwarnings("ignore")              # 關閉警告訊息,讓輸出乾淨

# pip install catboost
# pip install lightgbm
# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install scikit-learn
# pip install xgboost
# pip install MinMaxScaler

#############################
# Sample 編碼
#############################
#boston = pd.read_csv('boston.csv', header=0)
#boston.head()
#boston['CHAS_original'] = boston['CHAS']  
#boston = pd.get_dummies(boston, columns=['CHAS'], dtype=int)
#boston.head()
#cols_to_norm = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE',
#                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
#                'CHAS_0', 'CHAS_1', 'MEDV']

# 正規化公式: x' = (x - min) / (max - min) * 0.6 + 0.2  → 映射到 [0.2, 0.8]
#boston_min = boston[cols_to_norm].min()
#boston_max = boston[cols_to_norm].max()
#boston[cols_to_norm] = (boston[cols_to_norm] - boston_min) / (boston_max - boston_min) * 0.6 + 0.2
#boston.head()
#y = boston['MEDV']                        # 目標變數：房價中位數
#data = boston.drop(['MEDV','CHAS_original'], axis=1)      # 其餘欄位為自變數（含 CHAS_0 / CHAS_1）
#data.head()
#############################
#daily_gaming_hours
#screen_time_total
#exercise_hours
#sleep_hours
#age
#social_interaction_score
#stress_level
#online_friends
#eye_strain_score
#esports_interest
#back_pain_score
#anxiety_score
#loneliness_score
#microtransactions_spending

'''
3.1 人口統計特徵
age— 參與者的年齡
gender— 性別類別
income— 月收入估算
3.2 遊戲行為特徵
daily_gaming_hours— 平均每日遊戲時間
weekly_sessions— 每週遊戲次數
years_gaming— 總遊戲經驗年資
weekend_gaming_hours— 週末遊戲時長
multiplayer_ratio— 多人遊戲比例
violent_games_ratio— 暴力遊戲的比例
mobile_gaming_ratio— 行動遊戲份額
night_gaming_ratio— 夜間遊戲比例
competitive_rank— 競技技能排名
esports_interest— 對參與電競的興趣
streaming_hours— 每週遊戲直播時長
microtransactions_spending— 每月遊戲內消費
headset_usage— 是否使用耳機
3.3 心理健康特徵
stress_level—知覺壓力評分
anxiety_score—焦慮評估得分
depression_score憂鬱評分
addiction_level— 遊戲成癮指標
loneliness_score— 孤獨指數
aggression_score— 攻擊傾向分數
happiness_score— 整體幸福感
3.4 社會環境特徵
social_interaction_score— 社會活動指數
relationship_satisfaction— 關係品質評分
friends_gaming_count— 遊戲好友數量
online_friends— 線上社交聯繫
toxic_exposure— 接觸有毒社區
parental_supervision— 家長監控評分
3.5 生活方式特色
sleep_hours— 平均睡眠時間
exercise_hours— 每週運動時間
caffeine_intake— 每日咖啡因攝取量
screen_time_total— 每日總螢幕時間
internet_quality— 網路連線品質
3.6 身體健康指標
bmi— 體重指數
eye_strain_score— 眼睛疲勞程度
back_pain_score— 背痛嚴重程度
3.7 生產力/績效指標
academic_performance— 學業成績
work_productivity— 工作場所生產力評分
'''

##################################################
#Step 1 : 要改的地方                               #
##################################################
CallModel = (1,) #1:XGB, 2:DNN, 3:SVM, 4:KNN, 5:GBDT, 6:LightGBM, 7:CatBoost

(step01,step05,step001,step005)  = (0.1,0.5,0.01,0.05)

'''
n_estimators 推薦範圍： [100, 500, 1000, 1500, 2000]
step (100,400,10)(410,700,10)(710,1000,10)(1010,1300,10)(1310,1600,10)

learning_rate 推薦範圍： [0.01, 0.03, 0.05, 0.1, 0.3]
step(0.01,0.3,0.01)

gamma 推薦範圍： [0, 0.1, 0.2, 0.4, 0.8]
step(0,1,0.1)

reg_alpha (L1) 推薦範圍： [0, 0.01, 0.1, 1, 10]
pop(0, 0.01, 0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10)

reg_lambda (L2) 推薦範圍： [0.5, 1, 2, 5, 10]
step(0.5,10,0.5)

max_depth 推薦範圍： [3, 4, 5, 6, 7, 8]
step(3,8,1)

min_child_weight 推薦範圍： [1, 3, 5, 7, 10]
step(1,10,1)
'''

#(n_estimators_1, n_estimators_2)=(100, 400)                 # A 樹數。核心預設 100 /300/500/1000
#(learning_rate_1, learning_rate_2)=(0.01,0.31)              # A 收縮率。核心預設 0.3 /0.1/0.05（與樹數反向搭配）
#(max_depth_1,max_depth_2)=(3,9)                             # B 最大深度。核心預設 6 /3~10
#(min_child_weight_1,min_child_weight_2)=(1,11)              # B 葉節點 Hessian 總和下限。1預設 /3/5/10（越大越保守）
#(gamma_1,gamma_2)=(0, 1.1)                                  # B 分裂最小損失下降。0預設 /0.1/0.2/0.5（相當於剪枝）
#reg_alpha_1=(0, 0.01, 0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10)                        # D L1 正則化。0預設 /0.01/0.1/1（sklearn GBDT 沒有）
#(reg_lambda_1,reg_lambda_2)=(0.5, 10.5)                          # D L2 正則化。1預設 /5/10 (0.5,10.5)

# 參數設定
# XGB
estimator1 = 3000
#(n_estimators_1, n_estimators_2)=(200, 400)                 # A 樹數。核心預設 100 /300/500/1000
#(n_estimators_1, n_estimators_2)=(410, 700)                 # A 樹數。核心預設 100 /300/500/1000
#(n_estimators_1, n_estimators_2)=(710, 1000)                 # A 樹數。核心預設 100 /300/500/1000
#(n_estimators_1, n_estimators_2)=(1010, 1300)                 # A 樹數。核心預設 100 /300/500/1000
#(n_estimators_1, n_estimators_2)=(1310, 1600)                 # A 樹數。核心預設 100 /300/500/1000
#(learning_rate_1, learning_rate_2)=(0.03,0.05)              # A 收縮率。核心預設 0.3 /0.1/0.05（與樹數反向搭配）
learn_rate = 0.01
(max_depth_1,max_depth_2)=(3,9)                             # B 最大深度。核心預設 6 /3~10
(min_child_weight_1,min_child_weight_2)=(1,5)              # B 葉節點 Hessian 總和下限。1預設 /3/5/10（越大越保守）
(gamma_1,gamma_2)=(0, 0.6)                                  # B 分裂最小損失下降。0預設 /0.1/0.2/0.5（相當於剪枝）
#(reg_alpha_1,reg_alpha_2)=(0.0,0.03)                        # D L1 正則化。0預設 /0.01/0.1/1（sklearn GBDT 沒有）
reg_alpha_1=(0, 0.01, 0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09)                        # D L1 正則化。0預設 /0.01/0.1/1（sklearn GBDT 沒有）
(reg_lambda_1,reg_lambda_2)=(0.5, 2.5)                          # D L2 正則化。1預設 /5/10 (0.5,10.5)

# DNN
# SVM
# KNN
KNN_n_neighbors = (1,5,9,13,17,21)                # 01 鄰居數 K。5預設 /1/3/7/9/11/15/21（取奇數避免平手）
KNN_leaf_size = (10,30,50,80,100)               # 04 葉節點門檻，僅樹狀索引。30預設 /10/20/50/100
KNN_weights = ('uniform',)                                     # 02 投票權重。'uniform'預設 /'distance'(距離倒數加權)
KNN_algorithm = ('auto',)                                      # 03 搜尋索引。'auto'預設 /'ball_tree'/'kd_tree'/'brute'
KNN_p = (1,2)                                                  # 05 Minkowski 次方，僅 metric='minkowski'。2預設(歐氏)/1(曼哈頓)

# GBDT
# LightGBM
# CatBoost

#### addiction_level #daily_gaming_hours、screen_time_total、weekly_sessions、work_productivity 以及 academic_performance
mypath = r'C:\Users\Administrator\Desktop\專題\pjpy'  #school
#mypath = r'C:\Users\user\PycharmProjects\PythonProject'  #home

#csvFile = mypath + r'\gh_1_38083.csv'               #資料檔
#csvFile = mypath + r'\gh_1_1000.csv'               #資料檔
csvFile = mypath + r'\gaming_part1_100k.csv'               #資料檔

oXGBcsv  = mypath + r'\XGB_predict_1_1.csv'
oDNNcsv  = mypath + r'\DNN_predict.csv'
oSVMcsv  = mypath + r'\SVM_predict.csv'
oKNNcsv  = mypath + r'\KNN_predict.csv'
oGBDTcsv = mypath + r'\GBDT_predict.csv'
oLGBMcsv = mypath + r'\LGBM_predict.csv'
oCatBcsv = mypath + r'\CatB_predict.csv'

#daily_gaming_hours
#screen_time_total
#exercise_hours
#sleep_hours
#age
#social_interaction_score
#stress_level
#online_friends
#eye_strain_score
#esports_interest
#back_pain_score
#anxiety_score
#loneliness_score
#microtransactions_spending

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
proValue_cols = ['daily_gaming_hours', 'screen_time_total', 'exercise_hours', 'sleep_hours', 'age',
                 'social_interaction_score', 'stress_level', 'online_friends', 'eye_strain_score',
                 'esports_interest', 'back_pain_score', 'anxiety_score', 'loneliness_score',
                 'microtransactions_spending', 'addiction_level']                 #因為此程式無處理文字欄位，避免程式當掉先將gender欄位刪掉

predictCol = 'addiction_level'          #要預測的欄位即Y
dummies = []

##################################################
#Step 2                                          #
##################################################
data = pd.read_csv(csvFile, header=0)
data = data.drop(dropcolumn, axis=1) #因為此程式無處理文字欄位，避免程式當掉先將gender欄位刪掉
data = data.astype(np.float32)
data.head()

##################################################
#Step 3                                          #
##################################################
cols_to_norm = proValue_cols
scaler = MinMaxScaler(feature_range=(0.2, 0.8))
data[cols_to_norm] = scaler.fit_transform(data[cols_to_norm])
y    = data[predictCol]                                                            
data = data.drop([predictCol], axis=1)

##################################################
#Step 4                                          #
##################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.33, random_state=101)
print('Training:', X_train.shape, '  Testing:', X_test.shape)

##################################################
#Step 5                                          #
##################################################
def evaluate_reg(model, X_train, y_train, X_test, y_test, outFile, threshold=0.01):
    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test,  y_test)

    # Train metrics
    y_tr_pred = model.predict(X_train)
    tr_mse  = mean_squared_error(y_train, y_tr_pred)
    tr_rmse = np.sqrt(tr_mse)
    tr_mae  = mean_absolute_error(y_train, y_tr_pred)
    tr_r2   = 1 - tr_mse / np.var(y_train)

    # Test metrics
    y_te_pred = model.predict(X_test)
    te_mse  = mean_squared_error(y_test, y_te_pred)
    te_rmse = np.sqrt(te_mse)
    te_mae  = mean_absolute_error(y_test, y_te_pred)
    te_r2   = 1 - te_mse / np.var(y_test)

    # 通過門檻百分比 (誤差 < threshold)
    ytest = y_test.values
    err = np.abs((ytest - y_te_pred) / ytest)
    pass_count = (err < threshold).sum()


    # 💡 新增：取得最佳迭代次數 (如果模型支援的話，如 XGBoost)
    if hasattr(model, 'best_iteration'):
        best_iter = model.best_iteration
    else:
        best_iter = "N/A" # 傳統模型不支援則填 N/A

    with open(outFile, "a") as f:
        # 末端新增了 ,{best_iter}
        f.write(f"{best_iter:>8.4f},{tr_mse:>8.4f},{te_mse:>8.4f},{tr_rmse:>8.4f},{te_rmse:>8.4f},{tr_mae:>8.4f},{te_mae:>8.4f},{tr_r2:>8.4f},{te_r2:>8.4f},{threshold*100:.0f},{pass_count}/{len(ytest)}, {pass_count/len(ytest)*100:.1f},{train_score:>8.4f},{test_score:>8.4f}")
        f.write('\n')



 #   with open(outFile, "a") as f:
 #       f.write(f"{tr_mse:>8.4f},{te_mse:>8.4f},{tr_rmse:>8.4f},{te_rmse:>8.4f},{tr_mae:>8.4f},{te_mae:>8.4f},{tr_r2:>8.4f},{te_r2:>8.4f},{threshold*100:.0f},{pass_count}/{len(ytest)}, {pass_count/len(ytest)*100:.1f},{train_score:>8.4f},{test_score:>8.4f}")
 #       f.write('\n')
    return te_r2

##################################################
#Step 6 : Loop parameter for each model          #
##################################################
def SevenModel(SModel, oFile):
    try:
        #SModel.fit(X_train, y_train)
        SModel.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        evaluate_reg(SModel, X_train, y_train, X_test, y_test, oFile)
    except Exception as e:
        # print('Error:', estimator1, md3)
        print('錯誤原因:', e)
        with open(oFile, "a") as f:
            f.write('\n')

##################################################
#Step 7 : Loop parameter for each model          #
##################################################
def Call_XGB() :
    with open(oXGBcsv, "w") as f:
    #    f.write(f"{"estimator1":>8.4f}, {"md3":>8.4f},{"tr_mse":>8.4f},{"te_mse":>8.4f},{"tr_rmse":>8.4f},{"te_rmse":>8.4f},{"tr_mae":>8.4f},{"te_mae":>8.4f},{"tr_r2":>8.4f},{"te_r2":>8.4f}")
        f.write("{:<12}, {:<12}, {:<12}, {:<12}, {:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12}".format("n_estimators",'learning_rate',"max_depth",'min_child_weight','gamma','subsample','colsample_bytree','reg_alpha','reg_lambda',"best_iter","tr_mse","te_mse","tr_rmse","te_rmse","tr_mae","te_mae","tr_r2","te_r2","通過%","誤差門檻","誤差門檻%",'Train %','Test %'))
        f.write('\n')

    #要新增, 注意step01, step001, step05, step005
    #for estimator1 in range(n_estimators_1,n_estimators_2,100):
    #for learn_rate in np.arange(learning_rate_1, learning_rate_2, step001):
    for md3 in range(max_depth_1,max_depth_2):
        for min_child in range(min_child_weight_1,min_child_weight_2):
            for gamm in np.arange(gamma_1, gamma_2, step01):
                #for sbsample in np.arange(subsample_1, subsample_2, step01):
                    #for colsample_bt in np.arange(colsample_bytree_1, colsample_bytree_2, step01):
                        for reg_al in reg_alpha_1:
                            for reg_la in np.arange(reg_lambda_1,reg_lambda_2, step05):
                                xgbr1 = XGBRegressor(n_estimators=estimator1, learning_rate=learn_rate,
                                                    max_depth=md3, min_child_weight=min_child,
                                                    gamma=gamm, subsample=1.0,
                                                    colsample_bytree=1.0, colsample_bylevel=1.0,
                                                    colsample_bynode=1.0, reg_alpha=reg_al,
                                                    reg_lambda=reg_la, scale_pos_weight=1,
                                                    booster='gbtree', tree_method='auto',
                                                    n_jobs=None, random_state=42,
                                                    verbosity=1, objective='reg:squarederror',early_stopping_rounds = 100)
                                with open(oXGBcsv, "a") as f:
                                    # 要新增, 後面要逗號
                                    f.write(f"{estimator1:>8.4f}, {learn_rate:>8.4f},{md3:>8.4f}, {min_child:>8.4f},{gamm:>8.4f},{1.0:>8.4f},{1.0:>8.4f},{round(reg_al,2):>8.4f},{round(reg_la,2):>8.4f},")
                                SevenModel(xgbr1, oXGBcsv)

def Call_DNN() :
    with open(oDNNcsv, "w") as f:
        f.write("{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12}".format("tr_mse","te_mse","tr_rmse","te_rmse","tr_mae","te_mae","tr_r2","te_r2","通過%","誤差門檻","誤差門檻%",'Train %','Test %'))
        f.write('\n')
    xgbr1 = MLPRegressor(hidden_layer_sizes=(16, 8), alpha=0.0001, max_iter=800, activation='relu', solver='adam', random_state=42)
    #with open(oDNNcsv, "a") as f:
    #    f.write(f"{estimator1:>8.4f}, {learn_rate:>8.4f},{md3:>8.4f}, {min_child:>8.4f},{gamm:>8.4f},{1.0:>8.4f},{1.0:>8.4f},{round(reg_al,2):>8.4f},{round(reg_la,2):>8.4f},")

    SevenModel(xgbr1, oDNNcsv)

def Call_SVM():
    with open(oSVMcsv, "w") as f:
        f.write("{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12}".format("tr_mse","te_mse","tr_rmse","te_rmse","tr_mae","te_mae","tr_r2","te_r2","通過%","誤差門檻","誤差門檻%",'Train %','Test %'))
        f.write('\n')
    xgbr1 = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
    #with open(oSVMcsv, "a") as f:
    #    f.write(f"{estimator1:>8.4f}, {learn_rate:>8.4f},{md3:>8.4f}, {min_child:>8.4f},{gamm:>8.4f},{1.0:>8.4f},{1.0:>8.4f},{round(reg_al,2):>8.4f},{round(reg_la,2):>8.4f},")
    SevenModel(xgbr1, oSVMcsv)

def Call_KNN():
    with open(oKNNcsv, "w") as f:
        f.write("{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12}".format("n_neighbors","leaf_size","weights","algorithm","p","tr_mse","te_mse","tr_rmse","te_rmse","tr_mae","te_mae","tr_r2","te_r2","通過%","誤差門檻","誤差門檻%",'Train %','Test %'))
#        f.write("{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12}".format"tr_mse","te_mse","tr_rmse","te_rmse","tr_mae","te_mae","tr_r2","te_r2","通過%","誤差門檻","誤差門檻%",'Train %','Test %'))
        f.write('\n')
    for KN_n_neighbors in KNN_n_neighbors:
        for KN_leaf_size in KNN_leaf_size:
            for KN_weights in KNN_weights:
                for KN_algorithm in KNN_algorithm:
                    for KN_p in KNN_p:
                        xgbr1 = KNeighborsRegressor(n_neighbors=KN_n_neighbors, leaf_size=KN_leaf_size, weights=KN_weights, algorithm=KN_algorithm, p=KN_p)
                        with open(oKNNcsv, "a") as f:
                            f.write(f"{KN_n_neighbors:>8.4f}, {KN_leaf_size:>8.4f},{KN_weights:>8s}, {KN_algorithm:>8s},{KN_p:>8.4f},")
                        SevenModel(xgbr1, oKNNcsv)

def Call_GBDT():
    with open(oGBDTcsv, "w") as f:
        f.write("{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12}".format("tr_mse","te_mse","tr_rmse","te_rmse","tr_mae","te_mae","tr_r2","te_r2","通過%","誤差門檻","誤差門檻%",'Train %','Test %'))
        f.write('\n')
    xgbr1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1.0, loss='squared_error', random_state=42)
    #with open(oGBDTcsv, "a") as f:
    #    f.write(f"{estimator1:>8.4f}, {learn_rate:>8.4f},{md3:>8.4f}, {min_child:>8.4f},{gamm:>8.4f},{1.0:>8.4f},{1.0:>8.4f},{round(reg_al,2):>8.4f},{round(reg_la,2):>8.4f},")
    SevenModel(xgbr1, oGBDTcsv)

def Call_LightGBM():
    with open(oLGBMcsv, "w") as f:
        f.write("{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12}".format("tr_mse","te_mse","tr_rmse","te_rmse","tr_mae","te_mae","tr_r2","te_r2","通過%","誤差門檻","誤差門檻%",'Train %','Test %'))
        f.write('\n')
    xgbr1 = LGBMRegressor(n_estimators=200, num_leaves=31, max_depth=-1, learning_rate=0.1, n_jobs=1, verbose=-1, boosting_type='gbdt', random_state=42)
    #with open(oLGBMcsv, "a") as f:
    #    f.write(f"{estimator1:>8.4f}, {learn_rate:>8.4f},{md3:>8.4f}, {min_child:>8.4f},{gamm:>8.4f},{1.0:>8.4f},{1.0:>8.4f},{round(reg_al,2):>8.4f},{round(reg_la,2):>8.4f},")
    SevenModel(xgbr1, oLGBMcsv)

def Call_CatBoost():
    with open(oCatBcsv, "w") as f:
        f.write("{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12}".format("tr_mse","te_mse","tr_rmse","te_rmse","tr_mae","te_mae","tr_r2","te_r2","通過%","誤差門檻","誤差門檻%",'Train %','Test %'))
        f.write('\n')
    xgbr1 = CatBoostRegressor(iterations=200, depth=4, l2_leaf_reg=3.0, learning_rate=None, random_seed=42, verbose=0, allow_writing_files=False, thread_count=1, loss_function='RMSE')
    #with open(oCatBcsv, "a") as f:
    #    f.write(f"{estimator1:>8.4f}, {learn_rate:>8.4f},{md3:>8.4f}, {min_child:>8.4f},{gamm:>8.4f},{1.0:>8.4f},{1.0:>8.4f},{round(reg_al,2):>8.4f},{round(reg_la,2):>8.4f},")
    SevenModel(xgbr1, oCatBcsv)

##################################################
# Step 8 : Execute Model                         #
##################################################
print('Trainning Data : ', csvFile)
for i in CallModel:
    if i == 1 :
        print('Excuting XGB ...')        
        Call_XGB()
    elif i == 2 :
        print('Excuting DNN ...')
        Call_DNN()
    elif i == 3:
        print('Excuting SVM ...')
        Call_SVM()
    elif i == 4:
        print('Excuting KNN ...')
        Call_KNN()
    elif i == 5:
        print('Excuting GBDT ...')
        Call_GBDT()
    elif i == 6:
        print('Excuting LightGBM ...')
        Call_LightGBM()
    elif i == 7:
        print('Excuting CatBoost ...')
        Call_CatBoost()
    else:
        print('Excuting XGB ...')        
        Call_XGB()        


