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
#import warnings; warnings.filterwarnings("ignore")              # 關閉警告訊息,讓輸出乾淨

# pip install catboost
# pip install lightgbm
# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install scikit-learn
# pip install xgboost

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
#night_gaming_ratio
#online_friends
#internet_quality
#work_productivity
#toxic_exposure
#eye_strain_score
#esports_interest



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
microtransactios_spending— 每月遊戲內消費
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
CallModel = (1,2,3,4,5,6,7) #1:XGB, 2:DNN, 3:SVM, 4:KNN, 5:GBDT, 6:LightGBM, 7:CatBoost
#CallModel = (4,5) #1:XGB, 2:DNN, 3:SVM, 4:KNN, 5:GBDT, 6:LightGBM, 7:CatBoost

(step01,step05,step001,step005)  = (0.1,0.5,0.01,0.05)

# 參數設定
# XGB
(n_estimators_1, n_estimators_2)=(100, 101)                 # A 樹數。核心預設 100 /300/500/1000
(learning_rate_1, learning_rate_2)=(0.16,0.20)              # A 收縮率。核心預設 0.3 /0.1/0.05（與樹數反向搭配）
(max_depth_1,max_depth_2)=(1,2)                             # B 最大深度。核心預設 6 /3~10
(min_child_weight_1,min_child_weight_2)=(8,10)              # B 葉節點 Hessian 總和下限。1預設 /3/5/10（越大越保守）
(gamma_1,gamma_2)=(0, 0.2)                                  # B 分裂最小損失下降。0預設 /0.1/0.2/0.5（相當於剪枝）
#(subsample_1,subsample_2)=(0.7,0.9)                         # C 樣本抽樣比。1.0預設 /0.6~0.9
#(colsample_bytree_1,colsample_bytree_2)=(0.6,0.8)           # C 每棵樹的特徵抽樣比。1.0預設 /0.6~0.9
(reg_alpha_1,reg_alpha_2)=(0.0,0.03)                        # D L1 正則化。0預設 /0.01/0.1/1（sklearn GBDT 沒有）
(reg_lambda_1,reg_lambda_2)=(1, 3)                          # D L2 正則化。1預設 /5/10
###(colsample_bylevel_1,colsample_bylevel_2)=(1.0,1.0)         # C 每一層的特徵抽樣比。1.0預設（三者可疊乘）
###(colsample_bynode_1,colsample_bynode_2)=(1.0,1.0)           # C 每個節點的特徵抽樣比。1.0預設
###(scale_pos_weight_1,scale_pos_weight_2)=(1,1)               # D 類別不平衡權重。1預設 /負樣本數 ÷ 正樣本數
###(verbosity_1,verbosity_2)=(1,4)                             # H 輸出詳細度。1預設 /0(安靜)/2/3
###booster_1=('gbtree','gblinear','dart')                      # H 基學習器。'gbtree'預設 /'gblinear'/'dart'
###tree_method_1=('auto','hist','exact','approx')              # H 建樹演算法。'auto'預設 /'hist'(快)/'exact'/'approx'
###n_jobs_1=None                                               # H 平行核心數。None預設 /-1
###random_state=42                                             # I 亂數種子。0預設 /1/42

# DNN
#DNN_hidden = [(100,),(64,32),(50,),(128,64,32),(50,50,50)]         # 01 隱藏層結構。(100,)預設 /(64,32)/(50,)/(128,64,32)/(50,50,50)
#DNN_activation = ('relu', 'tanh', 'logistic','identity')           # 02 隱藏層激勵函數。'relu'預設 /'tanh'/'logistic'/'identity'
#DNN_alpha = (0,0.1,0.01,0.001,0.0001, 0.00001)                     # 04 L2 正則化強度。1e-4預設 /0/1e-5/1e-3/1e-2/0.1（對數尺度掃描）
#DNN_learning_rate_init = (0.1,0.01,0.001,0.0001)                   # 07 初始學習率，sgd/adam。0.001預設 /0.0001/0.01/0.1
#DNN_max_iter = (200, 500, 800, 1000, 1200, 1500, 1800, 2000)       # 09 迭代(epoch)上限。200預設 /500/1000~2000；先調到不再警告

# SVM
#SVM_kernel = ('linear', 'rbf', 'poly')                         # 01 核函數。'rbf'預設 /'linear'/'poly'/'sigmoid'/'precomputed'
#SVM_C = (0.01, 0.1, 1.0, 10, 100)                              # 06 懲罰係數。1.0預設 /0.01/0.1/10/100（對數尺度掃描）
#SVM_gamma = ('scale',)                                          # 03 核係數，rbf/poly/sigmoid。'scale'預設 /'auto'/0.001~10
#SVM_epsilon = (0.01, 0.1, 1.0, 10, 100)                        # 07 ε 管道寬度，SVR 專屬。0.1預設 /0.01(窄)/0.5/1.0(寬)

# KNN
KNN_n_neighbors = (1,3,5,7,9,11,13,15,17,19,21)                # 01 鄰居數 K。5預設 /1/3/7/9/11/15/21（取奇數避免平手）
KNN_leaf_size = (10,20,30,40,50,60,70,80,90,100)               # 04 葉節點門檻，僅樹狀索引。30預設 /10/20/50/100
KNN_weights = ('uniform',)                                     # 02 投票權重。'uniform'預設 /'distance'(距離倒數加權)
KNN_algorithm = ('auto',)                                      # 03 搜尋索引。'auto'預設 /'ball_tree'/'kd_tree'/'brute'
KNN_p = (1,2)                                                  # 05 Minkowski 次方，僅 metric='minkowski'。2預設(歐氏)/1(曼哈頓)

# GBDT
#GBDT_learning_rate = (0.01, 0.05, 0.1, 0.3, 0.5)               # A02 收縮率。0.1預設 /0.3/0.05/0.01（與 n_estimators 反向搭配）
#GBDT_n_estimators = (100,300,500,1000)                         # A01 樹的數量。100預設 /300/500/1000
#GBDT_max_depth = (1,2,3,4,5,6,7,10,20,30,50,100,200)           # B 樹參數：最大深度。3預設（Boosting 用淺樹 + 多棵）/5/8
#GBDT_subsample = (0.1, 0.5, 0.8, 1.0)                          # B06 樣本子採樣比。1.0預設 /0.8/0.5（< 1 即 Stochastic GB）
#GBDT_loss = ('squared_error',)                                 # C08 損失函數。'squared_error'預設 /'absolute_error'/'huber'/'quantile'
###random_state=42

# LightGBM
#LGBM_learning_rate = (0.01, 0.05, 0.1, 0.3, 0.5)               # A 收縮率。0.1預設 /0.05/0.01
#LGBM_n_estimators = (100,300,500,1000)                         # A 樹的數量。100預設 /300/500/1000
#LGBM_num_leaves = (15, 21, 26, 31, 63, 127)                    # B 葉子數 — LightGBM 最重要的參數。31預設 /15/63/127
#LGBM_max_depth = (-1,5,8)                                      # B 最大深度。-1預設(不限制，靠 num_leaves 控制)/5/8
#LGBM_min_child_samples = (20,50,100)                           # B 葉節點最少樣本。20預設 /50/100（小資料要調大）
### n_jobs = 1                                                  # E 平行核心數。None預設 /-1
### verbose = -1                                                  
### boosting_type = 'gbdt'                                      # E 提升類型。'gbdt'預設 /'dart'/'goss'/'rf'
### random_state = 42                                           # E 亂數種子。None預設 /1/0/42

# CatBoost
#CatB_learning_rate = (0.01, 0.05, 0.1, 0.3, 0.5)               # A 收縮率。None預設(依 iterations 自動推算)/0.03/0.1
#CatB_iterations = (100,300,500,1000)                           # A 樹的數量。1000預設 /100/500
#CatB_depth = (4,5,6,7,8,9,10)                                  # B 對稱樹深度。6預設 /4~10（同層用相同分裂條件）
#CatB_l2_leaf_reg = (1,3,10,20,30)                              # D L2 正則化。3.0預設 /1/10/30（CatBoost 沒有 L1）
#CatB_random_strength = (0,1,2)                                 # D 分裂評分的隨機擾動。1.0預設 /0/2（抗過擬合）
### random_seed=42                                              # G 亂數種子。None預設 /1/0/42
### verbose=0                                                   # G 訓練輸出。CatBoost 預設會逐輪洗版，建議 0 或 100
### allow_writing_files=False
### thread_count=1                                              # G 平行執行緒數。None預設(全部)/4
### loss_function='RMSE'                                        # F 損失函數。'RMSE'預設 /'MAE'/'Quantile'/'Poisson'

#### addiction_level #daily_gaming_hours、screen_time_total、weekly_sessions、work_productivity 以及 academic_performance
mypath = r'C:\Users\Administrator\Desktop\專題\pjpy'  #school
#mypath = r'C:\Users\user\PycharmProjects\PythonProject\pjpy'  #home

#csvFile = mypath + r'\gh_1_38083.csv'               #資料檔
#csvFile = mypath + r'\gh_1_1000.csv'               #資料檔
csvFile = mypath + r'\gaming_part1_100k.csv'               #資料檔

oXGBcsv  = mypath + r'\XGB_predict.csv'
oDNNcsv  = mypath + r'\DNN_predict.csv'
oSVMcsv  = mypath + r'\SVM_predict.csv'
oKNNcsv  = mypath + r'\KNN_predict.csv'
oGBDTcsv = mypath + r'\GBDT_predict.csv'
oLGBMcsv = mypath + r'\LGBM_predict.csv'
oCatBcsv = mypath + r'\CatB_predict.csv'

dropcolumn = ['gender']
proValue_cols = ['age',	'income',
    'academic_performance', 'work_productivity', 'multiplayer_ratio',	
    'toxic_exposure', 'violent_games_ratio', 'mobile_gaming_ratio',
    'night_gaming_ratio', 'weekend_gaming_hours', 'friends_gaming_count',
    'streaming_hours', 'esports_interest', 'headset_usage',	
    'microtransactions_spending', 'parental_supervision', 'bmi',		
    'eye_strain_score',	'back_pain_score', 'competitive_rank',
    'internet_quality', 'stress_level', 'anxiety_score', 'loneliness_score', 'aggression_score', 'happiness_score',
    'social_interaction_score', 'relationship_satisfaction', 'online_friends', 'sleep_hours', 'exercise_hours', 'caffeine_intake',
    'depression_score', 'addiction_level']                 #因為此程式無處理文字欄位，避免程式當掉先將gender欄位刪掉

'''
dropcolumn = ['age',	'gender', 'income',
    'academic_performance', 'work_productivity', 'multiplayer_ratio',	
    'toxic_exposure', 'violent_games_ratio', 'mobile_gaming_ratio',
    'night_gaming_ratio', 'weekend_gaming_hours', 'friends_gaming_count',
    'streaming_hours', 'esports_interest', 'headset_usage',	
    'microtransactions_spending', 'parental_supervision', 'bmi',		
    'eye_strain_score',	'back_pain_score', 'competitive_rank',
    'internet_quality', 'stress_level', 'anxiety_score', 'loneliness_score', 'aggression_score', 'happiness_score',
    'social_interaction_score', 'relationship_satisfaction', 'online_friends', 'sleep_hours', 'exercise_hours', 'caffeine_intake',
    'depression_score']                 #因為此程式無處理文字欄位，避免程式當掉先將gender欄位刪掉

proValue_cols = ['daily_gaming_hours', 'screen_time_total', 'weekly_sessions', 'years_gaming', 'addiction_level'] # 數值欄位的資料處理
'''
predictCol = 'addiction_level'          #要預測的欄位即Y
dummies = []

##################################################
#Step 2                                          #
##################################################
data = pd.read_csv(csvFile, header=0)
data = data.drop(dropcolumn, axis=1) #因為此程式無處理文字欄位，避免程式當掉先將gender欄位刪掉
data = data.astype(np.float32)
data.head()
#print(data.isnull().sum())    #計算空值數量
#print(data.isnull().count())  #計算非空值數量
#print(data.agg(['max','min']))

##################################################
#Step 3                                          #
##################################################
# 對所有欄位做 0.2 ~ 0.8 的 Min-Max 縮放
#data[] = data[dummies]                                         #編碼用 要去改dummies = [?]
#data = pd.get_dummies(data, columns=['CHAS'], dtype=int)       #編碼用

cols_to_norm = proValue_cols
data[cols_to_norm] = data[cols_to_norm].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()) * 0.6 + 0.2)
y    = data[predictCol]                                                            
data = data.drop([predictCol], axis=1)
#data = data.drop([predictCol, ''], axis=1)                      #編碼用
#data = data.drop([predictCol, 'CHAS_original'], axis=1)         #boston

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

    with open(outFile, "a") as f:
        f.write(f"{tr_mse:>8.4f},{te_mse:>8.4f},{tr_rmse:>8.4f},{te_rmse:>8.4f},{tr_mae:>8.4f},{te_mae:>8.4f},{tr_r2:>8.4f},{te_r2:>8.4f},{threshold*100:.0f},{pass_count}/{len(ytest)}, {pass_count/len(ytest)*100:.1f},{train_score:>8.4f},{test_score:>8.4f}")
        f.write('\n')
    return te_r2

##################################################
#Step 6 : Loop parameter for each model          #
##################################################
def SevenModel(SModel, oFile):
    try:
        SModel.fit(X_train, y_train)
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
        f.write("{:<12}, {:<12}, {:<12}, {:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12},{:<12}".format("n_estimators",'learning_rate',"max_depth",'min_child_weight','gamma','subsample','colsample_bytree','reg_alpha','reg_lambda',"tr_mse","te_mse","tr_rmse","te_rmse","tr_mae","te_mae","tr_r2","te_r2","通過%","誤差門檻","誤差門檻%",'Train %','Test %'))
        f.write('\n')

    #要新增, 注意step01, step001, step05, step005
    for estimator1 in range(n_estimators_1,n_estimators_2):
        for learn_rate in np.arange(learning_rate_1, learning_rate_2, step001):
            for md3 in range(max_depth_1,max_depth_2):
                for min_child in range(min_child_weight_1,min_child_weight_2):
                    for gamm in np.arange(gamma_1, gamma_2, step01):
                        #for sbsample in np.arange(subsample_1, subsample_2, step01):
                            #for colsample_bt in np.arange(colsample_bytree_1, colsample_bytree_2, step01):
                                for reg_al in np.arange(reg_alpha_1, reg_alpha_2, step001):
                                    for reg_la in range(reg_lambda_1,reg_lambda_2):
                                        xgbr1 = XGBRegressor(n_estimators=estimator1, learning_rate=learn_rate,
                                                             max_depth=md3, min_child_weight=min_child,
                                                             gamma=gamm, subsample=1.0,
                                                             colsample_bytree=1.0, colsample_bylevel=1.0,
                                                             colsample_bynode=1.0, reg_alpha=reg_al,
                                                             reg_lambda=reg_la, scale_pos_weight=1,
                                                             booster='gbtree', tree_method='auto',
                                                             n_jobs=None, random_state=42,
                                                             verbosity=1, objective='reg:squarederror')
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


