import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# 1. Streamlit 設定
# ============================================================

st.set_page_config(
    page_title="遊戲成癮分析 Gaming Addiction Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. 遊戲感 UI / CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #FFF8E7;
        color: #263238;
    }

    .stApp::before {
        display: none;
    }

    [data-testid="stSidebar"] {
        background: #FFE8A3;
        border-right: 5px solid #F4B942;
    }

    [data-testid="stSidebar"] * {
        color: #263238 !important;
    }

    h1, h2, h3, h4, h5, p, label {
        color: #263238 !important;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero-box {
        background: #FFD166;
        border: 6px solid #F4A261;
        border-radius: 28px;
        padding: 34px 38px;
        box-shadow: 10px 10px 0 #2A9D8F;
        margin-bottom: 28px;
    }

    .hero-title {
        color: #263238 !important;
        font-size: 42px;
        font-weight: 1000;
        letter-spacing: 2px;
        line-height: 1.05;
    }

    .hero-subtitle {
        color: #5D4037 !important;
        font-size: 20px;
        font-weight: 800;
        margin-top: 12px;
        line-height: 1.7;
    }

    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        background: #2A9D8F !important;
        color: white !important;
        border: 4px solid #1F776D !important;
        border-radius: 16px !important;
        font-weight: 900 !important;
        box-shadow: 5px 5px 0 #264653 !important;
    }

    .stButton > button:hover {
        background: #E76F51 !important;
        border-color: #C65339 !important;
    }

    /* ========================================================
       METRIC
       ======================================================== */

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 4px solid #F4B942;
        border-radius: 18px;
        padding: 14px;
        box-shadow: 5px 5px 0 #F6BD60;
    }

    /* ========================================================
       基本卡片
       ======================================================== */

    .cartoon-card {
        background: #FFFFFF;
        border: 5px solid #264653;
        border-radius: 22px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 7px 7px 0 #E9C46A;
    }

    .section-card {
        background: #BDE0FE;
        border: 5px solid #457B9D;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 7px 7px 0 #90BEDE;
    }

    .gold-card {
        background: #FFE5B4;
        border: 5px solid #F4A261;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 7px 7px 0 #FFD166;
    }

    .pink-card {
        background: #FFD6E0;
        border: 5px solid #E76F8F;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 7px 7px 0 #F4A6B7;
    }

    .green-card {
        background: #D8F3DC;
        border: 5px solid #52B788;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 7px 7px 0 #95D5B2;
    }

    /* ========================================================
       GitHub 程式碼連結
       ======================================================== */

    .flow-py-link {
        display: inline-block;
        margin-top: 8px;
        padding: 7px 12px;
        background: #2A9D8F;
        color: white !important;
        border: 3px solid #1F776D;
        border-radius: 10px;
        font-weight: 900;
        text-decoration: none !important;
        box-shadow: 3px 3px 0 #264653;
    }

    .flow-py-link:hover {
        background: #E76F51;
        border-color: #C65339;
        color: white !important;
    }

    /* ========================================================
       DataFrame
       ======================================================== */

    [data-testid="stDataFrame"] {
        border: 4px solid #F4B942;
        border-radius: 16px;
    }

    /* ========================================================
       關聯性
       ======================================================== */

    .relation-top4 {
        background: #FFF3CD;
        border: 4px solid #F4B942;
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 18px;
        box-shadow: 5px 5px 0 #E9C46A;
    }

    .relation-top4-title {
        font-size: 19px;
        font-weight: 900;
        color: #7A5610 !important;
        margin-bottom: 8px;
    }

    /* ========================================================
       專題流程
       ======================================================== */

    .flow-container {
        background: #FFFFFF;
        border: 5px solid #264653;
        border-radius: 22px;
        padding: 25px;
        box-shadow: 7px 7px 0 #E9C46A;
        margin-top: 15px;
        margin-bottom: 25px;
    }

    .flow-title {
        background: #FFD166;
        border: 4px solid #F4A261;
        border-radius: 16px;
        padding: 14px;
        text-align: center;
        font-size: 24px;
        font-weight: 1000;
        color: #263238 !important;
        margin-bottom: 22px;
    }

    .flow-step {
        background: #BDE0FE;
        border: 4px solid #457B9D;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        font-weight: 900;
        color: #263238 !important;
        margin: 8px auto;
        max-width: 850px;
        line-height: 1.6;
    }

    .flow-step-main {
        background: #FFD166;
        border-color: #F4A261;
    }

    .flow-step-green {
        background: #D8F3DC;
        border-color: #52B788;
    }

    .flow-step-pink {
        background: #FFD6E0;
        border-color: #E76F8F;
    }

    .flow-arrow {
        text-align: center;
        font-size: 28px;
        font-weight: 1000;
        color: #264653 !important;
        padding: 2px;
    }

    .flow-small {
        font-size: 14px;
        font-weight: 700;
    }

    /* ========================================================
       Dashboard 卡片
       ======================================================== */

    .game-card {
        background: #FFFFFF;
        border: 5px solid #264653;
        border-radius: 22px;
        padding: 22px;
        min-height: 165px;
        box-shadow: 7px 7px 0 #E9C46A;
        margin-bottom: 18px;
    }

    .game-card h3 {
        margin-top: 0;
        font-size: 20px;
        font-weight: 1000;
        color: #263238 !important;
    }

    .game-card p {
        margin-bottom: 0;
        line-height: 1.7;
        color: #263238 !important;
    }

    .game-card-yellow {
        background: #FFF3CD;
        border-color: #F4B942;
        box-shadow: 7px 7px 0 #FFD166;
    }

    .game-card-blue {
        background: #BDE0FE;
        border-color: #457B9D;
        box-shadow: 7px 7px 0 #90BEDE;
    }

    .game-card-green {
        background: #D8F3DC;
        border-color: #52B788;
        box-shadow: 7px 7px 0 #95D5B2;
    }

    .game-card-pink {
        background: #FFD6E0;
        border-color: #E76F8F;
        box-shadow: 7px 7px 0 #F4A6B7;
    }

    /* ========================================================
       Dashboard 流程
       ======================================================== */

    .game-flow {
        background: #264653;
        color: white;
        border: 5px solid #1F776D;
        border-radius: 22px;
        padding: 20px;
        text-align: center;
        box-shadow: 7px 7px 0 #F4B942;
        margin: 18px 0;
    }

    .game-flow-title {
        font-size: 18px;
        font-weight: 1000;
        margin-bottom: 12px;
        color: white !important;
    }

    .game-flow-step {
        display: inline-block;
        background: #FFD166;
        color: #263238 !important;
        border: 3px solid #F4A261;
        border-radius: 12px;
        padding: 10px 15px;
        margin: 5px;
        font-weight: 1000;
    }

    /* ========================================================
       Dashboard Section Title
       ======================================================== */

    .game-section-title {
        background: #FFD166;
        border: 4px solid #F4A261;
        border-radius: 15px;
        padding: 10px 16px;
        display: inline-block;
        font-weight: 1000;
        color: #263238 !important;
        box-shadow: 4px 4px 0 #2A9D8F;
        margin-bottom: 15px;
    }

    /* ========================================================
       PLAYER
       ======================================================== */

    .player-card {
        background: #FFFFFF;
        border: 5px solid #264653;
        border-radius: 20px;
        padding: 18px 10px;
        text-align: center;
        box-shadow: 5px 5px 0 #FFD166;
        min-height: 115px;
    }

    .player-icon {
        font-size: 30px;
    }

    .player-number {
        font-size: 13px;
        font-weight: 900;
        color: #E76F51 !important;
        margin-top: 5px;
    }

    .player-name {
        font-size: 16px;
        font-weight: 1000;
        color: #263238 !important;
        margin-top: 5px;
    }

    /* ========================================================
       壓力測試
       ======================================================== */

    .stress-image-title {
        background: #FFFFFF;
        border: 4px solid #F4B942;
        border-radius: 18px;
        padding: 14px 18px;
        margin-top: 20px;
        margin-bottom: 18px;
        box-shadow: 5px 5px 0 #E9C46A;
        font-weight: 900;
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. Session State
# ============================================================

if "user_result" not in st.session_state:
    st.session_state.user_result = None

if "xgb_package" not in st.session_state:
    st.session_state.xgb_package = None


# ============================================================
# 4. CSV / 分析欄位
# ============================================================

DATA_FILE = "gaming_part1_100k.csv"

feature_columns = [
    "daily_gaming_hours",
    "sleep_hours",
    "exercise_hours",
    "screen_time_total"
]

target_column = "addiction_level"


column_names = {
    "daily_gaming_hours":
        "平均每日遊戲時間",

    "sleep_hours":
        "平均每日睡眠時間",

    "exercise_hours":
        "平均每日運動時間",

    "screen_time_total":
        "平均每日總螢幕時間",

    "addiction_level":
        "遊戲成癮程度"
}


field_help = {
    "daily_gaming_hours":
        "平均每天遊戲時間，單位為小時。",

    "sleep_hours":
        "平均每日睡眠時間，單位為小時。",

    "exercise_hours":
        "平均每日運動時間，單位為小時。",

    "screen_time_total":
        "平均每日總螢幕使用時間，單位為小時。"
}


# ============================================================
# 5. 讀取 10 萬筆資料
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_FILE)


try:
    df = load_data()

except Exception as e:
    st.error(
        "❌ CSV 讀取失敗，請確認 gaming_part1_100k.csv "
        "與程式位於同一個資料夾。"
    )

    st.error(str(e))

    st.stop()


required_columns = feature_columns + [
    target_column
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:
    st.error("❌ CSV 缺少必要欄位")

    st.write(
        "缺少欄位：",
        missing_columns
    )

    st.stop()


analysis_df = df[
    required_columns
].copy()


analysis_df = analysis_df.dropna()

analysis_df = analysis_df.astype(float)


# 圖表只顯示前 1000 筆
chart_df = analysis_df.head(1000).copy()


# ============================================================
# 6. 關聯性
# ============================================================

relation_values = (
    analysis_df[
        feature_columns + [target_column]
    ]
    .corr(
        numeric_only=True
    )[target_column]
    .drop(target_column)
    .to_dict()
)


relation_df = pd.DataFrame({
    "變數": [
        column_names[col]
        for col in feature_columns
    ],

    "英文欄位":
        feature_columns,

    "關聯程度": [
        relation_values[col]
        for col in feature_columns
    ]
})


relation_df["絕對關聯程度"] = (
    relation_df["關聯程度"].abs()
)


relation_df = (
    relation_df
    .sort_values(
        "絕對關聯程度",
        ascending=False
    )
    .drop(
        columns="絕對關聯程度"
    )
)


relation_df.index = range(
    1,
    len(relation_df) + 1
)

relation_df.index.name = "排名"


# ============================================================
# 7. 成癮程度分級
# ============================================================

def addiction_level(score):

    if score < 2:
        return "低"

    elif score < 4:
        return "偏低"

    elif score < 6:
        return "中"

    elif score < 8:
        return "偏高"

    else:
        return "高"


color_map = {
    "低":
        "#4CAF50",

    "偏低":
        "#8BC34A",

    "中":
        "#FFC107",

    "偏高":
        "#FF9800",

    "高":
        "#F44336"
}


# ============================================================
# 8. XGBoost
# ============================================================

@st.cache_resource(show_spinner="🤖 XGBoost 模型訓練中，請稍候...")
def train_xgboost_model(
    X_data,
    y_data
):

    X = X_data.copy()

    y = y_data.copy()


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.33,
            random_state=101
        )
    )


    scaler_X = MinMaxScaler(
        feature_range=(0.2, 0.8)
    )


    scaler_y = MinMaxScaler(
        feature_range=(0.2, 0.8)
    )


    X_train_scaled = (
        scaler_X.fit_transform(
            X_train
        )
    )


    X_test_scaled = (
        scaler_X.transform(
            X_test
        )
    )


    y_train_scaled = (
        scaler_y.fit_transform(
            y_train
        ).ravel()
    )


    # ========================================================
    # 最新 XGBoost 參數
    # ========================================================

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.7,
        colsample_bytree=1,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        reg_alpha=0,
        reg_lambda=0,
        scale_pos_weight=1,
        booster="gbtree",
        tree_method="auto",
        n_jobs=None,
        random_state=42,
        verbosity=1,
        objective="reg:squarederror"
    )


    model.fit(
        X_train_scaled,
        y_train_scaled
    )


    # ========================================================
    # 模型預測
    # ========================================================

    predicted_scaled = (
        model.predict(
            X_test_scaled
        )
    )


    predicted = (
        scaler_y.inverse_transform(
            predicted_scaled.reshape(
                -1,
                1
            )
        ).ravel()
    )


    actual = y_test.values.ravel()


    # ========================================================
    # 模型評估
    # ========================================================

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )
    )


    mae = float(
        mean_absolute_error(
            actual,
            predicted
        )
    )


    r2 = float(
        r2_score(
            actual,
            predicted
        )
    )


    return {
        "model":
            model,

        "scaler_X":
            scaler_X,

        "scaler_y":
            scaler_y,

        "rmse":
            rmse,

        "mae":
            mae,

        "r2":
            r2
    }


# ============================================================
# 9. 模型準備
# ============================================================

# ⭐ 不再使用 ThreadPoolExecutor
# ⭐ 改成 Streamlit cache_resource
# ⭐ 可以避免 Dashboard 因背景執行緒而無法正常顯示

if st.session_state.xgb_package is None:

    try:
        st.session_state.xgb_package = (
            train_xgboost_model(
                analysis_df[
                    feature_columns
                ],
                analysis_df[
                    [target_column]
                ]
            )
        )

    except Exception as e:

        st.session_state.xgb_package = None

        model_error = str(e)

    else:

        model_error = None

else:

    model_error = None


model_ready = (
    st.session_state.xgb_package
    is not None
)


if model_ready:

    model_package = (
        st.session_state.xgb_package
    )

    xgb_model = (
        model_package["model"]
    )

    scaler_X = (
        model_package["scaler_X"]
    )

    scaler_y = (
        model_package["scaler_y"]
    )

    model_rmse = (
        model_package["rmse"]
    )

    model_mae = (
        model_package["mae"]
    )

    model_r2 = (
        model_package["r2"]
    )

else:

    xgb_model = None
    scaler_X = None
    scaler_y = None

    model_rmse = None
    model_mae = None
    model_r2 = None


# ============================================================
# 10. Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🎮 遊戲成癮分析"
    )

    st.caption(
        "Gaming Addiction Analysis System"
    )

    st.divider()


    page = st.radio(
        "功能選單",

        [
            "🏠 Dashboard",
            "ℹ️ 專題說明",
            "📊 資料與關係分析",
            "✅ 資料驗證",
            "🧪 壓力測試",
            "👤 使用者分析"
        ]
    )


    st.divider()


# ============================================================
# 11. Dashboard
# ============================================================

if page == "🏠 Dashboard":

    # ========================================================
    # HERO
    # ========================================================

    st.markdown(
        """
        <div class="hero-box">
            <div style="font-size:18px; font-weight:1000; color:#7A5610; letter-spacing:2px;">
                🎮 AI GAMING LAB｜MISSION START
            </div>
            <div class="hero-title">
                遊戲成癮分析
            </div>
            <div style="font-size:27px; font-weight:1000; color:#264653; margin-top:4px;">
                GAMING ADDICTION ANALYSIS
            </div>
            <div class="hero-subtitle">
                🎯 用資料探索遊戲行為<br>
                🤖 用 AI 預測遊戲成癮程度
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # GAME FLOW
    # ========================================================

    st.markdown(
        """
        <div class="game-flow">
            <div class="game-flow-title">
                🎮 GAME FLOW｜YOUR DATA → AI RESULT
            </div>
            <span class="game-flow-step">🎯 INPUT</span>
            <span style="font-size:20px; font-weight:1000; color:white;">→</span>
            <span class="game-flow-step">📊 ANALYZE</span>
            <span style="font-size:20px; font-weight:1000; color:white;">→</span>
            <span class="game-flow-step">🤖 XGBOOST AI</span>
            <span style="font-size:20px; font-weight:1000; color:white;">→</span>
            <span class="game-flow-step">🏆 RESULT</span>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # MISSION
    # ========================================================

    st.markdown(
        """
        <div class="section-card">
            <h3>🎯 MISSION｜這個專題在做什麼？</h3>
            <p>
                從遊戲行為與生活型態資料出發，
                找出與「遊戲成癮程度」最相關的關鍵因素，
                再透過 XGBoost 建立 AI 預測模型，
                最後讓使用者輸入自己的資料，
                取得個人化的遊戲成癮程度預測結果。
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    # ========================================================
    # LEVEL
    # ========================================================

    st.markdown(
        """
        <div class="game-section-title">
            🕹️ GAME LEVEL｜專題三大任務
        </div>
        """,
        unsafe_allow_html=True
    )


    level1, level2, level3 = st.columns(3)


    with level1:

        st.markdown(
            """
            <div class="game-card game-card-yellow">
                <h3>🎯 LEVEL 01｜DATA</h3>
                <p>
                    <b>資料探索任務</b><br><br>
                    📁 Kaggle 原始資料<br>
                    📊 271,807 筆資料<br>
                    🧹 資料清洗<br>
                    ✂️ 資料切分<br>
                    🔎 關聯性分析
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    with level2:

        st.markdown(
            """
            <div class="game-card game-card-blue">
                <h3>🤖 LEVEL 02｜AI</h3>
                <p>
                    <b>人工智慧任務</b><br><br>
                    🧠 XGBoost Regressor<br>
                    📐 MinMaxScaler<br>
                    🎯 4 個核心特徵<br>
                    ⚙️ 模型訓練<br>
                    🧪 壓力測試
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    with level3:

        st.markdown(
            """
            <div class="game-card game-card-green">
                <h3>🏆 LEVEL 03｜PREDICT</h3>
                <p>
                    <b>成癮程度預測任務</b><br><br>
                    👤 使用者輸入資料<br>
                    🔍 資料驗證<br>
                    🤖 AI 預測<br>
                    📊 0～10 分<br>
                    🏆 成癮程度分級
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # ========================================================
    # GAME STATUS
    # ========================================================

    st.markdown(
        """
        <div class="game-section-title">
            📊 GAME STATUS｜目前專題資料
        </div>
        """,
        unsafe_allow_html=True
    )


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.metric(
            "📁 分析資料",
            f"{len(analysis_df):,} 筆"
        )


    with m2:

        st.metric(
            "🎯 核心特徵",
            "4 個"
        )


    with m3:

        st.metric(
            "🤖 AI MODEL",
            "XGBoost"
        )


    with m4:

        st.metric(
            "🏆 預測範圍",
            "0～10"
        )


    st.divider()


    # ========================================================
    # CORE SKILLS
    # ========================================================

    st.markdown(
        """
        <div class="game-section-title">
            🎮 CORE SKILLS｜4 個核心分析特徵
        </div>
        """,
        unsafe_allow_html=True
    )


    feature1, feature2 = st.columns(2)


    with feature1:

        st.markdown(
            """
            <div class="game-card game-card-pink">
                <h3>🎮 SKILL 01</h3>
                <p>
                    <b>平均每日遊戲時間</b><br>
                    daily_gaming_hours
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="game-card game-card-yellow">
                <h3>😴 SKILL 02</h3>
                <p>
                    <b>平均每日睡眠時間</b><br>
                    sleep_hours
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    with feature2:

        st.markdown(
            """
            <div class="game-card game-card-green">
                <h3>🌱 SKILL 03</h3>
                <p>
                    <b>平均每日運動時間</b><br>
                    exercise_hours
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="game-card game-card-blue">
                <h3>🖥️ SKILL 04</h3>
                <p>
                    <b>平均每日總螢幕時間</b><br>
                    screen_time_total
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # ========================================================
    # AI BATTLE
    # ========================================================

    st.markdown(
        """
        <div class="game-section-title">
            🤖 AI BATTLE｜模型預測流程
        </div>
        """,
        unsafe_allow_html=True
    )


    model_status = (
        "🟢 AI READY"
        if model_ready
        else "🔴 AI ERROR"
    )


    st.markdown(
        f"""
        <div class="gold-card">
            <h3>🤖 XGBoost Regressor</h3>
            <p>
                4 個核心分析特徵
                → MinMaxScaler
                → XGBoost
                → 預測 addiction_level
                → 0～10 分
                → 成癮傾向百分比
                → 成癮等級
            </p>
            <br>
            <b>🎯 MODEL STATUS：</b>
            {model_status}
        </div>
        """,
        unsafe_allow_html=True
    )


    if not model_ready and model_error:

        st.warning(
            "⚠️ XGBoost 模型目前無法完成訓練。"
        )


    st.divider()


    # ========================================================
    # PLAYER
    # ========================================================

    st.markdown(
        """
        <div class="game-section-title">
            👥 PLAYER SELECT｜專題組員
        </div>
        """,
        unsafe_allow_html=True
    )


    members = [
        "賴柏宇",
        "陳奕均",
        "陳桃桃",
        "王韻涵",
        "莊中辰"
    ]


    cols = st.columns(5)


    for i, member in enumerate(
        members,
        start=1
    ):

        with cols[i - 1]:

            st.markdown(
                f"""
                <div class="player-card">
                    <div class="player-icon">🎮</div>
                    <div class="player-number">PLAYER {i}</div>
                    <div class="player-name">{member}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


    st.divider()


    # ========================================================
    # CTA
    # ========================================================

    st.markdown(
        """
        <div class="hero-box" style="text-align:center; margin-top:15px; margin-bottom:10px;">
            <div style="font-size:28px; font-weight:1000; color:#264653;">
                🎮 READY TO PLAY?
            </div>
            <div style="font-size:18px; font-weight:800; color:#5D4037; margin-top:10px;">
                前往「👤 使用者分析」輸入你的遊戲與生活型態資料
            </div>
            <div style="font-size:24px; font-weight:1000; color:#2A9D8F; margin-top:12px;">
                ▶ START YOUR AI CHALLENGE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 12. 專題說明
# ============================================================

elif page == "ℹ️ 專題說明":

    st.title("ℹ️ 專題說明")


    st.subheader(
        "🎮 遊戲與心理健康 × 遊戲成癮分析"
    )


    st.write(
        "本專題以 Kaggle「Gaming & Mental Health」資料集為基礎，"
        "經過資料清洗、欄位篩選、關聯性分析與資料視覺化，"
        "建立遊戲成癮程度分析平台。"
    )


    st.divider()


    # ========================================================
    # 資料來源
    # ========================================================

    st.subheader("📚 資料來源")


    st.markdown(
        "**資料集：** "
        "[Kaggle：Gaming & Mental Health]"
        "(https://www.kaggle.com/datasets/sharmajicoder/gaming-and-mental-health)"
    )


    source_df = pd.DataFrame({
        "項目": [
            "資料集",
            "檔案名稱",
            "原始欄位",
            "原始資料筆數"
        ],

        "內容": [
            "Kaggle：Gaming & Mental Health",
            "gaming_mental_health_10M_40features.csv",
            "39 欄",
            "10,000,000 筆"
        ]
    })


    source_df.index = range(
        1,
        len(source_df) + 1
    )


    st.dataframe(
        source_df,
        use_container_width=True
    )


    st.divider()


    # ========================================================
    # 專題技術
    # ========================================================

    st.subheader("🛠️ 專題技術")


    tech1, tech2, tech3 = st.columns(3)


    with tech1:

        st.info(
            "🐍 Python\n\n"
            "程式開發"
        )


    with tech2:

        st.info(
            "📊 Pandas\n\n"
            "資料處理"
        )


    with tech3:

        st.info(
            "🔢 NumPy\n\n"
            "數值運算"
        )


    tech4, tech5, tech6 = st.columns(3)


    with tech4:

        st.info(
            "🎨 Streamlit\n\n"
            "互動式網站"
        )


    with tech5:

        st.info(
            "📈 Plotly\n\n"
            "資料視覺化"
        )


    with tech6:

        st.info(
            "🤖 XGBoost\n\n"
            "機器學習"
        )


    st.divider()

    # ============================================================
    # 專題完整流程
    # ============================================================

    st.markdown(
        """
        <div class="section-card">
            <h2>🚀 專題完整流程</h2>
            <p>
                從原始資料整理、資料切割、特徵分析，
                到最後建立 XGBoost AI 預測模型。
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # 01 原始資料
    # ============================================================

    st.markdown(
        """
        <div style="
            background:#FFD166;
            border:4px solid #F4A261;
            border-radius:16px;
            padding:18px;
            text-align:center;
            color:#263238;
            margin:10px 0;
        ">
            <div style="font-size:28px; font-weight:1000;">01</div>
            <div style="font-size:21px; font-weight:1000;">
                📥 原始資料 
            </div>
            <div style="font-size:14px; font-weight:700;">
                Kaggle Gaming & Mental Health Dataset<br>
                gaming_mental_health_10M_40features.csv
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align:center;font-size:30px;font-weight:1000;">↓</div>',
        unsafe_allow_html=True
    ) 

    # ============================================================
    # 02 資料清理
    # ============================================================

    st.markdown(
        """
        <div style="
            background:#BDE0FE;
            border:4px solid #457B9D;
            border-radius:16px;
            padding:18px;
            text-align:center;
            color:#263238;
            margin:10px 0;
        ">
            <div style="font-size:28px; font-weight:1000;">02</div>
            <div style="font-size:21px; font-weight:1000;">
                🧹 資料清理
            </div>
            <div style="font-size:14px; font-weight:700;">
                缺失值、異常資料與欄位整理
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # 03 資料切割
    # ============================================================

    st.markdown(
        """
        <div style="
            background:#BDE0FE;
            border:4px solid #457B9D;
            border-radius:16px;
            padding:18px;
            text-align:center;
            color:#263238;
            margin:10px 0;
        ">
            <div style="font-size:28px; font-weight:1000;">03</div>
            <div style="font-size:21px; font-weight:1000;">
                ✂️ 資料切割
            </div>
            <div style="font-size:14px; font-weight:700;">
                將資料分成 100,000 / 100,000 / 71,807
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; margin:8px 0 15px 0;">
            <a href="https://github.com/fuwafuwa152123/gaming/blob/main/data_split.py"
            target="_blank"
            style="
                color:#264653;
                font-weight:900;
                text-decoration:none;
                background:#FFFFFF;
                padding:7px 16px;
                border-radius:10px;
                border:2px solid #264653;
            ">
                💻 查看 data_split.py
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align:center;font-size:30px;font-weight:1000;">↓</div>',
        unsafe_allow_html=True
    )


    # ============================================================
    # 04 特徵檢查
    # ============================================================

    st.markdown(
        """
        <div style="
            background: #D8F3DC;
            border: 4px solid #52B788;
            border-radius: 16px;
            padding: 18px;
            text-align: center;
            color: #263238;
            margin: 10px 0;
        ">
            <div style="font-size: 28px; font-weight: 900;">04</div>
            <div style="font-size: 21px; font-weight: 900;">
                🔍 特徵檢查
            </div>
            <div style="font-size: 14px; font-weight: 700;">
                找出與遊戲成癮程度最相關的重要特徵
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align: center; margin: 8px 0 15px 0;">
            <a href="https://github.com/fuwafuwa152123/gaming/blob/main/featureCheck_1.py"
            target="_blank"
            style="
                color: #264653;
                font-weight: 900;
                text-decoration: none;
                background: #FFFFFF;
                padding: 7px 16px;
                margin: 4px;
                border-radius: 10px;
                border: 2px solid #264653;
                display: inline-block;
            ">
                💻 查看 featureCheck_1.py
            </a>
            <a href="https://github.com/fuwafuwa152123/gaming/blob/main/featureCheck_2.py"
            target="_blank"
            style="
                color: #264653;
                font-weight: 900;
                text-decoration: none;
                background: #FFFFFF;
                padding: 7px 16px;
                margin: 4px;
                border-radius: 10px;
                border: 2px solid #264653;
                display: inline-block;
            ">
                💻 查看 featureCheck_2.py
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align: center; font-size: 30px; font-weight: 900;">↓</div>',
        unsafe_allow_html=True
    )


    # ============================================================
    # 05 關聯性分析
    # ============================================================

    st.markdown(
        """
        <div style="
            background:#D8F3DC;
            border:4px solid #52B788;
            border-radius:16px;
            padding:18px;
            text-align:center;
            color:#263238;
            margin:10px 0;
        ">
            <div style="font-size:28px; font-weight:1000;">05</div>
            <div style="font-size:21px; font-weight:1000;">
                📊 關聯性分析
            </div>
            <div style="font-size:14px; font-weight:700;">
                分析各項生活型態與遊戲行為和成癮程度的關係
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; margin:8px 0 15px 0;">
            <a href="https://github.com/fuwafuwa152123/gaming/blob/main/relation.py"
            target="_blank"
            style="
                color:#264653;
                font-weight:900;
                text-decoration:none;
                background:#FFFFFF;
                padding:7px 16px;
                border-radius:10px;
                border:2px solid #264653;
            ">
                💻 查看 relation.py
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align:center;font-size:30px;font-weight:1000;">↓</div>',
        unsafe_allow_html=True
    )


    # ============================================================
    # 06 壓力測試
    # ============================================================

    st.markdown(
        """
        <div style="
            background:#FFD6E0;
            border:4px solid #E76F8F;
            border-radius:16px;
            padding:18px;
            text-align:center;
            color:#263238;
            margin:10px 0;
        ">
            <div style="font-size:28px; font-weight:1000;">06</div>
            <div style="font-size:21px; font-weight:1000;">
                🧪 壓力測試
            </div>
            <div style="font-size:14px; font-weight:700;">
                使用不同資料批次測試模型穩定性
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; margin:8px 0 15px 0;">
            <a href="https://github.com/fuwafuwa152123/gaming/blob/main/stress_test.py"
            target="_blank"
            style="
                color:#264653;
                font-weight:900;
                text-decoration:none;
                background:#FFFFFF;
                padding:7px 16px;
                border-radius:10px;
                border:2px solid #264653;
            ">
                💻 查看 stress_test.py
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align:center;font-size:30px;font-weight:1000;">↓</div>',
        unsafe_allow_html=True
    )


    # ============================================================
    # 07 XGBoost AI 模型
    # ============================================================

    st.markdown(
        """
        <div style="
            background:#FFD166;
            border:4px solid #F4A261;
            border-radius:16px;
            padding:18px;
            text-align:center;
            color:#263238;
            margin:10px 0;
        ">
            <div style="font-size:28px; font-weight:1000;">07</div>
            <div style="font-size:21px; font-weight:1000;">
                🤖 XGBoost AI 模型
            </div>
            <div style="font-size:14px; font-weight:700;">
                使用 4 項核心特徵建立遊戲成癮程度預測模型
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; margin:8px 0 15px 0;">
            <a href="https://github.com/fuwafuwa152123/gaming/blob/main/xgboost_model.py"
            target="_blank"
            style="
                color:#264653;
                font-weight:900;
                text-decoration:none;
                background:#FFFFFF;
                padding:7px 16px;
                border-radius:10px;
                border:2px solid #264653;
            ">
                💻 查看 xgboost_model.py
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align:center;font-size:30px;font-weight:1000;">↓</div>',
        unsafe_allow_html=True
    )

    # ============================================================
    # 08 使用者輸入
    # ============================================================

    st.markdown(
        """
        <div style="
            background: #BDE0FE;
            border: 4px solid #457B9D;
            border-radius: 16px;
            padding: 18px;
            text-align: center;
            color: #263238;
            margin: 10px 0;
        ">
            <div style="font-size: 28px; font-weight: 950;">
                08
            </div>
            <div style="font-size: 21px; font-weight: 900;">
                👤 使用者輸入
            </div>
            <div style="font-size: 14px; font-weight: 700;">
                輸入 4 項生活型態與遊戲相關資料
            </div>
            <p style="margin: 10px 0 0 0; font-weight: 900; font-size: 15px;">
                🎮 遊戲時間 ｜ 😴 睡眠時間 ｜ 🏃 運動時間 ｜ 📱 螢幕時間
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align: center; font-size: 30px; font-weight: 900;">↓</div>',
        unsafe_allow_html=True
    )


    # ============================================================
    # 09 預測遊戲成癮程度
    # ============================================================

    st.markdown(
        """
        <div style="
            background: #FFD6E0;
            border: 4px solid #E76F8F;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            color: #263238;
            margin: 10px 0;
        ">
            <div style="font-size: 28px; font-weight: 900;">
                09
            </div>
            <div style="font-size: 21px; font-weight: 900;">
                🎯 預測遊戲成癮程度
            </div>
            <div style="font-size: 14px; font-weight: 700;">
                XGBoost 預測 → 還原 0～10 分 → 轉換成成癮傾向百分比
            </div>
            <div style="
                margin-top: 10px;
                font-size: 18px;
                font-weight: 900;
            ">
                🟢 低  🟡 中  🟠 偏高  🔴 高
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # 最終分析欄位
    # ========================================================

    st.divider()


    st.subheader(
        "🧩 本研究最終分析欄位"
    )


    st.write(
        "經過關聯性分析後，本專題最終選擇關聯程度最高的 "
        "4 個核心欄位作為 XGBoost 模型輸入。"
    )


    selected_df = pd.DataFrame({
        "分析欄位": [
            column_names[col]
            for col in feature_columns
        ],

        "英文欄位":
            feature_columns
    })


    selected_df.index = range(
        1,
        len(selected_df) + 1
    )


    st.dataframe(
        selected_df,
        use_container_width=True
    )


    st.info(
        "🎯 遊戲成癮程度 addiction_level 為預測目標，"
        "不列入 4 個 X 分析特徵。"
    )


    st.divider()


    # ========================================================
    # XGBoost 預測架構
    # ========================================================

    st.subheader(
        "🤖 XGBoost 預測架構"
    )


    st.write(
        "目前程式使用 4 個核心分析特徵，"
        "經過 MinMaxScaler 標準化後送入 XGBoost Regressor，"
        "再將預測結果轉換回原始 0～10 分數。"
    )


    model_flow = """
4 個分析特徵
↓
MinMaxScaler
↓
XGBoost Regressor
↓
predict()
↓
Inverse Transform
↓
0 ～ 10 分
↓
成癮傾向百分比
↓
低／偏低／中／偏高／高
"""


    st.code(
        model_flow,
        language="text"
    )


# ============================================================
# 13. 資料與關係分析
# ============================================================

elif page == "📊 資料與關係分析":

    st.title(
        "📊 資料與關係分析"
    )


    st.write(
        "本頁使用 gaming_part1_100k.csv 的 10 萬筆資料進行分析；"
        "為兼顧網頁效能，圖表只顯示前 1000 筆。"
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "📊 分析資料",
        f"{len(analysis_df):,} 筆"
    )


    c2.metric(
        "🎯 X 分析欄位",
        "4 個"
    )


    c3.metric(
        "📌 圖表顯示",
        "前 1,000 筆"
    )


    st.divider()


    # ========================================================
    # 散佈圖
    # ========================================================

    st.subheader(
        "📈 散佈圖：X 變數 × 遊戲成癮程度"
    )


    x_col_scatter = st.selectbox(
        "選擇要分析的 X 變數",

        feature_columns,

        format_func=lambda x:
            column_names[x],

        key="scatter_x"
    )


    st.caption(
        f"X 軸：{column_names[x_col_scatter]}　｜　"
        f"Y 軸：{column_names[target_column]}　｜　"
        f"顯示前 1,000 筆"
    )


    fig = px.scatter(
        chart_df,

        x=x_col_scatter,

        y=target_column,

        labels={
            x_col_scatter:
                column_names[x_col_scatter],

            target_column:
                column_names[target_column]
        },

        title=(
            f"{column_names[x_col_scatter]} × "
            f"{column_names[target_column]}"
        ),

        opacity=0.65
    )


    fig.update_traces(
        marker=dict(size=6)
    )


    fig.update_layout(
        height=560,

        paper_bgcolor="#ffffff",

        plot_bgcolor="#ffffff",

        font=dict(
            color="#263238"
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.info(
        "📌 每一個點代表一筆資料。"
        "Y 軸固定為遊戲成癮程度，"
        "X 軸可在 4 個分析欄位中切換。"
    )


    st.divider()


    # ========================================================
    # 氣泡圖
    # ========================================================

    st.subheader(
        "🫧 氣泡圖：4 個欄位自由組合"
    )


    col1, col2 = st.columns(2)


    with col1:

        bubble_x = st.selectbox(
            "🔵 X 軸變數",

            feature_columns,

            index=0,

            format_func=lambda x:
                column_names[x],

            key="bubble_x"
        )


    with col2:

        available_y = [
            col
            for col in feature_columns
            if col != bubble_x
        ]


        bubble_y = st.selectbox(
            "🟣 Y 軸變數",

            available_y,

            format_func=lambda x:
                column_names[x],

            key="bubble_y"
        )


    bubble_df = chart_df.copy()


    bubble_df["成癮等級"] = (
        bubble_df[target_column]
        .apply(addiction_level)
    )


    bubble_fig = px.scatter(
        bubble_df,

        x=bubble_x,

        y=bubble_y,

        size=target_column,

        color="成癮等級",

        size_max=30,

        opacity=0.72,

        color_discrete_map=color_map,

        hover_data={
            bubble_x: True,

            bubble_y: True,

            target_column: ":.2f",

            "成癮等級": True
        },

        labels={
            bubble_x:
                column_names[bubble_x],

            bubble_y:
                column_names[bubble_y],

            target_column:
                "遊戲成癮程度",

            "成癮等級":
                "成癮等級"
        },

        title=(
            f"{column_names[bubble_x]} × "
            f"{column_names[bubble_y]}"
        )
    )


    bubble_fig.update_layout(
        height=620,

        paper_bgcolor="#ffffff",

        plot_bgcolor="#ffffff",

        font=dict(
            color="#263238"
        ),

        legend_title_text=
            "遊戲成癮程度"
    )


    st.plotly_chart(
        bubble_fig,
        use_container_width=True
    )


    st.info(
        "🫧 泡泡大小代表遊戲成癮程度，"
        "泡泡顏色代表低／偏低／中／偏高／高五個等級；"
        "X、Y 軸只從本研究選定的 4 個欄位中選擇。"
    )


    st.divider()


    # ========================================================
    # 37 個欄位
    # ========================================================

    st.subheader(
        "📊 全部欄位與遊戲成癮程度的關聯性"
    )


    all_relation_data = [
        ["平均每日遊戲時間", "daily_gaming_hours", 0.866051],
        ["平均每日總螢幕時間", "screen_time_total", 0.578582],
        ["平均每日運動時間", "exercise_hours", 0.064452],
        ["平均每日睡眠時間", "sleep_hours", 0.038925],
        ["年齡", "age", 0.004475],
        ["社交活動指數", "social_interaction_score", 0.003425],
        ["生活壓力評分", "stress_level", 0.003324],
        ["夜間遊戲比例", "night_gaming_ratio", 0.003054],
        ["線上網友人數", "online_friends", 0.002733],
        ["網路連線品質", "internet_quality", 0.002725],
        ["工作場所生產力評分", "work_productivity", 0.002612],
        ["接觸有毒社區", "toxic_exposure", 0.002446],
        ["眼睛疲勞程度", "eye_strain_score", 0.001650],
        ["對參與電競的興趣", "esports_interest", 0.001480],
        ["關係品質評分", "relationship_satisfaction", 0.001454],
        ["背痛嚴重程度", "back_pain_score", 0.001399],
        ["焦慮評估得分", "anxiety_score", 0.001299],
        ["孤獨指數", "loneliness_score", 0.001046],
        ["暴力遊戲的比例", "violent_games_ratio", 0.000997],
        ["家長監控評分", "parental_supervision", 0.000926],
        ["每週遊戲直播時長", "streaming_hours", 0.000883],
        ["週末遊戲時長", "weekend_gaming_hours", 0.000879],
        ["憂鬱評分", "depression_score", 0.000818],
        ["行動遊戲份額", "mobile_gaming_ratio", 0.000812],
        ["是否使用耳機", "headset_usage", 0.000703],
        ["月收入估算", "income", 0.000626],
        ["遊戲好友數量", "friends_gaming_count", 0.000621],
        ["每日咖啡因攝取量", "caffeine_intake", 0.000587],
        ["每週遊戲次數", "weekly_sessions", 0.000551],
        ["身體質量指數", "bmi", 0.000342],
        ["競技技能排名", "competitive_rank", 0.000335],
        ["攻擊傾向分數", "aggression_score", 0.000275],
        ["學業成績", "academic_performance", 0.000187],
        ["每月遊戲內消費", "microtransactions_spending", 0.000152],
        ["整體幸福感", "happiness_score", 0.000130],
        ["多人遊戲比例", "multiplayer_ratio", 0.000087],
        ["總遊戲經驗年資", "years_gaming", 0.000053]
    ]


    all_relation_df = pd.DataFrame(
        all_relation_data,

        columns=[
            "欄位名稱",
            "英文",
            "關聯"
        ]
    )


    all_relation_df.index = range(
        1,
        len(all_relation_df) + 1
    )

    all_relation_df.index.name = "排名"


    def highlight_top4(row):

        if row.name <= 4:

            return [
                "background-color: #FFD166; "
                "color: #263238; "
                "font-weight: 900; "
                "border: 2px solid #F4A261;"
            ] * len(row)

        return [""] * len(row)


    styled_relation_df = (
        all_relation_df
        .style
        .apply(
            highlight_top4,
            axis=1
        )
        .format({
            "關聯": "{:.6f}"
        })
    )


    st.dataframe(
        styled_relation_df,

        use_container_width=True,

        height=850
    )


    st.info(
        "📌 關聯數值越接近 1，代表正向關聯越強；"
        "本表完整呈現 37 個欄位與遊戲成癮程度的關聯程度。"
        "其中前 4 名為本研究目前選定的核心分析特徵。"
    )


    st.divider()


    # ========================================================
    # 分級
    # ========================================================

    st.subheader(
        "🎯 遊戲成癮程度分級"
    )


    level_df = pd.DataFrame({
        "等級": [
            "低",
            "偏低",
            "中",
            "偏高",
            "高"
        ],

        "分數範圍": [
            "0 ～ <2",
            "2 ～ <4",
            "4 ～ <6",
            "6 ～ <8",
            "8 ～ 10"
        ]
    })


    level_df.index = range(
        1,
        len(level_df) + 1
    )


    st.dataframe(
        level_df,
        use_container_width=True
    )


# ============================================================
# 14. 資料驗證
# ============================================================

elif page == "✅ 資料驗證":

    st.title(
        "✅ 資料驗證"
    )


    st.write(
        "本頁呈現 featureCheck_1 與 featureCheck_2 "
        "的模型驗證結果，透過 XGBoost 特徵重要度觀察各欄位 "
        "對 addiction_level 的影響程度，確認資料特徵是否具有實際分析價值。"
    )


    st.divider()


    # ========================================================
    # featureCheck_1
    # ========================================================

    st.subheader(
        "🔍 Feature Check 1｜原始欄位特徵重要度"
    )


    st.markdown(
        """
        <div class="simple-card">

        <h3>📊 模型評估結果</h3>

        <p>
        FeatureCheck_1 使用 XGBoost 模型進行特徵重要度分析，
        模型 R² Score 為 <b>0.7486</b>。
        </p>

        <p>
        在前 15 個重要特徵中，
        <b>daily_gaming_hours</b> 的重要度達
        <b>0.757947</b>，明顯高於其他欄位。
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.metric(
        "🎯 R² Score",
        "0.7486"
    )


    st.markdown(
        "🔗 **FeatureCheck_1 原始程式碼：** "
        "[featureCheck_1.py](https://github.com/fuwafuwa152123/gaming/blob/main/featureCheck_1.py)"
    )


    featurecheck1_data = [
        ["1", "daily_gaming_hours", 0.757947],
        ["2", "income", 0.007276],
        ["3", "aggression_score", 0.007036],
        ["4", "eye_strain_score", 0.007033],
        ["5", "streaming_hours", 0.007025],
        ["6", "bmi", 0.007006],
        ["7", "academic_performance", 0.006961],
        ["8", "back_pain_score", 0.006950],
        ["9", "relationship_satisfaction", 0.006896],
        ["10", "caffeine_intake", 0.006895],
        ["11", "loneliness_score", 0.006776],
        ["12", "happiness_score", 0.006752],
        ["13", "anxiety_score", 0.006669],
        ["14", "social_interaction_score", 0.006534],
        ["15", "depression_score", 0.006416]
    ]


    featurecheck1_df = pd.DataFrame(
        featurecheck1_data,

        columns=[
            "排名",
            "英文欄位",
            "特徵重要度"
        ]
    )


    featurecheck1_df["中文欄位"] = (
        featurecheck1_df["英文欄位"]
        .map(column_names)
        .fillna(
            featurecheck1_df["英文欄位"]
        )
    )


    featurecheck1_df = featurecheck1_df[
        [
            "排名",
            "中文欄位",
            "英文欄位",
            "特徵重要度"
        ]
    ]


    st.dataframe(
        featurecheck1_df.style.format({
            "特徵重要度":
                "{:.6f}"
        }),

        use_container_width=True
    )


    st.info(
        "📌 FeatureCheck_1 的結果顯示，"
        "daily_gaming_hours 的特徵重要度明顯高於其他欄位，"
        "代表平均每日遊戲時間是目前資料集中最具預測價值的變數。"
    )


    st.divider()


    # ========================================================
    # featureCheck_1 結論
    # ========================================================

    st.subheader(
        "💡 FeatureCheck_1 驗證結論"
    )


    st.markdown(
        """
        <div class="simple-card">

        <h3>🎮 主要發現</h3>

        <p>
        <b>daily_gaming_hours</b> 的重要度為
        <b>0.7579</b>，明顯高於其他特徵。
        </p>

        <p>
        其餘欄位的重要度大多集中在
        <b>0.006～0.007</b> 附近，
        與 daily_gaming_hours 存在非常明顯的差距。
        </p>

        <p>
        這表示在目前資料與模型條件下，
        <b>平均每日遊戲時間是解釋 addiction_level
        最主要的特徵。</b>
        </p>

        <p>
        其他欄位的重要度非常接近，
        顯示它們在目前資料中沒有呈現出明顯的額外預測能力。
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.warning(
        "⚠️ 資料特徵觀察："
        "大量欄位的重要度集中在非常接近的範圍，"
        "高度符合合成資料（Synthetic Data）中部分欄位由隨機方式產生的常見特徵。"
    )


    st.divider()


    # ========================================================
    # featureCheck_2
    # ========================================================

    st.subheader(
        "🧪 Feature Check 2｜特徵工程驗證"
    )


    st.markdown(
        """
        <div class="simple-card">

        <h3>📊 模型評估結果</h3>

        <p>
        FeatureCheck_2 進一步加入特徵工程，
        建立 <b>gaming_hours_sq</b> 等衍生特徵，
        再透過 XGBoost 重新觀察特徵重要度。
        </p>

        <p>
        模型 R² Score 為 <b>0.7488</b>，
        與 FeatureCheck_1 的 0.7486 非常接近。
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "🎯 FeatureCheck_1 R²",
        "0.7486"
    )


    c2.metric(
        "🎯 FeatureCheck_2 R²",
        "0.7488"
    )


    st.markdown(
        "🔗 **FeatureCheck_2 原始程式碼：** "
        "[featureCheck_2.py](https://github.com/fuwafuwa152123/gaming/blob/main/featureCheck_2.py)"
    )


    featurecheck2_data = [
        ["1", "gaming_hours_sq", 0.379735],
        ["2", "daily_gaming_hours", 0.378795],
        ["3", "weekend_gaming_hours", 0.007227],
        ["4", "income", 0.007206],
        ["5", "screen_time_total", 0.007056],
        ["6", "work_productivity", 0.007043],
        ["7", "bmi", 0.007039],
        ["8", "happiness_score", 0.006962],
        ["9", "relationship_satisfaction", 0.006950],
        ["10", "social_interaction_score", 0.006922],
        ["11", "sleep_hours", 0.006919],
        ["12", "exercise_hours", 0.006882],
        ["13", "streaming_hours", 0.006869],
        ["14", "anxiety_score", 0.006846],
        ["15", "online_friends", 0.006802],
        ["16", "depression_score", 0.006623],
        ["17", "eye_strain_score", 0.006359],
        ["18", "mobile_gaming_ratio", 0.006295],
        ["19", "back_pain_score", 0.006273],
        ["20", "multiplayer_ratio", 0.006267]
    ]


    featurecheck2_df = pd.DataFrame(
        featurecheck2_data,

        columns=[
            "排名",
            "英文欄位",
            "特徵重要度"
        ]
    )


    featurecheck2_df["中文欄位"] = (
        featurecheck2_df["英文欄位"]
        .map(column_names)
        .fillna(
            featurecheck2_df["英文欄位"]
        )
    )


    # gaming_hours_sq 中文名稱

    featurecheck2_df.loc[
        featurecheck2_df["英文欄位"] == "gaming_hours_sq",
        "中文欄位"
    ] = "遊戲時間平方"


    featurecheck2_df = featurecheck2_df[
        [
            "排名",
            "中文欄位",
            "英文欄位",
            "特徵重要度"
        ]
    ]


    st.dataframe(
        featurecheck2_df.style.format({
            "特徵重要度":
                "{:.6f}"
        }),

        use_container_width=True,

        height=650
    )


    st.info(
        "📌 FeatureCheck_2 顯示，"
        "gaming_hours_sq 與 daily_gaming_hours "
        "兩個特徵的重要度分別為 0.379735 與 0.378795，"
        "兩者合計約占 75.85%，再次顯示遊戲時間是模型最主要的資訊來源。"
    )


    st.divider()


    # ========================================================
    # FeatureCheck 1 vs FeatureCheck 2
    # ========================================================

    st.subheader(
        "📊 FeatureCheck_1 vs FeatureCheck_2"
    )


    comparison_df = pd.DataFrame({
        "驗證項目": [
            "R² Score",
            "最高重要度特徵",
            "最高特徵重要度",
            "主要發現"
        ],

        "FeatureCheck_1": [
            "0.7486",
            "daily_gaming_hours",
            "0.757947",
            "遊戲時間為主要特徵"
        ],

        "FeatureCheck_2": [
            "0.7488",
            "gaming_hours_sq",
            "0.379735",
            "遊戲時間及其平方特徵為主要特徵"
        ]
    })


    st.dataframe(
        comparison_df,

        use_container_width=True,

        hide_index=True
    )


    st.markdown(
        """
        <div class="simple-card">

        <h3>🎯 驗證結果</h3>

        <p>
        FeatureCheck_1 與 FeatureCheck_2 的 R² Score
        分別為 <b>0.7486</b> 與 <b>0.7488</b>，
        兩者差異非常小。
        </p>

        <p>
        FeatureCheck_2 加入 gaming_hours_sq 後，
        主要重要度仍集中在 daily_gaming_hours
        及其平方衍生特徵。
        </p>

        <p>
        這表示進行更多特徵工程後，
        模型的整體解釋能力並沒有出現明顯提升。
        因此本專題後續模型選擇
        <b>4 個核心欄位</b>作為主要分析與預測輸入，
        具有資料驗證上的依據。
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.success(
        "✅ 最終驗證結論："
        "目前資料分析結果顯示，daily_gaming_hours "
        "是最具代表性的核心特徵；加入衍生特徵後，"
        "R² 僅由 0.7486 提升至 0.7488，"
        "因此沒有必要為了增加欄位而加入大量低重要度特徵。"
    )


    st.divider()


    # ========================================================
    # 與目前專題模型的關係
    # ========================================================

    st.subheader(
        "🔗 驗證結果與本專題模型的關係"
    )


    st.markdown(
        """
        <div class="flow-box">

        <div class="flow-step">
            FeatureCheck_1
            <br>
            原始特徵重要度分析
        </div>

        <div class="flow-arrow">↓</div>

        <div class="flow-step">
            找出主要影響特徵
            <br>
            daily_gaming_hours
        </div>

        <div class="flow-arrow">↓</div>

        <div class="flow-step">
            FeatureCheck_2
            <br>
            加入特徵工程再次驗證
        </div>

        <div class="flow-arrow">↓</div>

        <div class="flow-step">
            R² 仍維持約 0.75
        </div>

        <div class="flow-arrow">↓</div>

        <div class="flow-step">
            最終聚焦 4 個核心分析欄位
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.info(
        "💡 這一頁的重點不是單純列出模型分數，"
        "而是說明「為什麼最後選擇目前 4 個核心欄位」："
        "透過兩次特徵重要度驗證後，確認遊戲時間是最主要的資訊來源，"
        "而大量其他欄位沒有提供明顯額外的預測價值。"
    )


# ============================================================
# 15. 壓力測試
# ============================================================

elif page == "🧪 壓力測試":

    st.title(
        "🧪 壓力測試"
    )


    st.subheader(
        "⚡ 模型連續預測與系統效能測試"
    )


    st.write(
        "本頁以已完成訓練的 XGBoost 模型進行大量輸入預測，"
        "觀察連續執行時的穩定性與平均處理時間。"
    )


    st.divider()


    st.markdown(
        """
        <div class="stress-image-title">
            📊 資料切分、模型訓練與壓力測試結果
        </div>
        """,
        unsafe_allow_html=True
    )


    st.image(
        "https://raw.githubusercontent.com/"
        "fuwafuwa152123/gaming/main/"
        "stress_test_result.png",

        use_container_width=True
    )


# ============================================================
# 16. 使用者分析
# ============================================================

elif page == "👤 使用者分析":

    st.title(
        "👤 使用者分析"
    )


    st.write(
        "請輸入 4 項主要遊戲行為與生活型態資料，"
        "系統將透過 XGBoost 分析遊戲成癮程度。"
    )


    st.info(
        "📌 輸入限制：平均每日遊戲時間不得大於平均每日總螢幕時間；"
        "遊戲時間＋運動時間＋睡眠時間不得超過 24 小時。"
    )


    st.divider()


    st.subheader(
        "📝 使用者資料"
    )


    with st.form(
        "user_input_form"
    ):

        c1, c2 = st.columns(2)


        with c1:

            gaming_hours = st.number_input(
                "🎮 平均每日遊戲時間（小時）",

                min_value=0.0,

                max_value=24.0,

                value=3.0,

                step=0.5,

                help=field_help[
                    "daily_gaming_hours"
                ]
            )


            exercise_hours = st.number_input(
                "🌱 平均每日運動時間（小時）",

                min_value=0.0,

                max_value=24.0,

                value=1.0,

                step=0.5,

                help=field_help[
                    "exercise_hours"
                ]
            )


        with c2:

            screen_time = st.number_input(
                "🖥️ 平均每日總螢幕時間（小時）",

                min_value=0.0,

                max_value=24.0,

                value=6.0,

                step=0.5,

                help=field_help[
                    "screen_time_total"
                ]
            )


            sleep_hours = st.number_input(
                "😴 平均每日睡眠時間（小時）",

                min_value=0.0,

                max_value=24.0,

                value=7.0,

                step=0.5,

                help=field_help[
                    "sleep_hours"
                ]
            )


        st.divider()


        submitted = st.form_submit_button(
            "🚀 開始分析",

            use_container_width=True,

            disabled=not model_ready
        )


    if not model_ready:

        st.warning(
            "⏳ XGBoost 模型目前尚未準備完成。"
        )


    elif submitted:

        # ====================================================
        # 輸入限制
        # ====================================================

        if gaming_hours > screen_time:

            st.error(
                "❌ 輸入不合理："
                "平均每日遊戲時間不能大於平均每日總螢幕時間。"
            )

            st.stop()


        total_hours = (
            gaming_hours
            + exercise_hours
            + sleep_hours
        )


        if total_hours > 24:

            st.error(
                f"❌ 輸入不合理："
                f"遊戲＋運動＋睡眠時間目前為 "
                f"{total_hours:.1f} 小時，"
                f"不能超過 24 小時。"
            )

            st.stop()


        # ====================================================
        # 使用者資料
        # ====================================================

        user_input = pd.DataFrame({
            "daily_gaming_hours": [
                gaming_hours
            ],

            "sleep_hours": [
                sleep_hours
            ],

            "exercise_hours": [
                exercise_hours
            ],

            "screen_time_total": [
                screen_time
            ]
        })


        # ====================================================
        # MinMaxScaler
        # ====================================================

        scaled_user = (
            scaler_X.transform(
                user_input
            )
        )


        # ====================================================
        # XGBoost 預測
        # ====================================================

        predicted_scaled = (
            xgb_model.predict(
                scaled_user
            )
        )


        # ====================================================
        # Inverse Transform
        # ====================================================

        raw_score = (
            scaler_y.inverse_transform(
                predicted_scaled.reshape(
                    -1,
                    1
                )
            )[0][0]
        )


        raw_score = float(
            np.clip(
                raw_score,
                0,
                10
            )
        )


        # ====================================================
        # 百分比
        # ====================================================

        percentage = float(
            np.clip(
                (raw_score / 10) * 100,
                0,
                100
            )
        )


        # ====================================================
        # 成癮等級
        # ====================================================

        level = addiction_level(
            raw_score
        )


        # ====================================================
        # Session State
        # ====================================================

        st.session_state.user_result = {
            "user_values":
                user_input.iloc[0].to_dict(),

            "addiction_score":
                raw_score,

            "percentage":
                percentage,

            "level":
                level
        }


    # ========================================================
    # 顯示結果
    # ========================================================

    if st.session_state.user_result is not None:

        result = (
            st.session_state.user_result
        )


        user_values = (
            result["user_values"]
        )


        addiction_score = (
            result["addiction_score"]
        )


        percentage = (
            result["percentage"]
        )


        level = (
            result["level"]
        )


        st.divider()


        left, right = st.columns(
            [1.2, 1]
        )


        # ====================================================
        # Gauge
        # ====================================================

        with left:

            st.subheader(
                "🎯 遊戲成癮程度"
            )


            level_color = (
                color_map[level]
            )


            gauge_html = f"""
            <!DOCTYPE html>

            <html>

            <head>

                <meta charset="UTF-8">

                <script src="
                https://cdn.amcharts.com/lib/4/core.js
                "></script>

                <script src="
                https://cdn.amcharts.com/lib/4/charts.js
                "></script>

                <style>

                    html, body {{
                        margin:0;
                        padding:0;
                        width:100%;
                        height:100%;
                        background:transparent;
                    }}

                    #chartdiv {{
                        width:100%;
                        height:330px;
                    }}

                    #result {{
                        text-align:center;
                        font-family:Arial,sans-serif;
                        font-size:25px;
                        font-weight:bold;
                        padding:8px;
                        color:#263238;
                    }}

                </style>

            </head>

            <body>

                <div id="chartdiv"></div>

                <div id="result">

                    遊戲成癮程度：
                    {addiction_score:.2f} / 10

                    <br>

                    <span style="
                        color:{level_color};
                    ">

                        {level}

                    </span>

                </div>


                <script>

                am4core.ready(function() {{

                    var chart =
                        am4core.create(
                            "chartdiv",
                            am4charts.GaugeChart
                        );


                    var axis =
                        chart.xAxes.push(
                            new am4charts.ValueAxis()
                        );


                    axis.min = 0;

                    axis.max = 10;

                    axis.strictMinMax = true;


                    axis.renderer.radius =
                        am4core.percent(90);


                    axis.renderer.innerRadius =
                        am4core.percent(65);


                    axis.renderer.line.strokeOpacity = 0;


                    axis.renderer.ticks.template.length = 10;


                    axis.renderer.labels.template.fontSize = 14;


                    var range1 =
                        axis.axisRanges.create();

                    range1.value = 0;

                    range1.endValue = 2;

                    range1.axisFill.fill =
                        am4core.color("#4CAF50");

                    range1.axisFill.fillOpacity = 0.8;


                    var range2 =
                        axis.axisRanges.create();

                    range2.value = 2;

                    range2.endValue = 4;

                    range2.axisFill.fill =
                        am4core.color("#8BC34A");

                    range2.axisFill.fillOpacity = 0.8;


                    var range3 =
                        axis.axisRanges.create();

                    range3.value = 4;

                    range3.endValue = 6;

                    range3.axisFill.fill =
                        am4core.color("#FFC107");

                    range3.axisFill.fillOpacity = 0.8;


                    var range4 =
                        axis.axisRanges.create();

                    range4.value = 6;

                    range4.endValue = 8;

                    range4.axisFill.fill =
                        am4core.color("#FF9800");

                    range4.axisFill.fillOpacity = 0.8;


                    var range5 =
                        axis.axisRanges.create();

                    range5.value = 8;

                    range5.endValue = 10;

                    range5.axisFill.fill =
                        am4core.color("#F44336");

                    range5.axisFill.fillOpacity = 0.8;


                    var hand =
                        chart.hands.push(
                            new am4charts.ClockHand()
                        );


                    hand.axis = axis;

                    hand.innerRadius =
                        am4core.percent(20);

                    hand.startWidth = 8;

                    hand.pin.disabled = false;

                    hand.value =
                        {addiction_score};

                }});

                </script>

            </body>

            </html>
            """


            components.html(
                gauge_html,
                height=430
            )


        # ====================================================
        # 分析結果
        # ====================================================

        with right:

            st.subheader(
                "📊 分析結果"
            )


            st.metric(
                "🎯 成癮程度分數",
                f"{addiction_score:.2f} / 10"
            )


            st.metric(
                "📊 成癮傾向",
                f"{percentage:.1f}%"
            )


            st.metric(
                "📌 成癮等級",
                level
            )


            if level == "低":

                st.success(
                    "目前遊戲成癮程度：低"
                )

            elif level == "偏低":

                st.success(
                    "目前遊戲成癮程度：偏低"
                )

            elif level == "中":

                st.warning(
                    "目前遊戲成癮程度：中"
                )

            elif level == "偏高":

                st.warning(
                    "目前遊戲成癮程度：偏高"
                )

            else:

                st.error(
                    "目前遊戲成癮程度：高"
                )


        st.divider()


        # ====================================================
        # 使用者輸入資料
        # ====================================================

        st.subheader(
            "📋 使用者輸入資料"
        )


        input_table = pd.DataFrame({
            "分析項目": [
                column_names[col]
                for col in feature_columns
            ],

            "英文欄位":
                feature_columns,

            "使用者輸入值": [
                user_values[col]
                for col in feature_columns
            ]
        })


        input_table.index = range(
            1,
            len(input_table) + 1
        )


        st.dataframe(
            input_table,
            use_container_width=True
        )


        st.divider()


        # ====================================================
        # 使用者 4 項主要特徵
        # ====================================================

        st.subheader(
            "📊 使用者 4 項主要特徵"
        )


        chart_user_df = pd.DataFrame({
            "分析項目": [
                column_names[col]
                for col in feature_columns
            ],

            "使用者輸入值": [
                user_values[col]
                for col in feature_columns
            ]
        })


        fig_user = px.bar(
            chart_user_df,

            x="使用者輸入值",

            y="分析項目",

            orientation="h",

            title="使用者 4 項主要特徵"
        )


        fig_user.update_layout(
            height=400,

            paper_bgcolor="#ffffff",

            plot_bgcolor="#ffffff",

            font=dict(
                color="#263238"
            )
        )


        st.plotly_chart(
            fig_user,

            use_container_width=True
        )


        st.info(
            "📌 本版本的使用者分析與資料分析均統一使用 4 個欄位："
            "平均每日遊戲時間、平均每日睡眠時間、"
            "平均每日運動時間、平均每日總螢幕時間。"
        )