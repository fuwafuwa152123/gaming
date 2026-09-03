import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor

from concurrent.futures import ThreadPoolExecutor
import threading


# ============================================================
# 1. Streamlit 網頁設定
# ============================================================

st.set_page_config(
    page_title="GAME ANALYTICS",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. Session State
# ============================================================

if "user_result" not in st.session_state:
    st.session_state.user_result = None


# ============================================================
# 3. CSV
# ============================================================

DATA_FILE = "gaming_part1_100k.csv"


# ============================================================
# 4. 14 個特徵
# ============================================================

feature_columns = [

    "age",
    "daily_gaming_hours",
    "sleep_hours",
    "exercise_hours",
    "stress_level",
    "anxiety_score",
    "social_interaction_score",
    "online_friends",
    "esports_interest",
    "microtransactions_spending",
    "loneliness_score",
    "screen_time_total",
    "eye_strain_score",
    "back_pain_score"

]

target_column = "addiction_level"


# ============================================================
# 5. 中文名稱
# ============================================================

column_names = {

    "age":
        "年齡",

    "daily_gaming_hours":
        "平均每日遊戲時間",

    "sleep_hours":
        "平均睡眠時間",

    "exercise_hours":
        "每日運動時間",

    "stress_level":
        "生活壓力評分",

    "anxiety_score":
        "焦慮評估得分",

    "social_interaction_score":
        "社交活動指數",

    "online_friends":
        "線上網友人數",

    "esports_interest":
        "對參與電競的興趣",

    "microtransactions_spending":
        "每月遊戲內消費",

    "loneliness_score":
        "孤獨指數",

    "screen_time_total":
        "每日總螢幕時間",

    "eye_strain_score":
        "眼睛疲勞程度",

    "back_pain_score":
        "背痛嚴重程度",

    "addiction_level":
        "遊戲成癮程度"

}


# ============================================================
# 6. 欄位說明
# ============================================================

field_help = {

    "age":
        "成年人年齡，範圍 18～100 歲。",

    "daily_gaming_hours":
        "平均每天遊戲時間，單位為小時。",

    "sleep_hours":
        "平均每天睡眠時間，單位為小時。",

    "exercise_hours":
        "平均每日運動時間，單位為小時。",

    "stress_level":
        "生活壓力評分，0～10 分。",

    "anxiety_score":
        "焦慮評估得分，0～10 分。",

    "social_interaction_score":
        "社交活動指數，0～10 分。",

    "online_friends":
        "線上遊戲或社群中的網友人數。",

    "esports_interest":
        "對參與電競活動的興趣程度，0～10 分。",

    "microtransactions_spending":
        "每月遊戲內購買金額。",

    "loneliness_score":
        "孤獨指數，0～10 分。",

    "screen_time_total":
        "平均每日總螢幕使用時間，單位為小時。",

    "eye_strain_score":
        "眼睛疲勞程度，0～10 分。",

    "back_pain_score":
        "背痛嚴重程度，0～10 分。"

}


# ============================================================
# 7. 讀取資料
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_FILE)


# ============================================================
# 8. 先載入資料
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error("❌ CSV 讀取失敗")
    st.error(str(e))
    st.stop()


# ============================================================
# 9. 欄位檢查
# ============================================================

required_columns = feature_columns + [target_column]

missing_columns = [

    col
    for col in required_columns
    if col not in df.columns

]

if missing_columns:

    st.error("❌ CSV 缺少必要欄位")

    st.write("缺少欄位：")
    st.write(missing_columns)

    st.write("目前 CSV 欄位：")
    st.write(list(df.columns))

    st.stop()


# ============================================================
# 10. 分析資料
# ============================================================

analysis_df = df[required_columns].copy()

analysis_df = analysis_df.dropna()

analysis_df = analysis_df.astype(float)


# ============================================================
# 11. 關聯性資料
# ============================================================

relation_values = {

    "daily_gaming_hours":
        0.866051,

    "screen_time_total":
        0.578582,

    "exercise_hours":
        0.064452,

    "sleep_hours":
        0.038925,

    "age":
        0.004475,

    "social_interaction_score":
        0.003425,

    "stress_level":
        0.003324,

    "online_friends":
        0.002733,

    "eye_strain_score":
        0.001650,

    "esports_interest":
        0.001480,

    "back_pain_score":
        0.001399,

    "anxiety_score":
        0.001299,

    "loneliness_score":
        0.001046,

    "microtransactions_spending":
        0.000152

}


relation_df = pd.DataFrame({

    "變數":
        [
            column_names[col]
            for col in relation_values.keys()
        ],

    "英文欄位":
        list(relation_values.keys()),

    "關聯程度":
        list(relation_values.values())

})

relation_df.index = range(
    1,
    len(relation_df) + 1
)


# ============================================================
# 12. 成癮程度
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


# ============================================================
# 13. 成癮顏色
# ============================================================

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
# 14. XGBoost 模型
#
# ⭐ 模型本身只訓練一次
# ⭐ 使用 Streamlit cache 保留
# ============================================================

@st.cache_resource(show_spinner=False)
def train_xgboost_model(

    data_signature

):

    X = analysis_df[feature_columns].copy()

    y = analysis_df[[target_column]].copy()


    # --------------------------------------------------------
    # Train / Test
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.33,

        random_state=101

    )


    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    scaler_X = MinMaxScaler(

        feature_range=(0.2, 0.8)

    )

    scaler_y = MinMaxScaler(

        feature_range=(0.2, 0.8)

    )


    X_train_scaled = scaler_X.fit_transform(

        X_train

    )

    X_test_scaled = scaler_X.transform(

        X_test

    )


    y_train_scaled = scaler_y.fit_transform(

        y_train

    ).ravel()


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    model = XGBRegressor(

        n_estimators=100,

        learning_rate=0.01,

        max_depth=3,

        min_child_weight=1,

        gamma=0,

        subsample=1.0,

        colsample_bytree=1.0,

        colsample_bylevel=1.0,

        colsample_bynode=1.0,

        reg_alpha=0,

        reg_lambda=0.5,

        scale_pos_weight=1,

        booster="gbtree",

        tree_method="auto",

        n_jobs=-1,

        random_state=42,

        verbosity=0,

        objective="reg:squarederror"

    )


    # --------------------------------------------------------
    # 訓練
    # --------------------------------------------------------

    model.fit(

        X_train_scaled,

        y_train_scaled

    )


    # --------------------------------------------------------
    # 評估
    # --------------------------------------------------------

    predicted_scaled = model.predict(

        X_test_scaled

    )


    predicted = scaler_y.inverse_transform(

        predicted_scaled.reshape(-1, 1)

    ).ravel()


    actual = y_test.values.ravel()


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
# 15. 背景模型狀態
# ============================================================

if "model_executor" not in st.session_state:

    st.session_state.model_executor = (
        ThreadPoolExecutor(max_workers=1)
    )


if "model_future" not in st.session_state:

    data_signature = (

        len(analysis_df),

        tuple(feature_columns),

        target_column

    )

    st.session_state.model_future = (

        st.session_state.model_executor.submit(

            train_xgboost_model,

            data_signature

        )

    )


# ============================================================
# 16. 檢查模型是否完成
# ============================================================

model_ready = False

model_package = None


future = st.session_state.model_future


if future.done():

    try:

        model_package = future.result()

        model_ready = True

    except Exception as e:

        st.error("❌ AI 模型準備失敗")
        st.error(str(e))
        st.stop()


# ============================================================
# 17. 取得模型
# ============================================================

if model_ready:

    xgb_model = model_package["model"]

    scaler_X = model_package["scaler_X"]

    scaler_y = model_package["scaler_y"]

    model_rmse = model_package["rmse"]

    model_mae = model_package["mae"]

    model_r2 = model_package["r2"]

else:

    xgb_model = None

    scaler_X = None

    scaler_y = None

    model_rmse = None

    model_mae = None

    model_r2 = None


# ============================================================
# 18. Sidebar
# ============================================================

with st.sidebar:

    st.title("🎮 GAME ANALYTICS")

    st.caption(
        "Gaming & Mental Health"
    )

    st.divider()


    page = st.radio(

        "功能選單",

        [

            "🏠 Dashboard",

            "📊 資料分析",

            "🫧 關係分析",

            "👤 使用者分析",

            "ℹ️ 專題說明"

        ]

    )


    st.divider()

    st.caption("第三組專題")

    st.caption(
        "Data Analysis & Machine Learning"
    )


# ============================================================
# 19. Dashboard
# ============================================================

if page == "🏠 Dashboard":

    st.title("🎮 GAME ANALYTICS")

    st.subheader(
        "遊戲行為、生活型態與遊戲成癮程度分析平台"
    )

    st.write(

        "本專題以遊戲行為、生活型態及身心狀態資料為基礎，"
        "探索不同因素與遊戲成癮程度之間的關聯，"
        "並透過資料視覺化與機器學習建立分析流程。"

    )

    st.divider()


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "📊 分析資料",
            "271,807 筆"
        )


    with c2:

        st.metric(
            "🎯 分析特徵",
            "14 個"
        )


    with c3:

        st.metric(

            "🎮 平均遊戲時間",

            f"{analysis_df['daily_gaming_hours'].mean():.1f} 小時"

        )


    with c4:

        st.metric(

            "🧠 平均成癮程度",

            f"{analysis_df['addiction_level'].mean():.2f} / 10"

        )


    st.divider()


    st.subheader("🔎 從資料到分析結果")


    process_col1, process_col2, process_col3 = st.columns(3)


    with process_col1:

        st.info(

            """
### ① 資料準備

原始資料

↓

資料清洗

↓

建立乾淨資料集
"""

        )


    with process_col2:

        st.info(

            """
### ② 探索關聯

14 個特徵

↓

關聯性分析

↓

資料視覺化
"""

        )


    with process_col3:

        st.info(

            """
### ③ 預測分析

使用者資料

↓

XGBoost

↓

成癮程度預測
"""

        )


    st.divider()


    st.subheader(
        "📌 目前選出的 14 個分析特徵"
    )


    feature_display = pd.DataFrame({

        "中文欄位":
            [
                column_names[col]
                for col in feature_columns
            ],

        "英文欄位":
            feature_columns

    })


    feature_display.index = range(
        1,
        len(feature_display) + 1
    )


    st.dataframe(
        feature_display,
        use_container_width=True
    )


# ============================================================
# 20. 資料分析
# ============================================================

elif page == "📊 資料分析":

    st.title("📊 資料分析")

    st.write(
        "透過散佈圖觀察每一個分析變數與「遊戲成癮程度」之間的關係。"
    )

    st.divider()


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "📊 資料筆數",
            "271,807"
        )


    with c2:

        st.metric(
            "📋 X 變數",
            "14"
        )


    with c3:

        st.metric(
            "🎯 Y 變數",
            "遊戲成癮程度"
        )


    st.divider()


    st.subheader(
        "📈 X 變數 × 遊戲成癮程度"
    )


    x_col = st.selectbox(

        "選擇要分析的 X 變數",

        feature_columns,

        format_func=lambda x:
            column_names[x]

    )


    st.caption(

        f"X 軸：{column_names[x_col]}　｜　"
        f"Y 軸：{column_names[target_column]}"

    )


    fig = px.scatter(

        analysis_df,

        x=x_col,

        y=target_column,

        labels={

            x_col:
                column_names[x_col],

            target_column:
                column_names[target_column]

        },

        title=(

            f"{column_names[x_col]} × "
            f"{column_names[target_column]}"

        )

    )


    fig.update_traces(

        marker=dict(

            size=7,

            opacity=0.55

        )

    )


    fig.update_layout(

        height=600,

        plot_bgcolor="white",

        paper_bgcolor="white"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )


    st.info(

        "📌 散佈圖中的每一個點代表一筆資料，可觀察 X 變數與遊戲成癮程度之間的分布與趨勢。"

    )


    st.divider()


    st.subheader(
        "📊 14 個特徵與遊戲成癮程度關聯性"
    )


    display_relation = relation_df.copy()


    display_relation["關聯程度"] = (

        display_relation["關聯程度"]

        .map(
            lambda x:
                f"{x:.6f}"
        )

    )


    st.dataframe(
        display_relation,
        use_container_width=True
    )


# ============================================================
# 21. 關係分析
# ============================================================

elif page == "🫧 關係分析":

    st.title("🫧 變數關係分析")


    st.write(

        "自由選擇兩個分析變數，觀察兩個變數之間的分布，"
        "並透過泡泡大小與顏色呈現遊戲成癮程度。"

    )


    st.divider()


    col1, col2 = st.columns(2)


    with col1:

        x_col = st.selectbox(

            "🔵 X 軸變數",

            feature_columns,

            index=1,

            format_func=lambda x:
                column_names[x],

            key="bubble_x"

        )


    with col2:

        available_y = [

            col
            for col in feature_columns
            if col != x_col

        ]


        y_col = st.selectbox(

            "🟣 Y 軸變數",

            available_y,

            format_func=lambda x:
                column_names[x],

            key="bubble_y"

        )


    st.divider()


    bubble_df = analysis_df.copy()


    bubble_df["成癮等級"] = (

        bubble_df[target_column]

        .apply(addiction_level)

    )


    fig = px.scatter(

        bubble_df,

        x=x_col,

        y=y_col,

        size=target_column,

        color="成癮等級",

        size_max=32,

        opacity=0.70,

        color_discrete_map=color_map,

        hover_data={

            x_col:
                True,

            y_col:
                True,

            target_column:
                ":.2f",

            "成癮等級":
                True

        },

        labels={

            x_col:
                column_names[x_col],

            y_col:
                column_names[y_col],

            target_column:
                "遊戲成癮程度",

            "成癮等級":
                "成癮等級"

        },

        title=(

            f"{column_names[x_col]} × "
            f"{column_names[y_col]}"

        )

    )


    fig.update_traces(

        marker=dict(

            line=dict(
                width=0.5
            )

        )

    )


    fig.update_layout(

        height=650,

        plot_bgcolor="white",

        paper_bgcolor="white",

        legend_title_text="遊戲成癮程度"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )


    st.info(

        """
🫧 圖表判讀方式：

X 軸與 Y 軸可以從 14 個分析變數中自由選擇。

泡泡大小：代表遊戲成癮程度，越大代表分數越高。

泡泡顏色：代表遊戲成癮程度五個等級。
"""

    )


    st.divider()


    st.subheader(
        "🎯 遊戲成癮程度分級"
    )


    level_df = pd.DataFrame({

        "等級":
            [
                "低",
                "偏低",
                "中",
                "偏高",
                "高"
            ],

        "分數範圍":
            [
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
# 22. 使用者分析
# ============================================================

elif page == "👤 使用者分析":

    st.title("👤 使用者分析")


    st.write(

        "請輸入 4 項主要遊戲行為與生活型態資料，"
        "系統將搭配資料集中的其他特徵，"
        "使用 XGBoost 模型分析遊戲成癮程度。"

    )


    st.info(

        "💡 本頁只需要輸入 4 項主要特徵。"
        "模型仍維持原本 14 個特徵，"
        "未輸入的 10 個特徵會使用資料集 Median 補入。"

    )


    st.divider()


    main_features = [

        "daily_gaming_hours",

        "screen_time_total",

        "exercise_hours",

        "sleep_hours"

    ]


    st.subheader("📝 使用者資料")


    with st.form("user_input_form"):


        st.subheader("🎮 遊戲行為")


        c1, c2 = st.columns(2)


        with c1:

            gaming_hours = st.number_input(

                "平均每日遊戲時間（小時）",

                min_value=0.0,

                max_value=20.0,

                value=3.0,

                step=0.5,

                help=field_help["daily_gaming_hours"]

            )


        with c2:

            screen_time = st.number_input(

                "每日總螢幕時間（小時）",

                min_value=0.0,

                max_value=20.0,

                value=6.0,

                step=0.5,

                help=field_help["screen_time_total"]

            )


        st.divider()


        st.subheader("🌱 生活型態")


        c1, c2 = st.columns(2)


        with c1:

            exercise_hours = st.number_input(

                "每日運動時間（小時）",

                min_value=0.0,

                max_value=20.0,

                value=1.0,

                step=0.5,

                help=field_help["exercise_hours"]

            )


        with c2:

            sleep_hours = st.number_input(

                "平均睡眠時間（小時）",

                min_value=0.0,

                max_value=20.0,

                value=7.0,

                step=0.5,

                help=field_help["sleep_hours"]

            )


        st.divider()


        # ====================================================
        # ⭐ 模型完成後才解鎖
        # ====================================================

        submitted = st.form_submit_button(

            "🔍 開始分析",

            use_container_width=True,

            disabled=not model_ready

        )


    # ========================================================
    # 如果模型尚未完成
    # ========================================================

    if not model_ready:

        # 不顯示「模型正在訓練」
        # 只讓按鈕維持不可使用

        st.caption(
            "請稍候即可開始分析。"
        )

        st.stop()


    # ========================================================
    # 使用者送出
    # ========================================================

    if submitted:

        # ----------------------------------------------------
        # 1. Median
        # ----------------------------------------------------

        user_values = (

            analysis_df[feature_columns]

            .median()

            .to_dict()

        )


        # ----------------------------------------------------
        # 2. 覆蓋 4 個使用者輸入
        # ----------------------------------------------------

        user_values["daily_gaming_hours"] = gaming_hours

        user_values["screen_time_total"] = screen_time

        user_values["exercise_hours"] = exercise_hours

        user_values["sleep_hours"] = sleep_hours


        # ----------------------------------------------------
        # 3. 完整 14 X
        # ----------------------------------------------------

        user_input = pd.DataFrame(

            [user_values],

            columns=feature_columns

        )


        # ----------------------------------------------------
        # 4. Scaling
        # ----------------------------------------------------

        scaled_user = scaler_X.transform(

            user_input

        )


        # ----------------------------------------------------
        # 5. 直接預測
        #
        # ⭐ 這裡絕對不重新訓練
        # ----------------------------------------------------

        predicted_scaled = xgb_model.predict(

            scaled_user

        )


        # ----------------------------------------------------
        # 6. 還原
        # ----------------------------------------------------

        raw_score = scaler_y.inverse_transform(

            predicted_scaled.reshape(-1, 1)

        )[0][0]


        raw_score = float(

            np.clip(

                raw_score,

                0,

                10

            )

        )


        # ----------------------------------------------------
        # 7. 百分比
        # ----------------------------------------------------

        percentage = float(

            np.clip(

                (raw_score / 10) * 100,

                0,

                100

            )

        )


        # ----------------------------------------------------
        # 8. 等級
        # ----------------------------------------------------

        level = addiction_level(

            raw_score

        )


        # ----------------------------------------------------
        # 9. 儲存
        # ----------------------------------------------------

        st.session_state.user_result = {

            "user_values":
                user_values,

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

        result = st.session_state.user_result


        user_values = result["user_values"]

        addiction_score = result["addiction_score"]

        percentage = result["percentage"]

        level = result["level"]


        st.divider()


        left, right = st.columns([1.2, 1])


        # ====================================================
        # Gauge
        # ====================================================

        with left:

            st.subheader(
                "🎯 遊戲成癮程度"
            )


            level_color = color_map[level]


            gauge_html = f"""

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<script src="https://cdn.amcharts.com/lib/4/core.js"></script>

<script src="https://cdn.amcharts.com/lib/4/charts.js"></script>

<style>

html, body {{

    margin: 0;

    padding: 0;

    width: 100%;

    height: 100%;

    background: transparent;

}}

#chartdiv {{

    width: 100%;

    height: 330px;

}}

#result {{

    text-align: center;

    font-family: Arial, sans-serif;

    font-size: 25px;

    font-weight: bold;

    padding: 8px;

}}

</style>

</head>

<body>

<div id="chartdiv"></div>

<div id="result">

遊戲成癮程度：

{addiction_score:.2f} / 10

<br>

<span style="color:{level_color};">

{level}

</span>

</div>

<script>

am4core.ready(function() {{

    var chart = am4core.create(

        "chartdiv",

        am4charts.GaugeChart

    );

    var axis = chart.xAxes.push(

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

    var range1 = axis.axisRanges.create();

    range1.value = 0;

    range1.endValue = 2;

    range1.axisFill.fill =

        am4core.color("#4CAF50");

    range1.axisFill.fillOpacity = 0.8;

    var range2 = axis.axisRanges.create();

    range2.value = 2;

    range2.endValue = 4;

    range2.axisFill.fill =

        am4core.color("#8BC34A");

    range2.axisFill.fillOpacity = 0.8;

    var range3 = axis.axisRanges.create();

    range3.value = 4;

    range3.endValue = 6;

    range3.axisFill.fill =

        am4core.color("#FFC107");

    range3.axisFill.fillOpacity = 0.8;

    var range4 = axis.axisRanges.create();

    range4.value = 6;

    range4.endValue = 8;

    range4.axisFill.fill =

        am4core.color("#FF9800");

    range4.axisFill.fillOpacity = 0.8;

    var range5 = axis.axisRanges.create();

    range5.value = 8;

    range5.endValue = 10;

    range5.axisFill.fill =

        am4core.color("#F44336");

    range5.axisFill.fillOpacity = 0.8;

    var hand = chart.hands.push(

        new am4charts.ClockHand()

    );

    hand.axis = axis;

    hand.innerRadius =

        am4core.percent(20);

    hand.startWidth = 8;

    hand.pin.disabled = false;

    hand.value = {addiction_score};

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
        # 結果
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


        # ====================================================
        # 使用者輸入
        # ====================================================

        st.divider()


        st.subheader(
            "📋 使用者輸入資料"
        )


        input_table = pd.DataFrame({

            "分析項目":

                [

                    column_names[col]

                    for col in main_features

                ],

            "英文欄位":

                main_features,

            "使用者輸入值":

                [

                    user_values[col]

                    for col in main_features

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


        # ====================================================
        # 完整 14 特徵
        # ====================================================

        with st.expander(
            "🔍 查看模型實際使用的完整 14 項特徵"
        ):


            model_input_table = pd.DataFrame({

                "分析項目":

                    [

                        column_names[col]

                        for col in feature_columns

                    ],

                "英文欄位":

                    feature_columns,

                "模型輸入值":

                    [

                        user_values[col]

                        for col in feature_columns

                    ],

                "資料來源":

                    [

                        "使用者輸入"

                        if col in main_features

                        else "資料集 Median"

                        for col in feature_columns

                    ]

            })


            model_input_table.index = range(

                1,

                len(model_input_table) + 1

            )


            st.dataframe(

                model_input_table,

                use_container_width=True

            )


        # ====================================================
        # 主要特徵
        # ====================================================

        st.divider()


        st.subheader(
            "📊 使用者主要特徵概覽"
        )


        chart_user_df = pd.DataFrame({

            "分析項目":

                [

                    column_names[col]

                    for col in main_features

                ],

            "使用者輸入值":

                [

                    user_values[col]

                    for col in main_features

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

            plot_bgcolor="white",

            paper_bgcolor="white"

        )


        st.plotly_chart(

            fig_user,

            use_container_width=True

        )


        st.info(

            "📌 本頁以 4 項主要特徵作為使用者輸入，"
            "其餘 10 項特徵則以分析資料的中位數帶入原本的 14-X XGBoost 模型。"

        )


# ============================================================
# 23. 專題說明
# ============================================================

elif page == "ℹ️ 專題說明":

    st.title("ℹ️ 專題說明")


    st.subheader(
        "🎮 遊戲與心理健康 × 遊戲成癮分析"
    )


    st.write(

        """
本專題以 Kaggle: Gaming & Mental Health 資料集為基礎，
從原始資料進行資料清洗、特徵篩選與關聯性分析，
最後建立遊戲成癮程度分析與預測系統。
"""

    )


    st.divider()


    st.subheader("📚 資料來源")


    source_df = pd.DataFrame({

        "項目":

            [

                "資料集",
                "原始欄位",
                "原始資料筆數",
                "分析特徵",
                "目標變數"

            ],

        "內容":

            [

                "Kaggle : Gaming & Mental Health",
                "39 欄",
                "10,000,000 筆",
                "14 個",
                "addiction_level"

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


    st.subheader("🧹 1. 第一次資料清洗")


    st.write(
        "原始資料：**10,000,000 筆**"
    )


    st.markdown(

        """
### 清洗條件

**① 刪除未成年人的不合理收入**

`age < 18 且 income > 15,000`

**② 刪除不合常理的家長監控**

`age >= 18 且 parental_supervision > 3`

**③ 刪除不合理的時間分配資料**

- `screen_time_total > 24`
- `daily_gaming_hours > 24`
- `exercise_hours > 24`
- 每日遊戲時間 + 睡眠時間 + 運動時間 > 24 小時
- 螢幕使用時間 + 睡眠時間 + 運動時間 > 24 小時
- `weekend_gaming_hours > 40`

**④ 刪除太小開始打電動的資料**

`age - years_gaming >= 7`

**⑤ 刪除遊戲年資大於年齡**

`years_gaming > age`

**⑥ 刪除 BMI 離群值**

`bmi < 15 或 bmi > 40`
"""

    )


    st.metric(
        "第一次清洗後",
        "271,808 筆"
    )


    st.divider()


    st.subheader("🧹 2. 第二次資料清洗")


    st.write(

        """
第二次清洗進一步調整時間相關欄位的合理範圍，
將以下欄位的上限設定為 **20 小時**：
"""

    )


    st.markdown(

        """
- 螢幕使用時間
- 每日遊戲時間
- 運動時間
"""

    )


    st.metric(
        "第二次清洗後",
        "271,807 筆"
    )


    st.divider()


    st.subheader(
        "🧩 3. 特徵工程與關聯性分析"
    )


    st.write(

        """
清洗完成後，從資料集中分析各欄位與
`addiction_level` 的關聯程度，
最終選出 14 個特徵作為網站分析與模型輸入。
"""

    )


    st.dataframe(

        relation_df,

        use_container_width=True

    )


    st.divider()


    st.subheader(
        "🎯 4. 最終選出的 14 個特徵"
    )


    feature_final_df = pd.DataFrame({

        "中文名稱":

            [

                column_names[col]

                for col in feature_columns

            ],

        "英文欄位":

            feature_columns

    })


    feature_final_df.index = range(
        1,
        len(feature_final_df) + 1
    )


    st.dataframe(

        feature_final_df,

        use_container_width=True

    )


    st.divider()


    st.subheader("✂️ 5. 資料分割")


    split_df = pd.DataFrame({

        "資料集":

            [

                "第 1 份",
                "第 2 份",
                "第 3 份",
                "總計"

            ],

        "筆數":

            [

                "100,000",
                "100,000",
                "71,807",
                "271,807"

            ]

    })


    split_df.index = range(
        1,
        len(split_df) + 1
    )


    st.dataframe(

        split_df,

        use_container_width=True

    )


    st.divider()


    st.subheader("🏗️ 6. 專題製作流程")


    st.code(

        """
原始資料
39 個欄位
10,000,000 筆
        │
        ▼
🧹 資料清洗
        │
        ▼
271,807 筆
        │
        ▼
🧩 特徵工程
        │
        ▼
🔎 關聯性分析
        │
        ▼
🎯 14 個分析特徵
        │
        ├───────────────┐
        ▼               ▼
📊 資料視覺化       🫧 關係分析
        │               │
        └───────┬───────┘
                ▼
        👤 使用者輸入
                │
                ▼
        4 個主要特徵輸入
                │
                ▼
        其他 10 個特徵
        Median 補值
                │
                ▼
        完整 14 個 X
                │
                ▼
        🤖 XGBoost
                │
                ▼
        🎯 成癮程度預測
                │
                ▼
        📊 分數 / 百分比 / 等級
""",

        language="text"

    )


    st.divider()


    st.subheader("📊 資料分割比例")


    split_chart_df = pd.DataFrame({

        "資料集":
            [
                "第 1 份",
                "第 2 份",
                "第 3 份"
            ],

        "筆數":
            [
                100000,
                100000,
                71807
            ]

    })


    fig_split = px.pie(

        split_chart_df,

        names="資料集",

        values="筆數",

        hole=0.45,

        title="271,807 筆清洗資料分割比例"

    )


    fig_split.update_layout(
        height=500
    )


    st.plotly_chart(

        fig_split,

        use_container_width=True

    )


    st.divider()


    st.subheader("🛠️ 7. 使用技術")


    tech1, tech2, tech3, tech4 = st.columns(4)


    with tech1:

        st.info(
            "🐍 Python\n\n程式開發"
        )


    with tech2:

        st.info(
            "📊 Pandas\n\n資料處理"
        )


    with tech3:

        st.info(
            "🎨 Streamlit\n\n網站介面"
        )


    with tech4:

        st.info(
            "📈 Plotly\n\n資料視覺化"
        )


    st.divider()


    st.subheader("🤖 機器學習")


    st.info(
        "XGBoost Regressor：使用 14 個特徵預測遊戲成癮程度。"
    )


    st.divider()


    st.subheader(
        "🤖 XGBoost 預測模型"
    )


    st.write(

        "本專題使用 XGBoost Regressor 建立遊戲成癮程度預測模型。"
        "模型使用 14 項生活型態與遊戲相關特徵作為輸入，"
        "經過 MinMaxScaler 資料標準化後進行預測，"
        "最後將結果還原為 0～10 分。"

    )


    st.code(

        """
14 個分析特徵
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
""",

        language="text"

    )


    # ========================================================
    # ⭐ 修改後的說明文字
    # ========================================================

    st.write(

        "網站啟動後會在背景準備 XGBoost 預測模型，"
        "模型完成後即可直接進行預測。"
        "模型會透過 Streamlit Cache 保留，"
        "因此使用者輸入資料時不需要重新訓練模型。"
        "使用者只需輸入 4 項主要資料，"
        "其餘 10 項特徵則以資料集 Median 補入，"
        "再直接使用已訓練完成的 14-X XGBoost 模型進行預測。"

    )


    st.subheader("📈 模型評估")


    if (

        model_rmse is not None
        and model_mae is not None
        and model_r2 is not None

    ):

        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "RMSE",
                f"{model_rmse:.4f}"
            )


        with c2:

            st.metric(
                "MAE",
                f"{model_mae:.4f}"
            )


        with c3:

            st.metric(
                "R²",
                f"{model_r2:.4f}"
            )


    else:

        st.info(
            "模型準備完成後會顯示評估結果。"
        )


    st.subheader("⚙️ 模型設定")


    model_setting_df = pd.DataFrame({

        "參數":

            [

                "Model",
                "X 特徵數",
                "使用者輸入數",
                "未輸入特徵處理",
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_child_weight",
                "reg_lambda",
                "test_size",
                "random_state"

            ],

        "設定":

            [

                "XGBRegressor",
                "14",
                "4",
                "Median",
                "100",
                "0.01",
                "3",
                "1",
                "0.5",
                "0.33",
                "101 / 42"

            ]

    })


    model_setting_df.index = range(
        1,
        len(model_setting_df) + 1
    )


    st.dataframe(

        model_setting_df,

        use_container_width=True

    )
