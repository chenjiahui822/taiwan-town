"""
台湾小镇业态评估系统 - 个性化主题版
功能：个性化主题 + 完整统计分析 + 业态PK + 匹配测试 + 自定义推荐 + 排行榜
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, f_oneway, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="台湾小镇业态评估系统",
    page_icon="🎨",
    layout="wide"
)

# =========================================================
# 初始化 session_state
# =========================================================
# 主题设置
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = '浅色'
if 'bg_color' not in st.session_state:
    st.session_state.bg_color = '#f0f2f6'
if 'card_color' not in st.session_state:
    st.session_state.card_color = '#ffffff'
if 'accent_color' not in st.session_state:
    st.session_state.accent_color = '#667eea'
if 'text_color' not in st.session_state:
    st.session_state.text_color = '#333333'
if 'card_radius' not in st.session_state:
    st.session_state.card_radius = 12

# 数据设置
if 'df' not in st.session_state:
    st.session_state.df = None
if 'shop_types' not in st.session_state:
    st.session_state.shop_types = []
if 'quiz_result' not in st.session_state:
    st.session_state.quiz_result = None
if 'user_recommendation' not in st.session_state:
    st.session_state.user_recommendation = None


# =========================================================
# 应用自定义CSS（简洁版，无突兀色块）
# =========================================================
def apply_custom_theme():
    if st.session_state.theme_mode == '深色':
        bg = '#1e1e2e'
        card_bg = '#2d2d3f'
        text = '#e0e0e0'
        accent = '#c084fc'
        sidebar_bg = '#1e1e2e'
    else:
        bg = st.session_state.bg_color
        card_bg = st.session_state.card_color
        text = st.session_state.text_color
        accent = st.session_state.accent_color
        sidebar_bg = bg

    custom_css = f"""
    <style>
        /* 整体背景 */
        .stApp {{
            background-color: {bg} !important;
        }}

        /* 侧边栏背景 - 与主背景一致 */
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            border-right: 1px solid rgba(128,128,128,0.1) !important;
        }}

        /* 侧边栏内容 */
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stRadio,
        [data-testid="stSidebar"] .stCheckbox,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stButton,
        [data-testid="stSidebar"] label {{
            color: {text} !important;
        }}

        /* 主区域背景透明 */
        .main .block-container {{
            background-color: transparent !important;
        }}

        /* 标题颜色 */
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: {accent} !important;
        }}

        /* 普通文字 */
        p, li, span, div, .stMarkdown {{
            color: {text} !important;
        }}

        /* 按钮样式 */
        .stButton button {{
            background-color: {accent} !important;
            color: white !important;
            border-radius: 20px !important;
            border: none !important;
            transition: opacity 0.2s !important;
        }}
        .stButton button:hover {{
            opacity: 0.85 !important;
        }}

        /* 指标卡片 */
        [data-testid="stMetricValue"] {{
            color: {accent} !important;
        }}

        /* 选项卡样式 */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: transparent !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: rgba(128,128,128,0.05) !important;
            border-radius: 20px !important;
            padding: 8px 20px !important;
            color: {text} !important;
            border: none !important;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {accent} !important;
            color: white !important;
        }}

        /* 滑块 */
        .stSlider [data-baseweb="slider"] div {{
            background-color: {accent} !important;
        }}

        /* 选择框 */
        .stSelectbox div[data-baseweb="select"] {{
            background-color: rgba(128,128,128,0.05) !important;
            border-radius: 10px !important;
        }}

        /* 分隔线 */
        hr {{
            border-color: rgba(128,128,128,0.2) !important;
        }}

        /* 信息框透明化 */
        .stAlert {{
            background-color: rgba(128,128,128,0.05) !important;
            border-left: 3px solid {accent} !important;
        }}

        /* 表格 */
        .stDataFrame {{
            background-color: transparent !important;
        }}
        .stDataFrame table {{
            background-color: transparent !important;
        }}
        .stDataFrame th {{
            background-color: rgba(128,128,128,0.05) !important;
        }}

        /* 多选框 */
        .stMultiSelect div[data-baseweb="select"] {{
            background-color: rgba(128,128,128,0.05) !important;
        }}

        /* 数字输入框 */
        .stNumberInput input {{
            background-color: rgba(128,128,128,0.05) !important;
            border: none !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# =========================================================
# 数据生成函数
# =========================================================
def generate_sample_data():
    np.random.seed(42)
    n = 200
    shop_types = ['网红茶饮店', '亲子手作工坊', '两岸文创店', '汉服旅拍馆',
                  '台湾小吃摊', '轻食简餐店', '伴手礼店', 'VR体验馆', '民俗手作店', '独立书店']
    age = np.random.randint(18, 65, n)
    has_child = np.random.choice([0, 1], n, p=[0.6, 0.4])
    visit_times = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.3, 0.2, 0.1])
    income = np.random.choice(['3千-5千', '5千-1万', '1万-2万', '2万以上'], n, p=[0.3, 0.4, 0.2, 0.1])
    gender = np.random.choice(['男', '女'], n, p=[0.5, 0.5])

    data = {}
    for shop in shop_types:
        if '网红' in shop or '茶饮' in shop:
            base = 3.5 + (35 - age) / 30 * 0.8
        elif '亲子' in shop or '手作' in shop:
            base = 3.0 + has_child * 0.8
        elif '汉服' in shop or '旅拍' in shop:
            base = 3.0 + (35 - age) / 30 * 1.0
        elif '小吃' in shop:
            base = 3.5 + (visit_times - 2) * 0.3
        elif '文创' in shop:
            base = 3.8 + (age - 35) / 30 * 0.3
        else:
            base = 3.5
        data[shop] = np.clip(base + np.random.normal(0, 0.5, n), 1, 5)

    data['年龄'] = age
    data['带小孩'] = has_child
    data['来访次数'] = visit_times
    data['收入水平'] = income
    data['性别'] = gender
    return pd.DataFrame(data)


# =========================================================
# 分析函数
# =========================================================
def descriptive_statistics(df, shop_types):
    stats_list = []
    for shop in shop_types:
        data = df[shop].dropna()
        if len(data) > 0:
            stats_list.append({
                '业态': shop,
                '样本量': len(data),
                '均值': round(data.mean(), 3),
                '标准差': round(data.std(), 3),
                '中位数': round(data.median(), 3),
                '最小值': round(data.min(), 2),
                '最大值': round(data.max(), 2)
            })
    return pd.DataFrame(stats_list).sort_values('均值', ascending=False)


def correlation_with_demographics(df, shop_types, demo_cols):
    results = []
    for shop in shop_types:
        for demo in demo_cols:
            if demo in df.columns:
                data = df[[shop, demo]].dropna()
                if len(data) > 1:
                    corr, p_val = pearsonr(data[shop], data[demo])
                    results.append({
                        '业态': shop,
                        '变量': demo,
                        '相关系数': round(corr, 3),
                        'p值': round(p_val, 4)
                    })
    return pd.DataFrame(results)


def regression_analysis(df, shop_types):
    if len(df) < 10:
        return None
    df_clean = df[shop_types].dropna()
    if len(df_clean) < 10:
        return None
    y = df_clean.mean(axis=1)
    X_cols = []
    if '年龄' in df.columns:
        X_cols.append('年龄')
    if '带小孩' in df.columns:
        X_cols.append('带小孩')
    if '来访次数' in df.columns:
        X_cols.append('来访次数')
    if len(X_cols) == 0:
        return None
    X = df[X_cols].dropna()
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    if len(X) < 10:
        return None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    n = len(y)
    k = len(X_cols)
    r2 = model.score(X_scaled, y)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else r2
    coef_df = pd.DataFrame({'变量': X_cols, '系数': model.coef_})
    return {'r2': r2, 'adj_r2': adj_r2, '系数表': coef_df, 'y_true': y.values, 'y_pred': model.predict(X_scaled)}


# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.markdown("## 🎨 个性美化")

    # 主题快速切换
    col1, col2 = st.columns(2)
    with col1:
        if st.button("☀️ 浅色", use_container_width=True):
            st.session_state.theme_mode = '浅色'
            st.rerun()
    with col2:
        if st.button("🌙 深色", use_container_width=True):
            st.session_state.theme_mode = '深色'
            st.rerun()

    # 颜色设置（仅浅色模式）
    if st.session_state.theme_mode == '浅色':
        st.session_state.accent_color = st.color_picker("强调色", st.session_state.accent_color)

    # 重置
    if st.button("🔄 重置", use_container_width=True):
        st.session_state.theme_mode = '浅色'
        st.session_state.bg_color = '#f0f2f6'
        st.session_state.card_color = '#ffffff'
        st.session_state.accent_color = '#667eea'
        st.session_state.text_color = '#333333'
        st.rerun()

    st.divider()

    # 数据导入
    st.markdown("## 📁 数据导入")
    data_option = st.radio("数据来源", ["📊 示例数据", "📁 上传CSV"], label_visibility="collapsed")
    if data_option == "📁 上传CSV":
        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ 已加载 {len(df)} 条数据")
            st.session_state.df = df
            exclude = ['年龄', '带小孩', '来访次数', '收入', '收入水平', '性别']
            st.session_state.shop_types = [col for col in df.columns if
                                           col not in exclude and pd.api.types.is_numeric_dtype(df[col])]
    else:
        df = generate_sample_data()
        st.session_state.df = df
        st.session_state.shop_types = [col for col in df.columns if
                                       col not in ['年龄', '带小孩', '来访次数', '收入水平', '性别']]
        st.success(f"✅ 示例数据，{len(df)} 条，{len(st.session_state.shop_types)} 个业态")

    st.divider()

    # 游客模拟器
    st.markdown("## 🎮 游客模拟器")

    col1, col2 = st.columns(2)
    with col1:
        user_age = st.select_slider("年龄", options=["18-25岁", "26-35岁", "36-45岁", "46-60岁", "60岁以上"],
                                    value="26-35岁")
        user_gender = st.selectbox("性别", ["男", "女", "不愿透露"])
    with col2:
        user_child = st.selectbox("是否有小孩", ["无", "有"])
        user_travel_with = st.selectbox("出游同伴", ["独自一人", "朋友", "情侣/伴侣", "带小孩的家庭", "父母长辈"])

    user_interest = st.multiselect("兴趣爱好", ["拍照打卡", "品尝美食", "文化体验", "亲子互动", "休闲放松"],
                                   default=["拍照打卡"])

    if st.button("✨ 推荐", use_container_width=True, type="primary"):
        df_local = st.session_state.df
        shop_types_local = st.session_state.shop_types
        if df_local is not None and len(shop_types_local) > 0:
            means = df_local[shop_types_local].mean()
            user_rec = []
            for shop in shop_types_local:
                score = means[shop] * 0.5
                if user_age in ["18-25岁", "26-35岁"] and ('网红' in shop or '汉服' in shop):
                    score += 0.4
                if user_child == "有" and '亲子' in shop:
                    score += 0.5
                if user_travel_with == "情侣/伴侣" and ('汉服' in shop or '网红' in shop):
                    score += 0.3
                if '拍照打卡' in user_interest and ('汉服' in shop or '网红' in shop):
                    score += 0.3
                if '品尝美食' in user_interest and ('小吃' in shop or '轻食' in shop):
                    score += 0.3
                user_rec.append({'业态': shop, '匹配度': round(score, 2)})
            user_rec_df = pd.DataFrame(user_rec).sort_values('匹配度', ascending=False).head(6)
            st.session_state.user_recommendation = user_rec_df
            st.success("✅ 推荐生成成功")

    if st.session_state.user_recommendation is not None:
        st.markdown("---")
        st.markdown("**🎯 专属推荐**")
        for i, row in st.session_state.user_recommendation.iterrows():
            stars = "⭐" * min(5, int(row['匹配度']))
            st.write(f"{i + 1}. **{row['业态']}** {stars}")

# =========================================================
# 应用主题
# =========================================================
apply_custom_theme()

# =========================================================
# 主界面
# =========================================================
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🎨 台湾小镇业态评估系统</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 30px; opacity: 0.7;'>统计分析 · 业态推荐 · 个性化主题</p>",
            unsafe_allow_html=True)

if st.session_state.df is not None:
    df = st.session_state.df
    shop_types = st.session_state.shop_types
    means = df[shop_types].mean()

    if len(shop_types) == 0:
        st.error("❌ 未识别到业态列")
        st.stop()

    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总样本量", len(df))
    with col2:
        st.metric("业态数量", len(shop_types))
    with col3:
        st.metric("平均评分", f"{means.mean():.2f}")
    with col4:
        st.metric("最佳业态", means.idxmax())

    # 选项卡
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 描述统计", "📈 相关性分析", "🔬 回归分析", "🎮 业态PK", "🏆 排行榜"
    ])

    # Tab 1: 描述统计
    with tab1:
        st.subheader("业态评分统计")
        stats_df = descriptive_statistics(df, shop_types)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # 条形图
        fig = px.bar(
            stats_df.sort_values('均值', ascending=True),
            x='均值', y='业态',
            orientation='h',
            title='业态满意度排名',
            color='均值',
            color_continuous_scale='RdYlGn',
            range_color=[1, 5]
        )
        fig.add_vline(x=3.5, line_dash="dash", line_color="gray")
        fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: 相关性分析
    with tab2:
        st.subheader("业态间相关性矩阵")
        corr_matrix = df[shop_types].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='皮尔逊相关系数'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("人口变量相关性")
        demo_corr = correlation_with_demographics(df, shop_types, ['年龄', '带小孩', '来访次数'])
        if len(demo_corr) > 0:
            st.dataframe(demo_corr, use_container_width=True, hide_index=True)

    # Tab 3: 回归分析
    with tab3:
        reg_result = regression_analysis(df, shop_types)
        if reg_result:
            col1, col2 = st.columns(2)
            col1.metric("R²", f"{reg_result['r2']:.4f}")
            col2.metric("调整后 R²", f"{reg_result['adj_r2']:.4f}")
            st.dataframe(reg_result['系数表'], use_container_width=True, hide_index=True)

            fig = px.scatter(
                x=reg_result['y_true'], y=reg_result['y_pred'],
                labels={'x': '实际值', 'y': '预测值'},
                title=f'预测 vs 实际 (R² = {reg_result["r2"]:.3f})'
            )
            fig.add_trace(
                go.Scatter(x=[1, 5], y=[1, 5], mode='lines', name='完美预测', line=dict(color='red', dash='dash')))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("需要年龄、带小孩等变量")

    # Tab 4: 业态PK
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            shop1 = st.selectbox("业态 A", shop_types, key="pk1")
        with col2:
            shop2 = st.selectbox("业态 B", shop_types, key="pk2")

        if shop1 and shop2:
            dimensions = ['总体评分', '稳定性']
            score1 = [means[shop1], 1 / (df[shop1].std() + 0.1)]
            score2 = [means[shop2], 1 / (df[shop2].std() + 0.1)]

            max_vals = [max(score1[i], score2[i]) for i in range(len(dimensions))]
            score1_norm = [score1[i] / max_vals[i] * 5 if max_vals[i] > 0 else 0 for i in range(len(dimensions))]
            score2_norm = [score2[i] / max_vals[i] * 5 if max_vals[i] > 0 else 0 for i in range(len(dimensions))]

            fig = go.Figure()
            fig.add_trace(go.Bar(name=shop1, x=dimensions, y=score1_norm, marker_color='#ff6b6b',
                                 text=[f'{x:.1f}' for x in score1_norm], textposition='outside'))
            fig.add_trace(go.Bar(name=shop2, x=dimensions, y=score2_norm, marker_color='#4ecdc4',
                                 text=[f'{x:.1f}' for x in score2_norm], textposition='outside'))
            fig.update_layout(title=f"{shop1} vs {shop2}", yaxis_title="得分", barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 5: 排行榜
    with tab5:
        rank_df = pd.DataFrame({
            '排名': range(1, len(shop_types) + 1),
            '业态': means.index,
            '得分': means.values.round(2),
            '星级': ['⭐' * min(5, int(x)) for x in means.values]
        })
        st.dataframe(rank_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            rank_df.head(5),
            x='业态', y='得分',
            title='TOP5 业态',
            color='得分',
            color_continuous_scale='Viridis',
            range_color=[0, 5]
        )
        fig.add_hline(y=3.5, line_dash="dash", line_color="gray")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 请从左侧选择数据来源")

st.divider()
st.caption("© 平潭文旅课题组 | 个性化主题版")