"""
台湾小镇业态评估系统 - 个性化主题版
功能：自定义背景颜色、主题切换、卡片样式
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
import warnings

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="台湾小镇业态评估系统",
    page_icon="🎨",
    layout="wide"
)

# =========================================================
# 初始化 session_state（主题设置）
# =========================================================
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = '浅色'
if 'bg_color' not in st.session_state:
    st.session_state.bg_color = '#f5f7fa'
if 'card_color' not in st.session_state:
    st.session_state.card_color = '#ffffff'
if 'accent_color' not in st.session_state:
    st.session_state.accent_color = '#667eea'
if 'text_color' not in st.session_state:
    st.session_state.text_color = '#333333'
if 'card_radius' not in st.session_state:
    st.session_state.card_radius = 16

# 其他初始化
if 'df' not in st.session_state:
    st.session_state.df = None
if 'shop_types' not in st.session_state:
    st.session_state.shop_types = []
if 'quiz_result' not in st.session_state:
    st.session_state.quiz_result = None
if 'user_recommendation' not in st.session_state:
    st.session_state.user_recommendation = None


# =========================================================
# 应用自定义CSS
# =========================================================
def apply_custom_theme():
    """根据用户设置应用主题"""

    # 深色模式的颜色映射
    if st.session_state.theme_mode == '深色':
        bg = '#1e1e2e'
        card_bg = '#2d2d3f'
        text = '#e0e0e0'
        accent = '#c084fc'
    else:
        bg = st.session_state.bg_color
        card_bg = st.session_state.card_color
        text = st.session_state.text_color
        accent = st.session_state.accent_color

    custom_css = f"""
    <style>
        /* 整体背景 */
        .stApp {{
            background-color: {bg} !important;
        }}

        /* 主内容区背景 */
        .main .block-container {{
            background-color: transparent !important;
        }}

        /* 卡片样式 */
        .custom-card {{
            background-color: {card_bg};
            border-radius: {st.session_state.card_radius}px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .custom-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }}

        /* 标题颜色 */
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: {accent} !important;
        }}

        /* 普通文字颜色 */
        p, li, span, div {{
            color: {text} !important;
        }}

        /* 按钮样式 */
        .stButton button {{
            background-color: {accent} !important;
            color: white !important;
            border-radius: 30px !important;
            transition: all 0.3s !important;
        }}
        .stButton button:hover {{
            transform: scale(1.02);
            opacity: 0.9;
        }}

        /* 侧边栏样式 */
        [data-testid="stSidebar"] {{
            background-color: {card_bg} !important;
            border-right: 1px solid rgba(0,0,0,0.1);
        }}

        /* 选项卡样式 */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {card_bg};
            border-radius: 30px;
            padding: 8px 20px;
            color: {text};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {accent} !important;
            color: white !important;
        }}

        /* 指标卡片样式 */
        [data-testid="stMetricValue"] {{
            color: {accent} !important;
            font-size: 1.8rem !important;
        }}

        /* 信息框样式 */
        .stAlert {{
            background-color: {card_bg} !important;
            border-left: 4px solid {accent} !important;
        }}

        /* 滑块样式 */
        .stSlider [data-baseweb="slider"] div {{
            background-color: {accent} !important;
        }}

        /* 选择框样式 */
        .stSelectbox div[data-baseweb="select"] {{
            background-color: {card_bg} !important;
        }}

        /* 放射状渐变标题 */
        .gradient-title {{
            background: linear-gradient(135deg, {accent}, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: bold;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# =========================================================
# 生成示例数据
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


# =========================================================
# 侧边栏 - 个性化设置
# =========================================================
with st.sidebar:
    st.markdown("## 🎨 个性美化")

    # 主题快速切换
    st.markdown("### 🌓 快速主题")
    theme_col1, theme_col2 = st.columns(2)
    with theme_col1:
        if st.button("☀️ 浅色模式", use_container_width=True):
            st.session_state.theme_mode = '浅色'
            st.rerun()
    with theme_col2:
        if st.button("🌙 深色模式", use_container_width=True):
            st.session_state.theme_mode = '深色'
            st.rerun()

    st.divider()

    # 详细颜色设置（仅在浅色模式下显示）
    if st.session_state.theme_mode == '浅色':
        st.markdown("### 🎨 颜色定制")

        st.session_state.bg_color = st.color_picker(
            "背景颜色",
            st.session_state.bg_color,
            help="页面整体背景色"
        )

        st.session_state.card_color = st.color_picker(
            "卡片颜色",
            st.session_state.card_color,
            help="卡片和侧边栏背景色"
        )

        st.session_state.accent_color = st.color_picker(
            "强调色",
            st.session_state.accent_color,
            help="按钮、标题、图标的颜色"
        )

        st.session_state.text_color = st.color_picker(
            "文字颜色",
            st.session_state.text_color,
            help="正文文字颜色"
        )

        st.session_state.card_radius = st.slider(
            "卡片圆角",
            min_value=0,
            max_value=30,
            value=st.session_state.card_radius,
            step=2,
            help="数值越大，卡片越圆润"
        )

    # 重置默认
    if st.button("🔄 重置默认主题", use_container_width=True):
        st.session_state.theme_mode = '浅色'
        st.session_state.bg_color = '#f5f7fa'
        st.session_state.card_color = '#ffffff'
        st.session_state.accent_color = '#667eea'
        st.session_state.text_color = '#333333'
        st.session_state.card_radius = 16
        st.rerun()

    st.divider()

    # 数据导入（原有功能）
    st.markdown("## 📁 数据导入")
    data_option = st.radio("数据来源", ["📊 示例数据", "📁 上传CSV"])
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
        user_income = st.selectbox("月收入", ["3千以下", "3千-5千", "5千-1万", "1万-2万", "2万以上"])
    with col2:
        user_child = st.selectbox("是否有小孩", ["无", "有"])
        user_travel_with = st.selectbox("出游同伴",
                                        ["独自一人", "朋友", "情侣/伴侣", "带小孩的家庭", "父母长辈", "公司团建"])
        user_interest = st.multiselect("兴趣爱好",
                                       ["拍照打卡", "品尝美食", "文化体验", "亲子互动", "刺激冒险", "休闲放松", "购物"],
                                       default=["拍照打卡", "品尝美食"])

    if st.button("✨ 生成专属推荐", use_container_width=True, type="primary"):
        df_local = st.session_state.df
        shop_types_local = st.session_state.shop_types
        if df_local is not None and len(shop_types_local) > 0:
            means = df_local[shop_types_local].mean()
            user_rec = []
            for shop in shop_types_local:
                score = means[shop] * 0.4
                if user_age in ["18-25岁", "26-35岁"]:
                    if '网红' in shop or '茶饮' in shop or '汉服' in shop:
                        score += 0.4
                if user_child == "有" and ('亲子' in shop or '手作' in shop):
                    score += 0.6
                if user_travel_with == "情侣/伴侣" and ('汉服' in shop or '网红' in shop):
                    score += 0.4
                if '拍照打卡' in user_interest and ('汉服' in shop or '网红' in shop):
                    score += 0.3
                if '品尝美食' in user_interest and ('小吃' in shop or '轻食' in shop):
                    score += 0.4
                user_rec.append({'业态': shop, '匹配度': round(score, 2)})
            user_rec_df = pd.DataFrame(user_rec).sort_values('匹配度', ascending=False).head(8)
            st.session_state.user_recommendation = user_rec_df
            st.success(f"✨ 推荐生成成功！")

    if st.session_state.user_recommendation is not None:
        st.markdown("---")
        st.markdown(f"**🎯 你的专属推荐 TOP 8**")
        for i, row in st.session_state.user_recommendation.iterrows():
            stars = "⭐" * min(5, int(row['匹配度']))
            st.write(f"{i + 1}. **{row['业态']}** {stars} ({row['匹配度']}分)")

# =========================================================
# 应用主题
# =========================================================
apply_custom_theme()

# =========================================================
# 主界面
# =========================================================

# 渐变标题
st.markdown(f"""
<div style="text-align: center; margin-bottom: 30px;">
    <span class="gradient-title">🎨 台湾小镇业态评估系统</span>
    <p style="font-size: 1.1rem; opacity: 0.8;">个性化主题 · 统计分析 · 业态推荐</p>
</div>
""", unsafe_allow_html=True)

# 主内容
if st.session_state.df is not None:
    df = st.session_state.df
    shop_types = st.session_state.shop_types
    means = df[shop_types].mean()

    if len(shop_types) == 0:
        st.error("❌ 未识别到业态列")
        st.stop()

    # 统计卡片（使用自定义卡片样式）
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="custom-card" style="text-align: center;">
            <div style="font-size: 2rem;">📊</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{len(df)}</div>
            <div>总样本量</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="custom-card" style="text-align: center;">
            <div style="font-size: 2rem;">🏪</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{len(shop_types)}</div>
            <div>业态数量</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="custom-card" style="text-align: center;">
            <div style="font-size: 2rem;">⭐</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{means.mean():.2f}</div>
            <div>平均评分</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        best_shop = means.idxmax()
        st.markdown(f"""
        <div class="custom-card" style="text-align: center;">
            <div style="font-size: 2rem;">🏆</div>
            <div style="font-size: 1rem; font-weight: bold;">{best_shop}</div>
            <div>最佳业态</div>
        </div>
        """, unsafe_allow_html=True)

    # 选项卡
    tab_stats, tab_rank = st.tabs(["📊 统计分析", "🏆 业态排行榜"])

    with tab_stats:
        st.subheader("业态满意度排名")
        stats_df = descriptive_statistics(df, shop_types)

        # 使用 Plotly 条形图
        stats_plot = stats_df.sort_values('均值', ascending=True)
        fig = px.bar(
            stats_plot,
            x='均值',
            y='业态',
            orientation='h',
            title='业态满意度排名',
            labels={'均值': '平均评分', '业态': ''},
            color='均值',
            color_continuous_scale='RdYlGn',
            range_color=[1, 5]
        )
        fig.add_vline(x=3.5, line_dash="dash", line_color="gray", annotation_text="参考线3.5分")
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=st.session_state.text_color if st.session_state.theme_mode == '浅色' else '#e0e0e0')
        )
        st.plotly_chart(fig, use_container_width=True)

        # 数据表
        st.subheader("详细统计")
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with tab_rank:
        st.subheader("业态排行榜")
        rank_df = pd.DataFrame({
            '排名': range(1, len(shop_types) + 1),
            '业态': means.index,
            '得分': means.values.round(2),
            '星级': ['⭐' * min(5, int(x)) for x in means.values]
        })
        st.dataframe(rank_df, use_container_width=True, hide_index=True)

        # 条形图
        fig = px.bar(
            rank_df.head(5),
            x='业态',
            y='得分',
            title='综合评分 TOP5',
            labels={'得分': '平均评分'},
            color='得分',
            color_continuous_scale='Viridis',
            range_color=[0, 5]
        )
        fig.add_hline(y=3.5, line_dash="dash", line_color="gray", annotation_text="参考线3.5分")
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=st.session_state.text_color if st.session_state.theme_mode == '浅色' else '#e0e0e0')
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 请从左侧选择数据来源（示例数据或上传CSV）")

st.divider()
st.caption(f"© 平潭文旅课题组 | 个性化主题 v2.0 | 当前主题：{st.session_state.theme_mode}")