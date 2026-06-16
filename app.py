"""
台湾小镇业态评估系统 - 接入免费API版
功能：IP定位 + 中文旅游景点介绍
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import requests
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="台湾小镇业态评估系统",
    page_icon="🎮",
    layout="wide"
)

# ========== 免费API函数 ==========

@st.cache_data(ttl=3600)
def get_ip_location():
    """通过IP获取访客位置（中文显示）"""
    try:
        url = "http://ip-api.com/json/?lang=zh-CN"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return {
                    'city': data.get('city', '未知'),
                    'region': data.get('regionName', '未知'),
                    'country': data.get('country', '未知'),
                    'lat': data.get('lat', 0),
                    'lon': data.get('lon', 0)
                }
    except Exception as e:
        return None
    return None

@st.cache_data(ttl=86400)
def get_travel_info():
    """获取平潭旅游景点信息（中文）"""
    # 本地中文景点信息（更稳定）
    destinations = {
        "平潭岛": "福建第一大岛，有'千礁岛县'之称，68海里景区、龙王头沙滩、坛南湾等著名景点",
        "台湾小镇": "两岸特色文化街区，集文创、美食、手作体验于一体，适合拍照打卡",
        "坛南湾": "平潭最美海湾之一，沙滩细腻，适合亲子游玩和水上运动",
        "68海里景区": "祖国大陆距离台湾最近的地方，地标性打卡景点",
        "壳丘头遗址": "史前文化遗址，距今6000多年，平潭历史文化的重要见证",
        "龙王头": "平潭最大的沙滩浴场，观日出绝佳地点"
    }
    return destinations

# ========== 数据生成函数 ==========
def generate_sample_data():
    """生成示例数据"""
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

# ========== 分析函数 ==========
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
# 初始化session_state
# =========================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'shop_types' not in st.session_state:
    st.session_state.shop_types = []
if 'quiz_result' not in st.session_state:
    st.session_state.quiz_result = None
if 'user_recommendation' not in st.session_state:
    st.session_state.user_recommendation = None

# 标题
st.title("🎮 台湾小镇业态评估系统")
st.caption("统计分析 + 互动玩法 | 接入免费API · IP定位 · 景点介绍")

# =========================================================
# 侧边栏 - 显示API信息
# =========================================================
with st.sidebar:
    # ===== 显示IP定位信息 =====
    st.subheader("📍 访客位置")
    location = get_ip_location()
    if location:
        st.info(f"🌍 您来自：**{location['city']}市 {location['region']}省**\n\n🇨🇳 {location['country']}")
    else:
        st.info("🌍 正在获取位置信息...")

    st.divider()

    # ===== 显示旅游景点介绍（中文） =====
    st.subheader("🏖️ 平潭景点推荐")

    travel_info = get_travel_info()

    for spot_name, description in travel_info.items():
        with st.expander(f"📍 {spot_name}"):
            st.write(description)

    if st.button("🔄 刷新页面", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("数据来源：平潭文旅资料库")

    st.divider()

    # ===== 数据导入 =====
    st.subheader("📁 数据导入")
    data_option = st.radio("数据来源", ["📊 示例数据", "📁 上传CSV"])
    if data_option == "📁 上传CSV":
        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ 已加载 {len(df)} 条数据")
            st.session_state.df = df
            exclude = ['年龄', '带小孩', '来访次数', '收入', '收入水平', '性别']
            st.session_state.shop_types = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]
    else:
        df = generate_sample_data()
        st.session_state.df = df
        st.session_state.shop_types = [col for col in df.columns if col not in ['年龄', '带小孩', '来访次数', '收入水平', '性别']]
        st.success(f"✅ 示例数据，{len(df)} 条，{len(st.session_state.shop_types)} 个业态")

    st.divider()

    # ===== 游客模拟器 =====
    st.subheader("🎮 游客模拟器")

    col1, col2 = st.columns(2)
    with col1:
        user_age = st.select_slider("年龄", options=["18-25岁", "26-35岁", "36-45岁", "46-60岁", "60岁以上"], value="26-35岁")
        user_gender = st.selectbox("性别", ["男", "女", "不愿透露"])
        user_income = st.selectbox("月收入", ["3千以下", "3千-5千", "5千-1万", "1万-2万", "2万以上"])
    with col2:
        user_child = st.selectbox("是否有小孩", ["无", "有"])
        user_travel_with = st.selectbox("出游同伴", ["独自一人", "朋友", "情侣/伴侣", "带小孩的家庭", "父母长辈", "公司团建"])
        user_interest = st.multiselect("兴趣爱好", ["拍照打卡", "品尝美食", "文化体验", "亲子互动", "刺激冒险", "休闲放松", "购物"], default=["拍照打卡", "品尝美食"])

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
            st.write(f"{i+1}. **{row['业态']}** {stars} ({row['匹配度']}分)")

# =========================================================
# 主界面
# =========================================================
if st.session_state.df is not None:
    df = st.session_state.df
    shop_types = st.session_state.shop_types
    means = df[shop_types].mean()

    if len(shop_types) == 0:
        st.error("❌ 未识别到业态列")
        st.stop()

    # 选项卡
    tab_stats, tab_pk, tab_quiz, tab_custom, tab_rank = st.tabs([
        "📊 统计分析", "🎮 业态PK", "🎯 匹配测试", "⚙️ 自定义推荐", "🏆 排行榜"
    ])

    # Tab 1: 统计分析
    with tab_stats:
        st.subheader("📊 统计分析报告")

        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "描述统计", "相关性分析", "回归分析", "聚类分析", "假设检验"
        ])

        with sub_tab1:
            col1, col2 = st.columns(2)
            with col1:
                stats_df = descriptive_statistics(df, shop_types)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            with col2:
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
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        with sub_tab2:
            st.markdown("### 业态间相关性矩阵")
            corr_matrix = df[shop_types].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect='auto',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title='皮尔逊相关系数热力图'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 人口变量相关性")
            demo_corr = correlation_with_demographics(df, shop_types, ['年龄', '带小孩', '来访次数'])
            if len(demo_corr) > 0:
                st.dataframe(demo_corr, use_container_width=True, hide_index=True)

        with sub_tab3:
            reg_result = regression_analysis(df, shop_types)
            if reg_result:
                col1, col2 = st.columns(2)
                col1.metric("R²", f"{reg_result['r2']:.4f}")
                col2.metric("调整后 R²", f"{reg_result['adj_r2']:.4f}")
                st.dataframe(reg_result['系数表'], use_container_width=True, hide_index=True)

                fig = px.scatter(
                    x=reg_result['y_true'],
                    y=reg_result['y_pred'],
                    labels={'x': '实际值', 'y': '预测值'},
                    title=f'预测 vs 实际 (R² = {reg_result["r2"]:.3f})'
                )
                fig.add_trace(go.Scatter(
                    x=[1, 5], y=[1, 5],
                    mode='lines',
                    name='完美预测线',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("需要年龄、带小孩等变量进行回归分析")

        with sub_tab4:
            data_cluster = df[shop_types].dropna()
            if len(data_cluster) >= 10:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_cluster)

                sil_scores = []
                k_range = range(2, min(7, len(data_cluster)))
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(data_scaled)
                    if len(set(labels)) > 1:
                        from sklearn.metrics import silhouette_score
                        sil_scores.append(silhouette_score(data_scaled, labels))
                    else:
                        sil_scores.append(0)

                if len(sil_scores) > 0:
                    fig = px.line(
                        x=list(k_range), y=sil_scores,
                        markers=True,
                        labels={'x': '聚类数 K', 'y': '轮廓系数'},
                        title='轮廓系数图（越高越好）'
                    )
                    best_k = k_range[np.argmax(sil_scores)]
                    fig.add_vline(x=best_k, line_dash="dash", line_color="red", annotation_text=f'最佳 K={best_k}')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("数据不足，无法进行聚类分析")

        with sub_tab5:
            test_type = st.radio("选择检验", ["方差分析 ANOVA", "t检验"], horizontal=True)
            if test_type == "方差分析 ANOVA":
                if '年龄' in df.columns:
                    df['年龄分组'] = pd.cut(df['年龄'], bins=[0, 30, 45, 100], labels=['青年', '中年', '中老年'])
                    results = []
                    for shop in shop_types:
                        groups = []
                        for g in ['青年', '中年', '中老年']:
                            data_g = df[df['年龄分组'] == g][shop].dropna()
                            if len(data_g) > 0:
                                groups.append(data_g.values)
                        if len(groups) >= 2:
                            f_stat, p_val = f_oneway(*groups)
                            results.append({'业态': shop, 'F值': round(f_stat, 3), 'p值': round(p_val, 4)})
                    if results:
                        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                else:
                    st.info("需要年龄列")
            else:
                if '带小孩' in df.columns:
                    results = []
                    for shop in shop_types:
                        group1 = df[df['带小孩'] == 1][shop].dropna()
                        group2 = df[df['带小孩'] == 0][shop].dropna()
                        if len(group1) > 1 and len(group2) > 1:
                            t_stat, p_val = ttest_ind(group1, group2)
                            results.append({'业态': shop, '差异': round(group1.mean() - group2.mean(), 3), 'p值': round(p_val, 4)})
                    if results:
                        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                else:
                    st.info("需要带小孩列")

    # Tab 2: 业态PK
    with tab_pk:
        st.subheader("🎮 业态PK对战")

        col1, col2 = st.columns(2)
        with col1:
            shop1 = st.selectbox("业态 A", shop_types, key="pk1")
        with col2:
            shop2 = st.selectbox("业态 B", shop_types, key="pk2")

        if shop1 and shop2:
            dimensions = ['总体评分', '年轻人偏好', '家庭偏好', '复购吸引力', '稳定性', '差异化']
            score1 = [
                means[shop1],
                -pearsonr(df[shop1], df['年龄'])[0] if '年龄' in df.columns else 0,
                df[df['带小孩'] == 1][shop1].mean() if '带小孩' in df.columns else means[shop1],
                df[df['来访次数'] >= 3][shop1].mean() if '来访次数' in df.columns else means[shop1],
                1 / (df[shop1].std() + 0.1),
                abs(pearsonr(df[shop1], df['年龄'])[0]) if '年龄' in df.columns else 0
            ]
            score2 = [
                means[shop2],
                -pearsonr(df[shop2], df['年龄'])[0] if '年龄' in df.columns else 0,
                df[df['带小孩'] == 1][shop2].mean() if '带小孩' in df.columns else means[shop2],
                df[df['来访次数'] >= 3][shop2].mean() if '来访次数' in df.columns else means[shop2],
                1 / (df[shop2].std() + 0.1),
                abs(pearsonr(df[shop2], df['年龄'])[0]) if '年龄' in df.columns else 0
            ]

            max_vals = [max(score1[i], score2[i]) for i in range(len(dimensions))]
            score1_norm = [score1[i]/max_vals[i]*5 if max_vals[i] > 0 else 0 for i in range(len(dimensions))]
            score2_norm = [score2[i]/max_vals[i]*5 if max_vals[i] > 0 else 0 for i in range(len(dimensions))]

            fig = go.Figure()
            fig.add_trace(go.Bar(name=shop1, x=dimensions, y=score1_norm, marker_color='#ff6b6b', text=[f'{x:.1f}' for x in score1_norm], textposition='outside'))
            fig.add_trace(go.Bar(name=shop2, x=dimensions, y=score2_norm, marker_color='#4ecdc4', text=[f'{x:.1f}' for x in score2_norm], textposition='outside'))
            fig.update_layout(title=f"{shop1} vs {shop2}", yaxis_title="得分", barmode='group', height=500)
            st.plotly_chart(fig, use_container_width=True)

            win_count = sum(1 for a, b in zip(score1_norm, score2_norm) if a > b)
            lose_count = sum(1 for a, b in zip(score1_norm, score2_norm) if a < b)
            total1 = sum(score1_norm)
            total2 = sum(score2_norm)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(shop1, f"赢得 {win_count} 项", delta=f"总分 {total1:.1f}")
            with col2:
                st.metric(shop2, f"赢得 {lose_count} 项", delta=f"总分 {total2:.1f}")
            with col3:
                if win_count > lose_count:
                    st.success(f"🏆 胜者：{shop1}")
                elif lose_count > win_count:
                    st.success(f"🏆 胜者：{shop2}")
                else:
                    st.info("平局！")

    # Tab 3: 匹配测试
    with tab_quiz:
        st.subheader("🎯 业态匹配测试")

        with st.form("quiz_form"):
            col1, col2 = st.columns(2)
            with col1:
                q1 = st.radio("1. 你通常和谁一起出游？", ["独自一人", "朋友", "情侣/伴侣", "带小孩的家庭", "父母长辈"])
                q2 = st.radio("2. 你最喜欢的体验类型？", ["拍照打卡", "品尝美食", "动手制作体验", "文化学习", "刺激冒险"])
                q3 = st.radio("3. 你的预算范围？", ["50元以下", "50-100元", "100-200元", "200元以上"])
            with col2:
                q4 = st.radio("4. 你更看重什么？", ["性价比", "独特体验", "服务质量", "环境氛围", "社交属性"])
                q5 = st.radio("5. 你计划停留时间？", ["1小时以内", "1-2小时", "2-3小时", "半天以上"])
                q6 = st.radio("6. 你是第几次来平潭？", ["第1次", "第2-3次", "4次以上"])

            submitted = st.form_submit_button("🔮 开始匹配", use_container_width=True, type="primary")

            if submitted:
                match_scores = {shop: means[shop] * 0.3 for shop in shop_types}
                if q1 == "带小孩的家庭":
                    for s in shop_types:
                        if '亲子' in s:
                            match_scores[s] += 0.6
                if q1 == "情侣/伴侣":
                    for s in ['汉服旅拍馆', '网红茶饮店']:
                        if s in match_scores:
                            match_scores[s] += 0.5
                if q2 == "拍照打卡":
                    for s in ['汉服旅拍馆', '网红茶饮店']:
                        if s in match_scores:
                            match_scores[s] += 0.5
                if q2 == "品尝美食":
                    for s in ['台湾小吃摊', '轻食简餐店']:
                        if s in match_scores:
                            match_scores[s] += 0.5
                match_df = pd.DataFrame(list(match_scores.items()), columns=['业态', '匹配分']).sort_values('匹配分', ascending=False).head(3)
                st.session_state.quiz_result = match_df
                st.success(f"✨ 匹配完成！")

        if st.session_state.quiz_result is not None:
            st.markdown("---")
            st.markdown(f"### 🎉 你的专属匹配结果")
            col1, col2, col3 = st.columns(3)
            for i, col in enumerate([col1, col2, col3]):
                if i < len(st.session_state.quiz_result):
                    score = st.session_state.quiz_result.iloc[i]['匹配分']
                    with col:
                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; background:linear-gradient(135deg,#667eea,#764ba2); border-radius:15px; color:white;">
                            <div style="font-size:40px;">🏆</div>
                            <div style="font-size:18px; font-weight:bold;">{st.session_state.quiz_result.iloc[i]['业态']}</div>
                            <div>{score:.1f}分</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Tab 4: 自定义推荐
    with tab_custom:
        st.subheader("⚙️ 自定义推荐引擎")

        col1, col2 = st.columns([1, 1.5])
        with col1:
            w1 = st.slider("总体满意度", 0, 100, 40, key="cw1")
            w2 = st.slider("年轻人偏好", 0, 100, 25, key="cw2")
            w3 = st.slider("家庭偏好", 0, 100, 20, key="cw3")
            w4 = st.slider("差异化程度", 0, 100, 15, key="cw4")
            total = w1 + w2 + w3 + w4
            if total == 0:
                total = 1

        with col2:
            custom_scores = []
            for shop in shop_types:
                score = (means[shop] * w1/total +
                        (-pearsonr(df[shop], df['年龄'])[0] if '年龄' in df.columns else 0) * w2/total * 3 +
                        (df[df['带小孩'] == 1][shop].mean() if '带小孩' in df.columns else means[shop]) * w3/total +
                        df[shop].std() * w4/total)
                score = min(5, score)
                custom_scores.append({'业态': shop, '推荐分': round(score, 3)})
            custom_df = pd.DataFrame(custom_scores).sort_values('推荐分', ascending=False)
            st.dataframe(custom_df.head(10), use_container_width=True, hide_index=True)

            fig = px.bar(
                custom_df.head(6),
                x='推荐分',
                y='业态',
                orientation='h',
                title='自定义权重推荐 TOP6',
                labels={'推荐分': '推荐分', '业态': ''},
                color='推荐分',
                color_continuous_scale='Viridis',
                range_color=[0, 5]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 5: 排行榜
    with tab_rank:
        st.subheader("🏆 业态排行榜")

        rank_tab1, rank_tab2, rank_tab3 = st.tabs(["综合排名", "年轻人最爱", "家庭最爱"])

        with rank_tab1:
            rank_df = pd.DataFrame({
                '排名': range(1, len(shop_types)+1),
                '业态': means.index,
                '得分': means.values.round(2),
                '星级': ['⭐' * min(5, int(x)) for x in means.values]
            })
            st.dataframe(rank_df, use_container_width=True, hide_index=True)

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
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with rank_tab2:
            if '年龄' in df.columns:
                youth_scores = {shop: -pearsonr(df[shop], df['年龄'])[0] for shop in shop_types}
                youth_df = pd.DataFrame(list(youth_scores.items()), columns=['业态', '年轻偏好分']).sort_values('年轻偏好分', ascending=False)
                youth_df['排名'] = range(1, len(youth_df)+1)
                st.dataframe(youth_df, use_container_width=True, hide_index=True)
            else:
                st.info("需要年龄数据")

        with rank_tab3:
            if '带小孩' in df.columns:
                family_scores = {shop: df[df['带小孩'] == 1][shop].mean() for shop in shop_types}
                family_df = pd.DataFrame(list(family_scores.items()), columns=['业态', '家庭评分']).sort_values('家庭评分', ascending=False)
                family_df['排名'] = range(1, len(family_df)+1)
                st.dataframe(family_df, use_container_width=True, hide_index=True)
            else:
                st.info("需要带小孩数据")

else:
    st.info("👈 请从左侧选择数据来源（示例数据或上传CSV）")

st.divider()
st.caption("© 平潭文旅课题组 | 接入免费API · 中文景点介绍")