"""
台湾小镇业态评估系统 - 完整功能版
功能：描述统计、相关性、回归、聚类、假设检验、业态PK、匹配测试、自定义推荐、排行榜
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
    page_icon="📊",
    layout="wide"
)

# =========================================================
# 初始化 session_state
# =========================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'shop_types' not in st.session_state:
    st.session_state.shop_types = []
if 'quiz_result' not in st.session_state:
    st.session_state.quiz_result = None
if 'user_recommendation' not in st.session_state:
    st.session_state.user_recommendation = None


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
# 统计分析函数
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
                        'p值': round(p_val, 4),
                        '显著性': '***' if p_val < 0.001 else (
                            '**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
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
    coef_df = pd.DataFrame(
        {'变量': X_cols, '系数': round(model.coef_[0], 4) if len(X_cols) == 1 else [round(c, 4) for c in model.coef_]})
    return {'r2': r2, 'adj_r2': adj_r2, '系数表': coef_df, 'y_true': y.values, 'y_pred': model.predict(X_scaled)}


def cluster_analysis(df, shop_types):
    data = df[shop_types].dropna()
    if len(data) < 10:
        return None
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    sil_scores = []
    k_range = range(2, min(7, len(data)))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        if len(set(labels)) > 1:
            sil_scores.append(silhouette_score(data_scaled, labels))
        else:
            sil_scores.append(0)
    best_k = k_range[np.argmax(sil_scores)] if sil_scores else 3
    return {'best_k': best_k, 'sil_scores': sil_scores, 'k_range': list(k_range)}


def anova_analysis(df, shop_types):
    if '年龄' not in df.columns:
        return None
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
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            results.append({'业态': shop, 'F值': round(f_stat, 3), 'p值': round(p_val, 4), '显著性': sig})
    return pd.DataFrame(results)


def ttest_analysis(df, shop_types):
    if '带小孩' not in df.columns:
        return None
    results = []
    for shop in shop_types:
        group1 = df[df['带小孩'] == 1][shop].dropna()
        group2 = df[df['带小孩'] == 0][shop].dropna()
        if len(group1) > 1 and len(group2) > 1:
            t_stat, p_val = ttest_ind(group1, group2)
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            results.append({
                '业态': shop,
                '带小孩均值': round(group1.mean(), 3),
                '不带小孩均值': round(group2.mean(), 3),
                '差异': round(group1.mean() - group2.mean(), 3),
                'p值': round(p_val, 4),
                '显著性': sig
            })
    return pd.DataFrame(results)


# =========================================================
# 标题
# =========================================================
st.title("📊 台湾小镇业态评估系统")
st.caption("统计分析 · 业态推荐 · 数据驱动决策")

# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.subheader("📁 数据导入")
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
    st.subheader("🎮 游客模拟器")

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

    if st.button("✨ 生成专属推荐", use_container_width=True, type="primary"):
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
        st.markdown("**🎯 你的专属推荐**")
        for i, row in st.session_state.user_recommendation.iterrows():
            stars = "⭐" * min(5, int(row['匹配度']))
            st.write(f"{i + 1}. **{row['业态']}** {stars}")

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

    # =========================================================
    # 选项卡（完整版）
    # =========================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 描述统计", "📈 相关性", "🔬 回归分析", "📉 聚类分析", "📐 假设检验", "🎮 业态PK", "🏆 排行榜"
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
            labels={'均值': '平均评分', '业态': ''},
            color='均值',
            color_continuous_scale='RdYlGn',
            range_color=[1, 5]
        )
        fig.add_vline(x=3.5, line_dash="dash", line_color="gray", annotation_text="参考线3.5分")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: 相关性
    with tab2:
        st.subheader("业态间相关性矩阵")
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

        st.subheader("人口变量相关性")
        demo_corr = correlation_with_demographics(df, shop_types, ['年龄', '带小孩', '来访次数'])
        if len(demo_corr) > 0:
            st.dataframe(demo_corr, use_container_width=True, hide_index=True)

    # Tab 3: 回归分析
    with tab3:
        st.subheader("多元线性回归分析")
        reg_result = regression_analysis(df, shop_types)
        if reg_result:
            col1, col2 = st.columns(2)
            col1.metric("R² (决定系数)", f"{reg_result['r2']:.4f}")
            col2.metric("调整后 R²", f"{reg_result['adj_r2']:.4f}")
            st.dataframe(reg_result['系数表'], use_container_width=True, hide_index=True)

            fig = px.scatter(
                x=reg_result['y_true'], y=reg_result['y_pred'],
                labels={'x': '实际值', 'y': '预测值'},
                title=f'预测 vs 实际 (R² = {reg_result["r2"]:.3f})'
            )
            fig.add_trace(
                go.Scatter(x=[1, 5], y=[1, 5], mode='lines', name='完美预测线', line=dict(color='red', dash='dash')))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("需要年龄、带小孩、来访次数等变量进行回归分析")

    # Tab 4: 聚类分析
    with tab4:
        st.subheader("KMeans 聚类分析")
        cluster_result = cluster_analysis(df, shop_types)
        if cluster_result:
            st.metric("建议聚类数", f"K = {cluster_result['best_k']}")

            fig = px.line(
                x=cluster_result['k_range'], y=cluster_result['sil_scores'],
                markers=True,
                labels={'x': '聚类数 K', 'y': '轮廓系数'},
                title='轮廓系数图（越高越好）'
            )
            fig.add_vline(x=cluster_result['best_k'], line_dash="dash", line_color="red",
                          annotation_text=f'最佳 K={cluster_result["best_k"]}')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("数据不足，无法进行聚类分析")

    # Tab 5: 假设检验
    with tab5:
        test_type = st.radio("选择检验", ["方差分析 ANOVA", "独立样本 t检验"], horizontal=True)

        if test_type == "方差分析 ANOVA":
            st.subheader("方差分析 (不同年龄组的偏好差异)")
            anova_df = anova_analysis(df, shop_types)
            if anova_df is not None and len(anova_df) > 0:
                st.dataframe(anova_df, use_container_width=True, hide_index=True)
                st.caption("显著性标记：*** p<0.001, ** p<0.01, * p<0.05")
            else:
                st.info("需要「年龄」列进行方差分析")
        else:
            st.subheader("独立样本 t检验 (带小孩 vs 不带小孩)")
            ttest_df = ttest_analysis(df, shop_types)
            if ttest_df is not None and len(ttest_df) > 0:
                st.dataframe(ttest_df, use_container_width=True, hide_index=True)
                st.caption("显著性标记：*** p<0.001, ** p<0.01, * p<0.05")
            else:
                st.info("需要「带小孩」列进行t检验")

    # Tab 6: 业态PK
    with tab6:
        st.subheader("业态PK对战")

        col1, col2 = st.columns(2)
        with col1:
            shop1 = st.selectbox("业态 A", shop_types, key="pk1")
        with col2:
            shop2 = st.selectbox("业态 B", shop_types, key="pk2")

        if shop1 and shop2:
            dimensions = ['总体评分', '年轻人偏好', '家庭偏好', '稳定性']

            # 计算年轻人偏好（负相关）
            if '年龄' in df.columns:
                youth1 = -pearsonr(df[shop1], df['年龄'])[0]
                youth2 = -pearsonr(df[shop2], df['年龄'])[0]
            else:
                youth1, youth2 = 0, 0

            # 计算家庭偏好
            if '带小孩' in df.columns:
                family1 = df[df['带小孩'] == 1][shop1].mean() if len(df[df['带小孩'] == 1]) > 0 else means[shop1]
                family2 = df[df['带小孩'] == 2][shop2].mean() if len(df[df['带小孩'] == 1]) > 0 else means[shop2]
            else:
                family1, family2 = means[shop1], means[shop2]

            score1 = [means[shop1], youth1, family1, 1 / (df[shop1].std() + 0.1)]
            score2 = [means[shop2], youth2, family2, 1 / (df[shop2].std() + 0.1)]

            # 归一化到0-5
            max_vals = [max(score1[i], score2[i]) for i in range(len(dimensions))]
            score1_norm = [score1[i] / max_vals[i] * 5 if max_vals[i] > 0 else 0 for i in range(len(dimensions))]
            score2_norm = [score2[i] / max_vals[i] * 5 if max_vals[i] > 0 else 0 for i in range(len(dimensions))]

            fig = go.Figure()
            fig.add_trace(go.Bar(name=shop1, x=dimensions, y=score1_norm, marker_color='#ff6b6b',
                                 text=[f'{x:.1f}' for x in score1_norm], textposition='outside'))
            fig.add_trace(go.Bar(name=shop2, x=dimensions, y=score2_norm, marker_color='#4ecdc4',
                                 text=[f'{x:.1f}' for x in score2_norm], textposition='outside'))
            fig.update_layout(title=f"{shop1} vs {shop2}", yaxis_title="得分 (满分5分)", barmode='group', height=450)
            st.plotly_chart(fig, use_container_width=True)

            win_count = sum(1 for a, b in zip(score1_norm, score2_norm) if a > b)
            lose_count = sum(1 for a, b in zip(score1_norm, score2_norm) if a < b)

            if win_count > lose_count:
                st.success(f"🏆 胜者：{shop1}")
            elif lose_count > win_count:
                st.success(f"🏆 胜者：{shop2}")
            else:
                st.info("平局！")

    # Tab 7: 排行榜
    with tab7:
        st.subheader("业态排行榜")

        rank_tab1, rank_tab2, rank_tab3 = st.tabs(["综合排名", "年轻人最爱", "家庭最爱"])

        with rank_tab1:
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
                title='综合评分 TOP5',
                labels={'得分': '平均评分'},
                color='得分',
                color_continuous_scale='Viridis',
                range_color=[0, 5]
            )
            fig.add_hline(y=3.5, line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with rank_tab2:
            if '年龄' in df.columns:
                youth_scores = {shop: -pearsonr(df[shop], df['年龄'])[0] for shop in shop_types}
                youth_df = pd.DataFrame(list(youth_scores.items()), columns=['业态', '年轻偏好分']).sort_values(
                    '年轻偏好分', ascending=False)
                youth_df['排名'] = range(1, len(youth_df) + 1)
                st.dataframe(youth_df, use_container_width=True, hide_index=True)
                st.caption("得分越高，越受年轻人欢迎")
            else:
                st.info("需要年龄数据")

        with rank_tab3:
            if '带小孩' in df.columns:
                family_scores = {shop: df[df['带小孩'] == 1][shop].mean() for shop in shop_types}
                family_df = pd.DataFrame(list(family_scores.items()), columns=['业态', '家庭评分']).sort_values(
                    '家庭评分', ascending=False)
                family_df['排名'] = range(1, len(family_df) + 1)
                st.dataframe(family_df, use_container_width=True, hide_index=True)
                st.caption("得分越高，越受带小孩家庭欢迎")
            else:
                st.info("需要带小孩数据")

else:
    st.info("👈 请从左侧选择数据来源（示例数据或上传CSV）")

st.divider()
st.caption("© 平潭文旅课题组 | 完整统计分析版")