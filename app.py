"""
台湾小镇业态评估系统 - 完整版（统计分析 + 自定义玩法）
功能：描述统计、相关性、回归、聚类、假设检验、PCA + 业态PK、匹配测试、游客模拟器
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, f_oneway, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ========== 修复中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="台湾小镇业态评估系统",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS - 强制使用中文字体
st.markdown("""
<style>
    /* 侧边栏更宽 */
    [data-testid="stSidebar"] {
        min-width: 320px;
        width: 320px;
    }
    /* 修复所有label换行 */
    .stSelectbox label, .stRadio label, .stCheckbox label, 
    .stMultiSelect label, .stSlider label, .stNumberInput label {
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.4 !important;
    }
    /* expander内文字正常显示 */
    .streamlit-expanderContent {
        overflow-x: visible !important;
    }
    /* 修复radio按钮组 */
    div[role="radiogroup"] label {
        white-space: normal !important;
        word-break: break-word !important;
    }
    /* 强制所有文字使用中文字体 */
    * {
        font-family: 'Microsoft YaHei', 'SimHei', 'PingFang SC', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.title("🎮 台湾小镇业态评估系统")
st.caption("统计分析 + 丰富互动玩法 | 7维游客模拟 · 业态PK · 深度匹配测试 · 自定义推荐")

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
if 'pk_history' not in st.session_state:
    st.session_state.pk_history = []


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


def correlation_analysis(df, shop_types):
    n_shops = len(shop_types)
    pearson_matrix = np.zeros((n_shops, n_shops))
    p_matrix = np.zeros((n_shops, n_shops))
    for i, shop1 in enumerate(shop_types):
        for j, shop2 in enumerate(shop_types):
            data1 = df[shop1].dropna()
            data2 = df[shop2].dropna()
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) > 1:
                pearson_matrix[i, j], p_matrix[i, j] = pearsonr(data1[common_idx], data2[common_idx])
    return pearson_matrix, p_matrix


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
    coef_df = pd.DataFrame({'变量': X_cols, '系数': model.coef_})
    return {'r2': r2, 'adj_r2': adj_r2, '系数表': coef_df, 'y_true': y.values, 'y_pred': model.predict(X_scaled)}


def pca_analysis(df, shop_types):
    data = df[shop_types].dropna()
    if len(data) < 3:
        return None
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i + 1}' for i in range(len(shop_types))], index=shop_types)
    return {'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_), 'loadings': loadings}


def cluster_analysis(df, shop_types, max_clusters=6):
    data = df[shop_types].dropna()
    if len(data) < 10:
        return None
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(max_clusters + 1, len(data)))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        inertias.append(kmeans.inertia_)
        if len(set(labels)) > 1:
            silhouette_scores.append(silhouette_score(data_scaled, labels))
        else:
            silhouette_scores.append(0)
    best_k = k_range[np.argmax(silhouette_scores)] if silhouette_scores else 3
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(data_scaled)
    centers = pd.DataFrame(scaler.inverse_transform(kmeans_final.cluster_centers_), columns=shop_types)
    cluster_profiles = []
    for i in range(best_k):
        cluster_data = data[labels == i]
        profile = {'聚类': f'第{i + 1}类', '样本量': len(cluster_data),
                   '占比': f"{len(cluster_data) / len(data) * 100:.1f}%"}
        for shop in shop_types[:3]:
            profile[f'{shop}均分'] = round(cluster_data[shop].mean(), 2)
        cluster_profiles.append(profile)
    return {'best_k': best_k, 'inertias': inertias, 'silhouette_scores': silhouette_scores,
            'profiles': pd.DataFrame(cluster_profiles)}


def anova_analysis(df, shop_types):
    if '年龄' not in df.columns:
        return None
    df['年龄分组'] = pd.cut(df['年龄'], bins=[0, 30, 45, 100], labels=['青年', '中年', '中老年'])
    results = []
    for shop in shop_types:
        groups = [df[df['年龄分组'] == g][shop].dropna().values for g in ['青年', '中年', '中老年'] if
                  len(df[df['年龄分组'] == g][shop].dropna()) > 0]
        if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
            results.append({'业态': shop, 'F值': round(f_stat, 3), 'p值': round(p_val, 4),
                            '显著性': '***' if p_val < 0.001 else (
                                '**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))})
    return pd.DataFrame(results)


def t_test_family(df, shop_types):
    if '带小孩' not in df.columns:
        return None
    results = []
    for shop in shop_types:
        group1 = df[df['带小孩'] == 1][shop].dropna()
        group2 = df[df['带小孩'] == 0][shop].dropna()
        if len(group1) > 1 and len(group2) > 1:
            t_stat, p_val = ttest_ind(group1, group2)
            results.append(
                {'业态': shop, '带小孩平均': round(group1.mean(), 3), '不带小孩平均': round(group2.mean(), 3),
                 '差异': round(group1.mean() - group2.mean(), 3), 'p值': round(p_val, 4),
                 '显著性': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))})
    return pd.DataFrame(results)


def generate_sample_data():
    np.random.seed(42)
    n = 200
    shop_types = ['网红茶饮店', '亲子手作工坊', '两岸文创店', '汉服旅拍馆', '台湾小吃摊', '轻食简餐店', '伴手礼店',
                  'VR体验馆', '民俗手作店', '独立书店']
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
            exclude = ['年龄', '带小孩', '来访次数', '收入', '收入水平', '性别', '年龄分组']
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
    st.markdown("选择你的身份，获取专属业态推荐")

    # 使用更紧凑的布局，分两列
    col1, col2 = st.columns(2)
    with col1:
        user_age = st.select_slider(
            "年龄",
            options=["18-25岁", "26-35岁", "36-45岁", "46-60岁", "60岁以上"],
            value="26-35岁"
        )
        user_gender = st.selectbox("性别", ["男", "女", "不愿透露"])
        user_income = st.selectbox(
            "月收入",
            ["3千以下", "3千-5千", "5千-1万", "1万-2万", "2万以上"]
        )
    with col2:
        user_child = st.selectbox("是否有小孩", ["无", "有"])
        user_travel_with = st.selectbox(
            "出游同伴",
            ["独自一人", "朋友", "情侣/伴侣", "带小孩的家庭", "父母长辈", "公司团建"]
        )
        user_interest = st.multiselect(
            "兴趣爱好",
            ["拍照打卡", "品尝美食", "文化体验", "亲子互动", "刺激冒险", "休闲放松", "购物"],
            default=["拍照打卡", "品尝美食"]
        )

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
                    if 'VR' in shop:
                        score += 0.3
                elif user_age in ["46-60岁", "60岁以上"]:
                    if '文创' in shop or '伴手礼' in shop:
                        score += 0.4
                    if '两岸' in shop:
                        score += 0.3

                if user_child == "有" and ('亲子' in shop or '手作' in shop):
                    score += 0.6

                if user_travel_with == "情侣/伴侣" and ('汉服' in shop or '网红' in shop):
                    score += 0.4
                if user_travel_with == "带小孩的家庭" and ('亲子' in shop):
                    score += 0.5
                if user_travel_with == "朋友" and ('VR' in shop or '网红' in shop):
                    score += 0.3
                if user_travel_with == "父母长辈" and ('文创' in shop or '两岸' in shop):
                    score += 0.4

                if '拍照打卡' in user_interest and ('汉服' in shop or '网红' in shop):
                    score += 0.3
                if '品尝美食' in user_interest and ('小吃' in shop or '轻食' in shop):
                    score += 0.4
                if '文化体验' in user_interest and ('文创' in shop or '两岸' in shop or '民俗' in shop):
                    score += 0.4
                if '亲子互动' in user_interest and ('亲子' in shop or '手作' in shop):
                    score += 0.5
                if '刺激冒险' in user_interest and ('VR' in shop):
                    score += 0.3
                if '休闲放松' in user_interest and ('茶饮' in shop or '轻食' in shop):
                    score += 0.3
                if '购物' in user_interest and ('伴手礼' in shop):
                    score += 0.4

                if user_gender == '女' and ('汉服' in shop or '网红' in shop):
                    score += 0.2

                if user_income in ["1万-2万", "2万以上"] and ('VR' in shop or '汉服' in shop):
                    score += 0.2

                user_rec.append({'业态': shop, '匹配度': round(score, 2)})

            user_rec_df = pd.DataFrame(user_rec).sort_values('匹配度', ascending=False).head(8)
            max_score = user_rec_df['匹配度'].max()

            st.session_state.user_recommendation = user_rec_df
            st.session_state.user_max_score = max_score

            st.success(f"✨ 推荐生成成功！最高匹配度 {max_score}/5分")

    if st.session_state.user_recommendation is not None:
        st.markdown("---")
        st.markdown(f"**🎯 你的专属推荐 TOP 8** (满分5分)")

        for i, row in st.session_state.user_recommendation.iterrows():
            stars = "⭐" * min(5, int(row['匹配度']))
            st.write(f"{i + 1}. **{row['业态']}** {stars} ({row['匹配度']}/5分)")

        st.caption(
            f"💡 推荐基于：年龄、性别、收入、出游同伴、兴趣爱好等维度综合计算 | 最高匹配度 {st.session_state.user_max_score}/5分")

# =========================================================
# 主界面 - 选项卡
# =========================================================
if st.session_state.df is not None:
    df = st.session_state.df
    shop_types = st.session_state.shop_types
    means = df[shop_types].mean()

    if len(shop_types) == 0:
        st.error("❌ 未识别到业态列")
        st.stop()

    tab_stats, tab_pk, tab_quiz, tab_custom, tab_rank = st.tabs([
        "📊 统计分析", "🎮 业态PK", "🎯 匹配测试", "⚙️ 自定义推荐", "🏆 排行榜"
    ])

    # Tab 1: 统计分析
    with tab_stats:
        st.subheader("📊 统计分析报告")

        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "描述统计", "相关性", "回归分析", "聚类分析", "假设检验"
        ])

        with sub_tab1:
            col1, col2 = st.columns(2)
            with col1:
                stats_df = descriptive_statistics(df, shop_types)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                means_plot = stats_df.set_index('业态')['均值'].sort_values()
                colors = ['#2ecc71' if x >= 3.8 else '#f39c12' if x >= 3.5 else '#e74c3c' for x in means_plot.values]
                bars = ax.barh(means_plot.index, means_plot.values, color=colors)
                ax.set_xlabel('平均评分')
                ax.set_title('业态满意度排名')
                ax.axvline(x=3.5, color='gray', linestyle='--', label='参考线3.5分')
                for bar, val in zip(bars, means_plot.values):
                    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f'{val:.2f}', va='center',
                            fontsize=9)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        with sub_tab2:
            st.markdown("### 业态间相关性矩阵")
            pearson_matrix, _ = correlation_analysis(df, shop_types)
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(pearson_matrix, dtype=bool))
            sns.heatmap(pearson_matrix, mask=mask, annot=True, fmt='.2f',
                        xticklabels=shop_types, yticklabels=shop_types, cmap='RdBu_r', center=0, ax=ax)
            ax.set_title('皮尔逊相关系数')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

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

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(reg_result['y_true'], reg_result['y_pred'], alpha=0.5)
                ax.plot([1, 5], [1, 5], 'r--', label='完美预测线')
                ax.set_xlabel('实际值')
                ax.set_ylabel('预测值')
                ax.set_title(f'预测 vs 实际 (R² = {reg_result["r2"]:.3f})')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("需要年龄、带小孩等变量进行回归分析")

        with sub_tab4:
            cluster_result = cluster_analysis(df, shop_types)
            if cluster_result:
                st.metric("最佳聚类数", f"{cluster_result['best_k']} 类")
                fig, ax = plt.subplots(figsize=(10, 5))
                k_range = range(2, len(cluster_result['inertias']) + 2)
                ax.plot(k_range, cluster_result['silhouette_scores'], 'ro-')
                ax.set_xlabel('K值')
                ax.set_ylabel('轮廓系数')
                ax.set_title('轮廓系数图（越高越好）')
                best_idx = np.argmax(cluster_result['silhouette_scores'])
                best_k = k_range[best_idx]
                best_score = cluster_result['silhouette_scores'][best_idx]
                ax.plot(best_k, best_score, 'go', markersize=10)
                ax.annotate(f'最佳 K={best_k}\n系数={best_score:.3f}', xy=(best_k, best_score),
                            xytext=(best_k + 0.5, best_score + 0.05))
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(cluster_result['profiles'], use_container_width=True, hide_index=True)
            else:
                st.info("数据不足，无法聚类")

        with sub_tab5:
            test_type = st.radio("选择检验", ["方差分析 ANOVA", "t检验"], horizontal=True)
            if test_type == "方差分析 ANOVA":
                anova_df = anova_analysis(df, shop_types)
                if anova_df is not None and len(anova_df) > 0:
                    st.dataframe(anova_df, use_container_width=True, hide_index=True)
                else:
                    st.info("需要年龄列")
            else:
                ttest_df = t_test_family(df, shop_types)
                if ttest_df is not None and len(ttest_df) > 0:
                    st.dataframe(ttest_df, use_container_width=True, hide_index=True)
                else:
                    st.info("需要带小孩列")

    # Tab 2: 业态PK
    with tab_pk:
        st.subheader("🎮 业态PK对战")
        st.caption("选择两个业态对战，每项满分5分")

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
            score1_norm = [score1[i] / max_vals[i] * 5 if max_vals[i] > 0 else 0 for i in range(len(dimensions))]
            score2_norm = [score2[i] / max_vals[i] * 5 if max_vals[i] > 0 else 0 for i in range(len(dimensions))]

            fig = go.Figure()
            fig.add_trace(go.Bar(name=shop1, x=dimensions, y=score1_norm, marker_color='#ff6b6b',
                                 text=[f'{x:.1f}' for x in score1_norm], textposition='outside'))
            fig.add_trace(go.Bar(name=shop2, x=dimensions, y=score2_norm, marker_color='#4ecdc4',
                                 text=[f'{x:.1f}' for x in score2_norm], textposition='outside'))
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
        st.caption("回答6个问题，匹配最适合你的TOP3业态")

        with st.form("quiz_form"):
            col1, col2 = st.columns(2)
            with col1:
                q1 = st.radio("1. 你通常和谁一起出游？",
                              ["独自一人", "朋友", "情侣/伴侣", "带小孩的家庭", "父母长辈"])
                q2 = st.radio("2. 你最喜欢的体验类型？",
                              ["拍照打卡", "品尝美食", "动手制作体验", "文化学习", "刺激冒险"])
                q3 = st.radio("3. 你的预算范围？",
                              ["50元以下", "50-100元", "100-200元", "200元以上"])
            with col2:
                q4 = st.radio("4. 你更看重什么？",
                              ["性价比", "独特体验", "服务质量", "环境氛围", "社交属性"])
                q5 = st.radio("5. 你计划停留时间？",
                              ["1小时以内", "1-2小时", "2-3小时", "半天以上"])
                q6 = st.radio("6. 你是第几次来平潭？",
                              ["第1次", "第2-3次", "4次以上"])

            submitted = st.form_submit_button("🔮 开始匹配", use_container_width=True, type="primary")

            if submitted:
                match_scores = {shop: means[shop] * 0.3 for shop in shop_types}

                if q1 == "带小孩的家庭":
                    for s in shop_types:
                        if '亲子' in s or '手作' in s:
                            match_scores[s] += 0.6
                if q1 == "情侣/伴侣":
                    for s in ['汉服旅拍馆', '网红茶饮店']:
                        if s in match_scores:
                            match_scores[s] += 0.5
                if q1 == "朋友":
                    for s in ['网红茶饮店', 'VR体验馆']:
                        if s in match_scores:
                            match_scores[s] += 0.4
                if q1 == "父母长辈":
                    for s in ['两岸文创店', '伴手礼店']:
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
                if q2 == "动手制作体验":
                    for s in ['亲子手作工坊', '民俗手作店']:
                        if s in match_scores:
                            match_scores[s] += 0.5
                if q2 == "文化学习":
                    for s in ['两岸文创店', '独立书店']:
                        if s in match_scores:
                            match_scores[s] += 0.5
                if q2 == "刺激冒险":
                    for s in ['VR体验馆']:
                        if s in match_scores:
                            match_scores[s] += 0.5

                if q3 == "200元以上":
                    for s in ['VR体验馆', '汉服旅拍馆']:
                        if s in match_scores:
                            match_scores[s] += 0.3

                if q4 == "独特体验":
                    for s in ['VR体验馆', '汉服旅拍馆']:
                        if s in match_scores:
                            match_scores[s] += 0.3
                if q4 == "性价比":
                    for s in ['台湾小吃摊']:
                        if s in match_scores:
                            match_scores[s] += 0.3

                if q5 == "半天以上":
                    for s in ['轻食简餐店', '独立书店']:
                        if s in match_scores:
                            match_scores[s] += 0.2

                if q6 in ["第2-3次", "4次以上"]:
                    for s in ['伴手礼店']:
                        if s in match_scores:
                            match_scores[s] += 0.3
                    for s in ['台湾小吃摊']:
                        if s in match_scores:
                            match_scores[s] += 0.2

                match_df = pd.DataFrame(list(match_scores.items()), columns=['业态', '匹配分']).sort_values('匹配分',
                                                                                                            ascending=False).head(
                    3)
                st.session_state.quiz_result = match_df
                st.success(f"✨ 匹配完成！最高分 {match_df.iloc[0]['匹配分']:.1f}/5分")

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
                            <div>{score:.1f}/5分</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#f0f2f6; padding:12px; border-radius:10px; margin-top:15px;">
                <strong>💡 推荐理由：</strong><br>
                根据你的偏好，推荐 <strong>{st.session_state.quiz_result.iloc[0]['业态']}</strong>（{st.session_state.quiz_result.iloc[0]['匹配分']:.1f}/5分）
            </div>
            """, unsafe_allow_html=True)

    # Tab 4: 自定义推荐
    with tab_custom:
        st.subheader("⚙️ 自定义推荐引擎")
        st.caption("调整权重，推荐分满分5分")

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("**权重设置**")
            w1 = st.slider("总体满意度", 0, 100, 40, key="cw1")
            w2 = st.slider("年轻人偏好", 0, 100, 25, key="cw2")
            w3 = st.slider("家庭偏好", 0, 100, 20, key="cw3")
            w4 = st.slider("差异化程度", 0, 100, 15, key="cw4")
            w5 = st.slider("高收入偏好", 0, 100, 0, key="cw5")
            total = w1 + w2 + w3 + w4 + w5
            if total == 0:
                total = 1

        with col2:
            custom_scores = []
            for shop in shop_types:
                score = (means[shop] * w1 / total +
                         (-pearsonr(df[shop], df['年龄'])[0] if '年龄' in df.columns else 0) * w2 / total * 3 +
                         (df[df['带小孩'] == 1][shop].mean() if '带小孩' in df.columns else means[shop]) * w3 / total +
                         df[shop].std() * w4 / total)

                if '收入水平' in df.columns:
                    high_income_score = df[df['收入水平'].isin(['1万-2万', '2万以上'])][shop].mean()
                    score += (high_income_score if not pd.isna(high_income_score) else means[shop]) * w5 / total

                score = min(5, score)
                custom_scores.append({'业态': shop, '推荐分': round(score, 3)})

            custom_df = pd.DataFrame(custom_scores).sort_values('推荐分', ascending=False)
            st.dataframe(custom_df.head(10), use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            top6 = pd.DataFrame(custom_scores).sort_values('推荐分', ascending=False).head(6)
            bars = ax.barh(top6['业态'], top6['推荐分'], color='#ff6b6b')
            ax.set_xlabel('推荐分')
            ax.set_title('自定义权重推荐 TOP6')
            for bar, val in zip(bars, top6['推荐分']):
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f'{val:.2f}', va='center',
                        fontsize=9)
            st.pyplot(fig)
            plt.close(fig)

    # Tab 5: 排行榜
    with tab_rank:
        st.subheader("🏆 业态排行榜")

        rank_tab1, rank_tab2, rank_tab3, rank_tab4 = st.tabs(["综合排名", "年轻人最爱", "家庭最爱", "高收入最爱"])

        with rank_tab1:
            rank_df = pd.DataFrame({
                '排名': range(1, len(shop_types) + 1),
                '业态': means.index,
                '得分': means.values.round(2),
                '星级': ['⭐' * min(5, int(x)) for x in means.values]
            })
            st.dataframe(rank_df, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            top5 = means.head(5)
            bars = ax.bar(top5.index, top5.values, color=['gold', 'silver', '#cd7f32', '#3498db', '#95a5a6'])
            ax.set_ylabel('平均评分')
            ax.set_title('综合评分 TOP5')
            ax.axhline(y=3.5, color='gray', linestyle='--', label='参考线3.5分')
            for bar, val in zip(bars, top5.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{val:.2f}', ha='center',
                        fontsize=10)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with rank_tab2:
            if '年龄' in df.columns:
                youth_scores = {shop: -pearsonr(df[shop], df['年龄'])[0] for shop in shop_types}
                youth_df = pd.DataFrame(list(youth_scores.items()), columns=['业态', '年轻偏好分']).sort_values(
                    '年轻偏好分', ascending=False)
                youth_df['排名'] = range(1, len(youth_df) + 1)
                st.dataframe(youth_df, use_container_width=True, hide_index=True)
            else:
                st.info("需要年龄数据")

        with rank_tab3:
            if '带小孩' in df.columns:
                family_scores = {shop: df[df['带小孩'] == 1][shop].mean() for shop in shop_types}
                family_df = pd.DataFrame(list(family_scores.items()), columns=['业态', '家庭评分']).sort_values(
                    '家庭评分', ascending=False)
                family_df['排名'] = range(1, len(family_df) + 1)
                st.dataframe(family_df, use_container_width=True, hide_index=True)
            else:
                st.info("需要带小孩数据")

        with rank_tab4:
            if '收入水平' in df.columns:
                high_income_scores = {shop: df[df['收入水平'].isin(['1万-2万', '2万以上'])][shop].mean() for shop in
                                      shop_types}
                high_df = pd.DataFrame(list(high_income_scores.items()), columns=['业态', '高收入评分']).sort_values(
                    '高收入评分', ascending=False)
                high_df['排名'] = range(1, len(high_df) + 1)
                st.dataframe(high_df, use_container_width=True, hide_index=True)
            else:
                st.info("需要收入水平数据")

else:
    st.info("👈 请从左侧选择数据来源（示例数据或上传CSV）")

st.divider()
st.caption("© 平潭文旅课题组 | 所有评分均标注满分标准")