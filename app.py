"""실업자훈련 대시보드 — 국기훈련은 계좌제보다 '덜 수료하지만 더 취업시키는' 과정인가?

실행:  streamlit run app.py
데이터: data/*.csv (prep_data.py 로 원본 xlsx 에서 생성)

구성
- 머리말(공통): 제목·출처 → 과정·지표 정의
- 탭 1 「과정별 성과 분석」: ③ KPI → ④ 추이 → ⑤ 같은 직종 비교 → ⑥ 변화의 출처(+직종 상세)
                           → ⑦ 직종 구성 → ⑧ 결론 → ⑨ 한계
- 탭 2 「부록: 훈련 실시 현황」: 월별 훈련 시작 인원, 계절성, 시도별
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------ 상수
ACC, NAT = "실업자계좌제", "국가기간전략산업직종훈련"
COURSES = [ACC, NAT]
SHORT = {ACC: "계좌제", NAT: "국기훈련"}
COLOR = {ACC: "#2a78d6", NAT: "#eb6834"}          # 범주 슬롯 1·2 (색약 검증 통과)
GREY, INK, INK2, GRID = "#8a8985", "#1f1f1e", "#52514e", "#e6e6e3"
OTHER_GREYS = ["#5f5e5a", "#8a8985", "#aeaca5", "#c9c7c0", "#dcdad4"]
REST_COLOR = "#eeede9"
SWITCH = 2024                                      # 2025 직종코드가 적용되기 시작한 해
ALL_JOBS = "전체"
METRICS = {"취업률": "employment_rate", "수료율": "completion_rate"}
PEOPLE_NOTE = ("훈련 시작 인원 = 훈련시작일이 해당 월인 과정의 참여자 수. 동일인이 여러 과정을 수강하면 중복 산정(연인원)되어 "
               "실제 사람 수보다 많을 수 있습니다. 해당 월 값은 다음 해 1월까지 재산정됩니다(2026-11-24 조회).")


# ------------------------------------------------------------------ 폰트·스타일
def setup_style():
    for f in (ROOT / "fonts").glob("*.ttf"):
        fm.fontManager.addfont(str(f))
    sns.set_theme(style="whitegrid", font="NanumGothic", rc={
        "axes.unicode_minus": False,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
        "legend.frameon": False, "figure.dpi": 110,
    })


@st.cache_data
def load_data():
    rates = pd.read_csv(ROOT / "data/rates.csv", dtype={"code": "string"})
    monthly = pd.read_csv(ROOT / "data/monthly.csv", dtype={"code": "string"})
    monthly["date"] = pd.to_datetime(monthly["ym"])
    return rates, monthly


# ------------------------------------------------------------------ 보조 함수
def pct(x, digits=1):
    return "–" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def pp(x):
    return None if pd.isna(x) else f"{x * 100:+.1f}%p"


def short_job(name):
    """'금속·재료 설치·정비·생산직(판금·단조…)' → '금속·재료 설치·정비·생산직'."""
    return name.split("(")[0].strip() if "(" in name and not name.startswith("관리직") else name


def fmt_pp(x, contrib=False):
    """%p 막대 라벨. 작은 값이 '0.0'으로 뭉개지지 않게 자릿수를 늘린다.
    기여: 소수 둘째 자리(0.005 미만은 '<0.01'), 변화: 1%p 미만만 둘째 자리."""
    if x == 0:
        return "0"
    if contrib:
        return "<0.01" if abs(x) < 0.005 else f"{x:+.2f}"
    return f"{x:+.2f}" if abs(x) < 1 else f"{x:+.1f}"


def code_of(year):
    return "2025" if year >= SWITCH else "2018"


def set_rate_axis(ax, from_zero, min_span=0.10):
    """비율 y축: 0% 기준이거나, 최소 10%p 폭을 보장해 작은 변화가 과장되지 않게 한다."""
    if from_zero:
        ax.set_ylim(0, 1.05)
    else:
        lo, hi = ax.get_ylim()
        mid, span = (lo + hi) / 2, max(hi - lo, min_span) * 1.15
        ax.set_ylim(max(mid - span / 2, 0), min(mid + span / 2, 1.05))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def pairs_for(course, metric):
    """같은 직종코드 안에서 비교 가능한 (시작, 끝, 코드) 연도 쌍."""
    jr = rates[(rates.level == "직종") & (rates.course == course)].dropna(subset=[metric])
    out = []
    for cd in ("2018", "2025"):
        ys = sorted(int(y) for y in jr[jr.code == cd].year.unique())
        out += [(a, b, cd) for i, a in enumerate(ys) for b in ys[i + 1:]]
    return out


def decompose(course, metric, pa, pb, pcode):
    """직종별 비율 변화와 '시작 연도 인원 비중 × 변화'로 근사한 기여."""
    jr = rates[(rates.level == "직종") & (rates.course == course) & (rates.code == pcode)]
    v = jr[jr.year.isin([pa, pb])].pivot_table(index="job", columns="year", values=metric)
    v = v.reindex(columns=[pa, pb]).dropna()
    v["n"] = [starts.get((course, pcode, pa, j), 0) for j in v.index]
    v = v[v.n > 0]
    v["w"] = v.n / v.n.sum()
    v["delta"] = v[pb] - v[pa]
    v["contrib"] = v.w * v.delta
    official = totals.loc[pb, (metric, course)] - totals.loc[pa, (metric, course)]
    return v.sort_values("contrib", ascending=False), official


def biggest_change_pair(course, metric, y0, y1):
    """연도 범위 안의 '연속 연도' 쌍 중 공식 전체값 변화가 가장 큰 쌍. 없으면 None."""
    cands = [p for p in pairs_for(course, metric) if y0 <= p[0] and p[1] <= y1 and p[1] - p[0] == 1]
    if not cands:
        return None
    return max(cands, key=lambda p: abs(totals.loc[p[1], (metric, course)] - totals.loc[p[0], (metric, course)]))


# ================================================================== 데이터·사이드바
st.set_page_config(page_title="실업자훈련 성과 대시보드", page_icon="📊", layout="wide")
setup_style()
rates, monthly = load_data()
totals = rates[rates.level == "전체"].pivot_table(
    index="year", columns="course", values=["completion_rate", "employment_rate"])
starts = monthly.groupby(["course", "code", "year", "job"]).trainees.sum()
key_of = pd.concat([rates[["job", "job_key"]], monthly[["job", "job_key"]]]).dropna().drop_duplicates() \
    .set_index("job").job_key.to_dict()
name_of = (pd.concat([rates[["code", "job", "job_key"]], monthly[["code", "job", "job_key"]]]).dropna()
           .sort_values("code").groupby("job_key").job.last().map(short_job).to_dict())   # 표시명 = 최신 이름

st.sidebar.header("조건 선택")
y0, y1 = st.sidebar.select_slider("연도 범위", options=list(range(2021, 2026)), value=(2021, 2025))
st.sidebar.caption(f"KPI·같은 직종 비교의 기준 연도는 범위의 끝인 **{y1}년**입니다.")
job_keys = sorted(rates[(rates.level == "직종")].dropna(subset=["employment_rate"]).job_key.unique(),
                  key=lambda k: name_of[k])
sel = st.sidebar.selectbox("직종 (하나 선택)", [ALL_JOBS] + job_keys,
                           format_func=lambda k: k if k == ALL_JOBS else name_of[k])
st.sidebar.caption("직종을 고르면 KPI에 그 직종 값이 추가되고, ⑤·⑥·⑦ 차트에서 그 직종이 강조됩니다. "
                   "비율은 여러 직종을 합칠 수 없어(가중치 미공개) 한 번에 하나만 고릅니다.")
years = list(range(y0, y1 + 1))

# ================================================================== 머리말 (공통)
st.title("국기훈련은 '덜 수료하지만 더 취업시키는' 과정인가?")
st.markdown("실업자훈련의 두 과정의 수료율·취업률을 2021~2025년에 걸쳐 비교하고, "
            "국기훈련의 취업 우위가 유지되었는지, 달라졌다면 어디서 달라졌는지 확인합니다.")
st.caption("출처: 한국고용정보원 고용행정통계(원자료: HRD-Net) — 실업자훈련 수료율현황(년)·취업률현황(년)·실시현황(월) · 2026-11-24 조회")

d1, d2, d3 = st.columns(3)
with d1, st.container(border=True):
    st.markdown(f"<span style='color:{COLOR[ACC]};font-weight:700'>● 실업자계좌제 (계좌제)</span>", unsafe_allow_html=True)
    st.markdown("고용행정통계에서 **'내일배움카드' 과정**으로 분류되는 실업자훈련(통계 표기: 실업자계좌제). "
                "국기훈련 지정 직종에 해당하지 않는 **일반 과정**으로, 사무·서비스·IT·디자인 등 분야가 넓다.")
with d2, st.container(border=True):
    st.markdown(f"<span style='color:{COLOR[NAT]};font-weight:700'>● 국가기간전략산업직종훈련 (국기훈련)</span>",
                unsafe_allow_html=True)
    st.markdown("국가 기간·전략산업 등 인력 양성이 필요해 **고용노동부가 별도로 지정한 훈련직종**"
                "(국민내일배움카드 운영규정 별표 1)의 실업자훈련 과정.")
with d3, st.container(border=True):
    st.markdown("**지표 정의** (연도 = 훈련이 끝난 해)")
    st.markdown("- **수료율** = 수료인원 ÷ 실시인원\n"
                "- **취업률** = 수료 후 6개월 이내 취업인원 ÷ 수료인원\n"
                "- 연도별 값만 비교하며 여러 해를 합하거나 평균내지 않음")

st.caption("두 과정 모두 실제 수강 신청은 국민내일배움카드로 하며, 차이는 지정 직종 과정인지 여부입니다. "
           "이 통계의 실업자훈련은 두 과정으로만 구성되고 두 과정의 합이 총계와 같습니다.")

tab_main, tab_app = st.tabs(["📈 과정별 성과 분석", "📎 부록: 훈련 실시 현황"])

# ================================================================== 탭 1
with tab_main:
    # -------------------------------------------------------------- ③ KPI
    st.subheader(f"{y1}년 한눈에 보기")
    prev = y1 - 1 if (y1 - 1) in totals.index else None
    kc = st.columns(3)

    def job_rate(course, metric, year):
        r = rates[(rates.job_key == sel) & (rates.course == course) & (rates.year == year)
                  & (rates.code == code_of(year))]
        return r[metric].iloc[0] if len(r) else np.nan

    for col, c in zip(kc, COURSES):
        with col, st.container(border=True):
            st.markdown(f"<span style='color:{COLOR[c]};font-weight:700'>● {SHORT[c]}</span>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            for mc, (lab, mt) in zip((m1, m2), [("수료율", "completion_rate"), ("취업률", "employment_rate")]):
                v = totals.loc[y1, (mt, c)]
                mc.metric(lab, pct(v), pp(v - totals.loc[prev, (mt, c)]) if prev else None)
            if sel != ALL_JOBS:
                jc, je = job_rate(c, "completion_rate", y1), job_rate(c, "employment_rate", y1)
                st.caption(f"**{name_of[sel]}**: 수료율 {pct(jc)}, 취업률 {pct(je)}"
                           + ("" if not (pd.isna(jc) and pd.isna(je)) else " (자료 없음)"))
    with kc[2], st.container(border=True):
        st.markdown("<span style='font-weight:700'>격차 (국기훈련 - 계좌제)</span>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        for mc, (lab, mt) in zip((m1, m2), [("수료율", "completion_rate"), ("취업률", "employment_rate")]):
            g = totals[mt][NAT] - totals[mt][ACC]
            mc.metric(lab, f"{g[y1] * 100:+.1f}%p", pp(g[y1] - g[prev]) if prev else None, delta_color="off")
        if sel != ALL_JOBS:
            gc = job_rate(NAT, "completion_rate", y1) - job_rate(ACC, "completion_rate", y1)
            ge = job_rate(NAT, "employment_rate", y1) - job_rate(ACC, "employment_rate", y1)
            st.caption(f"**{name_of[sel]}**: 수료율 {pp(gc) or '–'}, 취업률 {pp(ge) or '–'}")
    st.caption("증감은 전년 대비. 값은 원자료의 과정 '전체' 값(직종 평균이 아님). 직종 값은 해당 연도 직종코드 기준.")

    # -------------------------------------------------------------- ④ 추이
    st.subheader("과정별 수료율·취업률 추이")
    y_from_zero = st.toggle("y축을 0%부터 표시", value=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=y_from_zero)
    for ax, mt, title in zip(axes, ["completion_rate", "employment_rate"], ["수료율", "취업률"]):
        for c in COURSES:
            s = totals[mt][c].loc[y0:y1]
            ax.plot(s.index, s.values, color=COLOR[c], lw=2, marker="o", ms=7, mec="white", mew=1.5, label=SHORT[c])
            for x in ((s.index[0], s.index[-1]) if len(s) > 1 else (s.index[0],)):
                ax.annotate(pct(s[x]), (x, s[x]), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=9, color=INK)
        if len(years) > 1:
            a, n = totals[mt][ACC].loc[y0:y1], totals[mt][NAT].loc[y0:y1]
            ax.fill_between(a.index, a.values, n.values, color=GREY, alpha=0.12, lw=0)
            for x in (y0, y1):
                ax.annotate(f"격차 {(n[x] - a[x]) * 100:+.1f}%p", (x, (n[x] + a[x]) / 2),
                            ha="left" if x == y0 else "right", va="center", fontsize=9, color=INK2,
                            xytext=(6 if x == y0 else -6, 0), textcoords="offset points")
        ax.set_title(title, loc="left")
        ax.set_xticks(years)
        set_rate_axis(ax, y_from_zero)
        ax.set_xlim(y0 - 0.4, y1 + 0.4)
    axes[0].legend(loc="lower left", ncol=2)
    fig.tight_layout()
    show(fig)
    st.caption("과정 '전체' 값은 직종분류 개편과 무관하게 5년이 이어진 계열입니다. 음영 = 두 과정의 차이."
               + ("" if y_from_zero else " y축이 0에서 시작하지 않아 차이가 확대되어 보입니다(위 토글로 0% 기준 전환)."))
    if len(years) > 1:
        n0, n1 = totals.loc[y0, ("employment_rate", NAT)], totals.loc[y1, ("employment_rate", NAT)]
        a0, a1 = totals.loc[y0, ("employment_rate", ACC)], totals.loc[y1, ("employment_rate", ACC)]
        st.info(f"**{y0}→{y1}년 취업률:** 국기훈련 {pct(n0)} → {pct(n1)} ({(n1 - n0) * 100:+.1f}%p), "
                f"계좌제 {pct(a0)} → {pct(a1)} ({(a1 - a0) * 100:+.1f}%p). "
                f"격차는 {(n0 - a0) * 100:.1f}%p → {(n1 - a1) * 100:.1f}%p.")

    # -------------------------------------------------------------- ⑤ 같은 직종 비교
    st.subheader(f"같은 직종이어도 국기훈련이 더 취업시키나 — {y1}년 직종별 비교")
    st.markdown("국기훈련이 원래 취업률 높은 직종에 몰려 있어서 생긴 차이인지, **같은 직종끼리** 비교해 확인합니다.")
    s1, s2 = st.columns([1, 2])
    with s1:
        metric_label = st.radio("비교 지표 (이 구역과 아래 구역에 적용)", list(METRICS), horizontal=True)
    metric = METRICS[metric_label]
    with s2:
        min_n = st.slider("두 과정 모두 이 인원 이상인 직종만 (해당 연도 전국 훈련 시작 인원)", 0, 1000, 50, step=10)
    code = code_of(y1)
    full = (rates[(rates.level == "직종") & (rates.year == y1) & (rates.code == code)]
            .pivot_table(index="job", columns="course", values=metric).reindex(columns=COURSES).dropna())
    for c in COURSES:
        full[f"n_{c}"] = [starts.get((c, code, y1, j), 0) for j in full.index]
    full["diff"] = full[NAT] - full[ACC]
    d = full[(full[f"n_{ACC}"] >= min_n) & (full[f"n_{NAT}"] >= min_n)].sort_values("diff")
    if d.empty:
        st.warning("조건에 맞는 직종이 없습니다. 최소 인원을 낮춰 보세요.")
    else:
        fig, ax = plt.subplots(figsize=(12, 0.42 * len(d) + 1.3))
        yy = np.arange(len(d))
        hl = [key_of.get(j) == sel for j in d.index]
        for i, h in enumerate(hl):
            if h:
                ax.axhspan(i - 0.45, i + 0.45, color="#fff1e0", zorder=0)
        ax.hlines(yy, d[ACC], d[NAT], color=GREY, lw=2, alpha=0.5, zorder=1)
        for c in COURSES:
            ax.scatter(d[c], yy, s=80, color=COLOR[c], edgecolor="white", linewidth=1.5, zorder=3, label=SHORT[c])
        for i, v in enumerate(d["diff"]):
            ax.annotate(f"{v * 100:+.1f}%p", (1.005, i), xycoords=("axes fraction", "data"), va="center",
                        fontsize=9, color=INK if v >= 0 else COLOR[NAT], fontweight="bold" if v < 0 else "normal")
        ax.set_yticks(yy, [short_job(j) for j in d.index])
        for lab, h in zip(ax.get_yticklabels(), hl):
            if h:
                lab.set_fontweight("bold")
                lab.set_color(INK)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(metric_label)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.7, len(d) - 0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 + 0.9 / (0.42 * len(d) + 1.3)), ncol=2)
        ax.annotate("국기 - 계좌제", (1.005, 1), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", fontsize=8.5, color=INK2)
        fig.tight_layout()
        show(fig)
        win, win_all = int((d["diff"] > 0).sum()), int((full["diff"] > 0).sum())
        st.markdown(f"**{len(d)}개 직종 중 {win}개**에서 국기훈련 {metric_label}이 더 높습니다 "
                    f"(차이 중앙값 {d['diff'].median() * 100:+.1f}%p).")
        hidden = full[~full.index.isin(d.index)]
        note = ""
        if len(hidden):
            lows = hidden[hidden["diff"] <= 0]
            note = (f" 인원 기준으로 제외된 {len(hidden)}개 직종까지 포함하면 {len(full)}개 중 {win_all}개"
                    + (f" (예외: {', '.join(short_job(j) for j in lows.index)})." if len(lows) else "."))
        sel_note = "" if sel == ALL_JOBS or any(hl) else f" 선택한 직종({name_of[sel]})은 이 조건에서 비교 대상이 아닙니다."
        st.caption(f"{y1}년, {code} 직종코드 기준으로 두 과정 모두에 값이 있는 직종만 비교.{note}{sel_note} "
                   "같은 직종이라도 과정 내용·훈련생 특성이 다를 수 있어 인과적 효과로 해석할 수는 없습니다.")
        with st.expander("표로 보기"):
            st.dataframe(pd.DataFrame({
                "직종": d.index, "계좌제": d[ACC].map(pct), "국기훈련": d[NAT].map(pct),
                "차이(%p)": (d["diff"] * 100).round(1),
                "계좌제 인원": d[f"n_{ACC}"].astype(int), "국기 인원": d[f"n_{NAT}"].astype(int)}).iloc[::-1],
                hide_index=True)

    # -------------------------------------------------------------- ⑥ 변화의 출처
    st.subheader(f"{metric_label} 변화는 어느 직종에서 왔나")
    st.markdown("직종코드가 같은 두 연도 사이에서만 비교합니다(2018 코드끼리 또는 2025 코드끼리).")
    c1, c2 = st.columns(2)
    with c1:
        dc = st.selectbox("과정 (이 구역과 아래 직종 구성에 적용)", COURSES, index=1, format_func=SHORT.get)
    pairs = pairs_for(dc, metric)
    in_range = [p for p in pairs if y0 <= p[0] and p[1] <= y1]
    opts = in_range or pairs
    best = biggest_change_pair(dc, metric, y0, y1)
    with c2:
        k = st.selectbox("비교 구간", range(len(opts)), index=opts.index(best) if best in opts else 0,
                         format_func=lambda i: f"{opts[i][0]} → {opts[i][1]}  ({opts[i][2]} 코드)",
                         help="기본값은 연도 범위 안에서 전체 값이 가장 크게 변한 연속 연도 구간")
    if not in_range:
        st.caption("현재 연도 범위 안에는 같은 직종코드의 연도가 2개 이상 없어 전체 기간의 구간을 보여 줍니다.")
    pa, pb, pcode = opts[k]
    v, official = decompose(dc, metric, pa, pb, pcode)

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.36 * len(v) + 1.6), sharey=True)
    yy = np.arange(len(v))
    hl = [key_of.get(j) == sel for j in v.index]
    for ax, col, title in zip(axes, ["delta", "contrib"],
                              [f"직종별 {metric_label} 변화 (%p)", "전체 변화에 대한 기여 (%p, 근사)"]):
        vals = v[col] * 100
        ax.barh(yy, vals, color=[COLOR[dc] if (h or sel == ALL_JOBS) else "#f4b79c" for h in hl], height=0.65)
        ax.axvline(0, color=INK2, lw=0.8)
        lim = max(abs(vals).max(), 0.5) * 1.35
        ax.set_xlim(-lim, lim)
        for i, x in enumerate(vals):
            ax.annotate(fmt_pp(x, contrib=(col == "contrib")), (x, i), xytext=(4 if x >= 0 else -4, 0),
                        textcoords="offset points", ha="left" if x >= 0 else "right", va="center",
                        fontsize=8.5, color=INK)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(yy, [f"{short_job(j)}  ({int(n):,}명)" for j, n in zip(v.index, v.n)])
    for lab, h in zip(axes[0].get_yticklabels(), hl):
        if h:
            lab.set_fontweight("bold")
            lab.set_color(INK)
    axes[0].invert_yaxis()
    fig.tight_layout()
    show(fig)
    top = v.contrib.abs().idxmax()
    st.markdown(f"**{SHORT[dc]} 전체 {metric_label}: {pct(totals.loc[pa, (metric, dc)])} → "
                f"{pct(totals.loc[pb, (metric, dc)])} ({official * 100:+.1f}%p).** 직종별 기여를 더하면 "
                f"{v.contrib.sum() * 100:+.1f}%p이고, 그중 **{short_job(top)}**이 {v.contrib[top] * 100:+.1f}%p를 차지합니다.")
    st.caption(f"기여 = ({pa}년 훈련 시작 인원 비중) × (직종의 {metric_label} 변화). 실제 가중치(수료인원 등)가 공개되지 않아 "
               "시작 인원 비중으로 근사했으므로 기여의 합은 공식 전체 변화와 정확히 같지 않습니다(직종 구성 변화·가중치 근사). "
               "괄호 안은 시작 연도의 전국 훈련 시작 인원(연인원).")

    # 직종 상세 추이
    dj = sel if sel != ALL_JOBS else key_of.get(top)
    st.markdown(f"##### 직종 상세: {name_of[dj]}의 5년 추이"
                + ("" if sel != ALL_JOBS else "  \n<small>사이드바에서 직종을 고르면 바뀝니다. 기본값은 위에서 기여가 가장 큰 직종.</small>"),
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=y_from_zero)
    jd = rates[(rates.job_key == dj) & rates.year.between(y0, y1)]
    for ax, mt, title in zip(axes, ["completion_rate", "employment_rate"], ["수료율", "취업률"]):
        for c in COURSES:
            for cd, ls, face in (("2018", "--", "white"), ("2025", "-", None)):
                s = jd[(jd.course == c) & (jd.code == cd)].set_index("year")[mt].dropna().sort_index()
                if s.empty:
                    continue
                ax.plot(s.index, s.values, ls=ls, lw=2, color=COLOR[c], marker="o", ms=7,
                        mfc=face or COLOR[c], mec=COLOR[c] if face else "white", mew=1.5, label=f"{SHORT[c]} · {cd} 코드")
        if y0 < SWITCH <= y1:
            ax.axvline(SWITCH - 0.5, color=GREY, lw=1, ls=":")
            ax.annotate("KECO 개편", (SWITCH - 0.5, 0), xycoords=("data", "axes fraction"), xytext=(-4, 4),
                        ha="right", textcoords="offset points", fontsize=8, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xticks(years)
        ax.set_xlim(y0 - 0.4, y1 + 0.4)
        set_rate_axis(ax, y_from_zero)
    h, l = axes[1].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    show(fig)
    st.caption("점선·속 빈 점 = 2018 직종코드, 실선·속 찬 점 = 2025 직종코드. 2025년 KECO 개편으로 직종의 세부 구성이 달라져 "
               "두 코드를 선으로 잇지 않았습니다. 2024년 취업률은 원자료에 두 코드 값이 함께 있고(수료율은 2025 코드만) "
               "같은 직종명도 값이 다릅니다. 표시되지 않는 점은 원자료 값이 없는 경우입니다.")

    # -------------------------------------------------------------- ⑦ 직종 구성
    st.subheader(f"왜 그 직종의 영향이 큰가 — {SHORT[dc]} 훈련 시작 인원의 직종 구성")
    comp = (monthly[(monthly.course == dc) & monthly.year.between(y0, y1)]
            .groupby(["year", "job_key"]).trainees.sum().unstack(fill_value=0))
    share = comp.div(comp.sum(axis=1), axis=0)
    order = comp.sum().sort_values(ascending=False).index
    focus = sel if (sel != ALL_JOBS and sel in comp.columns) else order[0]
    others = [j for j in order if j != focus][:4]
    cats = [focus] + others
    plot = share[cats].copy()
    plot["기타"] = 1 - plot.sum(axis=1)
    colors = [COLOR[dc]] + OTHER_GREYS[:len(others)] + [REST_COLOR]
    ypos = {y: i + (0.6 if y >= SWITCH else 0) for i, y in enumerate(plot.index)}
    fig, ax = plt.subplots(figsize=(12, 0.62 * len(plot) + 1.6))
    left = np.zeros(len(plot))
    yv = [ypos[y] for y in plot.index]
    for col, colr in zip(plot.columns, colors):
        vals = plot[col].values
        ax.barh(yv, vals, left=left, color=colr, height=0.72, edgecolor="white", linewidth=2,
                label=col if col == "기타" else name_of[col])
        for yy_, l_, v_ in zip(yv, left, vals):
            if v_ >= 0.06:
                dark = colr in (COLOR[dc], OTHER_GREYS[0], OTHER_GREYS[1])
                ax.annotate(f"{v_ * 100:.0f}%", (l_ + v_ / 2, yy_), ha="center", va="center", fontsize=9,
                            color="white" if dark else INK, fontweight="bold" if col == focus else "normal")
        left += vals
    ax.set_yticks(yv, [str(y) for y in plot.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    if y0 < SWITCH <= y1:
        ysw = (ypos[SWITCH - 1] + ypos[SWITCH]) / 2
        ax.axhline(ysw, color=INK2, lw=1, ls=":")
        ax.annotate("KECO 개편: 위 2018 직종코드 / 아래 2025 직종코드", (1, ysw), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08 * 6 / (0.62 * len(plot) + 1.6) - 0.02),
              ncol=3, fontsize=9)
    fig.tight_layout()
    show(fig)
    f0, f1 = share.loc[y0, focus], share.loc[y1, focus]
    st.markdown(f"{SHORT[dc]} 훈련 시작 인원 중 **{name_of[focus]}** 비중: "
                + (f"{y0}년 {pct(f0, 0)} → {y1}년 {pct(f1, 0)}." if y0 != y1 else f"{y1}년 {pct(f1, 0)}."))
    st.caption("각 막대 = 그 해 전국 훈련 시작 인원(연인원) 중 직종별 비율, 인원 상위 4개 직종 외는 '기타'. 같은 이름의 직종을 "
               "같은 색으로 두었지만 개편 앞뒤는 코드 체계가 달라 구성이 정확히 같지 않습니다. "
               "비율(훈련이 끝난 해 기준)과 달리 인원은 훈련을 시작한 해 기준입니다.")

    # -------------------------------------------------------------- ⑧ 결론
    st.subheader("결론")
    gN = totals["employment_rate"][NAT] - totals["employment_rate"][ACC]
    gC = totals["completion_rate"][NAT] - totals["completion_rate"][ACC]
    emp_full = (rates[(rates.level == "직종") & (rates.year == y1) & (rates.code == code)]
                .pivot_table(index="job", columns="course", values="employment_rate").reindex(columns=COURSES).dropna())
    emp_win = int((emp_full[NAT] > emp_full[ACC]).sum())
    bp = biggest_change_pair(NAT, "employment_rate", y0, y1)
    lines = [f"**덜 수료하고 더 취업한다:** {y1}년 국기훈련은 계좌제보다 수료율이 {abs(gC[y1]) * 100:.1f}%p "
             f"{'낮고' if gC[y1] < 0 else '높고'}, 취업률은 {abs(gN[y1]) * 100:.1f}%p {'높다' if gN[y1] > 0 else '낮다'}.",
             f"**직종 구성 탓만은 아니다:** 같은 직종끼리 비교해도 {len(emp_full)}개 중 {emp_win}개 직종에서 국기훈련 취업률이 더 높다({y1}년)."]
    if len(years) > 1:
        lines.append(f"**그러나 우위는 {'줄었다' if gN[y1] < gN[y0] else '유지·확대되었다'}:** 취업률 격차 "
                     f"{gN[y0] * 100:.1f}%p({y0}) → {gN[y1] * 100:.1f}%p({y1}).")
    if bp:
        vb, ob = decompose(NAT, "employment_rate", *bp)
        tb = vb.contrib.abs().idxmax()
        lines.append(f"**변화의 출처:** 국기훈련 취업률이 가장 크게 변한 {bp[0]}→{bp[1]}년({ob * 100:+.1f}%p)의 기여 중 "
                     f"{short_job(tb)}이 {vb.contrib[tb] * 100:+.1f}%p로 가장 크다. 이 직종은 국기훈련 시작 인원의 "
                     f"{pct(vb.w[tb], 0)}({bp[0]}년)를 차지한다.")
    st.success("\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines)))
    st.caption("결론 문장은 사이드바 연도 범위에 따라 데이터에서 다시 계산됩니다(같은 직종 비교는 인원 필터 없이 계산).")

    # -------------------------------------------------------------- ⑨ 한계
    with st.expander("데이터 출처 · 집계 기준 · 한계"):
        st.markdown("""
**출처** 한국고용정보원 고용행정통계(eis.work24.go.kr, 원자료: 고용정보통합분석시스템 HRD-Net) — 실업자훈련
수료율현황(년), 취업률현황(년), 실시현황(월). 2026-11-24 다운로드.
원본 xlsx → `prep_data.py` 로 정리(원본 셀 64,000여 개 역대조 불일치 0건).

**지표 정의 (제공처 설명)**
- 수료율: 훈련종료일이 해당 연도인 과정의 실시인원 대비 수료인원. 다음 해 1월 말 기준 산정.
- 취업률: 훈련종료일이 해당 연도인 과정의 수료생 중 종료일로부터 6개월 이내 취업한 비율. 다음 해 7월 말 기준 산정·8월 말 공개
  → 2025년 취업률(2026년 8월 공개)까지가 현재 받을 수 있는 최신 확정치.
- 훈련 시작 인원: 훈련시작일이 해당 월인 과정의 참여자 수(동일인 중복 산정), 다음 해 1월까지 재산정. 시도는 훈련기관 소재지.

**집계 기준**
- 비율은 연도별 값만 사용하고 여러 연도를 합하거나 평균하지 않음(제공처 주의사항). 과정 '전체' 값은 원자료 값 그대로.
- 원자료에서 해당 직종코드가 쓰이지 않은 연도·월의 0은 결측으로 처리.
- 직종별 기여는 공개되지 않은 실제 가중치 대신 훈련 시작 인원 비중으로 근사.

**한계**
- 과정 구분(계좌제/국기훈련)은 제공처 분류를 그대로 따른다. K-디지털 트레이닝 등 별도 훈련사업이 이 통계에 포함되는지,
  포함된다면 어느 과정으로 집계되는지는 제공처가 밝히지 않아 확인하지 못했다(제공처 문의 예정).
- 2025년 KECO(한국고용직업분류) 개편으로 직종 단위 계열은 2023/2024 사이에서 끊긴다. 2024년 취업률은 두 코드 값이 함께
  제공되는데 같은 직종명도 값이 달라, 이유는 확인하지 못했고 잇지 않았다.
- 비율 값이 0인 칸은 '자료 없음'과 '실제 0%'를 구분할 수 없어 모두 결측으로 처리했다.
- 인원이 적은 직종은 비율이 크게 흔들린다 → 같은 직종 비교에 최소 인원 필터.
- 비율(훈련이 끝난 과정 기준)과 인원(훈련을 시작한 과정 기준)은 대상 집단이 달라 직접 연결하면 근사가 된다.
- 취업률 격차는 과정의 '효과'를 뜻하지 않는다. 같은 직종 안에서도 과정 내용·훈련생 특성이 다를 수 있다.
- 연령·성별 등 훈련생 특성은 데이터 제공처에서 함께 받을 수 없어(차원 4개 제한) 다루지 못했다.
""")

# ================================================================== 탭 2: 부록
with tab_app:
    st.markdown("분석과 별도로, 실업자훈련이 **언제·어디서 얼마나** 시작되는지 보여 주는 참고 자료입니다.")
    a1, a2 = st.columns(2)
    with a1:
        sido = st.selectbox("훈련기관 시도", ["전국"] + sorted(monthly.sido.unique()))
    with a2:
        ac = st.multiselect("과정", COURSES, default=COURSES, format_func=SHORT.get) or COURSES
    job_txt = "전체" if sel == ALL_JOBS else name_of[sel]

    def cond_text(period, with_sido=True):
        parts = [f"기간 **{period}**"] + ([f"시도 **{sido}**"] if with_sido else []) + \
                [f"과정 **{' · '.join(SHORT[c] for c in ac)}**", f"직종 **{job_txt}** (사이드바)"]
        return " · ".join(parts)

    def why_empty(ya, yb, with_sido=True):
        """현재 필터에서 데이터가 비는 이유를 찾아 문장으로 돌려준다."""
        reasons = []
        if sel != ALL_JOBS:
            jm = monthly[monthly.job_key == sel]
            missing = [c for c in ac if jm[jm.course == c].trainees.sum() == 0]
            for c in missing:
                reasons.append(f"'{job_txt}'은(는) 원자료에 **{SHORT[c]} 과정이 없는 직종**입니다.")
            if len(missing) < len(ac):
                jm = jm[jm.course.isin(ac) & jm.year.between(ya, yb)]
                if jm.trainees.sum() == 0:
                    reasons.append(f"이 기간({ya}~{yb})에 '{job_txt}' 훈련이 시작된 기록이 없습니다.")
                elif with_sido and sido != "전국" and jm[jm.sido == sido].trainees.sum() == 0:
                    reasons.append(f"**{sido}**에서는 이 기간에 '{job_txt}' 훈련이 시작된 기록이 없습니다.")
        if not reasons:
            reasons.append("선택한 조건의 훈련 시작 인원이 0명입니다.")
        tip = " 사이드바 직종을 '전체'로 바꾸면 모든 직종을 합쳐 볼 수 있습니다." if sel != ALL_JOBS else ""
        return " ".join(reasons) + tip

    def partial_note():
        """일부 과정만 비는 경우(예: 계좌제에만 있는 직종) 알림."""
        if sel == ALL_JOBS:
            return
        jm = monthly[monthly.job_key == sel]
        missing = [c for c in ac if jm[jm.course == c].trainees.sum() == 0]
        if missing and len(missing) < len(ac):
            st.info(f"'{job_txt}'은(는) {', '.join(SHORT[c] for c in missing)} 과정이 없는 직종이라 "
                    f"{', '.join(SHORT[c] for c in ac if c not in missing)}만 표시됩니다.")

    st.caption("현재 조건: " + cond_text(f"{y0}~{y1}"))
    partial_note()
    m_sel = monthly[monthly.course.isin(ac) & monthly.year.between(y0, y1)]
    if sido != "전국":
        m_sel = m_sel[m_sel.sido == sido]
    if sel != ALL_JOBS:
        m_sel = m_sel[m_sel.job_key == sel]

    st.subheader(f"월별 훈련 시작 인원 ({sido})")
    mp = (m_sel.pivot_table(index="date", columns="course", values="trainees", aggfunc="sum")
          .reindex(columns=ac).fillna(0))
    if mp.empty or mp.values.sum() == 0:
        st.warning(f"표시할 데이터가 없습니다.  \n현재 조건: {cond_text(f'{y0}~{y1}')}  \n{why_empty(y0, y1)}")
    else:
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.stackplot(mp.index, [mp[c] for c in ac], colors=[COLOR[c] for c in ac],
                     labels=[SHORT[c] for c in ac], alpha=0.85, edgecolor="white", lw=1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylabel("명 (연인원)")
        ax.set_xlim(mp.index.min(), mp.index.max())
        ax.legend(loc="upper right", ncol=2)
        fig.tight_layout()
        show(fig)

        b1, b2 = st.columns([3, 2])
        hm = m_sel.pivot_table(index="year", columns="month", values="trainees", aggfunc="sum").fillna(0)
        with b1:
            big = hm.values.max() >= 10000
            fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(hm) + 1.2))
            sns.heatmap(hm / 1000 if big else hm, annot=True, fmt=".1f" if big else ".0f", cmap="Blues",
                        cbar=False, linewidths=2, linecolor="white", annot_kws={"fontsize": 8.5}, ax=ax)
            ax.set_xlabel("월")
            ax.set_ylabel("")
            ax.set_xticklabels([f"{m}월" for m in hm.columns], rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_title(f"연도 × 월 훈련 시작 인원 ({'천 명' if big else '명'})", loc="left")
            fig.tight_layout()
            show(fig)
        with b2:
            mavg = hm.mean()
            st.markdown(f"**계절 패턴** ({y0}~{y1}년 평균)")
            st.markdown(f"- 가장 많이 시작하는 달: **{mavg.idxmax()}월** (월평균 {mavg.max():,.0f}명)\n"
                        f"- 가장 적게 시작하는 달: **{mavg.idxmin()}월** (월평균 {mavg.min():,.0f}명)")
            yearly = m_sel.groupby(["year", "course"]).trainees.sum().unstack().reindex(columns=ac).fillna(0)
            tbl = yearly.rename(columns=SHORT)
            if len(ac) == 2:
                tbl["국기 비중"] = (yearly[NAT] / yearly.sum(axis=1).replace(0, np.nan)).map(pct)
            st.dataframe(tbl.style.format({SHORT[c]: "{:,.0f}" for c in ac}), width="stretch")
        st.caption(PEOPLE_NOTE)

    st.subheader(f"{y1}년 시도별 훈련 시작 인원")
    base = monthly[monthly.course.isin(ac) & (monthly.year == y1)]
    if sel != ALL_JOBS:
        base = base[base.job_key == sel]
    sd = (base.pivot_table(index="sido", columns="course", values="trainees", aggfunc="sum")
          .reindex(columns=ac).fillna(0))
    sd = sd[sd.sum(axis=1) > 0]
    if sd.empty:
        st.warning(f"표시할 데이터가 없습니다.  \n현재 조건: {cond_text(f'{y1}년', with_sido=False)}  \n"
                   f"{why_empty(y1, y1, with_sido=False)}")
    else:
        sd = sd.loc[sd.sum(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(12, max(0.32 * len(sd) + 1.2, 2.5)))
        left = np.zeros(len(sd))
        for c in ac:
            bars = ax.barh(sd.index, sd[c], left=left, color=COLOR[c], height=0.7, label=SHORT[c],
                           edgecolor="white", linewidth=1)
            for b, s in zip(bars, sd.index):
                b.set_alpha(1.0 if (sido == "전국" or s == sido) else 0.3)
            left += sd[c].values
        for i, (s, val) in enumerate(zip(sd.index, left)):
            ax.annotate(f"{val:,.0f}", (val, i), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK if (sido == "전국" or s == sido) else GREY)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("명 (연인원)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        fig.tight_layout()
        show(fig)
        zero = sorted(set(monthly.sido.unique()) - set(sd.index))
        notes = []
        if zero:
            notes.append(f"0명이라 막대가 없는 시도: {', '.join(zero)}.")
        if sido != "전국" and sido in zero:
            notes.append(f"선택한 시도({sido})는 {y1}년에 이 조건의 훈련 시작 인원이 없습니다.")
        if notes:
            st.info(f"현재 조건: {cond_text(f'{y1}년', with_sido=False)}  \n" + " ".join(notes))
        shr = sd.sum(axis=1) / sd.values.sum()
        st.caption(f"서울·경기 합계가 전체의 {(shr.get('서울', 0) + shr.get('경기', 0)) * 100:.0f}%. 시도는 훈련기관 소재지 "
                   "기준(훈련생 거주지 아님). 수료율·취업률은 시도별 자료가 없어 분석 탭과 연결할 수 없습니다.")
