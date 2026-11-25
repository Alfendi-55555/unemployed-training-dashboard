"""실업자훈련 대시보드 — 국기훈련은 계좌제보다 '덜 수료하지만 더 취업하는' 과정인가?

실행:  streamlit run app.py
데이터: data/*.csv (prep_data.py 로 원본 xlsx 에서 생성)

코드 순서
  0. 설정·상수 → 데이터 불러오기 → 표시용 보조 함수 → 분석 함수
  1. 데이터 준비      2. 사이드바      3. 머리말(제목·출처·정의, 두 탭 공통)
  4. 탭 1 「과정별 성과 분석」: ① 요약 카드 → ② 추이(+격차 분해) → ③ 같은 직종 비교 → ④ 변화의 출처
                              → ⑤ 직종 구성 → ⑥ 결론·한계 요약 → ⑦ (접이식) 직종 상세 → ⑧ 출처·기준·한계
  5. 탭 2 「부록: 훈련 실시 현황」: 월별 훈련 시작 인원, 계절성, 시도별
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent

# ================================================================== 0. 설정·상수
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
PLOTLY_FONT = dict(family="Pretendard, 'Noto Sans KR', 'Malgun Gothic', sans-serif", size=13, color=INK)
PLOTLY_CONFIG = {"displayModeBar": False}                # 툴바 숨김(마우스 툴팁은 유지)
KECO_WARN = ("2025년 직종분류(KECO) 개편 구간(2023→2024)이 포함되어 있습니다. 개편 앞뒤는 같은 중분류라도 "
             "세부 구성과 직종 배정이 달라 값이 다르므로(2024년은 두 코드 값이 함께 있고 서로 다름), "
             "두 체계의 값을 직접 비교하지 않습니다.")
PEOPLE_NOTE = ("훈련 시작 인원 = 훈련시작일이 해당 월인 과정의 참여자 수. 동일인이 여러 과정을 수강하면 중복 산정(연인원)되어 "
               "실제 사람 수보다 많을 수 있습니다. 해당 월 값은 다음 해 1월까지 재산정됩니다(2026-11-24 조회).")


# ------------------------------------------------------------------ 데이터 불러오기
@st.cache_data
def load_data():
    rates = pd.read_csv(ROOT / "data/rates.csv", dtype={"code": "string"})
    monthly = pd.read_csv(ROOT / "data/monthly.csv", dtype={"code": "string"})
    monthly["date"] = pd.to_datetime(monthly["ym"])
    keco = pd.read_csv(ROOT / "data/keco_map.csv", dtype=str)      # 직종코드 신구연계표(세분류 단위)
    return rates, monthly, keco


# ------------------------------------------------------------------ 표시용 보조 함수 (숫자 형식·축)
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


def rate_range(values, from_zero, min_span=0.10):
    """비율 축 범위(0~1 단위): 0% 기준이거나, 최소 10%p 폭을 보장해 작은 변화가 과장되지 않게 한다."""
    if from_zero:
        return 0, 1.05
    vals = [v for v in np.ravel(values) if pd.notna(v)]
    lo, hi = min(vals), max(vals)
    mid, span = (lo + hi) / 2, max(hi - lo, min_span) * 1.25
    return max(mid - span / 2, 0), min(mid + span / 2, 1.05)


def trend_word(diff_pp):
    """%p 차이 → '줄었다/늘었다'. 소수 첫째 자리로 같아 보이는 차이(0.05%p 미만)는 '거의 같았다'."""
    return "거의 같았다" if abs(diff_pp) < 0.05 else ("줄었다" if diff_pp < 0 else "늘었다")


def keco_note(detail="", label="직종코드 개편(2023→2024) 구간 포함"):
    """KECO 개편 안내: 화면에는 한 줄 표시만, 자세한 설명은 마우스를 올리면(ⓘ) 보인다."""
    st.caption(f":orange[**⚠ {label}**]", help=KECO_WARN + (" " + detail if detail else ""))


# ------------------------------------------------------------------ 분석 함수 (직종별 변화 분해)
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


# ================================================================== 1. 데이터 준비
st.set_page_config(page_title="실업자훈련 성과 대시보드", page_icon="📊", layout="wide")
rates, monthly, keco = load_data()
totals = rates[rates.level == "전체"].pivot_table(
    index="year", columns="course", values=["completion_rate", "employment_rate"])
starts = monthly.groupby(["course", "code", "year", "job"]).trainees.sum()
key_of = pd.concat([rates[["job", "job_key"]], monthly[["job", "job_key"]]]).dropna().drop_duplicates() \
    .set_index("job").job_key.to_dict()
name_of = (pd.concat([rates[["code", "job", "job_key"]], monthly[["code", "job", "job_key"]]]).dropna()
           .sort_values("code").groupby("job_key").job.last().map(short_job).to_dict())   # 표시명 = 최신 이름

# ================================================================== 2. 사이드바
st.sidebar.header("조건 선택")
y0, y1 = st.sidebar.select_slider("연도 범위", options=list(range(2021, 2026)), value=(2021, 2025),
                                  help="요약 카드와 같은 직종 비교의 기준 연도는 범위의 끝 연도입니다.")
job_keys = sorted(rates[(rates.level == "직종")].dropna(subset=["employment_rate"]).job_key.unique(),
                  key=lambda k: name_of[k])
sel = st.sidebar.selectbox(
    "직종 (하나 선택)", [ALL_JOBS] + job_keys, format_func=lambda k: k if k == ALL_JOBS else name_of[k],
    help="고르면 요약 카드에 그 직종 값이 추가되고, 직종별 차트에서 그 직종이 강조됩니다. "
         "수료율·취업률은 직종별 인원이 공개되지 않아 여러 직종을 합쳐 계산할 수 없으므로 한 번에 하나만 고릅니다.\n\n"
         "2018·2025 직종코드의 같은 중분류를 하나로 묶어 표시합니다(신구연계표 기준 중분류 체계 동일). "
         "다만 세부 구성과 값은 달라 연도별 선은 개편 지점에서 잇지 않습니다.")
st.sidebar.caption(f"기준 연도: **{y1}년** · 항목 옆 ⓘ에 마우스를 올리면 설명")
years = list(range(y0, y1 + 1))
period = f"{y0}~{y1}년" if y0 < y1 else f"{y1}년"          # 제목에 쓰는 기간 (사이드바 범위를 따름)
trend_lbl = f"{period} 추이" if y0 < y1 else period          # 한 해만 고르면 '추이'를 빼고 연도만

# ================================================================== 3. 머리말 (두 탭 공통)
st.title("국기훈련은 '덜 수료하지만 더 취업하는' 과정인가?")
# 한 줄 답 (맨 아래 결론 상자의 첫 줄과 같은 문장 — 두괄식)
gN = totals["employment_rate"][NAT] - totals["employment_rate"][ACC]
gC = totals["completion_rate"][NAT] - totals["completion_rate"][ACC]
yes = gC[y1] < 0 and gN[y1] > 0
answer = (f"**답: {'그렇다' if yes else '이 기간에는 그렇지 않다'}.** {y1}년 국기훈련은 계좌제보다 "
          f"수료율 {gC[y1] * 100:+.1f}%p, 취업률 {gN[y1] * 100:+.1f}%p.")
if len(years) > 1:
    answer += (f" 다만 취업 우위는 **{gN[y0] * 100:.1f}%p({y0}) → {gN[y1] * 100:.1f}%p({y1})로 "
               f"{trend_word(gN[y1] * 100 - gN[y0] * 100)}**.")
st.markdown(answer)

h1, h2 = st.columns([4, 1], vertical_alignment="center")
h1.caption("수료율 = 수료 ÷ 실시 · 취업률 = 수료 후 6개월 내 취업 ÷ 수료 · 연도 = 훈련이 끝난 해  \n"
           "출처: 한국고용정보원 고용행정통계(원자료: HRD-Net) — 실업자훈련 수료율·취업률현황(년)·실시현황(월) · 2026-11-24 조회")
with h2.popover("📖 정의 자세히"):
    st.markdown(f"<span style='color:{COLOR[ACC]};font-weight:700'>● 실업자계좌제 (계좌제)</span>", unsafe_allow_html=True)
    st.markdown("고용행정통계에서 **'내일배움카드' 과정**으로 분류되는 실업자훈련(통계 표기: 실업자계좌제). "
                "국기훈련 지정 직종에 해당하지 않는 **일반 과정**으로, 사무·서비스·IT·디자인 등 분야가 넓다.")
    st.markdown(f"<span style='color:{COLOR[NAT]};font-weight:700'>● 국가기간전략산업직종훈련 (국기훈련)</span>",
                unsafe_allow_html=True)
    st.markdown("국가 기간·전략산업 등 인력 양성이 필요해 **고용노동부가 별도로 지정한 훈련직종**"
                "(국민내일배움카드 운영규정 별표 1)의 실업자훈련 과정.")
    st.markdown("**지표 정의** (연도 = 훈련이 끝난 해)\n"
                "- **수료율** = 수료인원 ÷ 실시인원\n"
                "- **취업률** = 수료 후 6개월 이내 취업인원 ÷ 수료인원\n"
                "- 연도별 값만 비교하며 여러 해를 합하거나 평균내지 않음")
    st.caption("두 과정 모두 실제 수강 신청은 국민내일배움카드로 하며, 차이는 지정 직종 과정인지 여부입니다. "
               "이 통계의 실업자훈련은 두 과정으로만 구성되고 두 과정의 합이 총계와 같습니다.")

tab_main, tab_app = st.tabs(["📈 과정별 성과 분석", "📎 부록: 훈련 실시 현황 · 직종코드"])

# ================================================================== 4. 탭 1 — 과정별 성과 분석
with tab_main:
    # -------------------------------------------------------------- ① 요약 카드
    st.subheader(f"{y1}년 한눈에 보기")
    # 증감 기준: 사이드바 범위의 시작 연도(한 해만 고르면 직전 연도) — 추이·결론과 같은 기간을 말하도록
    prev = y0 if y0 < y1 else (y1 - 1 if (y1 - 1) in totals.index else None)
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
    st.caption(f"▲▼ = {prev}년 대비 증감" if prev else "증감 표시 없음")

    # -------------------------------------------------------------- ② 추이
    # 구역마다 같은 틀: 제목(질문) → 답 상자(사실 1~2개) → 위젯 → 차트 → 캡션 한 줄(읽는 법)
    st.divider()
    st.subheader(f"덜 수료하고 더 취업하나 — {trend_lbl}")
    emp = totals["employment_rate"] * 100
    gapC = (totals["completion_rate"][NAT] - totals["completion_rate"][ACC]).loc[y0:y1] * 100
    gapE = (emp[NAT] - emp[ACC]).loc[y0:y1]
    if len(years) > 1:
        st.info(f"수료율은 매년 국기훈련이 {abs(gapC).min():.1f}~{abs(gapC).max():.1f}%p 낮다. "
                f"취업률은 국기훈련이 높지만, 격차가 **{gapE[y0]:.1f}%p({y0}) → {gapE[y1]:.1f}%p({y1})로 "
                f"{trend_word(gapE[y1] - gapE[y0])}**.")
    else:
        st.info(f"{y1}년 국기훈련은 계좌제보다 수료율 {gapC[y1]:+.1f}%p, 취업률 {gapE[y1]:+.1f}%p.")
    y_from_zero = st.toggle("y축을 0%부터 표시", value=False)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>수료율</b>", "<b>취업률</b>"),
                        horizontal_spacing=0.08, shared_yaxes=y_from_zero)
    for ci, mt in enumerate(["completion_rate", "employment_rate"], start=1):
        for c in COURSES:
            s = totals[mt][c].loc[y0:y1] * 100
            ends = {s.index[0], s.index[-1]}
            fig.add_scatter(
                x=s.index, y=s.values, mode="lines+markers+text", name=SHORT[c], legendgroup=c, showlegend=ci == 1,
                line=dict(color=COLOR[c], width=2.5), marker=dict(size=9, line=dict(color="white", width=1.5)),
                text=[f"{y_:.1f}%" if x_ in ends else "" for x_, y_ in s.items()], textposition="top center",
                fill="tonexty" if c == NAT and len(years) > 1 else None, fillcolor="rgba(138,137,133,0.12)",
                hovertemplate=f"{SHORT[c]} %{{x}}년: %{{y:.1f}}%<extra></extra>", row=1, col=ci)
        if len(years) > 1:
            a, n = totals[mt][ACC].loc[y0:y1] * 100, totals[mt][NAT].loc[y0:y1] * 100
            for x_, sh in ((y0, 40), (y1, -40)):
                fig.add_annotation(x=x_, y=(a[x_] + n[x_]) / 2, text=f"격차 {n[x_] - a[x_]:+.1f}%p", showarrow=False,
                                   xshift=sh, font=dict(size=11, color=INK2), row=1, col=ci)
        lo, hi = rate_range(totals[mt].loc[y0:y1].values.ravel(), y_from_zero)
        fig.update_yaxes(range=[lo * 100, hi * 100], ticksuffix="%", row=1, col=ci)
    fig.update_xaxes(tickmode="array", tickvals=years, range=[y0 - 0.4, y1 + 0.4])
    fig.update_layout(height=420, font=PLOTLY_FONT, template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.2, yanchor="bottom", x=0), margin=dict(l=10, r=10, t=80, b=30))
    st.plotly_chart(fig, config=PLOTLY_CONFIG)
    if not y_from_zero:
        st.caption("y축이 0%에서 시작하지 않아 차이가 확대되어 보임(토글로 전환)")

    # 취업률 격차 분해: 격차 변화 = 국기훈련 변화 - 계좌제 변화 (해마다)
    stages = []                                  # (연도, 국기 변화, 계좌 변화, 격차 변화, 주된 원인) — 결론에서도 사용
    for y in range(y0 + 1, y1 + 1):
        sN, sA = emp[NAT][y] - emp[NAT][y - 1], emp[ACC][y] - emp[ACC][y - 1]
        who = (f"국기훈련 {'상승' if sN > 0 else '하락'}" if abs(sN) >= abs(sA)
               else f"계좌제 {'상승' if sA > 0 else '하락'}")
        stages.append((y, sN, sA, sN - sA, who))
    nat_main = True
    if len(years) > 1:
        dN, dA = emp[NAT][y1] - emp[NAT][y0], emp[ACC][y1] - emp[ACC][y0]
        nat_main = abs(dN) >= abs(dA)
        verb = {"줄었다": "줄었나", "늘었다": "늘었나"}.get(trend_word(dN - dA), "변했나")   # 필터에 따라 방향이 바뀜
        st.markdown(f"##### 격차는 언제, 왜 {verb} — 연도별 분해")
        big = [f"{y}년 격차 {g:+.1f}%p({who})" for y, sN, sA, g, who in stages if abs(g) >= 1]
        st.info(f"{y0}→{y1}년 격차 변화 {dN - dA:+.1f}%p = 국기훈련 {dN:+.1f}%p − 계좌제 {dA:+.1f}%p. "
                f"해마다 보면 **{', '.join(big) if big else '1%p 넘게 움직인 해 없음'}**.")
        with st.expander("연도별 격차 변화 표로 보기"):
            st.dataframe(pd.DataFrame(
                [{"연도": f"{y - 1}→{y}", "국기훈련 변화(%p)": round(sN, 1), "계좌제 변화(%p)": round(sA, 1),
                  "격차 변화(%p)": round(g, 1), "주된 원인": who} for y, sN, sA, g, who in stages]),
                hide_index=True, width="stretch")
        if nat_main:
            st.caption("기간 전체로는 국기훈련 쪽 변화가 더 커서, "
                       "다음 구역에서 국기훈련 취업률을 직종별로 쪼개 봅니다(계좌제는 직종이 30개 안팎으로 흩어져 있어 쪼개지 않음).")
        else:
            st.warning("이 기간 격차 변화는 **주로 계좌제의 변화** 때문입니다. 아래 '직종별 분해'는 국기훈련 쪽만 본 것이라 "
                       "이 기간 격차 변화의 주된 원인을 설명하지 않습니다.")

    # -------------------------------------------------------------- ③ 같은 직종 비교
    st.divider()
    title3, lead3 = st.empty(), st.empty()      # 제목·설명은 지표 선택 뒤에 채운다
    s1, s2 = st.columns([1, 2])
    with s1:
        metric_label = st.radio("비교 지표", list(METRICS), horizontal=True)
    metric = METRICS[metric_label]
    title3.subheader("같은 직종이어도 더 취업하나 — 직종별 비교" if metric_label == "취업률" else
                     "같은 직종이어도 덜 수료하나 — 직종별 비교")
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
        hl = [key_of.get(j) == sel for j in d.index]
        ylab = [f"<b>{short_job(j)}</b>" if h else short_job(j) for j, h in zip(d.index, hl)]
        fig = go.Figure()
        # 두 점을 잇는 회색 선 (None 으로 끊어 한 번에)
        fig.add_scatter(x=[x_ for a_, n_ in zip(d[ACC], d[NAT]) for x_ in (a_ * 100, n_ * 100, None)],
                        y=[y_ for lab in ylab for y_ in (lab, lab, None)], mode="lines",
                        line=dict(color=GREY, width=2), opacity=0.5, hoverinfo="skip", showlegend=False)
        for c in COURSES:
            fig.add_scatter(x=d[c] * 100, y=ylab, mode="markers", name=SHORT[c],
                            marker=dict(size=12, color=COLOR[c], line=dict(color="white", width=1.5)),
                            customdata=[[f"{x_:,}"] for x_ in d[f"n_{c}"].astype(int)],
                            hovertemplate=f"<b>%{{y}}</b><br>{SHORT[c]} {metric_label} %{{x:.1f}}%"
                                          "<br>훈련 시작 인원 %{customdata[0]}명<extra></extra>")
        xmax = max(d[ACC].max(), d[NAT].max()) * 100
        fig.add_scatter(x=[xmax + 7] * len(d), y=ylab, mode="text", showlegend=False, hoverinfo="skip", cliponaxis=False,
                        text=[f"<b>{v_ * 100:+.1f}%p</b>" if v_ < 0 else f"{v_ * 100:+.1f}%p" for v_ in d["diff"]],
                        textfont=dict(color=[COLOR[NAT] if v_ < 0 else INK for v_ in d["diff"]], size=12))
        fig.add_annotation(x=xmax + 7, y=1, yref="paper", yanchor="bottom", text="국기 - 계좌제",
                           showarrow=False, font=dict(size=11, color=INK2))
        fig.update_layout(height=32 * len(d) + 120, font=PLOTLY_FONT, template="plotly_white",
                          legend=dict(orientation="h", y=1.04, yanchor="bottom", x=0),
                          margin=dict(l=10, r=10, t=50, b=40),
                          xaxis=dict(title=metric_label, ticksuffix="%",
                                     range=[min(d[ACC].min(), d[NAT].min()) * 100 - 4, xmax + 14]),
                          yaxis=dict(title=None))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)
        win, win_all = int((d["diff"] > 0).sum()), int((full["diff"] > 0).sum())
        # 답 상자 (제목 바로 아래 자리에 채움)
        lead3.info(f"직종 구성 탓인지 보려고 **같은 직종끼리** 비교하면, {len(d)}개 직종 중 **{win}개**에서 "
                   f"국기훈련 {metric_label}이 더 높다 (차이 중앙값 {d['diff'].median() * 100:+.1f}%p).")
        hidden = full[~full.index.isin(d.index)]
        note = ""
        if len(hidden):
            lows = hidden[hidden["diff"] <= 0]
            note = (f" 인원 기준으로 제외된 {len(hidden)}개 직종까지 포함하면 {len(full)}개 중 {win_all}개"
                    + (f" (예외: {', '.join(short_job(j) for j in lows.index)})." if len(lows) else "."))
        sel_note = "" if sel == ALL_JOBS or any(hl) else f" 선택한 직종({name_of[sel]})은 이 조건에서 비교 대상이 아닙니다."
        st.caption(f"{y1}년·{code} 직종코드 기준."
                   f"{note}{sel_note}")
        with st.expander("표로 보기"):
            st.dataframe(pd.DataFrame({
                "직종": d.index, "계좌제": d[ACC].map(pct), "국기훈련": d[NAT].map(pct),
                "차이(%p)": (d["diff"] * 100).round(1),
                "계좌제 인원": d[f"n_{ACC}"].astype(int), "국기 인원": d[f"n_{NAT}"].astype(int)}).iloc[::-1],
                hide_index=True)

    # -------------------------------------------------------------- ④ 변화의 출처
    # 격차 변화의 주된 원인인 '국기훈련 취업률'에 고정 (위 정보 상자의 격차 분해가 이유)
    dc, metric, metric_label = NAT, "employment_rate", "취업률"
    st.divider()
    st.subheader("국기훈련 취업률은 왜 변했나 — 직종별 분해")
    # 같은 직종코드 안의 '연속 연도' 구간만 선택지로 (연도 범위 안)
    consec = [p for p in pairs_for(dc, metric) if p[1] - p[0] == 1 and y0 <= p[0] and p[1] <= y1]
    skipped = [f"{y}→{y + 1}" for y in range(y0, y1)
               if not any(p[0] == y for p in consec)]
    top = None
    if not consec:
        st.warning("현재 연도 범위 안에는 같은 직종코드로 비교할 수 있는 연속 연도가 없습니다. 사이드바 연도 범위를 넓혀 보세요.")
    else:
        best = biggest_change_pair(dc, metric, y0, y1)
        labels = {f"{a}→{b}": (a, b, cd) for a, b, cd in consec}
        pick = st.segmented_control("비교 구간", list(labels), default=f"{best[0]}→{best[1]}",
                                    help="기본값은 전체 값이 가장 크게 변한 구간") or f"{best[0]}→{best[1]}"
        pa, pb, pcode = labels[pick]
        if skipped:
            st.caption(f"{', '.join(skipped)}: 직종코드(KECO)가 바뀌어 직종 단위로 비교할 수 없어 선택지에서 뺐습니다.")
        v, official = decompose(dc, metric, pa, pb, pcode)
        v = v.reindex(v.contrib.abs().sort_values(ascending=False).index)   # 기여 큰 순 (차트·표 공통)
        down = int((v.delta < 0).sum())
        top = v.contrib.abs().idxmax()
        share_top = v.contrib[top] / official if abs(official) >= 0.01 else np.nan
        main_txt = (f"그중 **약 {share_top * 100:.0f}%가 {short_job(top)}**(인원의 {v.w[top] * 100:.0f}%)에서 나왔다"
                    "(시작 인원 기준 추정)."
                    if 0 < share_top <= 1 else
                    f"가장 큰 기여는 **{short_job(top)}**({v.contrib[top] * 100:+.2f}%p, 인원의 {v.w[top] * 100:.0f}%, "
                    "시작 인원 기준 추정).")
        # 이 한 구간이 연도 범위 전체 변화에서 차지하는 몫 (0~100%일 때만)
        totN = (totals.loc[y1, (metric, dc)] - totals.loc[y0, (metric, dc)]) * 100
        link = (f" 이 구간이 {y0}→{y1}년 전체 변화({totN:+.1f}%p)의 약 {official * 100 / totN * 100:.0f}%다."
                if (pa, pb) != (y0, y1) and abs(totN) >= 1 and 0 < official * 100 / totN <= 1 else "")
        st.info(f"{pa}→{pb}년 국기훈련 취업률 **{official * 100:+.1f}%p** — {main_txt} "
                f"{len(v)}개 직종 중 {down}개에서 떨어졌다.{link}")
        if abs(official) < 0.01:                  # 전체 변화가 1%p 미만이면 분해 결과가 쉽게 흔들림
            st.caption(f"이 구간은 전체 변화가 {official * 100:+.2f}%p로 1%p 미만이라, 직종별 기여는 작은 차이에도 "
                       "순위·비중이 크게 달라질 수 있습니다.")
        if pcode == "2018" and pb >= SWITCH:      # 2024년 값을 2018 코드로 쓰는 구간
            keco_note(f"{pb}년 값은 원자료에 2018 직종코드로도 함께 제공된 값을 썼습니다(같은 해 2025 코드 값과 다름). "
                      f"{pa}년과 같은 코드 체계라 비교는 가능하지만, 다른 구역의 {pb}년 직종 값(2025 코드)과는 기준이 다릅니다.",
                      label=f"{pb}년 값은 2018 직종코드 기준 (다른 구역의 {pb}년 값과 다름)")

        hl = [key_of.get(j) == sel for j in v.index]
        names = [(f"<b>{short_job(j)}</b>" if h else short_job(j)) + (f" ({w * 100:.0f}%)" if w >= 0.005 else " (<1%)")
                 for j, h, w in zip(v.index, hl, v.w)]
        wtxt = [f"{w * 100:.1f}%" if w >= 0.0005 else "<0.1%" for w in v.w]
        cv = v.contrib * 100
        share = cv / (official * 100) * 100 if abs(official) >= 0.0001 else cv * np.nan
        # 직종으로 나눌 수 없는 부분 → 막대 합계가 실제 변화와 같아지도록 맨 아래에 따로 표시
        #   직종 구성 변화 = Σ (pb년 비중 − pa년 비중) × pb년 비율 (인원이 다른 직종으로 옮겨 간 효과)
        #   근사 오차 = 나머지 (실제 가중치인 수료인원 대신 시작 인원 비중을 쓴 데서 생김)
        nb = pd.Series([starts.get((dc, pcode, pb, j), 0) for j in v.index], index=v.index)
        resid = official - v.contrib.sum()
        if nb.sum() > 0:
            between = ((nb / nb.sum() - v.w) * v[pb]).sum()
            res = {"직종 구성 변화": between * 100, "근사 오차 (가중치)": (resid - between) * 100}
        else:                                 # pb년 시작 인원이 이 코드로 없으면(2018 코드의 2024년) 나눌 수 없음
            between = np.nan
            res = {"구성 변화 + 근사 오차": resid * 100}
        cats = names + [" "] + list(res)      # " " = 직종 막대와 한 칸 띄우기
        # 한 그림에 좌우 두 패널, 직종 이름 축 공유 (위에서부터 기여 큰 순)
        #   왼쪽: 직종별 두 해 비율(묶음 막대) · 오른쪽: 전체 변화에 대한 기여 추정(인원 비중 × 변화)
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.58, 0.42], horizontal_spacing=0.04,
                            subplot_titles=(f"직종별 {metric_label} (막대 끝: 변화 %p)",
                                            "전체 변화에 대한 기여 추정<br><sup>%p · 시작 인원 비중 기준</sup>"))
        for yr, colr in ((pa, "#c9c7c0"), (pb, COLOR[dc])):
            fig.add_bar(
                y=names, x=v[yr] * 100, orientation="h", name=f"{yr}년", marker_color=colr,
                text=[fmt_pp(d_) for d_ in v.delta * 100] if yr == pb else None,
                textposition="outside", textfont=dict(size=11), cliponaxis=False,
                # 툴팁 숫자는 미리 문자열로 (부호 있는 형식은 plotly 서식이 적용되지 않음)
                customdata=[[w_, f"{a_:.2f}", f"{b_:.2f}", f"{d_:+.2f}"]
                            for w_, a_, b_, d_ in zip(wtxt, v[pa] * 100, v[pb] * 100, v.delta * 100)],
                hovertemplate=(f"<b>%{{y}}</b><br>{pa}년 인원 비중 %{{customdata[0]}}<br>"
                               f"{pa}년 {metric_label} %{{customdata[1]}}%<br>"
                               f"{pb}년 {metric_label} %{{customdata[2]}}%<br>"
                               "변화 %{customdata[3]}%p<extra></extra>"), row=1, col=1)
        fig.add_bar(
            y=names, x=cv.values, orientation="h", showlegend=False,
            marker_color=[COLOR[dc] if j == top else "#f4b79c" for j in v.index],
            text=[fmt_pp(x, contrib=True) for x in cv.values], textposition="outside", textfont=dict(size=11),
            cliponaxis=False,
            customdata=[[f"{c_:+.3f}", f"{w_:.1f}", f"{d_:+.2f}", "–" if pd.isna(s_) else f"{s_:.0f}"]
                        for c_, w_, d_, s_ in zip(cv.values, v.w * 100, v.delta * 100, share)],
            hovertemplate=("<b>%{y}</b><br>기여 %{customdata[0]}%p<br>= 인원 비중 %{customdata[1]}% "
                           f"× {metric_label} 변화 %{{customdata[2]}}%p<br>"
                           "전체 변화 중 %{customdata[3]}%<extra></extra>"), row=1, col=2)
        res_tip = {"직종 구성 변화": "인원이 다른 직종으로 옮겨 간 효과",
                   "근사 오차 (가중치)": "실제 가중치(수료인원) 대신 시작 인원 비중을 써서 남는 차이",
                   "구성 변화 + 근사 오차": f"{pb}년 시작 인원이 이 코드로 없어 둘을 나눌 수 없음"}
        fig.add_bar(
            y=list(res), x=list(res.values()), orientation="h", showlegend=False,
            marker=dict(color="#e4e2dd", pattern=dict(shape="/", fgcolor="#a8a6a0", size=6),
                        line=dict(color="#a8a6a0", width=1)),
            text=[fmt_pp(x, contrib=True) for x in res.values()], textposition="outside", textfont=dict(size=11),
            cliponaxis=False, customdata=[[f"{x:+.3f}", res_tip[k]] for k, x in res.items()],
            hovertemplate="<b>%{y}</b> %{customdata[0]}%p<br>%{customdata[1]}<extra></extra>", row=1, col=2)
        allx = list(cv.values) + list(res.values())
        lo, hi = min(min(allx), 0), max(max(allx), 0)
        pad = (hi - lo or 0.3) * 0.45               # 막대 끝 숫자가 들어갈 여백
        fig.update_layout(
            barmode="group", bargap=0.25, height=34 * len(cats) + 130, font=PLOTLY_FONT, template="plotly_white",
            legend=dict(orientation="h", y=1.08, yanchor="bottom", x=0, title=None),
            margin=dict(l=10, r=10, t=95, b=40))
        fig.update_annotations(font_size=14)
        fig.update_xaxes(title=metric_label, ticksuffix="%",
                         range=[0, max(v[pa].max(), v[pb].max()) * 100 * 1.22], row=1, col=1)
        fig.update_xaxes(title="기여 (%p)", range=[lo - pad, hi + pad], zeroline=True, zerolinecolor=INK2, row=1, col=2)
        fig.update_yaxes(categoryorder="array", categoryarray=cats, autorange="reversed", title=None)
        st.plotly_chart(fig, config=PLOTLY_CONFIG)
        st.caption(f"괄호 = {pa}년 인원 비중 · 기여 = 인원 비중 × 변화(추정) · "
                   "빗금 = 직종으로 나눌 수 없는 부분, 오른쪽 막대를 모두 더하면 실제 변화")
        with st.expander("표로 보기 (정확한 값)"):
            split = (f"직종 구성 변화 {between * 100:+.2f}%p + 근사 오차 {(resid - between) * 100:+.2f}%p"
                     if pd.notna(between) else f"구성 변화 + 근사 오차 {resid * 100:+.2f}%p({pb}년 시작 인원이 이 코드로 없어 나눌 수 없음)")
            st.caption(f"직종별 기여의 합 {v.contrib.sum() * 100:+.2f}%p + {split} = 실제 전체 변화 {official * 100:+.2f}%p. "
                       f"기여는 실제 가중치(수료인원)가 공개되지 않아 {pa}년 훈련 시작 인원 비중으로 추정했습니다.")
            st.dataframe(pd.DataFrame({
                "직종": [short_job(j) for j in v.index],
                f"{pa}년 인원 비중": (v.w * 100).round(1).astype(str) + "%",
                f"{pa}년 {metric_label}": v[pa].map(lambda x: pct(x, 2)),
                f"{pb}년 {metric_label}": v[pb].map(lambda x: pct(x, 2)),
                "변화(%p)": (v.delta * 100).round(2),
                "기여(%p)": (v.contrib * 100).round(3)}), hide_index=True)

    # -------------------------------------------------------------- ⑤ 직종 구성
    comp = (monthly[(monthly.course == dc) & monthly.year.between(y0, y1)]
            .groupby(["year", "job_key"]).trainees.sum().unstack(fill_value=0))
    share = comp.div(comp.sum(axis=1), axis=0)
    order = comp.sum().sort_values(ascending=False).index
    focus = sel if (sel != ALL_JOBS and sel in comp.columns) else order[0]
    st.divider()
    st.subheader("그 쏠림은 계속되나 — 국기훈련 직종 구성" if focus == order[0] else
                 f"{name_of[focus]} 비중은 어떻게 변했나 — 국기훈련 직종 구성")
    # 답: 직종코드 체계별로 나눠서 (개편 앞뒤 값은 직접 비교하지 않음)
    eras = []
    for era, ys in (("2018 코드", [y for y in share.index if y < SWITCH]), ("2025 코드", [y for y in share.index if y >= SWITCH])):
        if ys:
            eras.append(f"{era} {share.loc[ys[0], focus] * 100:.0f}%({ys[0]})"
                        + (f" → {share.loc[ys[-1], focus] * 100:.0f}%({ys[-1]})" if len(ys) > 1 else ""))
    low = share[focus].min() * 100
    st.info(f"국기훈련 시작 인원 중 **{name_of[focus]}** 비중: {' / '.join(eras)}."
            + (f" 모든 해에 **{low:.0f}% 이상**을 차지한다." if focus == order[0] and len(share) > 1 else ""))
    if y0 < SWITCH <= y1:
        keco_note("그래서 비중도 각 코드 체계 안에서만 비교했습니다.")
    others = [j for j in order if j != focus][:4]
    cats = [focus] + others
    plot = share[cats].copy()
    plot["기타"] = 1 - plot.sum(axis=1)
    colors = [COLOR[dc]] + OTHER_GREYS[:len(others)] + [REST_COLOR]
    sep = "── KECO 개편 ──"                       # 2023과 2024 사이에 빈 행을 넣어 개편 지점을 표시
    rows = [str(y) for y in plot.index if y < SWITCH] + \
           ([sep] if (y0 < SWITCH <= y1) else []) + [str(y) for y in plot.index if y >= SWITCH]
    fig = go.Figure()
    for col, colr in zip(plot.columns, colors):
        label = col if col == "기타" else name_of[col]
        vals = plot[col] * 100
        cnt = comp[col] if col != "기타" else comp.sum(axis=1) - comp[cats].sum(axis=1)
        dark = colr in (COLOR[dc], OTHER_GREYS[0], OTHER_GREYS[1])
        fig.add_bar(y=[str(y) for y in plot.index], x=vals, orientation="h", name=label, marker_color=colr,
                    marker_line=dict(color="white", width=2),
                    text=[f"<b>{x_:.0f}%</b>" if col == focus else f"{x_:.0f}%" if x_ >= 6 else "" for x_ in vals],
                    textposition="inside", insidetextanchor="middle", textangle=0,
                    textfont=dict(color="white" if dark else INK, size=12),
                    customdata=[[f"{n_:,.0f}"] for n_ in cnt.reindex(plot.index)],
                    hovertemplate=f"<b>{label}</b><br>%{{y}}년: %{{x:.1f}}% (%{{customdata[0]}}명)<extra></extra>")
    fig.update_layout(barmode="stack", height=46 * len(rows) + 140, font=PLOTLY_FONT, template="plotly_white",
                      uniformtext=dict(minsize=10, mode="hide"),      # 칸에 안 들어가는 숫자는 숨김(툴팁으로 확인)
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal"),   # 위로: 막대와 겹치지 않게
                      margin=dict(l=10, r=10, t=70, b=30),
                      xaxis=dict(range=[0, 100], ticksuffix="%", title=None),
                      yaxis=dict(categoryorder="array", categoryarray=rows, autorange="reversed", title=None))
    st.plotly_chart(fig, config=PLOTLY_CONFIG)
    st.caption("그 해 국기훈련 시작 인원(연인원) 기준 · 인원 상위 4개 외는 '기타'")

    # -------------------------------------------------------------- ⑥ 결론 + 한계 요약
    st.divider()
    st.subheader("결론")
    emp_full = (rates[(rates.level == "직종") & (rates.year == y1) & (rates.code == code)]
                .pivot_table(index="job", columns="course", values="employment_rate").reindex(columns=COURSES).dropna())
    emp_win = int((emp_full[NAT] > emp_full[ACC]).sum())
    bp = biggest_change_pair(NAT, "employment_rate", y0, y1)
    lines = [f"**직종 구성 탓만은 아니다:** 같은 직종끼리 비교해도 {len(emp_full)}개 중 {emp_win}개 직종에서 "
             f"국기훈련 취업률이 더 높다({y1}년)."]
    if len(years) > 1:
        big_stages = [f"{y}년 격차 {g:+.1f}%p({who})"
                      for y, sN, sA, g, who in stages if abs(g) >= 1]          # 격차가 1%p 이상 움직인 해만
        lines.append(f"**우위가 {'줄어든' if gN[y1] < gN[y0] else '달라진'} 과정:** "
                     + (f"{', '.join(big_stages)}." if big_stages else "해마다 1%p 미만의 작은 변화."))
    if len(years) > 1 and not nat_main:          # 이 기간 격차 변화의 주된 원인이 계좌제인 경우
        lines.append(f"**변화의 출처:** 이 기간 격차 변화는 주로 계좌제 취업률 변화({dA:+.1f}%p) 때문이다. "
                     f"국기훈련 취업률은 {dN:+.1f}%p 변했다.")
    elif bp:
        vb, ob = decompose(NAT, "employment_rate", *bp)
        tb = vb.contrib.abs().idxmax()
        lines.append(f"**변화의 출처:** 국기훈련 취업률이 가장 크게 변한 {bp[0]}→{bp[1]}년({ob * 100:+.1f}%p)의 기여 중 "
                     f"{short_job(tb)}이 {vb.contrib[tb] * 100:+.1f}%p로 가장 크다(시작 인원 기준 추정). 이 직종은 국기훈련 시작 인원의 "
                     f"{pct(vb.w[tb], 0)}({bp[0]}년)를 차지한다.")
    # 제목 질문에 대한 한 줄 답(머리말에서 만든 answer) + 근거 항목, 한 상자에
    st.success(answer + "\n\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines)))
    st.markdown("**한계**\n"
                "- 취업률 격차는 과정의 '효과'가 아니다. 같은 직종 안에서도 과정 내용·훈련생 특성(지원 동기 등)이 다를 수 있다.\n"
                "- 2025년 직종분류(KECO) 개편으로 직종 단위 값은 같은 코드 체계 안에서만 비교했다.\n"
                "- 직종별 기여는 공개되지 않은 실제 가중치 대신 훈련 시작 인원 비중으로 근사했다.\n\n"
                "자세한 출처·기준·한계는 맨 아래 상자에 있습니다.")

    # -------------------------------------------------------------- ⑦ 직종 하나 자세히 보기 (접이식)
    with st.expander("직종 하나 자세히 보기 — 추이와 코드 개편"):
        # 기본 직종: 쪼개 보기에서 기여가 가장 큰 직종, 비교 구간이 없으면 직종 구성에서 인원이 가장 많은 직종
        dj = sel if sel != ALL_JOBS else (key_of.get(top) if top else order[0])
        why = "위에서 기여가 가장 큰 직종" if top else "위 직종 구성에서 인원이 가장 많은 직종"
        st.markdown(f"##### 직종 상세: {name_of[dj]} — {trend_lbl}"
                    + ("" if sel != ALL_JOBS else f"  \n<small>사이드바에서 직종을 고르면 바뀝니다. 기본값은 {why}.</small>"),
                    unsafe_allow_html=True)
        if y0 < SWITCH <= y1:
            keco_note("아래 차트에서 점선(2018 코드)과 실선(2025 코드)을 잇지 않은 이유입니다.")
        y0_detail = st.toggle("y축을 0%부터 표시", value=False, key="y0_detail")   # 위 추이 차트 토글과 별개
        fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>수료율</b>", "<b>취업률</b>"),
                            horizontal_spacing=0.08, shared_yaxes=y0_detail)
        jd = rates[(rates.job_key == dj) & rates.year.between(y0, y1)]
        shown_legend = set()
        for ci, mt in enumerate(["completion_rate", "employment_rate"], start=1):
            vals = []
            for c in COURSES:
                for cd, dash, sym in (("2018", "dash", "circle-open"), ("2025", "solid", "circle")):
                    s = jd[(jd.course == c) & (jd.code == cd)].set_index("year")[mt].dropna().sort_index() * 100
                    if s.empty:
                        continue
                    vals += list(s.values / 100)
                    key = f"{SHORT[c]} · {cd} 코드"
                    fig.add_scatter(x=s.index, y=s.values, mode="lines+markers", name=key, legendgroup=key,
                                    showlegend=key not in shown_legend,
                                    line=dict(color=COLOR[c], width=2.5, dash=dash),
                                    marker=dict(size=9, symbol=sym, color=COLOR[c], line=dict(width=2)),
                                    hovertemplate=f"{key} %{{x}}년: %{{y:.1f}}%<extra></extra>", row=1, col=ci)
                    shown_legend.add(key)
            if y0 < SWITCH <= y1:
                fig.add_vline(x=SWITCH - 0.5, line=dict(color=GREY, width=1, dash="dot"), row=1, col=ci)
                fig.add_annotation(x=SWITCH - 0.5, y=0, yref=f"y{'' if ci == 1 else ci} domain", text="KECO 개편",
                                   showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=10, color=INK2),
                                   row=1, col=ci)
            if vals:
                lo, hi = rate_range(vals, y0_detail)
                fig.update_yaxes(range=[lo * 100, hi * 100], row=1, col=ci)
            fig.update_yaxes(ticksuffix="%", row=1, col=ci)
        fig.update_xaxes(tickmode="array", tickvals=years, range=[y0 - 0.4, y1 + 0.4])
        fig.update_layout(height=380, font=PLOTLY_FONT, template="plotly_white",
                          legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"), margin=dict(l=10, r=10, t=50, b=40))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)
        st.caption("2024년 취업률은 원자료에 두 코드 값이 함께 있고(수료율은 2025 코드만) 서로 다릅니다. "
                   "빠진 점은 원자료 값이 없는 경우입니다.")

    # -------------------------------------------------------------- ⑧ 출처·기준·한계
    with st.expander("데이터 출처 · 집계 기준 · 한계"):
        st.markdown("""
**출처** 한국고용정보원 고용행정통계(eis.work24.go.kr, 원자료: 고용정보통합분석시스템 HRD-Net) — 실업자훈련
수료율현황(년), 취업률현황(년), 실시현황(월). 2026-11-24 다운로드.
원본 xlsx → `prep_data.py` 로 정리(원본 셀 64,000여 개 역대조 불일치 0건).
직종코드 대응: 고용노동부, [한국고용직업분류(KECO) 2025 개정 항목표 및 신구연계표](https://www.moel.go.kr/info/lawinfo/instruction/view.do?bbs_seq=20241200770).

**지표 정의 (제공처 설명)**
- 수료율: 훈련종료일이 해당 연도인 과정의 실시인원 대비 수료인원. 다음 해 1월 말 기준 산정.
- 취업률: 훈련종료일이 해당 연도인 과정의 수료생 중 종료일로부터 6개월 이내 취업한 비율. 다음 해 7월 말 기준 산정·8월 말 공개
  → 2025년 취업률(2026년 8월 공개)까지가 현재 받을 수 있는 최신 확정치.
- 훈련 시작 인원: 훈련시작일이 해당 월인 과정의 참여자 수(동일인 중복 산정), 다음 해 1월까지 재산정. 시도는 훈련기관 소재지.

**집계 기준**
- 비율은 연도별 값만 사용하고 여러 연도를 합하거나 평균하지 않음(제공처 주의사항). 과정 '전체' 값은 원자료 값 그대로.
- 원자료에서 해당 직종코드가 쓰이지 않은 연도·월의 0은 결측으로 처리.
- 2018·2025 직종코드는 신구연계표상 중분류(35개)의 번호·구성이 같아 같은 중분류끼리 묶었다(이름이 바뀐 것은
  51번 '미용·예식 서비스직' → '미용·예식 및 반려동물 서비스직' 하나). 중분류를 넘어 옮겨 간 세분류는 소수다
  (예: 교사보조 일부 → 사회복지·종교직, 반려동물 미용·관리 일부 → 보건·의료직).

**한계**
- 과정 구분(계좌제/국기훈련)은 제공처 분류를 그대로 따른다. K-디지털 트레이닝 등 별도 훈련사업이 이 통계에 포함되는지,
  포함된다면 어느 과정으로 집계되는지는 제공처가 밝히지 않아 확인하지 못했다(제공처 문의 예정).
- 2025년 KECO(한국고용직업분류) 개편으로 직종 단위 계열은 2023/2024 사이에서 끊긴다. 2024년 취업률은 두 코드 값이 함께
  제공되는데, 중분류 체계가 거의 같은데도 값이 다르다(예: 국기훈련 정보통신 연구개발직 56.9% vs 52.6%). 과정·훈련생을
  직종에 배정하는 방식이 바뀌었을 수 있으나 제공처가 밝히지 않아 확인하지 못했고, 그래서 잇지 않았다.
- 원자료 비율의 0은 결측으로 처리했다. 대부분(수료율 247칸, 취업률 194칸)은 그 직종코드가 쓰이지 않은 기간이라
  '자료 없음'이 확실하다. 코드 사용 기간 안의 0(수료율 8칸, 취업률 7칸)은 모두 인원이 매우 적은 직종이라 '실제 0%'와
  구분할 수 없다. 이 중 국기훈련 농림어업직 2024년 취업률은 수료율(96.4%)이 있어 실제 0%일 수도 있지만, 인원 비중이
  0.03%라 결과에 주는 영향은 무시할 수준이다.
- 인원이 적은 직종은 비율이 크게 흔들린다 → 같은 직종 비교에 최소 인원 필터.
- 직종별 기여('왜 변했나 — 직종별 분해' 구역)는 비율의 실제 가중치인 직종별 수료인원이 공개되지 않아, 시작 연도의 **훈련 시작 인원
  비중**으로 대신했다. 비율은 그 해 **끝난** 과정, 인원은 그 해 **시작한** 과정 기준(연인원)이라 대상 집단이 달라, 기여의 합이
  실제 변화와 정확히 맞지 않는다. 맞지 않는 부분은 '직종 구성 변화'와 '근사 오차' 막대로 따로 표시했다.
  가중치 기준을 바꾸면 기여 크기도 달라진다(예: 2022→2023년을 전년도 시작 인원 × 수료율로 계산하면 직종별 기여 합이
  −7.6%p 대신 −6.6%p).
- 취업률 격차는 과정의 '효과'를 뜻하지 않는다. 같은 직종 안에서도 과정 내용·훈련생 특성이 다를 수 있다.
- 연령·성별 등 훈련생 특성은 데이터 제공처에서 함께 받을 수 없어(차원 4개 제한) 다루지 못했다.
""")

# ================================================================== 5. 탭 2 — 부록: 훈련 실시 현황
with tab_app:
    st.markdown("분석과 별도로, 실업자훈련이 **언제·어디서 얼마나** 시작되는지 보여 주는 참고 자료입니다. "
                "맨 아래에는 직종코드 신구 대응표가 있습니다.")
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
        # 한 플롯에 과정별 선을 겹친다. 범례를 누르면 선이 숨겨지고 y축이 남은 선에 맞춰 다시 잡힌다.
        show_roll = st.checkbox("12개월 이동평균 함께 보기 (계절 변동을 걷어낸 추세)", value=False)
        m_all = monthly[monthly.course.isin(ac)]                 # 이동평균은 범위 앞 달까지 필요해 전 기간으로 계산
        if sido != "전국":
            m_all = m_all[m_all.sido == sido]
        if sel != ALL_JOBS:
            m_all = m_all[m_all.job_key == sel]
        full_m = (m_all.pivot_table(index="date", columns="course", values="trainees", aggfunc="sum")
                  .reindex(columns=ac).fillna(0))
        roll = full_m.rolling(12, min_periods=12).mean()
        in_rng = (full_m.index.year >= y0) & (full_m.index.year <= y1)
        fig = go.Figure()
        for c in ac:
            fig.add_scatter(x=full_m.index[in_rng], y=full_m[c][in_rng], mode="lines+markers", name=SHORT[c],
                            legendgroup=c,                      # 같은 과정의 월별·평균 선을 범례 한 번 클릭으로 함께 숨김
                            line=dict(color=COLOR[c], width=2), marker=dict(size=4),
                            opacity=0.45 if show_roll else 1.0,
                            hovertemplate=f"{SHORT[c]} %{{x|%Y년 %m월}}: %{{y:,.0f}}명<extra></extra>")
            if show_roll:
                fig.add_scatter(x=roll.index[in_rng], y=roll[c][in_rng], mode="lines",
                                name=f"{SHORT[c]} 12개월 평균", line=dict(color=COLOR[c], width=4), legendgroup=c,
                                hovertemplate=f"{SHORT[c]} 12개월 평균 %{{x|%Y년 %m월}}: %{{y:,.0f}}명<extra></extra>")
        keco_cross = sel != ALL_JOBS and y0 < SWITCH <= y1       # 직종을 골랐고 개편 시점이 범위 안
        if keco_cross:
            sw = pd.Timestamp(f"{SWITCH}-01-01") - pd.Timedelta(days=15)
            fig.add_vline(x=sw, line=dict(color=GREY, width=1, dash="dot"))
            fig.add_annotation(x=sw, y=1, yref="paper", yanchor="bottom", xanchor="right", showarrow=False,
                               text="KECO 개편", font=dict(size=11, color=INK2))
        fig.update_layout(height=420, font=PLOTLY_FONT, template="plotly_white", hovermode="x unified",
                          legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                          yaxis=dict(tickformat=",", ticksuffix="명", rangemode="tozero"),
                          margin=dict(l=10, r=10, t=40, b=30))
        if keco_cross:
            keco_note(f"'{name_of[sel]}'의 {SWITCH}년 1월 앞뒤 값은 이어서 읽으면 안 됩니다.")
        st.plotly_chart(fig, config=PLOTLY_CONFIG)
        summary = []
        for c in ac:
            yt = full_m[c][full_m.index.year == y1].sum()
            yp = full_m[c][full_m.index.year == y1 - 1].sum()
            summary.append(f"{SHORT[c]} {y1}년 {yt:,.0f}명" + (f"(전년 대비 {(yt / yp - 1) * 100:+.1f}%)" if yp > 0 else ""))
        st.markdown(" · ".join(summary))
        st.caption("범례에서 계좌제를 눌러 숨기면 국기훈련 증감이 잘 보입니다(규모가 월 4만 명대 vs 5천 명대)."
                   + (" 굵은 선 = 직전 12개월 평균(2021년 12월부터 표시)." if show_roll else ""))
        b1, b2 = st.columns([3, 2])
        hm = m_sel.pivot_table(index="year", columns="month", values="trainees", aggfunc="sum").fillna(0)
        with b1:
            big = hm.values.max() >= 10000
            z = hm / 1000 if big else hm
            fig = go.Figure(go.Heatmap(
                z=z.values, x=[f"{m}월" for m in hm.columns], y=[str(y) for y in hm.index],
                colorscale="Blues", showscale=False, xgap=2, ygap=2,
                text=z.values, texttemplate="%{text:.1f}" if big else "%{text:,.0f}",
                customdata=hm.values,
                hovertemplate="%{y}년 %{x}: %{customdata:,.0f}명<extra></extra>"))
            fig.update_layout(height=46 * len(hm) + 110, font=PLOTLY_FONT, template="plotly_white",
                              title=dict(text=f"연도 × 월 훈련 시작 인원 ({'천 명' if big else '명'})", x=0,
                                         font=dict(size=14)),
                              margin=dict(l=10, r=10, t=50, b=30), yaxis=dict(autorange="reversed"),
                              xaxis=dict(tickangle=0))
            st.plotly_chart(fig, config=PLOTLY_CONFIG)
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
        fig = go.Figure()
        for c in ac:
            fig.add_bar(y=sd.index, x=sd[c], orientation="h", name=SHORT[c], marker_color=COLOR[c],
                        marker_opacity=[1.0 if (sido == "전국" or s == sido) else 0.3 for s in sd.index],
                        marker_line=dict(color="white", width=1),
                        hovertemplate=f"%{{y}} {SHORT[c]}: %{{x:,.0f}}명<extra></extra>")
        tot = sd.sum(axis=1)
        fig.add_scatter(y=sd.index, x=tot, mode="text", text=[f"{t:,.0f}" for t in tot], textposition="middle right",
                        textfont=dict(size=11, color=[INK if (sido == "전국" or s == sido) else GREY for s in sd.index]),
                        showlegend=False, hoverinfo="skip")
        fig.update_layout(barmode="stack", height=max(26 * len(sd) + 100, 220), font=PLOTLY_FONT,
                          template="plotly_white", legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1,
                                                               xanchor="right", traceorder="normal"),
                          margin=dict(l=10, r=10, t=30, b=30),
                          xaxis=dict(title="명 (연인원)", tickformat=",", range=[0, tot.max() * 1.15]))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)
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

    # -------------------------------------------------------------- 직종코드 신구 대응표
    st.divider()
    st.subheader("직종코드 신구 대응표 — 2018 코드와 2025 코드")
    st.markdown("분석 탭의 '직종'은 KECO **중분류**입니다. 중분류를 고르면 그 안의 세분류가 2018 코드와 2025 코드로 "
                "어떻게 대응하는지 신구연계표 그대로 보여 줍니다. 위의 시도·과정 선택과는 관계없습니다.")
    mids = keco.drop_duplicates("mid").set_index("mid")
    mid_ids = list(mids.index)
    hit = mids.index[mids.job_key == sel]
    pick = st.selectbox("중분류 (2025 코드 기준)", mid_ids, index=mid_ids.index(hit[0]) if len(hit) else 0,
                        format_func=lambda m: f"{m} {mids.at[m, 'mid_name']}")
    km = keco[(keco.mid == pick) | (keco.code_2018.str[:2] == pick)]     # 2018 코드로 이 중분류에 속했던 행도 함께
    st.dataframe(km.rename(columns={"code_2025": "2025 코드", "name_2025": "2025 세분류",
                                    "code_2018": "2018 코드", "name_2018": "2018 세분류"})
                 [["2025 코드", "2025 세분류", "2018 코드", "2018 세분류"]].fillna(""),
                 hide_index=True, width="stretch", height=min(35 * len(km) + 38, 600))
    st.caption("출처: 고용노동부, [한국고용직업분류(KECO) 2025 개정 항목표 및 신구연계표]"
               "(https://www.moel.go.kr/info/lawinfo/instruction/view.do?bbs_seq=20241200770). 한 줄 = 연계표의 한 줄이며, 같은 코드가 여러 줄에 "
               "나오면 나뉘거나 합쳐진 것입니다. 코드 앞 두 자리가 중분류 번호이므로, 앞자리가 고른 중분류와 다른 줄은 "
               "다른 중분류와 주고받은 세분류입니다. 우리 데이터는 중분류 단위라 세분류별 인원·비율은 알 수 없습니다.")
