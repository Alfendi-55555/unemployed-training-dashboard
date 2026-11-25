"""원본 xlsx(고용행정통계 다운로드) → 정리본 CSV 변환 스크립트.

원본 파일은 건드리지 않고 data/ 폴더에 새 CSV를 만든다.
    python prep_data.py

출력
- data/rates.csv   : 연도 × 훈련과정 × 직종별 수료율·취업률 (tidy, wide)
- data/monthly.csv : 월 × 시도 × 훈련과정 × 직종별 훈련실시인원 (tidy)

처리 원칙
- 원본의 0 중 '그 직종코드 체계가 쓰이지 않은 기간'의 0은 결측(NaN)으로 바꾼다.
  · 비율(수료율·취업률): 0 은 모두 결측으로 본다(값이 없는 칸이 0으로 채워져 있음).
  · 인원(월): 해당 코드 체계 기간 밖의 0만 결측, 기간 안의 0은 실제 0명.
- 원본의 소계 행('… 전체', '총계')은 상세 행과 분리한다. 인원은 상세 행 합계가
  소계와 일치하는지 검증한 뒤 상세 행만 저장한다(앱에서 필요한 수준으로 다시 합산).
"""
from pathlib import Path
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
HEADER_ROW = 11   # 연도/년월 라벨이 있는 행 (0-based)
DATA_START = 13   # 첫 데이터 행 ('총계')

# 2024-01 부터 월 데이터가 2025 직종코드로 바뀜
MONTHLY_CODE_SWITCH = pd.Timestamp("2024-01-01")


def find_file(keyword: str) -> Path:
    hits = glob.glob(str(ROOT / f"*{keyword}*.xlsx"))
    if len(hits) != 1:
        raise FileNotFoundError(f"'{keyword}' 파일을 하나만 찾아야 하는데 {len(hits)}개: {hits}")
    return Path(hits[0])


def split_job(label):
    """'2025직종_교육직' → ('2025', '교육직')."""
    if not isinstance(label, str) or "직종_" not in label:
        return None, None
    code, name = label.split("직종_", 1)
    return code, name.strip()


def job_key(name):
    """코드 체계가 달라도 '같은 이름'으로 볼 직종끼리 묶는 키 (비교용, 이어붙이기용 아님)."""
    if name is None:
        return None
    key = name.replace(" ", "")
    if key.startswith("미용·예식"):      # 2025: '미용·예식 및 반려동물 서비스직'
        key = "미용·예식서비스직"
    return key


# ---------------------------------------------------------------- 연도 비율
def parse_rates(path: Path, metric: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    years = [int(str(y).replace("년", "")) for y in raw.iloc[HEADER_ROW, 2:]]
    rows, course = [], None
    for _, r in raw.iloc[DATA_START:].iterrows():
        c0, c1 = r[0], r[1]
        if isinstance(c0, str):
            c0 = c0.strip()
            if c0 == "총계":
                course, level = "총계", "전체"
            elif c0.endswith(" 전체"):
                course, level = c0[: -len(" 전체")], "전체"
            else:
                course, level = c0, "직종"
        else:
            level = "직종"
        code, name = split_job(c1)
        if level == "직종" and name is None:
            raise ValueError(f"직종 행인데 직종명이 없음: {r.tolist()}")
        for y, v in zip(years, r.iloc[2:]):
            rows.append(dict(year=y, course=course, level=level, code=code,
                             job=name, job_key=job_key(name), **{metric: v}))
    df = pd.DataFrame(rows)
    df[metric] = pd.to_numeric(df[metric], errors="raise").replace(0, np.nan)
    return df


# ---------------------------------------------------------------- 월 인원
def parse_monthly(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    months = pd.to_datetime([m.replace("년", "-").replace("월", "")
                             for m in raw.iloc[HEADER_ROW, 3:]], format="%Y-%m")
    recs = []
    sido = course = None
    for _, r in raw.iloc[DATA_START:].iterrows():
        c0, c1, c2 = r[0], r[1], r[2]
        if isinstance(c0, str):
            c0 = c0.strip()
            if c0 == "총계":
                sido, course, level = "총계", None, "총계"
            elif c0.endswith(" 전체"):
                sido, course, level = c0[: -len(" 전체")], None, "시도"
            else:
                sido = c0
        if isinstance(c1, str):
            c1 = c1.strip()
            if c1.endswith(" 전체"):
                course, level = c1[: -len(" 전체")], "과정"
            else:
                course = c1
        if isinstance(c2, str):
            level = "직종"
        code, name = split_job(c2)
        recs.append(dict(sido=sido, course=course if level in ("과정", "직종") else None,
                         level=level, code=code, job=name, job_key=job_key(name),
                         values=pd.to_numeric(r.iloc[3:], errors="raise").to_numpy()))
    wide = pd.DataFrame(recs)
    vals = pd.DataFrame(np.vstack(wide["values"]), columns=months).astype(int)
    wide = pd.concat([wide.drop(columns="values"), vals], axis=1)
    long = wide.melt(id_vars=["sido", "course", "level", "code", "job", "job_key"],
                     var_name="ym", value_name="trainees")
    long["ym"] = pd.to_datetime(long["ym"])
    return long


def check_monthly(long: pd.DataFrame) -> None:
    """상세(직종) 합계 = 과정 소계, 과정 합계 = 시도 소계, 시도 합계 = 총계 인지 검증."""
    lv = {k: g for k, g in long.groupby("level")}
    job_sum = lv["직종"].groupby(["sido", "course", "ym"])["trainees"].sum()
    course_tot = lv["과정"].set_index(["sido", "course", "ym"])["trainees"]
    pd.testing.assert_series_equal(job_sum.sort_index(), course_tot.sort_index(), check_names=False)

    course_sum = lv["과정"].groupby(["sido", "ym"])["trainees"].sum()
    sido_tot = lv["시도"].set_index(["sido", "ym"])["trainees"]
    pd.testing.assert_series_equal(course_sum.sort_index(), sido_tot.sort_index(), check_names=False)

    sido_sum = lv["시도"].groupby("ym")["trainees"].sum()
    grand = lv["총계"].set_index("ym")["trainees"]
    pd.testing.assert_series_equal(sido_sum.sort_index(), grand.sort_index(), check_names=False)

    # 코드 체계 기간 밖에는 0만 있어야 한다
    j = lv["직종"]
    outside = ((j.code == "2018") & (j.ym >= MONTHLY_CODE_SWITCH)) | \
              ((j.code == "2025") & (j.ym < MONTHLY_CODE_SWITCH))
    assert (j.loc[outside, "trainees"] == 0).all(), "코드 체계 기간 밖에 0이 아닌 값이 있음"
    print("  [OK] 월 인원: 직종→과정→시도→총계 합계 일치, 코드 기간 밖은 모두 0")


def parse_keco_map(path: Path) -> pd.DataFrame:
    """KECO 2025 신구연계표(두 번째 시트) → 2025 세분류 ↔ 2018 세분류 대응 행.

    병합 셀은 값을 병합 범위 전체에 채운다(한 2018 코드가 여러 2025 코드로 나뉘는 등).
    '삭제'(2025에 없음)·'신설'(2018에 없음)은 원문 그대로 둔다.
    """
    import openpyxl
    ws = openpyxl.load_workbook(path).worksheets[1]
    for rng in list(ws.merged_cells.ranges):
        v = ws.cell(rng.min_row, rng.min_col).value
        ws.unmerge_cells(rng.coord)
        for r in range(rng.min_row, rng.max_row + 1):
            for c in range(rng.min_col, rng.max_col + 1):
                ws.cell(r, c).value = v
    cols = ["major", "major_name", "mid", "mid_name", "minor", "minor_name",
            "code_2025", "name_2025", "code_2018", "name_2018"]
    rows = [[ws.cell(i, c).value for c in range(1, 11)] for i in range(4, ws.max_row + 1)]
    d = pd.DataFrame(rows, columns=cols).astype("string").apply(lambda s: s.str.strip())
    d = d[d.code_2025.notna() & ~d.code_2025.str.endswith("개", na=False)]      # 빈 줄·맨 끝 합계 줄 제거
    d[["mid", "mid_name"]] = d[["mid", "mid_name"]].ffill()
    d.loc[d.code_2025 == "삭제", "name_2025"] = pd.NA
    d.loc[d.code_2018 == "신설", "name_2018"] = pd.NA
    assert d.code_2025.str.fullmatch(r"\d{4}|삭제").all() and d.code_2018.str.fullmatch(r"\d{4}|신설").all()
    assert d.mid.nunique() == 35, "중분류는 35개여야 함"
    d["job_key"] = d.mid_name.map(job_key)
    return d[["mid", "mid_name", "job_key", "code_2025", "name_2025", "code_2018", "name_2018"]]


def main():
    OUT.mkdir(exist_ok=True)

    # 직종코드 신구연계표 ---------------------------------------
    keco = parse_keco_map(find_file("신구연계표"))
    keco.to_csv(OUT / "keco_map.csv", index=False, encoding="utf-8-sig")
    print(f"keco_map.csv: {len(keco)}행 (중분류 {keco.mid.nunique()}개)")

    # 연도 비율 ------------------------------------------------
    comp = parse_rates(find_file("수료율"), "completion_rate")
    emp = parse_rates(find_file("취업률"), "employment_rate")
    keys = ["year", "course", "level", "code", "job", "job_key"]
    rates = comp.merge(emp, on=keys, how="outer", validate="one_to_one")
    rates = rates.dropna(subset=["completion_rate", "employment_rate"], how="all")
    rates = rates.sort_values(["course", "level", "code", "job", "year"], na_position="first")
    rates.to_csv(OUT / "rates.csv", index=False, encoding="utf-8-sig")
    print(f"rates.csv   : {len(rates)}행")

    # 월 인원 --------------------------------------------------
    long = parse_monthly(find_file("실시"))
    check_monthly(long)
    jobs = long[long.level == "직종"].copy()
    outside = ((jobs.code == "2018") & (jobs.ym >= MONTHLY_CODE_SWITCH)) | \
              ((jobs.code == "2025") & (jobs.ym < MONTHLY_CODE_SWITCH))
    jobs = jobs[~outside]          # 구조적 0(해당 코드 미사용 기간) 제거
    jobs = jobs.drop(columns="level")
    jobs.insert(1, "year", jobs.ym.dt.year)
    jobs.insert(2, "month", jobs.ym.dt.month)
    jobs["ym"] = jobs.ym.dt.strftime("%Y-%m")
    jobs = jobs[["ym", "year", "month", "sido", "course", "code", "job", "job_key", "trainees"]]
    jobs.to_csv(OUT / "monthly.csv", index=False, encoding="utf-8-sig")
    print(f"monthly.csv : {len(jobs)}행, 총 {jobs.trainees.sum():,}명(연인원)")


if __name__ == "__main__":
    main()
