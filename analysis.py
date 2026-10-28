"""와인 리뷰 점수(전반적 만족도) 다중 회귀분석.

입력: data/winemag-data-130k-v2.csv (Kaggle에서 내려받아 넣기)
출력: results.json (보고서 페이지가 쓰는 통계 결과)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "winemag-data-130k-v2.csv"
OUT = HERE / "results.json"
SEED = 42

KEYWORDS = {
    # 긍정적으로 쓰일 것으로 예상되는 표현
    "complex": r"\bcomplex",
    "elegant": r"\belegan",
    "long": r"\blong\b",
    "concentrated": r"\bconcentrat",
    "age": r"\bage(?:d|s|ing|worthy)?\b|\bcellar",
    "rich": r"\brich",
    # 부정적으로 쓰일 것으로 예상되는 표현
    "simple": r"\bsimple",
    "light": r"\blight",
    "soft": r"\bsoft",
    "thin": r"\bthin\b",
    "short": r"\bshort\b",
    "easy": r"\beasy\b",
}


# ── 1. 데이터 정제 ────────────────────────────────────────────
raw = pd.read_csv(SRC, index_col=0)
n_raw = len(raw)
df = raw.drop_duplicates(subset=["title", "description"]).copy()
n_dedup = len(df)
n_no_price = int(df["price"].isna().sum())
df = df.dropna(subset=["price", "country", "variety"])

df["vintage"] = df["title"].str.extract(r"\b(19[89]\d|20[01]\d)\b")[0].astype(float)
df = df[df["vintage"].between(1995, 2016)]
df["words"] = df["description"].str.split().str.len()
df["log2_price"] = np.log2(df["price"])

desc = df["description"].str.lower()
for k, pat in KEYWORDS.items():
    df[f"kw_{k}"] = desc.str.contains(pat, regex=True).astype(int)

top_countries = df["country"].value_counts().head(10).index
df["country_g"] = np.where(df["country"].isin(top_countries), df["country"], "Other")
top_varieties = df["variety"].value_counts().head(12).index
df["variety_g"] = np.where(df["variety"].isin(top_varieties), df["variety"], "Other")
df["taster"] = df["taster_name"].fillna("Unknown")
big_tasters = df["taster"].value_counts()
big_tasters = big_tasters[big_tasters >= 2000].index
df["taster_g"] = np.where(df["taster"].isin(big_tasters), df["taster"], "Other")
df = df.reset_index(drop=True)
n_model = len(df)

# ── 2. 회귀모형 (단계적으로 변수 추가) ─────────────────────────
kw_terms = " + ".join(f"kw_{k}" for k in KEYWORDS)
cat_terms = ("C(country_g, Treatment('US')) + C(variety_g, Treatment('Pinot Noir'))"
             " + C(taster_g, Treatment('Unknown'))")
formulas = {
    "M1 가격": "points ~ log2_price",
    "M2 + 산지·품종·평가자·빈티지": f"points ~ log2_price + vintage + {cat_terms}",
    "M3 + 리뷰 텍스트": f"points ~ log2_price + vintage + words + {kw_terms} + {cat_terms}",
}

rng = np.random.default_rng(SEED)
test_mask = rng.random(n_model) < 0.2
train, test = df[~test_mask], df[test_mask]

models, fits = [], {}
for name, f in formulas.items():
    fit = smf.ols(f, data=train).fit(cov_type="HC3")  # BP 검정에서 이분산 확인 → 강건 표준오차
    pred = fit.predict(test)
    fits[name] = fit
    models.append({
        "name": name,
        "k": int(fit.df_model),
        "r2": fit.rsquared,
        "adj_r2": fit.rsquared_adj,
        "test_r2": 1 - ((test.points - pred) ** 2).sum() / ((test.points - test.points.mean()) ** 2).sum(),
        "test_rmse": float(np.sqrt(((test.points - pred) ** 2).mean())),
        "test_mae": float((test.points - pred).abs().mean()),
        "f": fit.fvalue,
        "aic": fit.aic,
    })

final = fits["M3 + 리뷰 텍스트"]
ci = final.conf_int()
coefs = []
for term in final.params.index:
    if term == "Intercept":
        continue
    m = re.match(r"C\((\w+), Treatment\('([^']+)'\)\)\[T\.(.+)\]", term)
    if m:
        group = {"country_g": "나라", "variety_g": "품종", "taster_g": "평가자"}[m.group(1)]
        label, ref = m.group(3), m.group(2)
    else:
        group, label, ref = ("키워드", term[3:], "미포함") if term.startswith("kw_") else ("수치형", term, None)
    coefs.append({
        "group": group, "term": label, "ref": ref,
        "b": final.params[term], "se": final.bse[term],
        "lo": ci.loc[term, 0], "hi": ci.loc[term, 1],
        "t": final.tvalues[term], "p": final.pvalues[term],
    })

# 수치형 변수 표준화 계수 (1 SD 변화당 점수 변화를 점수 SD로 나눈 값)
num_vars = ["log2_price", "vintage", "words"]
std_beta = {v: float(final.params[v] * train[v].std() / train.points.std()) for v in num_vars}

# VIF (수치형 + 키워드)
vif_cols = num_vars + [f"kw_{k}" for k in KEYWORDS]
X = train[vif_cols].assign(const=1.0).to_numpy()
vif = {c: float(variance_inflation_factor(X, i)) for i, c in enumerate(vif_cols)}

# 잔차 진단
resid = final.resid
bp = het_breuschpagan(resid, final.model.exog)
diag = {
    "durbin_watson": float(durbin_watson(resid)),
    "skew": float(stats.skew(resid)),
    "kurtosis": float(stats.kurtosis(resid)),
    "bp_stat": float(bp[0]), "bp_p": float(bp[1]),
    "resid_sd": float(resid.std()),
}
edges = np.arange(-8, 8.5, 0.5)
h, _ = np.histogram(resid.clip(-8, 7.99), bins=edges)
diag["resid_hist"] = {"edges": edges.tolist(), "counts": h.tolist()}

# 예측 vs 실제 (테스트셋, 실제 점수별 평균 예측)
test_pred = final.predict(test)
pva = (pd.DataFrame({"actual": test.points, "pred": test_pred})
       .groupby("actual")["pred"].agg(["mean", "count",
                                       lambda s: s.quantile(.1), lambda s: s.quantile(.9)]))
pva.columns = ["mean", "n", "p10", "p90"]
pred_vs_actual = [{"actual": int(a), **{k: float(v) for k, v in r.items()}} for a, r in pva.iterrows()]

# ── 3. 기술통계 ──────────────────────────────────────────────
pts = df["points"]
points_hist = pts.value_counts().sort_index()
price_edges = [4, 10, 15, 20, 30, 40, 50, 75, 100, 200, 3301]
price_bins = pd.cut(df["price"], price_edges, right=False)
price_curve = df.groupby(price_bins, observed=True)["points"].agg(["mean", "count"]).reset_index()
price_curve["label"] = [f"${int(i.left)}–{int(i.right) if i.right < 3000 else ''}" for i in price_curve["price"]]
price_curve["label"] = price_curve["label"].str.replace("–$", "+", regex=True)

country_tbl = (df[df.country_g != "Other"].groupby("country")
               .agg(n=("points", "size"), points=("points", "mean"), price=("price", "median"))
               .sort_values("points", ascending=False).reset_index())
variety_tbl = (df[df.variety_g != "Other"].groupby("variety")
               .agg(n=("points", "size"), points=("points", "mean"), price=("price", "median"))
               .sort_values("points", ascending=False).reset_index())
taster_tbl = (df[df.taster != "Unknown"].groupby("taster")
              .agg(n=("points", "size"), points=("points", "mean"), sd=("points", "std"))
              .query("n >= 500").sort_values("points", ascending=False).reset_index())

kw_tbl = []
for k in KEYWORDS:
    col = df[f"kw_{k}"]
    kw_tbl.append({"kw": k, "share": float(col.mean()),
                   "with": float(df.loc[col == 1, "points"].mean()),
                   "without": float(df.loc[col == 0, "points"].mean())})

wb = df.groupby(pd.cut(df["words"], [0, 20, 30, 40, 50, 60, 80, 200]), observed=True)["points"].agg(["mean", "count"])
words_curve = [{"label": f"{int(i.left)}–{int(i.right)}", "mean": float(r["mean"]), "n": int(r["count"])}
               for i, r in wb.iterrows()]

# 가성비: 모형 예측보다 점수가 높게 나온 와인 중 $25 이하, 92점 이상
df_pred = final.predict(df)
df["gap"] = df["points"] - df_pred
value = (df[(df.price <= 25) & (df.points >= 92)].sort_values("gap", ascending=False)
         .drop_duplicates("title").head(8))
value_list = [{"title": r.title, "country": r.country, "variety": r.variety,
               "points": int(r.points), "price": float(r.price), "gap": float(r.gap)}
              for r in value.itertuples()]

corr_price = float(stats.pearsonr(df.log2_price, df.points)[0])
corr_words = float(stats.pearsonr(df.words, df.points)[0])

result = {
    "counts": {"raw": n_raw, "dedup": n_dedup, "no_price": n_no_price, "model": n_model,
               "train": int((~test_mask).sum()), "test": int(test_mask.sum())},
    "points": {"mean": float(pts.mean()), "sd": float(pts.std()), "median": float(pts.median()),
               "min": int(pts.min()), "max": int(pts.max()),
               "hist": {str(k): int(v) for k, v in points_hist.items()}},
    "price": {"median": float(df.price.median()), "mean": float(df.price.mean()),
              "p90": float(df.price.quantile(.9)), "max": float(df.price.max())},
    "corr": {"log2_price": corr_price, "words": corr_words},
    "price_curve": [{"label": r.label, "mean": float(r["mean"]), "n": int(r["count"])}
                    for _, r in price_curve.iterrows()],
    "words_curve": words_curve,
    "countries": country_tbl.to_dict("records"),
    "varieties": variety_tbl.to_dict("records"),
    "tasters": taster_tbl.to_dict("records"),
    "keywords": kw_tbl,
    "models": models,
    "coefs": coefs,
    "intercept": float(final.params["Intercept"]),
    "std_beta": std_beta,
    "vif": vif,
    "diag": diag,
    "pred_vs_actual": pred_vs_actual,
    "value_wines": value_list,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(final.summary().tables[0])
for m in models:
    print(m["name"], round(m["r2"], 3), round(m["test_r2"], 3), round(m["test_rmse"], 3))
print(json.dumps({"std_beta": std_beta, "vif": vif, "diag": {k: v for k, v in diag.items() if k != "resid_hist"}}, indent=1))
