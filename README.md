# 와인 리뷰 데이터 분석: 와인 점수는 무엇으로 결정되는가

Wine Enthusiast 리뷰 약 10만 건으로 **가격·산지·품종·평가자·빈티지·리뷰 표현**이 와인 점수(80~100점)에 미치는 영향을 다중 회귀분석했습니다.

📊 **[인포그래픽 보고서 보기](https://hyunholee9204.github.io/Wine-review_data-analysis/)**

## 주요 결과

| 모형 | 변수 | 테스트 R² |
|---|---|---|
| M1 | log₂(가격) | 0.377 |
| M2 | + 나라·품종·평가자·빈티지 | 0.446 |
| M3 | + 리뷰 길이·키워드 12개 | **0.598** |

- 가격이 2배가 되면 점수가 약 **+1.41점** 오릅니다. 가격은 가장 영향력이 큰 변수입니다(표준화 계수 β = 0.43).
- 리뷰 표현이 설명력을 크게 높입니다. "thin" −2.1점, "simple" −1.0점, "complex" +1.0점, "elegant" +1.0점입니다.
- 같은 가격이라면 미국 와인 점수가 가장 낮습니다. 오스트리아는 미국보다 +1.6점 높습니다.
- 다른 조건이 같아도 평가자에 따라 최대 약 2.5점 차이가 납니다.

## 분석 방법

- **데이터**: 원본 129,971건에서 중복을 제거하고 가격·빈티지 결측을 제외해 107,586건을 분석했습니다. 학습 80%, 테스트 20%로 나눴습니다.
- **모형**: OLS 다중 회귀이며, 이분산을 보정하기 위해 HC3 강건 표준오차를 사용했습니다.
- **진단**: VIF(다중공선성), 잔차 정규성, Breusch–Pagan(등분산성), 학습·테스트 성능 비교를 확인했습니다.

## 실행 방법

```bash
pip install -r requirements.txt
# Kaggle에서 winemag-data-130k-v2.csv를 받아 data/ 폴더에 넣습니다
python analysis.py   # 회귀분석 → results.json
python build.py      # results.json → index.html (보고서)
```

## 파일 구성

| 파일 | 내용 |
|---|---|
| `analysis.py` | 데이터 정제, 회귀분석, 진단 |
| `results.json` | 분석 결과 수치 |
| `template.html` | 보고서 템플릿 |
| `build.py` | 템플릿에 결과를 넣어 `index.html` 생성 |
| `index.html` | 인포그래픽 보고서 (GitHub Pages) |

## 데이터 출처

[Kaggle – Wine Reviews (zackthoutt)](https://www.kaggle.com/datasets/zackthoutt/wine-reviews). 원본 데이터는 용량과 라이선스 문제로 저장소에 포함하지 않았습니다.
