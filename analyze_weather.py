"""
weather_data.csv を集計し、Ollama上のローカルLLMに解釈・比較させて
自然言語のレポートを生成するプロトタイプ。

設計方針:
  - 672行の生データをそのままLLMに渡すとコンテキストを圧迫し、数値の読み違いも起きやすい。
  - そこで pandas で都市ごとの統計量と異常値(zスコア)を先に計算し、
    要約済みの構造化データとしてLLMに渡す。LLMの役割は計算ではなく「解釈・言語化・比較」に絞る。
"""

import json
import re
import sys

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
CSV_PATH = "weather_data.csv"
Z_THRESHOLD = 2.0
MAX_ANOMALIES_PER_CITY = 5
NUM_CTX = 8192
NUMBER_TOLERANCE = 0.15
NUMBER_UNIT_RE = re.compile(r"(-?\d+\.?\d*)\s*(℃|°C|mm|m/s|時間)")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    return df


def summarize_city(df: pd.DataFrame, city: str) -> dict:
    g = df[df["city"] == city].sort_values("time")
    temp = g["temperature_2m"]
    precip = g["precipitation"]
    wind = g["windspeed_10m"]

    temp_z = (temp - temp.mean()) / temp.std(ddof=0)
    wind_z = (wind - wind.mean()) / wind.std(ddof=0)

    anomaly_rows = g.assign(temp_z=temp_z.values, wind_z=wind_z.values)
    anomaly_rows = anomaly_rows.assign(
        max_abs_z=anomaly_rows[["temp_z", "wind_z"]].abs().max(axis=1)
    )
    anomaly_rows = anomaly_rows[anomaly_rows["max_abs_z"] >= Z_THRESHOLD]
    anomaly_rows = anomaly_rows.sort_values("max_abs_z", ascending=False).head(
        MAX_ANOMALIES_PER_CITY
    )

    return {
        "city": city,
        "period": f"{g['time'].min()} ~ {g['time'].max()}",
        "temperature": {
            "mean": round(temp.mean(), 1),
            "min": round(temp.min(), 1),
            "max": round(temp.max(), 1),
        },
        "precipitation": {
            "total_mm": round(precip.sum(), 1),
            "rain_hours": int((precip > 0).sum()),
        },
        "windspeed": {
            "mean": round(wind.mean(), 1),
            "max": round(wind.max(), 1),
        },
        "anomalies": [
            {
                "time": str(r.time),
                "temperature_2m": r.temperature_2m,
                "windspeed_10m": r.windspeed_10m,
            }
            for r in anomaly_rows.itertuples()
        ],
    }


def build_prompt(summaries: list[dict]) -> str:
    data_json = json.dumps(summaries, ensure_ascii=False, indent=2)
    return f"""あなたは気象データ分析の専門家です。以下は日本の4都市（秋田・新潟・金沢・東京）の
2026年5月20日〜26日の気象統計データ（気温・降水量・風速）です。数値はすでに集計済みです。

```json
{data_json}
```

このデータをもとに、日本語で以下の構成のレポートを書いてください:

1. 【都市ごとの特徴】各都市の気温・降水・風速の傾向を1〜2文で
2. 【異常値・注目ポイント】anomaliesに挙がっている時刻について、何が起きていた可能性があるか
3. 【都市間比較】4都市の中で最も対照的な2都市を挙げ、その違いを説明
4. 【総括】この週の気象パターンから言えることを2〜3文で

必ず4都市（新潟・東京・秋田・金沢）すべてに言及し、"city"フィールドの値をそのまま都市名として使ってください。
「都市A」のような言い換えはしないでください。数値を捏造せず、与えられたデータの範囲内で述べてください。"""


def collect_known_values(summaries: list[dict]) -> set[float]:
    """summarize_city() が計算した「本当の」数値を1個の集合にまとめる。
    LLMの出力に登場する数値がこの集合に含まれなければ、元データにない値=誤りの疑いとみなす。"""
    known: set[float] = set()
    for s in summaries:
        known.add(round(s["temperature"]["mean"], 1))
        known.add(round(s["temperature"]["min"], 1))
        known.add(round(s["temperature"]["max"], 1))
        known.add(round(s["precipitation"]["total_mm"], 1))
        known.add(float(s["precipitation"]["rain_hours"]))
        known.add(round(s["windspeed"]["mean"], 1))
        known.add(round(s["windspeed"]["max"], 1))
        for a in s["anomalies"]:
            known.add(round(a["temperature_2m"], 1))
            known.add(round(a["windspeed_10m"], 1))
    return known


def verify_numbers(report_text: str, known_values: set[float]) -> list[dict]:
    """report_text中の「数値+単位」をすべて拾い、known_valuesに（許容誤差内で）
    一致するものが無ければ、捏造・誤読の疑いがある数値として記録する。"""
    findings = []
    for m in NUMBER_UNIT_RE.finditer(report_text):
        value = float(m.group(1))
        unit = m.group(2)
        if any(abs(value - k) <= NUMBER_TOLERANCE for k in known_values):
            continue
        start, end = max(0, m.start() - 15), min(len(report_text), m.end() + 15)
        findings.append(
            {
                "value": value,
                "unit": unit,
                "context": report_text[start:end].replace("\n", " "),
            }
        )
    return findings


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": NUM_CTX},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def main():
    df = load_data(CSV_PATH)
    cities = sorted(df["city"].unique())
    summaries = [summarize_city(df, c) for c in cities]

    prompt = build_prompt(summaries)
    print(f"[info] {len(cities)}都市のデータを要約し、{MODEL} に送信します...", file=sys.stderr)
    report = call_ollama(prompt)

    known_values = collect_known_values(summaries)
    findings = verify_numbers(report, known_values)
    print(f"[info] 数値検証: {len(findings)}件の疑わしい数値を検出", file=sys.stderr)

    with open("analysis_report.md", "w", encoding="utf-8") as f:
        f.write("# 気象データLLM分析レポート（プロトタイプ）\n\n")
        f.write(f"- 使用モデル: {MODEL} (Ollama, ローカル実行)\n")
        f.write(f"- 対象都市: {', '.join(cities)}\n")
        f.write("- 手法: pandasで集計 → LLMが解釈・比較を言語化\n\n")
        f.write("## 集計データ（LLMへの入力）\n\n```json\n")
        f.write(json.dumps(summaries, ensure_ascii=False, indent=2))
        f.write("\n```\n\n## LLMによる分析\n\n")
        f.write(report)

        f.write("\n\n## 数値検証（自動チェック）\n\n")
        f.write(
            "上記レポート中の「数値+単位」を正規表現で抽出し、元データ（集計JSON）に\n"
            f"±{NUMBER_TOLERANCE}の許容誤差で一致する値が存在するかを機械的に確認した結果。\n\n"
        )
        if findings:
            f.write(f"**{len(findings)}件、元データに存在しない数値を検出しました:**\n\n")
            for fnd in findings:
                f.write(
                    f"- `{fnd['value']}{fnd['unit']}` … 文脈: 「...{fnd['context']}...」\n"
                )
        else:
            f.write("元データに存在しない数値は検出されませんでした。\n")

    print("[done] analysis_report.md を出力しました", file=sys.stderr)


if __name__ == "__main__":
    main()
