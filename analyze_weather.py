"""
weather_data.csv を集計し、Ollama上のローカルLLMに解釈・比較させて
自然言語のレポートを生成するプロトタイプ (v2)。

設計方針（v1からの変更点）:
  - v1では集計済みの数値JSONをそのままLLMに渡していたが、7B級モデルは
    複数都市にまたがる数値を横断的に正確に扱うのが苦手で、数値の取り違えが発生した。
  - v2では具体的な数値（℃・mm・m/s）を一切LLMに渡さない。数値の集計・表示はすべて
    コードが直接行い、LLMには「どの都市がどの項目で何位か」という順位関係と、
    異常が見られた時刻（すでに正しい値）だけを渡す。
  - LLMの役割を「数値の記憶・再現」から「定性的な解釈・比較の言語化」に完全に限定することで、
    数値誤りが構造的に起こらないようにする。
  - それでも指示を無視して数値を書いてしまうケースに備え、出力に数値+単位が
    含まれていないかを自動チェックする。
"""

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
NUMBER_UNIT_RE = re.compile(r"(-?\d+\.?\d*)\s*(℃|°C|mm|m/s|時間)")


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"])


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


def render_data_table(summaries: list[dict]) -> str:
    """数値はここでコードが直接書く。LLMはこの表を書く工程に一切関与しない。"""
    header = (
        "| 都市 | 平均気温(℃) | 最低(℃) | 最高(℃) | 総降水量(mm) | 降雨時間 "
        "| 平均風速(m/s) | 最大風速(m/s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    rows = []
    for s in summaries:
        t, p, w = s["temperature"], s["precipitation"], s["windspeed"]
        rows.append(
            f"| {s['city']} | {t['mean']} | {t['min']} | {t['max']} | "
            f"{p['total_mm']} | {p['rain_hours']}時間 | {w['mean']} | {w['max']} |"
        )
    return "\n".join([header, sep, *rows])


def render_anomaly_list(summaries: list[dict]) -> str:
    lines = []
    for s in summaries:
        if not s["anomalies"]:
            continue
        lines.append(f"**{s['city']}**")
        for a in s["anomalies"]:
            lines.append(
                f"- {a['time']}: 気温 {a['temperature_2m']}℃ / 風速 {a['windspeed_10m']}m/s"
            )
    return "\n".join(lines) if lines else "（該当なし）"


def build_rankings(summaries: list[dict]) -> str:
    """生の数値の代わりに「順位」だけを渡す。LLMの手元には数値が一切無いので、
    数値を捏造しようにも参照するものがない。"""

    def rank_line(label: str, key) -> str:
        ordered = sorted(summaries, key=key, reverse=True)
        return f"- {label}: " + " > ".join(s["city"] for s in ordered)

    lines = [
        rank_line("平均気温が高い順", lambda s: s["temperature"]["mean"]),
        rank_line("総降水量が多い順", lambda s: s["precipitation"]["total_mm"]),
        rank_line("降雨時間が長い順", lambda s: s["precipitation"]["rain_hours"]),
        rank_line("平均風速が強い順", lambda s: s["windspeed"]["mean"]),
        rank_line("最大風速が強い順", lambda s: s["windspeed"]["max"]),
    ]

    anomaly_lines = ["【異常が観測された時刻（都市別、数値は伏せてある）】"]
    for s in summaries:
        if not s["anomalies"]:
            continue
        times = "、".join(a["time"] for a in s["anomalies"])
        anomaly_lines.append(f"- {s['city']}: {times}")

    return "\n".join(lines) + "\n\n" + "\n".join(anomaly_lines)


def build_prompt(summaries: list[dict]) -> str:
    rankings = build_rankings(summaries)
    cities = "、".join(s["city"] for s in summaries)
    return f"""あなたは気象データ分析の専門家です。日本の4都市（{cities}）の
2026年5月20日〜26日の気象傾向について、以下の「都市間の順位・相対比較」と
「異常が観測された時刻」だけをもとに、日本語でレポートを書いてください。

{rankings}

■ 絶対的な制約（重要）
- 気温(℃)・降水量(mm)・風速(m/s)などの具体的な数値は、あなたには一切与えられていません。
  したがって数値は絶対に書かないでください。書けば、それは全て捏造です。
- 使ってよいのは「最も高い/低い」「〜より強い/穏やか」のような相対表現、都市名、
  上記に示された時刻データのみです。

レポートの構成:
1. 【都市ごとの特徴】各都市の気温・降水・風速の傾向を定性的に1〜2文で
2. 【異常が見られた時間帯】示された時刻について、何が起きていた可能性があるか（推測）
3. 【都市間比較】最も対照的な2都市を挙げ、その違いを定性的に説明
4. 【総括】この週の気象パターンから言えることを2〜3文で"""


def check_narrative_compliance(narrative: str) -> list[dict]:
    """LLMには数値を書くなと指示した。それでも数値+単位が出現していれば、
    指示違反=捏造とみなして全て記録する（v1の「既知の値と照合」より厳しい基準）。"""
    violations = []
    for m in NUMBER_UNIT_RE.finditer(narrative):
        start, end = max(0, m.start() - 15), min(len(narrative), m.end() + 15)
        violations.append(
            {
                "text": m.group(0),
                "context": narrative[start:end].replace("\n", " "),
            }
        )
    return violations


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
    print(
        f"[info] {len(cities)}都市の「順位関係」のみを {MODEL} に送信します"
        "（生の数値は渡しません）...",
        file=sys.stderr,
    )
    narrative = call_ollama(prompt)

    violations = check_narrative_compliance(narrative)
    print(f"[info] 指示違反（数値の記載）チェック: {len(violations)}件", file=sys.stderr)

    with open("analysis_report.md", "w", encoding="utf-8") as f:
        f.write("# 気象データLLM分析レポート（プロトタイプ v2）\n\n")
        f.write(f"- 使用モデル: {MODEL} (Ollama, ローカル実行)\n")
        f.write(f"- 対象都市: {', '.join(cities)}\n")
        f.write(
            "- 手法: pandasで集計 → **数値はコードが直接**表・リストに出力し、"
            "LLMには数値を一切渡さず、順位関係と時刻だけから定性的な解釈・比較を書かせる\n\n"
        )

        f.write("## 集計データ（数値はすべてコードが出力・LLM不介入）\n\n")
        f.write(render_data_table(summaries))
        f.write(f"\n\n### 異常値（上位{MAX_ANOMALIES_PER_CITY}件/都市、zスコア基準）\n\n")
        f.write(render_anomaly_list(summaries))

        f.write("\n\n## LLMによる定性的な解釈・比較\n\n")
        f.write(narrative)

        f.write("\n\n## 指示遵守チェック（自動）\n\n")
        f.write(
            "LLMには「具体的な数値を書くな」と明示的に指示した。"
            "上記のLLM出力に数値+単位が出現していないかを機械的に確認した結果。\n\n"
        )
        if violations:
            f.write(f"**{len(violations)}件、指示に反して数値が記載されていました:**\n\n")
            for v in violations:
                f.write(f"- `{v['text']}` … 文脈: 「...{v['context']}...」\n")
        else:
            f.write("数値の記載は検出されませんでした。指示に従っています。\n")

    print("[done] analysis_report.md を出力しました", file=sys.stderr)


if __name__ == "__main__":
    main()
