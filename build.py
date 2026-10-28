"""results.json을 template.html에 넣어 index.html을 만든다."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = (HERE / "results.json").read_text(encoding="utf-8")
html = (HERE / "template.html").read_text(encoding="utf-8").replace("__DATA__", data)
(HERE / "index.html").write_text(html, encoding="utf-8")
print("index.html", len(html.encode("utf-8")) // 1024, "KB")
