"""HTML report renderer — Jinja2 layout, inline Plotly scatter chart."""
from __future__ import annotations

from pathlib import Path

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>aphex report</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      background: #0d0d0d; color: #d0d0d0;
      max-width: 1080px; margin: 48px auto; padding: 0 24px;
    }
    header { margin-bottom: 40px; }
    header h1 { font-size: 0.75rem; font-weight: 500; letter-spacing: 0.18em;
                text-transform: uppercase; color: #444; }
    h2 { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.16em;
         text-transform: uppercase; color: #444; margin: 40px 0 16px; }
    .rec {
      border: 1px solid #1e1e1e; border-radius: 4px;
      padding: 20px 24px; margin-top: 24px;
    }
    .rec-name { font-size: 1.1rem; font-weight: 600; color: #f0f0f0;
                margin-bottom: 14px; }
    .rec-grid { display: flex; gap: 32px; flex-wrap: wrap; }
    .rec-kv { display: flex; flex-direction: column; gap: 2px; }
    .rec-label { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
                 color: #444; }
    .rec-value { font-size: 0.95rem; color: #d0d0d0; }
    .rec-rationale { margin-top: 16px; font-size: 0.8rem; color: #555; line-height: 1.5; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th { text-align: left; padding: 6px 10px; border-bottom: 1px solid #1e1e1e;
         font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
         color: #444; font-weight: 500; }
    td { padding: 7px 10px; border-bottom: 1px solid #141414; }
    tr.ok:hover td { background: #111; }
    tr.failed td { color: #333; }
    .rank { color: #4ade80; font-weight: 700; }
    .err  { color: #333; font-size: 0.78rem; max-width: 340px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  </style>
</head>
<body>
<header>
  <h1>aphex · benchmark report</h1>
</header>

<div class="rec">
  <div class="rec-name">{{ rec.backend }}</div>
  <div class="rec-grid">
    <div class="rec-kv">
      <span class="rec-label">batch</span>
      <span class="rec-value">{{ rec.batch_size }}</span>
    </div>
    <div class="rec-kv">
      <span class="rec-label">p50 latency</span>
      <span class="rec-value">{{ "%.2f" | format(rec.latency_p50_ms) }} ms</span>
    </div>
    <div class="rec-kv">
      <span class="rec-label">p95 latency</span>
      <span class="rec-value">{{ "%.2f" | format(rec.latency_p95_ms) }} ms</span>
    </div>
    <div class="rec-kv">
      <span class="rec-label">throughput</span>
      <span class="rec-value">{{ rec.throughput_rps | int }} req/s</span>
    </div>
    <div class="rec-kv">
      <span class="rec-label">memory</span>
      <span class="rec-value">{{ rec.memory_mb | int }} MB</span>
    </div>
  </div>
  {% if rec.rationale %}
  <div class="rec-rationale">{{ rec.rationale }}</div>
  {% endif %}
</div>

<h2>latency vs throughput</h2>
{{ chart_html }}

<h2>all candidates</h2>
<table>
  <thead>
    <tr>
      <th></th><th>backend</th><th>batch</th>
      <th>p50</th><th>p95</th><th>req/s</th><th>MB</th><th>note</th>
    </tr>
  </thead>
  <tbody>
    {% set ns = namespace(rank=0) %}
    {% for r in results %}
    <tr class="{{ 'ok' if r.ok else 'failed' }}">
      <td>
        {% if r.ok %}{% set ns.rank = ns.rank + 1 %}<span class="rank">#{{ ns.rank }}</span>
        {% else %}—{% endif %}
      </td>
      <td>{{ r.backend }}</td>
      <td>{{ r.batch_size }}</td>
      {% if r.ok %}
      <td>{{ "%.2f" | format(r.latency_p50_ms) }} ms</td>
      <td>{{ "%.2f" | format(r.latency_p95_ms) }} ms</td>
      <td>{{ r.throughput_rps | int }}</td>
      <td>{{ r.memory_mb | int }}</td>
      <td></td>
      {% else %}
      <td>—</td><td>—</td><td>—</td><td>—</td>
      <td class="err" title="{{ r.error or '' }}">{{ r.error or '' }}</td>
      {% endif %}
    </tr>
    {% endfor %}
  </tbody>
</table>
</body>
</html>
"""


def render_report(payload: dict, output: Path) -> None:
    """Render an HTML benchmark report from an aphex JSON metrics payload."""
    from jinja2 import Template

    rec = payload.get("recommendation") or {}
    results = payload.get("results", [])
    chart_html = _build_chart(results, rec)

    html = Template(_TEMPLATE).render(
        rec=_Obj(rec),
        results=[_Obj(r) for r in results],
        chart_html=chart_html,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def _build_chart(results: list[dict], rec: dict) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return "<p style='color:#444;padding:1rem'>Install plotly to enable chart rendering.</p>"

    ok = [r for r in results if r.get("ok")]
    if not ok:
        return "<p style='color:#444;padding:1rem'>No successful results to plot.</p>"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[r["latency_p50_ms"] for r in ok],
        y=[r["throughput_rps"] for r in ok],
        mode="markers+text",
        text=[f"{r['backend']}<br>bs={r['batch_size']}" for r in ok],
        textposition="top center",
        textfont=dict(size=10, color="#666"),
        marker=dict(
            size=[max(10, min(28, (r["memory_mb"] or 0) / 10 + 8)) for r in ok],
            color="#1e1e1e",
            line=dict(color="#444", width=1),
        ),
        hovertemplate="<b>%{text}</b><br>p50: %{x:.2f} ms<br>req/s: %{y:.0f}<extra></extra>",
        name="candidates",
    ))

    rec_x, rec_y = rec.get("latency_p50_ms"), rec.get("throughput_rps")
    if rec_x is not None and rec_y is not None:
        fig.add_trace(go.Scatter(
            x=[rec_x], y=[rec_y],
            mode="markers",
            marker=dict(size=16, color="#4ade80", symbol="star"),
            hovertemplate=f"<b>recommended</b><br>p50: {rec_x:.2f} ms<br>req/s: {rec_y:.0f}<extra></extra>",
            name="recommendation",
        ))

    fig.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font=dict(color="#888", size=11),
        xaxis=dict(title="p50 latency (ms)", gridcolor="#1a1a1a", zerolinecolor="#222"),
        yaxis=dict(title="throughput (req/s)", gridcolor="#1a1a1a", zerolinecolor="#222"),
        legend=dict(bgcolor="#0d0d0d", bordercolor="#222"),
        margin=dict(l=60, r=20, t=20, b=60),
        height=380,
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})


class _Obj:
    """Wrap a dict for attribute-style access in Jinja2 templates."""
    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)

    def __getattr__(self, name: str):
        return None
