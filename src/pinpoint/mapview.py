"""Render search results as a self-contained Leaflet map."""

import html
import json
from pathlib import Path

import pandas as pd

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>pinpoint: __TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
 integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="anonymous">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
 integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin="anonymous"></script>
<style>
 html,body{margin:0;height:100%;font:14px/1.4 system-ui,sans-serif}
 #map{position:absolute;inset:0}
 #panel{position:absolute;z-index:1000;top:10px;right:10px;max-width:360px;background:#fff;color:#111;
   padding:10px 12px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.3)}
 #panel h1{font-size:15px;margin:0 0 6px} #panel table{border-collapse:collapse;width:100%}
 #panel td{padding:2px 4px;border-top:1px solid #eee;vertical-align:top} .k{color:#555;white-space:nowrap}
 @media (max-width:600px){#panel{left:10px;right:10px;max-width:none;top:auto;bottom:10px}}
</style></head><body><div id="map"></div><div id="panel"></div>
<script>
const R = __DATA__;
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 19, attribution: 'Imagery &copy; Esri'});
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'});
const map = L.map('map', {layers: [sat]}).setView([R.best.lat, R.best.lon], 13);
L.control.layers({'Satellite': sat, 'Streets': osm}).addTo(map);
const pts = L.layerGroup().addTo(map), tracks = L.layerGroup().addTo(map);
const smax = Math.max(...R.points.map(p => p.score)), smin = Math.min(...R.points.map(p => p.score));
for (const p of R.points.slice().reverse()) {
  const x = (p.score - smin) / ((smax - smin) || 1);
  L.circleMarker([p.lat, p.lon], {radius: 3 + 5 * x, weight: 0, fillOpacity: .35 + .5 * x,
    fillColor: `hsl(${Math.round(240 - 240 * x)},90%,50%)`}).bindPopup(p.html).addTo(pts);
}
const colors = ['#00e5ff', '#ff4081', '#ffd740', '#69f0ae'];
R.tracks.forEach((t, i) => L.polyline(t.path, {color: colors[i % 4], weight: 2, opacity: .9})
  .bindTooltip(t.label).addTo(tracks));
const b = R.best;
L.marker([b.lat, b.lon]).bindPopup(b.html).addTo(map).openPopup();
L.circle([b.lat, b.lon], {radius: R.radius_m, color: '#fff', weight: 1, fill: false}).addTo(map);
document.getElementById('panel').innerHTML = R.panel;
L.control.layers(null, {'Candidates': pts, 'Matched flight tracks': tracks}, {position: 'bottomleft'}).addTo(map);
</script></body></html>"""


def _row_html(r, n):
    s = " + ".join(f"{r[f's{i + 1}']:.2f}" for i in range(n))
    fl = ", ".join(str(r[f"icao{i + 1}"]) for i in range(n))
    when = pd.to_datetime(r["start"], unit="s").strftime("%Y-%m-%d %H:%M:%S UTC")
    return (f"<b>{r['lat']:.5f}, {r['lon']:.5f}</b><br>score {r['score']:.2f} ({s})<br>"
            f"aircraft {html.escape(fl)}<br>recording start {when}")


def render(res: pd.DataFrame, out: Path, title: str, info: dict, tracks=(), top=3000, radius_m=150):
    n = sum(c.startswith("icao") for c in res.columns)
    pts = res.head(top)
    best = res.iloc[0]
    rows = "".join(f"<tr><td class=k>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in info.items())
    data = {
        "points": [{"lat": r.lat, "lon": r.lon, "score": r.score, "html": _row_html(r, n)} for _, r in pts.iterrows()],
        "best": {"lat": best.lat, "lon": best.lon, "html": "<b>Best estimate</b><br>" + _row_html(best, n)},
        "tracks": list(tracks),
        "radius_m": radius_m,
        "panel": f"<h1>{html.escape(title)}</h1><table>{rows}</table>",
    }
    page = TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out.write_text(page)
    return out


def render_pin(out: Path, title: str, lat: float, lon: float, note: str, radius_m: float = 500):
    """A single judged location (e.g. from visual/metadata clues) on the same map page."""
    gm = f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}"
    data = {"points": [], "tracks": [], "radius_m": radius_m,
            "best": {"lat": lat, "lon": lon, "html": f"<b>{lat:.5f}, {lon:.5f}</b><br>{html.escape(note)}"},
            "panel": f"<h1>{html.escape(title)}</h1><table><tr><td class=k>Location</td>"
                     f"<td><a href='{gm}' target=_blank>{lat:.5f}, {lon:.5f}</a></td></tr>"
                     f"<tr><td class=k>Basis</td><td>{html.escape(note)}</td></tr></table>"}
    out.write_text(TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    return out
