r"""
TSP Assignment -- Charleston, SC to Atlanta, GA
================================================
Solves the Traveling Salesman Problem for 30 real cities between
Charleston and Atlanta using five optimization algorithms:

  1. Simulated Annealing  - explores worse solutions to escape local optima
  2. Hill Climbing         - strictly greedy, only accepts improvements
  3. 2-Opt Inversion       - reverses route segments to remove costly edges
  4. Convex Hull Insertion - builds a hull and inserts the remaining cities
  5. Brute Force (DFS)     - search capped at 2,777 complete routes

Outputs:
  tsp_visual.html          - Interactive dark-map visualizer + convergence graph
  tsp_comparison_map.html  - Folium map with toggleable routes

Run inside the virtual environment:
    .\venv\Scripts\python.exe TSP_Assignment.py
"""

import numpy as np
import folium
import json
import os
import webbrowser
import http.server
import threading
import argparse
import math
from functools import partial
from time import perf_counter

# ==============================================================================
# 1. CITY DATA
#    30 real cities/towns along the Charleston, SC -> Atlanta, GA corridor.
#    Index 0 (Charleston) is the start and end point for the round trip.
# ==============================================================================
locations = {
    "Charleston, SC":      (32.7765, -79.9311),   # 0  - START / END
    "Mt Pleasant, SC":     (32.8323, -79.8284),   # 1
    "Summerville, SC":     (33.0185, -80.1757),   # 2
    "Moncks Corner, SC":   (33.1960, -80.0131),   # 3
    "Walterboro, SC":      (32.9052, -80.6666),   # 4
    "Orangeburg, SC":      (33.4918, -80.8556),   # 5
    "Bamberg, SC":         (33.2968, -81.0348),   # 6
    "Aiken, SC":           (33.5604, -81.7196),   # 7
    "N. Augusta, SC":      (33.5019, -81.9651),   # 8
    "Augusta, GA":         (33.4735, -81.9748),   # 9
    "Columbia, SC":        (34.0007, -81.0348),   # 10
    "Lexington, SC":       (33.9813, -81.2364),   # 11
    "Irmo, SC":            (34.0868, -81.1831),   # 12
    "Newberry, SC":        (34.2743, -81.6187),   # 13
    "Clinton, SC":         (34.4724, -81.8805),   # 14
    "Laurens, SC":         (34.4990, -82.0142),   # 15
    "Greenwood, SC":       (34.1954, -82.1618),   # 16
    "Abbeville, SC":       (34.1782, -82.3790),   # 17
    "Anderson, SC":        (34.5034, -82.6501),   # 18
    "Clemson, SC":         (34.6834, -82.8374),   # 19
    "Seneca, SC":          (34.6857, -82.9532),   # 20
    "Greenville, SC":      (34.8526, -82.3940),   # 21
    "Spartanburg, SC":     (34.9496, -81.9320),   # 22
    "Gaffney, SC":         (35.0718, -81.6497),   # 23
    "Toccoa, GA":          (34.5773, -83.3324),   # 24
    "Cornelia, GA":        (34.5115, -83.5271),   # 25
    "Gainesville, GA":     (34.2979, -83.8241),   # 26
    "Athens, GA":          (33.9519, -83.3576),   # 27
    "Lawrenceville, GA":   (33.9562, -83.9880),   # 28
    "Atlanta, GA":         (33.7490, -84.3880),   # 29
}

# Convert to arrays the algorithms can index into
location_names = list(locations.keys())
coords = np.array(list(locations.values()))
num_cities = len(coords)   # 30

# ==============================================================================
# 2. HAVERSINE DISTANCE MATRIX
#    Calculates "as the crow flies" distance (km) between every pair of cities
#    using the Haversine formula for spherical Earth geometry.
# ==============================================================================
def haversine_distance(coord1, coord2):
    """Return great-circle distance in km between two (lat, lon) points."""
    R = 6371.0                              # Earth's radius in km
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Build NxN matrix so any pair-wise lookup is O(1)
dist_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        dist_matrix[i][j] = haversine_distance(coords[i], coords[j])


def calculate_total_distance(route):
    """Sum the total round-trip distance for a given ordering of cities."""
    d = sum(dist_matrix[route[k]][route[k + 1]] for k in range(len(route) - 1))
    d += dist_matrix[route[-1]][route[0]]   # return to start
    return d


# ==============================================================================
# 3. SIMULATED ANNEALING
#    Key idea: use a "temperature" that lets the algorithm accept WORSE solutions
#    early on (to escape local optima), then gradually cool so it converges.
#
#    Records two outputs for the visualizer:
#      curve     - the current (accepted) distance at each temperature step
#                  (this oscillates because SA accepts worse solutions)
#      snapshots - route snapshots whenever the best-ever distance improves
# ==============================================================================
def simulated_annealing(start_route, initial_temp=10000, cooling_rate=0.995,
                        min_temp=0.01, max_iterations=200):
    np.random.seed(42)                  # reproducible results

    # Start from the given random route (same for both algorithms)
    current_route = list(start_route)
    current_dist  = calculate_total_distance(current_route)
    best_route    = list(current_route)
    best_dist     = current_dist
    temp          = initial_temp

    # --- Convergence tracking (do not modify) ---
    curve = [round(current_dist, 2)]
    snapshots = [{"idx": 0, "route": list(current_route),
                  "dist": round(current_dist, 2)}]
    prev_best = best_dist
    step = 0

    # Simulated Annealing Algorithm

    while temp > min_temp:
        for _ in range(max_iterations):
            i, j = np.random.choice(len(current_route), 2, replace=False)
            new_route = list(current_route)
            new_route[i], new_route[j] = new_route[j], new_route[i]
            new_distance = calculate_total_distance(new_route)
            delta = new_distance - current_dist

            # Decision rule
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_route = new_route
                current_dist = new_distance

                if current_dist < best_dist:
                    best_route = list(current_route)
                    best_dist = current_dist

        step += 1
        # Record current distance
        curve.append(round(current_dist, 2))

        if best_dist < prev_best:
            snapshots.append({"idx": step, "route": list(best_route),
            "dist": round(best_dist, 2)})
            prev_best = best_dist

        #reduce temperature
        temp *= cooling_rate
 
    return best_route, best_dist, curve, snapshots


# ==============================================================================
# 4. HILL CLIMBING
#    Key idea: only ever move to a strictly better neighbor.
#    Simple but easily trapped in local optima (no escape mechanism).
#
#    Same recording scheme as SA for the visualizer.
# ==============================================================================
def hill_climbing(start_route, max_iterations=100000, num_curve_points=None):
    np.random.seed(42)                  # same starting point as SA

    # Start from the given random route (same for both algorithms)
    current_route = list(start_route)
    current_dist  = calculate_total_distance(current_route)
    best_route    = list(current_route)
    best_dist     = current_dist

    # Sample every N iterations so curve length matches SA
    target_points = num_curve_points or 2000
    sample_interval = max(1, max_iterations // target_points)

    curve = [round(current_dist, 2)]
    snapshots = [{"idx": 0, "route": list(current_route),
                  "dist": round(current_dist, 2)}]
    prev_best = best_dist
    curve_step = 0

    for iteration in range(max_iterations):
        # Random 2-swap neighbor
        i, j = np.random.choice(len(current_route), 2, replace=False)
        new_route = list(current_route)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_dist = calculate_total_distance(new_route)

        # ONLY accept strictly better solutions
        if new_dist < current_dist:
            current_route = new_route
            current_dist  = new_dist
            if current_dist < best_dist:
                best_route = list(current_route)
                best_dist  = current_dist

        # Sample the current distance for the convergence graph
        if (iteration + 1) % sample_interval == 0:
            curve_step += 1
            curve.append(round(current_dist, 2))
            if best_dist < prev_best:
                snapshots.append({"idx": curve_step, "route": list(best_route),
                                  "dist": round(best_dist, 2)})
                prev_best = best_dist

    return best_route, best_dist, curve, snapshots


# ==============================================================================
# 5. ALTERNATIVE ALGORITHMS
# Each returns (route, distance, curve, snapshots), like SA and HC.
# Research references and the comparison methodology are in README.md.
# ==============================================================================
def _validate_route(start_route):
    """Validate city IDs without changing the supplied order."""
    route = list(start_route)
    if not route or len(set(route)) != len(route):
        raise ValueError("A route must contain distinct city IDs and cannot be empty.")
    if any(not isinstance(city, (int, np.integer)) or
           not 0 <= city < num_cities for city in route):
        raise ValueError("City IDs must index the supplied city data.")
    return route


def _record_route(route, curve, snapshots):
    """Store a copy so later edits cannot change an earlier animation frame."""
    distance = calculate_total_distance(route)
    curve.append(round(float(distance), 2))
    snapshots.append({"idx": len(curve) - 1, "route": list(route),
                      "dist": round(float(distance), 2)})
    return distance


DFS_ROUTE_LIMIT = 2777


def brute_force_dfs(start_route):
    """Explore tours by DFS, stopping after 2,777 complete routes.

    Fix the first city to remove rotational duplicates. Backtrack through
    permutations of the remaining cities. Only a fully exhausted search proves
    optimality; after the cutoff, return the best route found so far.
    """
    route = _validate_route(start_route)
    best_route = list(route)
    curve, snapshots = [], []
    best_dist = _record_route(best_route, curve, snapshots)
    total_tours = math.factorial(len(route) - 1)
    sample_interval = max(1, min(total_tours, DFS_ROUTE_LIMIT) // 1000)
    completed = 0
    path = [route[0]]

    def visit(remaining, partial_distance):
        nonlocal best_route, best_dist, completed
        if not remaining:
            completed += 1
            distance = partial_distance + dist_matrix[path[-1], path[0]]
            improved = distance < best_dist
            if improved:
                best_route, best_dist = list(path), float(distance)
            # Sample regularly AND at every improvement to retain the optimum.
            if improved or completed % sample_interval == 0 or completed == min(total_tours, DFS_ROUTE_LIMIT):
                curve.append(round(best_dist, 2))
                if improved:
                    snapshots.append({"idx": len(curve) - 1,
                                      "route": list(best_route),
                                      "dist": round(best_dist, 2)})
            return
        for index, city in enumerate(remaining):
            if completed >= DFS_ROUTE_LIMIT:
                return
            edge = dist_matrix[path[-1], city]
            path.append(city)
            visit(remaining[:index] + remaining[index + 1:], partial_distance + edge)
            path.pop()  # backtrack before exploring the next branch

    visit(route[1:], 0.0)
    if completed < total_tours:
        print(f"  Brute Force (DFS) stopped after {completed:,} complete routes: "
              "completing the search requires too much computational power. "
              "Returning the best route found before cutoff; optimality is not guaranteed.")
    else:
        print(f"  Brute Force (DFS) completed all {completed:,} routes; optimum proven.")
    return best_route, best_dist, curve, snapshots


def two_opt_inversion(start_route):
    """Apply the best improving segment reversal until a 2-opt local optimum.

    For a symmetric distance matrix, reversing the segment changes only the
    two boundary edges. Keep position zero fixed to avoid rotated duplicates.
    """
    route = _validate_route(start_route)
    curve, snapshots = [], []
    distance = _record_route(route, curve, snapshots)
    n = len(route)
    while True:
        best_delta, best_swap = -1e-10, None
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = route[i - 1], route[i]
                c, d = route[j], route[(j + 1) % n]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]
                         - dist_matrix[a, b] - dist_matrix[c, d])
                if delta < best_delta:
                    best_delta, best_swap = delta, (i, j)
        if best_swap is None:
            break
        i, j = best_swap
        route[i:j + 1] = reversed(route[i:j + 1])
        distance = _record_route(route, curve, snapshots)
    return route, distance, curve, snapshots


def convex_hull_insertion(start_route):
    """Build a planar convex hull, then insert every other city at least cost.

    Longitude scaled by cos(mean latitude) and latitude form a local planar
    approximation suitable for this regional dataset. All insertion costs and
    reported distances use the original Haversine matrix, not planar distance.
    This is a construction heuristic, not an exact TSP solver.
    """
    cities = _validate_route(start_route)
    scale = np.cos(np.radians(np.mean(coords[cities, 0])))
    points = {city: (float(coords[city, 1] * scale), float(coords[city, 0]))
              for city in cities}
    ordered = sorted(cities, key=lambda city: (*points[city], city))

    def cross(a, b, c):
        ax, ay = points[a]
        bx, by = points[b]
        cx, cy = points[c]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    # Andrew's monotone-chain hull. Collinear interior points are inserted later.
    def half_hull(sequence):
        half = []
        for city in sequence:
            while len(half) >= 2 and cross(half[-2], half[-1], city) <= 0:
                half.pop()
            half.append(city)
        return half

    if len(ordered) == 1:
        route = ordered[:]
    else:
        route = half_hull(ordered)[:-1] + half_hull(reversed(ordered))[:-1]
    remaining = set(cities) - set(route)
    curve, snapshots = [], []
    distance = _record_route(route, curve, snapshots)
    while remaining:
        # Cheapest insertion across every unused city and every tour edge.
        choices = []
        for city in sorted(remaining):
            for position, a in enumerate(route):
                b = route[(position + 1) % len(route)]
                extra = (dist_matrix[a, city] + dist_matrix[city, b]
                         - dist_matrix[a, b])
                choices.append((extra, city, position + 1))
        _, city, position = min(choices)
        route.insert(position, city)
        remaining.remove(city)
        distance = _record_route(route, curve, snapshots)
    return route, distance, curve, snapshots


ALGORITHMS = [
    ("sa", "Simulated Annealing", "#ff6600", simulated_annealing),
    ("hc", "Hill Climbing", "#4285f4", hill_climbing),
    ("two_opt", "2-Opt Inversion", "#40d9a0", two_opt_inversion),
    ("hull", "Convex Hull Insertion", "#c084fc", convex_hull_insertion),
    ("dfs", "Brute Force (DFS)", "#facc15", brute_force_dfs),
]


def run_comparison(city_ids):
    """Use the same city set and initial route for each applicable algorithm."""
    rng = np.random.RandomState(0)
    start_route = [int(city) for city in rng.permutation(city_ids)]
    results = []
    print(f"\n{len(city_ids)}-city comparison; initial route: "
          f"{calculate_total_distance(start_route):.2f} km")
    for key, name, color, algorithm in ALGORITHMS:
        started = perf_counter()
        route, distance, curve, snapshots = algorithm(start_route)
        elapsed = perf_counter() - started
        if key == "dfs":
            total = math.factorial(len(city_ids) - 1)
            name += " — best before cutoff" if total > DFS_ROUTE_LIMIT else " — exact optimum"
        print(f"  {name:24} {distance:9.2f} km  ({elapsed:.3f} s)")
        # Rotate only the printed route; snapshot ordering is otherwise preserved.
        origin = route.index(0)
        display = route[origin:] + route[:origin]
        print("    " + " -> ".join(location_names[c] for c in display + display[:1]))
        results.append({"key": key, "name": name, "color": color,
                        "route": route, "best_dist": round(float(distance), 2),
                        "seconds": round(elapsed, 4), "curve": curve,
                        "snapshots": snapshots,
                        "routes_evaluated": min(total, DFS_ROUTE_LIMIT) if key == "dfs" else None,
                        "search_complete": total <= DFS_ROUTE_LIMIT if key == "dfs" else None})
    status = "DFS capped at 2,777 routes" if math.factorial(len(city_ids) - 1) > DFS_ROUTE_LIMIT else "exact DFS baseline"
    return {"label": f"{len(city_ids)} cities ({status})",
            "city_ids": [int(c) for c in city_ids], "algorithms": results}


html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TSP Visualizer - Charleston to Atlanta</title>

<!-- Leaflet (map library) -->
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<!-- Chart.js (graphing library) -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>

<style>
/* -- Reset --------------------------------------------------------- */
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}

/* -- Page layout (dark theme) -------------------------------------- */
body{
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
  background:#0d1117; color:#c9d1d9;
  height:100vh; display:flex; flex-direction:column;
  overflow:hidden;
}

/* -- Header -------------------------------------------------------- */
header{text-align:center;padding:10px 16px 4px;border-bottom:1px solid #21262d}
header h1{
  font-size:1.25rem;font-weight:700;
  background:linear-gradient(135deg,#ff6600,#4285f4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
header p{color:#8b949e;font-size:.78rem;margin-top:2px}

/* -- Controls bar -------------------------------------------------- */
.controls{
  display:flex;align-items:center;justify-content:center;
  gap:8px;padding:6px 16px;background:#161b22;
  border-bottom:1px solid #21262d;flex-wrap:wrap;
}
.controls button{
  background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  border-radius:6px;padding:4px 12px;font-size:.78rem;cursor:pointer;
  transition:.2s;
}
.controls button:hover{border-color:#58a6ff;background:#1f2937}
.controls button.active{background:#1a7f37;border-color:#3fb950}

.stats{flex-wrap:wrap;display:flex;gap:14px;font-size:.78rem;color:#8b949e;margin-left:8px}
.stats .val{font-weight:700;color:#e6edf3}
.stats .sa-val{color:#ff6600}
.stats .hc-val{color:#4285f4}

.speed-wrap{display:flex;align-items:center;gap:4px;font-size:.72rem;color:#8b949e}
.speed-wrap input[type=range]{width:70px;accent-color:#58a6ff}

/* -- Map ----------------------------------------------------------- */
#map{flex:3;min-height:0}

/* -- Chart section ------------------------------------------------- */
.chart-section{
  flex:2;min-height:0;padding:6px 16px 4px;
  background:#0d1117;border-top:1px solid #21262d;
  display:flex;flex-direction:column;
}
.chart-section h2{font-size:.82rem;font-weight:600;color:#8b949e;margin-bottom:2px}
.chart-wrap{flex:1;min-height:0;position:relative}

/* -- Footer -------------------------------------------------------- */
select{background:#21262d;color:#c9d1d9;padding:4px;border:1px solid #30363d;border-radius:6px}
footer{
  text-align:center;padding:5px;font-size:.68rem;
  color:#484f58;border-top:1px solid #21262d;
}
</style>
</head>
<body>

<!-- -- Header ------------------------------------------------------ -->
<header>
  <h1>TSP Visualizer - Charleston, SC to Atlanta, GA</h1>
  <p>Compare route improvement, construction, and exact search</p>
</header>

<!-- -- Controls ---------------------------------------------------- -->
<div class="controls">
  <label for="comparison">Comparison</label>
  <select id="comparison" onchange="selectComparison()"></select>
  <button id="btnPlay" onclick="togglePlay()">&#9654; Play</button>
  <button onclick="stepForward()">&#9193; Step</button>
  <button onclick="showFinal()">&#9193; Final</button>
  <button onclick="resetAnimation()">&#8634; Reset</button>
  <div class="speed-wrap">
    <label>Speed</label>
    <input type="range" id="speed" min="1" max="30" value="5">
  </div>
  <div class="stats" id="distances"></div>
  <span id="stepLabel"></span>
</div>

<!-- -- Leaflet map ------------------------------------------------- -->
<div id="map"></div>

<!-- -- Chart.js convergence graph ---------------------------------- -->
<div class="chart-section">
  <h2>Route distance vs. relative algorithm progress</h2>
  <div class="chart-wrap">
    <canvas id="chart"></canvas>
  </div>
</div>

<footer id="explanation"></footer>

<!-- ================================================================ -->
<!-- JAVASCRIPT                                                       -->
<!-- ================================================================ -->
<script>
const DATA = __DATA_JSON__;
const cities = DATA.cities;
const selector = document.getElementById('comparison');
DATA.comparisons.forEach((group, index) => selector.add(new Option(group.label, index)));
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors', maxZoom: 19
}).addTo(map);
const markers = L.layerGroup().addTo(map);
let lines = [], layerControl = null, algorithms = [];
let frame = 0, playing = false, animId = null;
const maxFrames = 1001;
const chart = new Chart(document.getElementById('chart'), {
  type: 'line', data: {labels: [], datasets: []},
  options: {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {legend: {labels: {color: '#c9d1d9'}}},
    scales: {
      x: {title: {display: true, text: 'Relative progress (%) — not runtime', color: '#8b949e'},
          ticks: {color: '#8b949e', maxTicksLimit: 11}},
      y: {title: {display: true, text: 'Route distance (km)', color: '#8b949e'},
          ticks: {color: '#8b949e'}}
    }
  }
});
function stopAnimation() {
  playing = false;
  if (animId !== null) cancelAnimationFrame(animId);
  animId = null;
  document.getElementById('btnPlay').textContent = '▶ Play';
  document.getElementById('btnPlay').classList.remove('active');
}
function selectComparison() {
  stopAnimation();
  frame = 0;
  lines.forEach(line => map.removeLayer(line));
  if (layerControl) map.removeControl(layerControl);
  markers.clearLayers();
  const group = DATA.comparisons[Number(selector.value)];
  algorithms = group.algorithms;
  group.city_ids.forEach(id => {
    const city = cities[id];
    L.circleMarker([city.lat, city.lon], {radius: 5, color: id === 0 ? '#ff4081' : '#4285f4',
      fillOpacity: 0.9}).bindTooltip(city.name).addTo(markers);
  });
  map.fitBounds(group.city_ids.map(id => [cities[id].lat, cities[id].lon]), {padding: [25, 25]});
  const overlays = {};
  lines = algorithms.map(algorithm => {
    const line = L.polyline([], {color: algorithm.color, weight: 3, opacity: 0.8}).addTo(map);
    overlays[algorithm.name] = line;
    return line;
  });
  layerControl = L.control.layers(null, overlays, {collapsed: false}).addTo(map);
  document.getElementById('distances').innerHTML = algorithms.map(a =>
    `<span style="color:${a.color}">${a.name}: <b id="distance-${a.key}"></b> km` +
    ` <small>(final ${a.best_dist.toFixed(2)}; ${a.seconds.toFixed(3)} s)</small></span>`).join('');
  chart.data.datasets = algorithms.map(a => ({label: a.name, borderColor: a.color,
    borderWidth: 1.8, data: [], pointRadius: 0, tension: 0}));
  document.getElementById('explanation').textContent =
    'Progress is normalized separately for each algorithm, not equal work or time. ' +
    'Map: best routes (hull: partial construction). Chart: SA current distance; HC/2-Opt/DFS best distance. ' +
    'Hull distance can rise as cities are added; compare final complete routes. DFS stops at 2,777 complete routes if unfinished: completing the search requires too much computational power. Use layer checkboxes to isolate routes.';
  renderFrame();
}
function nativeIndex(algorithm, progress) {
  return Math.floor(progress * (algorithm.curve.length - 1) / (maxFrames - 1));
}
function renderFrame() {
  algorithms.forEach((algorithm, index) => {
    const target = nativeIndex(algorithm, frame);
    let snapshot = algorithm.snapshots[0];
    for (const candidate of algorithm.snapshots) {
      if (candidate.idx > target) break;
      snapshot = candidate;
    }
    const points = snapshot.route.map(id => [cities[id].lat, cities[id].lon]);
    lines[index].setLatLngs([...points, points[0]]);
    document.getElementById('distance-' + algorithm.key).textContent =
      snapshot.dist.toFixed(2) + (algorithm.key === 'hull' ? ` (${snapshot.route.length} cities)` : '');
    chart.data.datasets[index].data = Array.from({length: frame + 1}, (_, k) =>
      algorithm.curve[nativeIndex(algorithm, k)]);
  });
  chart.data.labels = Array.from({length: frame + 1}, (_, k) => (k / 10).toFixed(1));
  chart.update('none');
  document.getElementById('stepLabel').textContent = (frame / 10).toFixed(1) + '%';
}
function animate() {
  if (!playing) return;
  frame = Math.min(frame + Number(document.getElementById('speed').value), maxFrames - 1);
  renderFrame();
  if (frame === maxFrames - 1) stopAnimation();
  else animId = requestAnimationFrame(animate);
}
function togglePlay() {
  if (playing) return stopAnimation();
  if (frame === maxFrames - 1) frame = 0;
  playing = true;
  document.getElementById('btnPlay').textContent = '⏸ Pause';
  document.getElementById('btnPlay').classList.add('active');
  animId = requestAnimationFrame(animate);
}
function stepForward() {stopAnimation(); frame = Math.min(frame + 10, maxFrames - 1); renderFrame();}
function showFinal() {stopAnimation(); frame = maxFrames - 1; renderFrame();}
function resetAnimation() {stopAnimation(); frame = 0; renderFrame();}
selectComparison();
</script>
</body>
</html>
"""


def write_visualizations(comparisons, output_dir):
    """Write both HTML views; no browser or server is needed for generation."""
    m = folium.Map(location=coords.mean(axis=0).tolist(), zoom_start=7)
    for name, coord in locations.items():
        folium.CircleMarker(coord.tolist() if hasattr(coord, "tolist") else coord,
                            radius=5, popup=name).add_to(m)
    for group_index, group in enumerate(comparisons):
        for result in group["algorithms"]:
            layer = folium.FeatureGroup(
                name=f'{group["label"]}: {result["name"]} ({result["best_dist"]:.2f} km)',
                show=group_index == 0)
            route = result["route"]
            points = [coords[city].tolist() for city in route + route[:1]]
            folium.PolyLine(points, color=result["color"], weight=3).add_to(layer)
            layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(os.path.join(output_dir, "tsp_comparison_map.html"))
    data = {"cities": [{"name": name, "lat": float(coord[0]), "lon": float(coord[1])}
                       for name, coord in locations.items()], "comparisons": comparisons}
    with open(os.path.join(output_dir, "tsp_visual.html"), "w", encoding="utf-8") as file:
        file.write(html_template.replace("__DATA_JSON__", json.dumps(data)))
    print("\n[OK] Saved tsp_comparison_map.html and tsp_visual.html")


def main():
    parser = argparse.ArgumentParser(description="Compare five TSP algorithms.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Generate both HTML files and exit without starting a server.")
    parser.add_argument("--dfs-cities", type=int, choices=range(3, 11), default=7,
                        help="City count for the small comparison (default: 7; DFS still capped at 2,777 tours).")
    args = parser.parse_args()
    full = run_comparison(list(range(num_cities)))
    # Spread the subset through the city list; always include Charleston and Atlanta.
    subset = [int(city) for city in np.linspace(0, num_cities - 1, args.dfs_cities)]
    small = run_comparison(subset)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    write_visualizations([full, small], output_dir)
    if args.no_browser:
        return

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    # An available loopback port avoids conflicts and serves only this directory.
    handler = partial(QuietHandler, directory=output_dir)
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/tsp_visual.html"
        webbrowser.open(url)
        print(f"Visualizer: {url}\nPress Enter to stop the server...")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()