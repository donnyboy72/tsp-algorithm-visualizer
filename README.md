# Traveling Salesman Problem: five algorithms

Compare round trips through 30 cities between Charleston, SC and Atlanta, GA.
The program prints routes, distances, and measured runtimes, then generates an
animated Leaflet/Chart.js visualizer and a Folium map with toggleable layers.
Distances are great-circle (Haversine) distances in kilometers, not road mileage.

## Run

Use Python 3 with NumPy and Folium:

```sh
python3 -m venv venv
source venv/bin/activate
pip install numpy folium
python TSP_Assignment.py
```

On Windows, activate with `venv\Scripts\activate` instead.
The default command opens a local browser view. Press Enter to stop its server.
To generate HTML without a browser or server:

```sh
python TSP_Assignment.py --no-browser
```

Outputs are written beside the script:

- `tsp_visual.html`: animation, convergence chart, runtime and distance summaries.
- `tsp_comparison_map.html`: final routes with independently toggleable layers.

Internet access is needed to load the map tiles and visualization libraries.
No additional Python packages are required for the new algorithms.

## Algorithms

| Algorithm | What it does | Guarantee |
| --- | --- | --- |
| Simulated Annealing | Random city swaps; temperature controls acceptance of worse routes | Heuristic |
| Hill Climbing | Accepts only improving random swaps | Heuristic |
| 2-Opt Inversion | Finds the best improving segment reversal, repeats until no improving reversal remains | 2-opt local optimum, within floating-point tolerance |
| Convex Hull Insertion | Builds an outer hull, then inserts remaining cities at the globally cheapest insertion position | Heuristic |
| Brute Force (DFS) | Enumerates routes recursively with a fixed starting city and backtracking | Exact only if enumeration completes before the cutoff |

### Brute-force cutoff: 2,777 complete routes

One iteration means **one complete round-trip candidate evaluated**, not a
recursive call or an edge traversal. DFS stops immediately after evaluating
2,777 complete routes if additional routes remain. It prints:

> Brute Force (DFS) stopped after 2,777 complete routes: completing the search requires too much computational power. Returning the best route found before cutoff; optimality is not guaranteed.

This is a configured computational budget, not a measurement of CPU capacity.
The best route found is still displayed, explicitly labeled **best before cutoff**.
The initial route is the first DFS leaf; initializing its distance does not count
as an additional distinct tour. The algorithm explores `(n - 1)!` permutations
when unrestricted, including reverse tours; 30 cities would require `29!` tours.
The capped result can be poor because DFS initially explores only a small part
of the permutation space. It is not a claimed exact solution for 30 cities.

### Two comparison groups

The default run evaluates all five algorithms on all 30 cities, then on a
seven-city subset spread across the city list (including Charleston and Atlanta).
Seven cities require only `6! = 720` tours, so DFS finishes and establishes an
exact baseline there. Use the visualizer's comparison selector to switch groups.

```sh
python TSP_Assignment.py --dfs-cities 9
```

This changes the small comparison's city count (3–10). The same 2,777-route
cutoff applies; eight or more cities will hit it. Within each group, SA, HC,
2-Opt, and DFS receive the same initial route. Convex Hull constructs its own
route from the same city set.

### Read the animation correctly

- Each algorithm's recorded steps are normalized to 0–100% progress; these are
  **not equal amounts of work or elapsed time**. Printed runtimes measure the
  solver calls, including tracking and DFS status output, not HTML generation.
- The map shows best-known routes for SA/HC/2-Opt/DFS. The SA chart shows current
  accepted distance, so its line can rise while its best-known map route stays put.
- Convex Hull begins with a partial tour. Its curve may rise as interior cities
  are inserted, and its distance label includes the current city count. Compare
  final complete tours, not its partial-tour distances against complete routes.
- Use the map layer checkboxes to distinguish overlapping routes.

### Geometry and implementation

2-Opt reverses a segment and evaluates the change using its two boundary edges.
This relies on the symmetric Haversine distance matrix. Each neighborhood scan
is quadratic in the number of cities; multiple scans may be needed.

Convex Hull uses Andrew's monotone-chain algorithm on a local planar approximation:
longitude scaled by the cosine of mean latitude, and latitude. This is appropriate
as a regional heuristic, not a global spherical hull algorithm. Collinear and
coincident points are retained through the insertion phase. Insertions minimize
`distance(a, city) + distance(city, b) - distance(a, b)` using Haversine distances.
The straightforward insertion implementation takes cubic time overall.

The original SA loop is retained; SA and HC now sample positions using the actual
route length so they also work on subsets. Importing the module does not run the
comparisons or open a browser.

## Verification

```sh
python -m unittest -v test_tsp
python TSP_Assignment.py --no-browser
```

Tests compare DFS against independent permutation enumeration, count its exact
2,777-leaf cutoff, verify that 2-Opt has no remaining improving reversals, and
check route validity, recorded distances, hull insertion, degenerate geometry,
and subset support.

## Research references

- [Simon Fraser University: Solving the Traveling Salesman Problem](https://www2.cs.sfu.ca/CourseCentral/125/tjd/tsp_example.html)
  explains exhaustive permutation search, fixing the initial city, and factorial growth.
- [Cornell CS681: Problem Set 3, 2-Opt tour improvement](https://www.cs.cornell.edu/courses/cs681/2003fa/ps3.pdf)
  describes the two-edge replacement and segment reversal operation.
- [Goutham et al.: A Convex Hull Cheapest Insertion Heuristic for the Non-Euclidean TSP](https://arxiv.org/abs/2302.06582)
  provides the hull-plus-cheapest-insertion approach. This project implements a
  simple regional version, not a reproduction of the paper's full experiments.

These extensions go beyond the original assignment. Its stated submission is
only `TSP_Assignment.py`; the README, tests, environment, and generated HTML are
project files rather than required submission attachments.
