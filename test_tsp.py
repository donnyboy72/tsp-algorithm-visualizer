"""Algorithm correctness checks; importing the program does not launch a browser."""
import contextlib
import io
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import TSP_Assignment as tsp


class AlgorithmTests(unittest.TestCase):
    def check_result(self, result, cities):
        route, distance, curve, snapshots = result
        self.assertEqual(sorted(route), sorted(cities))
        self.assertAlmostEqual(distance, tsp.calculate_total_distance(route))
        self.assertTrue(curve)
        self.assertEqual(snapshots[0]['idx'], 0)
        self.assertEqual([s['idx'] for s in snapshots],
                         sorted(set(s['idx'] for s in snapshots)))
        for snapshot in snapshots:
            self.assertLess(snapshot['idx'], len(curve))
            self.assertEqual(snapshot['dist'], round(tsp.calculate_total_distance(snapshot['route']), 2))
        self.assertEqual(snapshots[-1]['dist'], round(distance, 2))

    def test_dfs_matches_independent_exhaustive_enumeration(self):
        cities = [0, 4, 10, 15, 24, 29]
        expected = min(tsp.calculate_total_distance([cities[0], *tail])
                       for tail in itertools.permutations(cities[1:]))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = tsp.brute_force_dfs(cities)
        self.check_result(result, cities)
        self.assertAlmostEqual(result[1], expected)
        self.assertIn('completed all 120 routes; optimum proven', output.getvalue())

    def test_dfs_stops_at_exactly_2777_complete_routes(self):
        cities = list(range(30))
        # Independent enumeration of precisely the permitted prefix of DFS leaves.
        expected = min(tsp.calculate_total_distance([cities[0], *tail])
                       for tail in itertools.islice(itertools.permutations(cities[1:]), 2777))
        output = io.StringIO()
        leaf_count = 0
        import sys

        def trace(frame, event, arg):
            nonlocal leaf_count
            if (event == 'call' and frame.f_code.co_name == 'visit'
                    and not frame.f_locals['remaining']):
                leaf_count += 1
            return trace

        sys.settrace(trace)
        try:
            with contextlib.redirect_stdout(output):
                result = tsp.brute_force_dfs(cities)
        finally:
            sys.settrace(None)
        self.assertEqual(leaf_count, 2777)
        self.assertAlmostEqual(result[1], expected)
        self.check_result(result, cities)
        self.assertIn('requires too much computational power', output.getvalue())
        self.assertIn('optimality is not guaranteed', output.getvalue())

    def test_two_opt_finishes_at_local_optimum(self):
        cities = [int(c) for c in np.random.RandomState(3).permutation(30)]
        result = tsp.two_opt_inversion(cities)
        self.check_result(result, cities)
        route, distance, curve, _ = result
        self.assertLessEqual(distance, tsp.calculate_total_distance(cities))
        self.assertTrue(all(a >= b for a, b in zip(curve, curve[1:])))
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                self.assertGreaterEqual(tsp.calculate_total_distance(candidate), distance - 1e-8)

    def test_hull_inserts_all_interior_cities_and_keeps_snapshots(self):
        cities = list(range(30))
        result = tsp.convex_hull_insertion(cities)
        self.check_result(result, cities)
        snapshots = result[3]
        self.assertLess(len(snapshots[0]['route']), len(cities))
        for previous, current in zip(snapshots, snapshots[1:]):
            self.assertEqual(len(current['route']), len(previous['route']) + 1)
            self.assertTrue(set(previous['route']).issubset(current['route']))

    def test_small_and_degenerate_inputs(self):
        for cities in ([0], [0, 29], [0, 9, 29]):
            for algorithm in (tsp.two_opt_inversion, tsp.convex_hull_insertion, tsp.brute_force_dfs):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.check_result(algorithm(cities), cities)
        # Collinear and coincident coordinates must not lose any city IDs.
        for points in (np.array([[0., 0.], [0., 1.], [0., 2.], [0., 3.]]), np.zeros((4, 2))):
            matrix = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            with patch.object(tsp, 'coords', points), patch.object(tsp, 'dist_matrix', matrix):
                self.check_result(tsp.convex_hull_insertion([0, 1, 2, 3]), [0, 1, 2, 3])

    def test_sa_and_hc_accept_city_subsets(self):
        cities = [0, 9, 18, 29]
        for result in (tsp.simulated_annealing(cities, initial_temp=10, min_temp=1,
                                              cooling_rate=0.5, max_iterations=20),
                       tsp.hill_climbing(cities, max_iterations=100, num_curve_points=20)):
            self.check_result(result, cities)


if __name__ == '__main__':
    unittest.main()
