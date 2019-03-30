import numpy as np
from multiprocessing import Pool
from functools import partial

class ACO:

    def __init__(self, n_ants=10, n_best=3, alpha=1, beta=1, evaporation_rate=0.05, init_pheromone=1, max_iterations=500):
        self._n_ants = n_ants
        self._n_best = n_best
        self._init_pheromone = init_pheromone
        self._alpha = alpha
        self._beta = beta
        self._evaporation_rate = evaporation_rate

    def _generate_solution(self):
        transition_matrix = np.nan_to_num(self._pheromone_matrix**self._alpha * (1/self._cost_matrix)**self._beta)
        size = len(transition_matrix)
        solution = np.zeros(size, dtype=np.int32)
        ar = np.arange(size)
        idx = 0
        for i in range(size-1):
            transition_matrix[:, idx] *= 0
            idx = np.random.choice(ar, p=transition_matrix[idx] / transition_matrix[idx].sum())
            solution[i] = idx
        cost = sum([self._cost_matrix[x,y] for x,y in zip(solution, solution[1:])]) + self._cost_matrix[0, solution[0]]
        return solution, cost

    def _daemon_actions(self, *args, **kwargs):
        pass

    def _update_function(self, idx, path_cost):
        return (1/path_cost) * len(self._cost_matrix)

    def _pheromone_update(self, paths, costs):
        self._pheromone_matrix *= 1-self._evaporation_rate
        for path, cost in zip(paths, costs):
            x = 0
            for y in path:
                self._pheromone_matrix[x,y] += self._update_function((x,y), cost)
                x = y

    def run(self, X, *args, **kwargs):
        self._cost_matrix = np.array(X, dtype=np.float64)
        np.fill_diagonal(self._cost_matrix, np.inf)
        self._pheromone_matrix = np.full_like(self._cost_matrix, self._init_pheromone, dtype=np.float64)

        all_time_shortest_path = None
        all_time_lowest_cost = np.inf
        last_iteration_lowest_cost = np.inf
        convergence = 0
        while convergence < 10:
            solutions = np.array([self._generate_solution() for _ in range(self._n_ants)])
            routes, costs = zip(*sorted(solutions, key=lambda x: x[1]))

            lowest_cost_in_run = costs[0]
            if lowest_cost_in_run == last_iteration_lowest_cost:
                convergence += 1
            else:
                convergence = 0
                last_iteration_lowest_cost = lowest_cost_in_run
            if lowest_cost_in_run < all_time_lowest_cost:
                all_time_shortest_path = routes[0]

            self._daemon_actions(*args, **kwargs)
            self._pheromone_update(routes[:self._n_best], costs[:self._n_best])
        return routes[0], lowest_cost_in_run

    def _wrapper(self, *wrap_args, **kwargs):
        X, seed, *args = wrap_args[0]
        np.random.seed(seed)
        return self.run(X, *args, **kwargs)

    def run_parallel(self, X, n_cores=None, *args, **kwargs):
        seeds = np.random.randint(2**32 - 1, size=len(X))
        with Pool(n_cores) as p:
            ant_map = p.map(partial(self._wrapper, *args, **kwargs), zip(X, seeds))
            return ant_map
