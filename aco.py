import numpy as np
from numpy.random import SeedSequence, default_rng
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class ACO:
    """
    Ant Colony Optimization algorithm for square matrices,
    e.g. distance matrices

    Attributes
    ----------
    n_ants : int
        the number of ant processes used
    n_best : int
        the number of best results in each iteration used to _pheromone_update
        the pheromone matrix
    init_pheromone : float
        the initial value of the pheromone matrix
    alpha : int
        a number alpha >= 0 that controls the influence of the pheromone
    beta : int
        a number beta >= 1 that controls the influence of the a priori known
        distance between two points
    evaporation_rate : float
        a coefficient 0 <= x <= 1 controlling how much pheromone disappears
        in each iteration. E.g. a value of 0.1 would mean 10% of the pheromone
        dissipates.
    max_iterations: int
        the number of iterations the algorithm maximally runs

    Methods
    -------
    run(self, X, n_jobs=None, verbose=False)
        executes the algorithm and returns the result
    """
    def __init__(self, n_ants=50, n_best=5, alpha=1, beta=2, evaporation_rate=0.1, init_pheromone=1, max_iterations=100):

        # hyperparameters
        self.n_ants = n_ants
        self.n_best = n_best
        self.init_pheromone = init_pheromone
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.max_iterations = max_iterations

        # global runtime variables
        self.init_location = 0
        self._pheromone = None
        self._cost = None

        # constants set at start of run
        self._INDICES = None
        self._SIZE = 0

    def _generate_solution(self, rng):
        # generates a single solution (or path), simulating the walk of a single
        # ant given the current pheromone levels

        # transition table to determine ant pathing probabilities.
        # exponents can cause numerical issues leading to NaN or inf entries in
        # extreme cases - hence nan_to_num is called as as a precaution
        transition = np.nan_to_num(self._pheromone**self.alpha * (1/self._cost)**self.beta)

        solution = np.zeros(self._SIZE, dtype=np.int32)

        # main loop
        idx = self.init_location
        for i in range(self._SIZE-1):
            transition[:, idx] = 0  # remove current location from table
            # choose next location; normalise transition to probability
            idx = rng.choice(self._INDICES, p=transition[idx] / transition[idx].sum())
            solution[i] = idx

        cost = sum([self._cost[x,y] for x,y in zip(solution, solution[1:])]) + self._cost[0, solution[0]]
        return solution, cost


    def _intensification(self, idx, path_cost):
        # pheromone increase is weighted by the cost of the path to offset the
        # cost factor (beta) when the transition matrix is calculated

        # *** TODO: it might be better to normalise the distance matrix at the
        # beginning and recalculate true cost later to control the influcne of
        # varying cost matrices ***

        return (1/path_cost) * len(self._cost)


    def _pheromone_update(self, paths, costs):
        # walks through the path and updates the pheromone based on cost

        self._pheromone *= 1-self.evaporation_rate
        for path, cost in zip(paths, costs):
            x = self.init_location
            for y in path:  # path goes from x to y
                self._pheromone[x,y] += self._intensification((x,y), cost)
                x = y


    def run(self, data, init_location=0, n_jobs=1, verbose=False):
        """Gets and prints the spreadsheet's header columns

        Parameters
        ----------
        data : array_like or list
            Square matrix of the input data, e.g. a distance matrix.
        init_location : int, optional
            the index of the starting point, e.g. the first node of a path
            (default is 0).
        n_jobs : int, optional
            The number of parallel processes. Use none or a negative number to
            automatically determine the maximum number of cores available
            (default is 1).
        verbose : bool, optional
            for additional output information during runtime
            (default is False).

        Returns
        -------
        (array_like, int)
            The indices of the best best solution (excluding the starting
            position) and its cost.
        """

        # initialize cost matrix and pheromone matrix
        self._cost = np.array(data, dtype=np.float64)
        np.fill_diagonal(self._cost, np.inf)
        self._pheromone = np.full_like(data, self.init_pheromone, dtype=np.float64)

        # set constants
        self._SIZE = len(data)
        self._INDICES = np.arange(self._SIZE)
        self.init_location = init_location

        # number of processes
        if n_jobs < 0 or n_jobs is None:
            n_jobs = cpu_count()
        n_jobs = max(1, round(n_jobs))

        # instanciate random number generators
        ss = SeedSequence()
        if n_jobs > 1:
            seeds = ss.spawn(self.n_ants)
            streams = [default_rng(s) for s in seeds]
        else:
            rng = default_rng()

        # main loop
        all_time_shortest_path = None
        all_time_lowest_cost = np.inf
        last_iteration_lowest_cost = np.inf
        convergence = 0
        for i in tqdm(range(self.max_iterations)):
            # generate a solution for each ant
            if n_jobs > 1:
                with Pool(n_jobs) as p:
                    solutions = p.map(self._generate_solution, streams)
            else:
                solutions = [self._generate_solution(rng) for _ in range(self.n_ants)]

            # sort solutions ascending by cost
            routes, costs = zip(*sorted(solutions, key=lambda x: x[1]))
            lowest_cost_in_iter = costs[0]

            # update and check convergence criterion
            if lowest_cost_in_iter == last_iteration_lowest_cost:
                convergence += 1
                if convergence >= 10:   # *** TODO: find better way ***
                    break
            else:
                convergence = 0
                last_iteration_lowest_cost = lowest_cost_in_iter

            # update cost and variables if necessary
            if lowest_cost_in_iter < all_time_lowest_cost:
                all_time_lowest_cost = lowest_cost_in_iter
                all_time_shortest_path = routes[0]

            # update pheromone matrix based on best solutions
            self._pheromone_update(routes[:self.n_best], costs[:self.n_best])

            # misc
            if(verbose):
                print(f'EPOCH {i + 1}: {all_time_lowest_cost}')

            # end of main loop

        print(f'converged after {i + 1} iterations with result: {all_time_lowest_cost} \n')
        return all_time_shortest_path, all_time_lowest_cost
