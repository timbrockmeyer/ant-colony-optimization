import numpy as np
from aco import ACO

aco = ACO(
    n_ants=50,
    n_best=5,
    init_pheromone=1,
    alpha=1,
    beta=2,
    evaporation_rate=0.1,
    max_iterations=100)

data = np.genfromtxt('distance.txt')

route, cost = aco.run(data, init_location=0, n_jobs=-1, verbose=False)

print(f'Best route leads along the path: {route}.')
print(f'The cost is: {cost}')
