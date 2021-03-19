from ACO import ACO
import numpy as np

aco = ACO()

X = np.genfromtxt('distance.txt')

res = aco.run(X)

print(res)
