import numpy as np

f = np.linspace(1,10,10)

g = np.zeros(len(f))

for i in range(len(g)):
    index_val = i - 2
    print(index_val)
    g[i] = f[(i-2) % len(f)]

print(f)
print(g)

