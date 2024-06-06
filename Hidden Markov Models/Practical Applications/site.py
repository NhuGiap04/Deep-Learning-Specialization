import numpy as np

transitions = {}
row_sums = {}

# collect counts
for line in open('../Datasets/site_data.csv'):
    s, e = line.rstrip().split(',')
    transitions[(s, e)] = transitions.get((s, e), 0.) + 1
    row_sums[s] = row_sums.get(s, 0.) + 1

# normalize
for k, v in transitions.items():
    s, e = k
    transitions[k] = v / row_sums[s]

# Calculate initial state distribution
print("Initial State Distribution")
for k, v in transitions.items():
    s, e = k
    if s == '-1':
        print(e, v)

# which has the highest bounce rate
for k, v in transitions.items():
    s, e = k
    if e == 'B':
        print(f"Bounce rate for {s}: {v}")
