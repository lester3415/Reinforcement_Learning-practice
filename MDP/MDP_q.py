import numpy as np
import pandas as pd
import random

reward = pd.read_csv("MDP\\reward.csv")
# print(reward)

data = reward.iloc[:, 1:].values
print(data)

pi = 0.5
y = 1

status = {"C1": random.randint(1, 10),
          "C2": random.randint(1, 10),
          "C3": random.randint(1, 10),
          "FB": random.randint(1, 10),
          "Sleep": 0}

for i in range(100):
    # C1
    status["C1"] = max(status["C2"]+data[0, 1], status["FB"]+data[0, 3])
    # C2
    status["C2"] = max(status["C3"]+data[1, 2], status["Sleep"]+data[1, 5])
    # C3
    status["C3"] = max(status["C1"]*0.2 + status["C2"]*0.4 + status["C3"]*0.4,
                       status["Sleep"]+data[2, 5])
    # FB
    status["FB"] = max(status["C1"]+data[3, 0], status["FB"]+data[3, 3])
    # Sleep
    # status[4] = 0

print(status)
