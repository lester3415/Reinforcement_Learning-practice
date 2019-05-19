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
    status["C1"] = pi * (data[0, 3] + y*status["FB"] + data[0, 1] + y*status["C2"])
    # C2
    status["C2"] = pi * (data[1, 2] + y*status["C3"] + data[1, 5] + y*status["Sleep"])
    # C3
    status["C3"] = pi * (data[2, 4] + y*0.2*status["C1"] + y*0.4*status["C2"] + y*0.4*status["C3"] +
                      data[2, 5] + y*status["Sleep"])
    # FB
    status["FB"] = pi * (data[3, 0] + y*status["C1"] + data[3, 3] + y*status["FB"])
    # Sleep
    # status[4] = 0

print(status)
