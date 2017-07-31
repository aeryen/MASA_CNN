import numpy as np
from sklearn import metrics

for i in range(5):
    f = open("./data_123g/test_123g_aspect_" + str(i) + ".txt", mode="r")
    lines = f.readlines()
    lines = [l.split(" ", 1)[0] for l in lines]
    lines = [float(n) for n in lines]
    tru = np.array(lines)

    f = open("./data_123g/test_" + str(i) + ".output", mode="r")
    lines = f.readlines()
    lines = [float(n) for n in lines]
    pred = np.array(lines)

    mse_result = metrics.mean_squared_error(tru, pred)
    print(str(i) + "\tMSE")
    print(mse_result)
    r2_result = metrics.r2_score(tru, pred)
    print(str(i) + "\tR2")
    print(r2_result)

    print("  ")
