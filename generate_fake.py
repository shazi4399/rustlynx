import os
import sys
    



args = sys.argv[1:]

rows=args[0]
cols=args[1]
party=args[2]

rows = 8192
cols = 11

folds = 5

for k in range(folds):
    with open("/home/XT{}/data/Party{}_{}/{}X_train.csv".format(str(int(party) + 1), party, "fake", k + 1), 'w') as f:
        vals = ["0"] * cols
        for r in range(rows):
            f.write(",".join(vals) + "\n")

    with open("/home/XT{}/data/Party{}_{}/{}y_train.csv".format(str(int(party) + 1), party, "fake", k + 1), 'w') as f:
        vals = ["0"] * 2
        for r in range(rows):
            f.write(",".join(vals) + "\n")
