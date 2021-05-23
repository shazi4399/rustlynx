import os
import sys

args = sys.argv[1:]

test_size=args[0]
party=args[1]

toml_learn_path = "cfg/ml/test_DT_protocols/learning{}.toml".format(party)

with open(toml_learn_path, 'w') as file:
    file.write("test_size = {}\n".format(test_size))
