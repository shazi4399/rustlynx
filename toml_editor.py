import os
import sys

party = 0
toml_learn_path = "cfg/ml/extratrees/learning{}.toml".format(party)
toml_inference_path = "cfg/ml/extratrees/inference{}.toml".format(party)

args = sys.argv[1:]

rows=args[0]
attr=args[1]
trees=args[2]
depth=args[3]
feat=args[4]
fold=args[5]
party=args[6]
key_word=args[7]

with open(toml_learn_path, 'w') as file:
    file.write("class_label_count = 2\n")
    file.write("attribute_count = {}\n".format(attr))
    file.write("instance_count = {}\n".format(rows))
    file.write("tree_count = {}\n".format(trees))
    file.write("max_depth = {}\n".format(depth))
    file.write("feature_count = {}\n".format(feat))
    file.write("epsilon = 0.05\n")
    file.write("data = \"data/Party{}_{}/{}X_train.csv\"\n".format(party, key_word, fold))
    file.write("classes = \"data/Party{}_{}/{}y_train.csv\"\n".format(party, key_word, fold))
    file.write("save_location = \"treedata/Party{}_trees.json\"".format(party))

with open(toml_inference_path, 'w') as file:

    file.write("class_label_count = 2\n")
    file.write("attribute_count = {}\n".format(attr))
    file.write("instance_count = {}\n".format(rows))
    file.write("max_depth = {}\n".format(depth))
    file.write("data = \"data/Party{}_{}/{}X_test.csv\"\n".format(party, key_word, fold))
    file.write("classes = \"data/Party{}_{}/{}y_test.csv\"\n".format(party, key_word, fold))
    file.write("save_location = \"treedata/Party{}_trees.json\"".format(party))


