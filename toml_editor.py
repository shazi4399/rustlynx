import os
import sys

party = 0
toml_learn_path = "learning{}_temp.toml".format(party)
toml_inference_path = "inference{}_temp.toml".format(party)

args = sys.argv[1:]

rows=args[0]
attr=args[1]
trees=args[2]
depth=args[3]
feat=args[4]
fold=args[5]
party=args[6]

with open(toml_learn_path, 'w') as file:
    file.write("class_label_count = 2\n")
    file.write("attribute_count = %s\n".format(attr))
    file.write("instance_count = %s\n".format(rows))
    file.write("tree_count = %s\n".format(trees))
    file.write("max_depth = %s\n".format(depth))
    file.write("feature_count = %s\n".format(feat))
    file.write("epsilon = 0.05\n")
    file.write("data = data/Party%s_breast/%sX_train.csv\n".format(party, fold))
    file.write("classes = data/Party%s_breast/%sy_train.csv\n".format(party, fold))
    file.write("save_location = treedata/Party%s_trees.json".format(party))

with open(toml_inference_path, 'w') as file:

    file.write("class_label_count = 2\n")
    file.write("attribute_count = %s\n".format(attr))
    file.write("instance_count = %s\n".format(rows))
    file.write("max_depth = %s\n".format(depth))
    file.write("data = data/Party%s_breast/%sX_test.csv\n".format(party, fold))
    file.write("classes = data/Party%s_breast/%sy_test.csv\n".format(party, fold))
    file.write("save_location = treedata/Party%s_trees.json".format(party))


