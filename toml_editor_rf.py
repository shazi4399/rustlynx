import os
import sys

args = sys.argv[1:]

rows=args[0]
attr=args[1]
trees=args[2]
depth=args[3]
feat=args[4]
fold=args[5]
party=args[6]
key_word=args[7]
epsilon=args[8]
attr_value_count=args[9]
instance_selected_count=args[10]

toml_learn_path = "cfg/ml/randomforest/learning{}.toml".format(party)
toml_inference_path = "cfg/ml/randomforest/inference{}.toml".format(party)

delim = "---------------------------------------"

printout = "\n{}\n\nlearn path: {} \n inference path: {} \n\n rows = {}, attr = {} for dataset with codename {}\n{}\n".format(
    delim,toml_learn_path,toml_inference_path,rows,attr,key_word,delim)

print(printout)


with open(toml_learn_path, 'w') as file:
    file.write("class_label_count = 2\n")
    file.write("attribute_count = {}\n".format(attr))
    file.write("instance_count = {}\n".format(rows))
    file.write("tree_count = {}\n".format(trees))
    file.write("max_depth = {}\n".format(depth))
    file.write("feature_count = {}\n".format(feat))
    file.write("epsilon = {}\n".format(epsilon))
    file.write("attr_value_count = {}\n".format(attr_value_count))
    file.write("instance_selected_count = {}\n".format(instance_selected_count))
    file.write("data = \"data/Party{}_{}/{}X_train.csv\"\n".format(party, key_word, fold))
    file.write("classes = \"data/Party{}_{}/{}y_train.csv\"\n".format(party, key_word, fold))
    file.write("save_location = \"treedata/Party{}_trees.json\"".format(party))
    

with open(toml_inference_path, 'w') as file:
    file.write("class_label_count = 2\n")
    file.write("attribute_count = {}\n".format(attr))
    file.write("instance_count = {}\n".format(rows))
    file.write("max_depth = {}\n".format(depth))
    file.write("bin_count = {}\n".format(attr_value_count))
    file.write("data = \"data/Party{}_{}/{}X_test.csv\"\n".format(party, key_word, fold))
    file.write("classes = \"data/Party{}_{}/{}y_test.csv\"\n".format(party, key_word, fold))
    file.write("save_location = \"treedata/Party{}_trees.json\"".format(party))

