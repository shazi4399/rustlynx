import os
import sys
import traceback
import pandas as pd
from sklearn.model_selection import KFold
from random import randint
from math import floor


def generate_shares(out_directory, filename, ringsize, decimalAcc, code_word):

    clear_path = out_directory + "/clear_" + code_word

    infile = open(os.path.join(clear_path, filename), 'r')

    dataset = []

    for line in infile:
        dataset.append(list(map(float, line.split(','))))

    infile.close()

    party0_path = out_directory + "/Party0_" + code_word
    party1_path = out_directory + "/Party1_" + code_word

    if not os.path.exists(party0_path):
        os.makedirs(party0_path)

    if not os.path.exists(party1_path):
        os.makedirs(party1_path)

    outfile0 = open(os.path.join(party0_path, filename[:-4] + ".csv"), 'w')
    outfile1 = open(os.path.join(party1_path, filename[:-4] + ".csv"), 'w')

    ringModulus = 2 ** ringsize
    shift = 2 ** decimalAcc

    for line in dataset:

        val = line[0]
        if val < 0:
            val = ringModulus - floor((-1 * val) * shift)
        else:
            val = floor(val * shift)

        z0 = randint(0, ringModulus - 1)
        z1 = (int(val) - z0) % ringModulus

        outfile0.write(str(z0))
        outfile1.write(str(z1))

        for i in range(1, len(line)):

            val = line[i]
            if val < 0:
                val = ringModulus - floor((-1 * val) * shift)
            else:
                val = floor(val * shift)

            z0 = randint(0, ringModulus - 1)
            z1 = (int(val) - z0) % ringModulus

            outfile0.write(',' + str(z0))
            outfile1.write(',' + str(z1))

        outfile0.write('\n')
        outfile1.write('\n')

    outfile0.close()
    outfile1.close()


# Secret shares every data set in in_path between two parties into the outpath, with ringsize and decimalAcc in mind.
def format_data(in_path, filename, out_path, class_col_index, class_0_name, k_fold, code_word, scale, truncate):

    epsilon = 2**-40

    dataframe = pd.read_csv(os.path.join(in_path, filename))

    classifications = dataframe.get(dataframe.columns[class_col_index])
    dataframe = dataframe.drop(dataframe.columns[class_col_index], axis=1)

    class_0_col = []
    class_1_col = []

    # OHE classification column
    for val in classifications:
        is_0 = int((val == class_0_name))
        is_1 = 1 - is_0

        class_0_col.append(is_0)
        class_1_col.append(is_1)

    features = []
    labels = [class_0_col, class_1_col]

    # remove columns that are constant, or nearly constant (max - min < 2^-(40))
    for i in range(len(dataframe.iloc[0])):
        column = dataframe.iloc[:, i].tolist()

        column = [float(el) for el in column]

        # If we truncate, check if we want to scale the value too.
        # If we don't wish to truncate, check if we still want to scale
        if truncate:
            if scale > 1:
                column = [int(el * scale) for el in column]
            else:
                column = [int(el) for el in column]
        else:
            if scale > 1:
                column = [int(el) for el in column]

        max_value = max(column)
        min_value = min(column)

        if max_value - min_value > epsilon:
            features.append(column)

    features_dataframe = pd.DataFrame(data=features).transpose()
    labels_dataframe = pd.DataFrame(data=labels).transpose()

    kf = KFold(n_splits=k_fold, shuffle=True)

    path = out_path + "/clear_" + code_word

    if not os.path.exists(path):
        os.makedirs(path)

    k = 0

    for train_index, test_index in kf.split(features_dataframe):
        k += 1

        X_train, X_test = features_dataframe.loc[train_index].values.tolist(), features_dataframe.loc[test_index].values.tolist()
        y_train, y_test = labels_dataframe.loc[train_index].values.tolist(), labels_dataframe.loc[test_index].values.tolist()

        train_X_file = open(os.path.join(path, str(k) + "X_" + "train" + '.csv'),'w')
        test_X_file  = open(os.path.join(path, str(k) + "X_" + "test" + '.csv'), 'w')
        train_y_file = open(os.path.join(path, str(k) + "y_" + "train" + '.csv'), 'w')
        test_y_file  = open(os.path.join(path, str(k) + "y_" + "test" + '.csv'), 'w')

        # Change the data to strings
        X_train = [[str(c) for c in d] for d in X_train]
        y_train = [[str(c) for c in d] for d in y_train]
        X_test = [[str(c) for c in d] for d in X_test]
        y_test = [[str(c) for c in d] for d in y_test]

        for i in range(len(X_train)):
            train_X_file.write(",".join(X_train[i]) + "\n")
            train_y_file.write(",".join(y_train[i]) + "\n")

        for i in range(len(X_test)):
            test_X_file.write(",".join(X_test[i]) + "\n")
            test_y_file.write(",".join(y_test[i]) + "\n")

        train_X_file.close()
        test_X_file.close()
        train_y_file.close()
        test_y_file.close()

def print_error():
    print("\nThere was an error with the command line arguments. Make sure you have the following: \n\n"
          "data_directory: The directory with the csv file containing the data you wish to secret share \n"
          "out_directory: The directory we output secret shared data to \n"
          "code_word: A unique word that describes the data set such as lower_back_pain + \n"
          "ringsize: The size of the ring of our additive sharing scheme \n"
          "decimalAcc: How many bits we reserve for decimal precision \n"
          "class_col_index: The column that contains the class labels \n"
          "class_0_name: The name of the 0'th class label \n"
          "k_fold: How many folds the data should be broken up into for k way cross validation (set to 1 if you don't "
          "want to break up the data) \n"
          "scale: A constant to multiply data by. This could be useful if the data values are really small (e.g. > "
          "0.0001, <= 0) \n"
          "truncate: Should be 1 if you want to truncate the data, 0 otherwise\n"
          "already_formatted: Should likely be 0 for most cases. Let it be 1 if the in-the-clear data is already in "
          "the correct form (has gone throughh the format_data function in this script)\n\n")


data_directory = None
out_directory = None
code_word = None
ringsize = None
decimalAcc = None
class_col_index = None
class_0_name = None
k_fold = None
scale = None
truncate = None
already_formatted = None

try:
    args = sys.argv[1:]

    data_directory = args[0]
    out_directory = args[1]
    code_word = args[2]
    ringsize = int(args[3])
    decimalAcc = int(args[4])
    class_col_index = int(args[5])
    class_0_name = args[6]
    k_fold = int(args[7])
    scale = int(args[8])
    truncate = int(args[9]) # 0 for false, 1 for true
    already_formatted = int(args[10])  # 0 for false, 1 for true

except:
    print_error()
    traceback.print_exc()


# If data is already formatted, just secret share it. Otherwise, format it correctly, then secret share
if already_formatted:
    for filename_formatted in os.listdir(out_directory + "/clear_" + code_word):
        generate_shares(out_directory, filename_formatted, int(ringsize), int(decimalAcc), code_word)
else:
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            # creates k-folded datasets in the output directory that are ready to be split between two parties
            format_data(data_directory, filename, out_directory, int(class_col_index), class_0_name, int(k_fold), code_word, scale, bool(truncate))
            for filename_formatted in os.listdir(out_directory + "/clear_" + code_word):
                generate_shares(out_directory, filename_formatted, int(ringsize), int(decimalAcc), code_word)
