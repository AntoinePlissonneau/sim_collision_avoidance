from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd



def learn_tree(X, Y, max_depth=5, criterion="gini", weights=[1,1,1], max_features=None, max_leaf_nodes=None, spliter='best'):

    clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth, criterion=criterion,
                                max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                splitter=spliter, class_weight={0 : weights[0], 1 : weights[1]})

    clf.fit(X, Y)
    return clf



def test_tree(clf,test_X, test_Y):
    """
    test a decision tree on a test dataset.
    clf : a decision tree
    test_X : decision tree's test input
    test_Y : labels

    return :
    """

    predictions = clf.predict(test_X)

    True_0 =0
    True_1 = 0
    True_2 = 0
    one_become_zero = 0
    one_become_two = 0
    zero_become_one = 0
    zero_become_two = 0
    two_become_zero = 0
    two_become_one = 0
    error_impact_secu = 0
    error_impact_perf = 0

    for pred, true in zip(predictions, test_Y):
        if pred == 0 and true == 0:
            True_0 += 1
        elif pred == 1 and true == 1:
            True_1 += 1
        elif pred == 2 and true == 2:
            True_2 += 1

        elif pred == 0 and true == 1:
            one_become_zero += 1
        elif pred == 2 and true == 1:
            one_become_two += 1
        elif pred == 1 and true == 0:
            zero_become_one += 1
        elif pred == 2 and true == 0:
            zero_become_two += 1
        elif pred == 0 and true == 2:
            two_become_zero += 1
        elif pred == 1 and true == 2:
            two_become_one += 1

        if pred < true :
            error_impact_perf += 1
        if pred > true:
            error_impact_secu += 1


    good = True_0 + True_1 + True_2

    return [good, error_impact_secu, error_impact_perf, True_0, True_1, True_2,
            one_become_zero, one_become_two, zero_become_one, zero_become_two,
            two_become_zero, two_become_one]







csv_file = "dataset.csv"

with open(csv_file) as f:
    nbr_line = sum(1 for line in f)

data = pd.read_csv(csv_file, nrows=int(nbr_line*0.8))
data = data.dropna()
Y = data.iloc[:, -1].to_numpy(dtype=np.int8)
X = data.iloc[:, :-1].to_numpy(dtype=np.int8)

data = None

data = pd.read_csv(csv_file, skiprows = range(1, int(nbr_line*0.8)), nrows=int(nbr_line*0.2))
data = data.dropna()
test_Y = data.iloc[:, -1].to_numpy(dtype=np.int8)

sata = None
test_X = data.iloc[:, :-1].to_numpy()

i = 0
results = []


clf = learn_tree(X, Y, 5, "entropy", [2, 3], 0.6)

test_tree(clf, test_X, test_Y)
