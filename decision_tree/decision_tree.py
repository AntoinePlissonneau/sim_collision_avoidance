from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import json
from sklearn import tree
from sklearn.tree import export_graphviz


class DT:

    def __init__(self, max_depth=5, criterion="gini", weights=[1,1,1], max_features=None, max_leaf_nodes=None, spliter='best'):
        self.max_depth =max_depth
        self.criterion =criterion
        self.weigths = {0 : weights[0], 1 : weights[1], 2 : weights[2]}
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.spliter = spliter

    def load_config(self, json_file):
        f = open(json_file)
        data = json.load(f)

        self.max_depth = data['max_depth']
        self.criterion = data['criterion']
        weights = data['weigths']
        self.weigths = {0 : weights[0], 1 : weights[1], 2 : weights[2]}
        self.spliter = data['spliter']

        if data['max_features'] == "None":
            self.max_features = None
        else :
            self.max_features = data['max_features']

        if data['max_leaf_nodes'] == "None":
            self.max_leaf_nodes = None
        else :
            self.max_leaf_nodes = data['max_leaf_nodes']

        self.load_data(data["dataset"])


    def load_data(self, csv_file):
        data = pd.read_csv(csv_file)
        data = data.dropna()
        self.Y = data.iloc[:, -1].to_numpy()
        self.X = data.iloc[:, :-1].to_numpy()


    def learn_tree(self, X=None, Y=None):
        if X is None:
            X = self.X
            Y = self.Y

        self.clf = DecisionTreeClassifier(random_state=0, max_depth=self.max_depth, criterion=self.criterion,
                                    max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes,
                                    splitter=self.spliter, class_weight=self.weigths)

        self.clf.fit(X, Y)


    def print_tree(self, features_name):
        tree.plot_tree(self.clf, feature_names=feature_name)
        export_graphviz(self.clf, "output_tree.dot", feature_names = feature_name, class_names = ['deceleration', 'do nothing', 'accelerate'])


    def test_tree(self, test_X, test_Y):
        """
        test a decision tree on a test dataset.
        clf : a decision tree
        test_X : decision tree's test input
        test_Y : labels

        return :
        """

        predictions = self.clf.predict(test_X)

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


    def compute_action(self, obs, train_coord, train_speed):
        min_action = 2

        for i in range(3):
            #print(i)
            action = self.clf.predict([[obs.coord[i][1], obs.coord[i][0] - float(train_coord[0]), train_speed]])
            min_action = min(min_action, action)

        return min_action




if __name__ == '__main__':

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


    dt = DT()

    dt.learn_tree(X, Y)
    dt.test_tree(test_X, test_Y)
