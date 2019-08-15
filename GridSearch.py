import matplotlib.pyplot as plt
import numpy as np
import datetime
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score


class GridSearch:
    """
    Class that calculates accuracy values for provided models and data and creates a 2d list, that can be displayed as an
    heatmap. Currently only uses XGBoost and SVC Classifier but over time more will be added. Does not have the capability
    of choosing which parameters are to be compared, but uses the most significant ones for the corresponding models.

    """

    def __init__(self, grid_size, cv, features, target, cross_validated):
        """
        :param grid_size: x*x size of the grid
        :param cv: amount of cross validations to be calculated
        :param features: parameters for training
        :param target: value to be predicted
        :param cross_validated: cross validation activated or not
        """
        self.grid_size = grid_size
        self.cv = cv
        self.features = features
        self.target = target
        self.cross_validated = cross_validated
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features, self.target, test_size=0.4, random_state=114)
        self.model = None           # name of the model used
        self.algorithm = None           # algorithm/kernel used for the model
        self.axises = None          # axis designators

    def svc_clf_gs(self, c_list, gamma_list, kernel):
        """
        Creates list of trained svc classifiers and returns 2d list of accuracy values, created by
        acc_gridsearch function.

        :param c_list: List of C-parameter values
        :param gamma_list: List of gamma-parameter values
        :param kernel: Algorithm/kernel used for training
        :return: 2d list of accuracy values
        """

        print("svc clf gs")
        self.model = "svc classifier"
        self.algorithm = kernel
        self.axises = ['gamma', 'C']

        clf_list = []
        for i in range(self.grid_size):
            print("i: {}".format(i))
            temp_list = []
            for j in range(self.grid_size):
                print("j: {}".format(j))
                try:
                    clf = svm.SVC(kernel=kernel, C=c_list[i], gamma=gamma_list[j]).fit(self.x_train, self.y_train)
                    temp_list.append(clf)
                except IndexError:          # To prevent different sized parameter-lists breaking the program.
                    pass
            clf_list.insert(0, temp_list)

        return self.acc_gridsearch(clf_list, self.x_test, self.y_test)

    # def knn_clf_gs(self, n_neighbors, weights, algorithm, leaf_size, metric, p):
    #
    #     print("knn")
    #     self.model = "knn classifier"
    #
    #     knn_list = []
    #     for i in range(self.grid_size):
    #         temp_list = []
    #         for j in range(self.grid_size):
    #             try:
    #                 knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors[i], weights=weights, algorithm=algorithm, leaf_size=leaf_size,
    #                                                      metric=metric, p=p[i]).fit(self.x_train, self.y_train)
    #                 temp_list.append(knn)
    #             except IndexError:
    #                 pass
    #         knn_list.insert(0, temp_list)
    #
    #     return self.acc_gridsearch(knn_list, self.x_test, self.y_test)

    def xgboost_clf_gs(self, lr_list, gamma_list, booster='gbtree'):
        """
        Creates list of trained xgboost classifiers and returns 2d list of accuracy values, created by
        acc_gridsearch function.

        :param booster: Algorithm/kernel used for the model
        :param lr_list: List of learning rate values
        :param gamma_list: List of gamma-parameter values
        :return: 2d list of accuracy values
        """

        self.model = "xgboost classifier"
        self.algorithm = booster

        clf_list = []
        for i in range(self.grid_size):
            temp_list = []
            for j in range(self.grid_size):
                try:
                    clf = XGBClassifier(booster=booster, learning_rate=lr_list[i], gamma=gamma_list[j]).fit(self.x_train, self.y_train)
                    temp_list.append(clf)
                except IndexError:
                    pass
            clf_list.insert(0, temp_list)

        return self.acc_gridsearch(clf_list, self.x_test, self.y_test)

    def acc_gridsearch(self, clf_list, x_test, y_test):
        """
        Creates and returns 2d list of accuracy values for provided models.

        :param clf_list: List of trained models   
        :param x_test: Feature testing values
        :param y_test: Target testing values
        :return: 2d np array of accuracy scores
        """

        print('acc gridsearch')

        acc_scores = []
        for i in range(self.grid_size):
            print("i: {}".format(i))
            temp_list = []
            for j in range(self.grid_size):
                print("j: {}".format(j))
                try:
                    if self.cross_validated:
                        cross_score = cross_val_score(clf_list[i][j], x_test, y_test, cv=self.cv).mean()
                        temp_list.append(np.around(cross_score, 2))
                    else:
                        accuracy = clf_list[i][j].score(x_test, y_test)
                        temp_list.append(np.around(accuracy, 2))
                except IndexError:
                    pass
            acc_scores.insert(0, temp_list)

        return np.array(acc_scores)

    def heatmap(self, y_axis, x_axis, grid_values, cmap_color='Wistia'):
        """
        Function that creates heatmap of gridsearch and saves it as a file.

        :param y_axis: Corresponding values for y-axis
        :param x_axis: Corresponding values for x-axis
        :param grid_values: 2d list of accuracy values to be displayed
        :param cmap_color: color scheme of heatmap
        """

        fig, ax = plt.subplots()
        im = ax.imshow(grid_values)
        im.set_cmap(cmap_color)

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy", rotation="-90", va="bottom")

        ax.set_xticks(np.arange(len(x_axis)))
        ax.set_yticks(np.arange(len(y_axis)))

        ax.set_xticklabels(x_axis)
        ax.set_yticklabels(y_axis)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(y_axis)):
            for j in range(len(x_axis)):
                try:
                    ax.text(j, i, grid_values[i, j], ha="center", va="center", color="black")
                except IndexError:
                    pass

        ax.set_title("Accuracy of {}\n(x-axis: {}; y-axis: {}; kernel: {})".format(self.model, self.axises[0], self.axises[1], self.algorithm))
        fig.tight_layout()
        plt.savefig(("{}.png".format(datetime.datetime.now())).replace(':', '_'), bbox_inches="tight")
        plt.show()
