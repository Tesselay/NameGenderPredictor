import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import datetime
import sklearn.neighbors
import GridSearch
import sklearn.svm
from sklearn.preprocessing import LabelEncoder


def consonant_vowel_count(word):
    """
    Returns the number of consonants and vowels for a given word.
    :param word: String to be operated on
    :return: number_of_consonants: integer value of count of consonants
             number_of_vowels: integer value of count of vowels
    """

    vowel_list = list("aeiouAEIOU")
    consonant_list = list("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")

    number_of_consonants = sum(word.count(c) for c in consonant_list)
    number_of_vowels = sum(word.count(c) for c in vowel_list)

    return number_of_consonants, number_of_vowels


def plotter(dataframe, group_by, var):
    grouped = dataframe.groupby(group_by)

    fig, ax = plt.subplots()

    plt.bar(0, grouped.get_group(1)[var].mean(), 0.35, label=var, color='blue')
    plt.bar(0.35, grouped.get_group(0)[var].mean(), 0.35, label=var, color='lightblue')

    plt.title('Mean of {} for male and female'.format(var))
    plt.legend(['{} (male)'.format(var), '{} (female)'.format(var)])
    plt.savefig(("{}.png".format(datetime.datetime.now())).replace(':', '_'), bbox_inches="tight")
    plt.show()
    plt.close()


"""Import the edited dataset"""
df = pd.read_csv('E:/Schule/2. Halbjahr/Wahlpflicht/Projekt2/Datasets/Final/complete_edited.csv')


"""Since the models only work with integer values, I replaced every name with the sum of all ascii values in the name
   and the sex with 1 for 'boy' and 0 for 'girl'"""
df['Gender'] = [1 if i == 'Masculine' else 0 for i in df['Gender']]


"""Calculate all vowels and consonants for every name and adds it as a new column"""
consonants = []
vowels = []
for index, row in df.iterrows():
    con, vow = consonant_vowel_count(row['Name'])
    consonants.append(con)
    vowels.append(vow)


"""Calculate new features based on the Name"""
lengths = [len(i) for i in df['Name']]

vowels_list = list("aeiouAEIOU")

last_letter = [i[-1] for i in df['Name']]
is_vowel = [1 if i in vowels_list else 0 for i in last_letter]
last_letter = [ord(i[-1]) for i in df['Name']]

lab = LabelEncoder()
last_two_letters = [i[-2:] for i in df['Name']]
last_two_letters = lab.fit_transform(last_two_letters)


# The result per loop is taken *100 and converted to int to save calculation time later on, since ints are smaller than floats
vow_con_ratios = [int((consonants[i]/lengths[i])*100) for i in range(len(lengths))]


"""Create dataframes out of said features and combines them with the main dataframe"""
df_con = pd.DataFrame(consonants, columns=['Consonants'])
df_vow = pd.DataFrame(vowels, columns=['Vowels'])
df_len = pd.DataFrame(lengths, columns=['Length'])
df_rat = pd.DataFrame(vow_con_ratios, columns=['Con ratio'])
df_llv = pd.DataFrame(is_vowel, columns=['Last letter vowel'])
df_lle = pd.DataFrame(last_letter, columns=['Last letter'])
df_ltl = pd.DataFrame(last_two_letters, columns=['Last two letters'])

df = pd.concat([df, df_con, df_vow, df_len, df_rat, df_llv, df_lle, df_ltl], axis=1, sort=False, ignore_index=False, join='outer')


"""Commented part for plotting of df"""
# variable = 'Urban'
# plotter(df, ['Gender'], variable)
#
# variable = 'Classic'
# plotter(df, ['Gender'], variable)
#
# variable = 'Masculine'
# plotter(df, ['Gender'], variable)
#
# variable = 'Mature'
# plotter(df, ['Gender'], variable)
#
# variable = 'Refined'
# plotter(df, ['Gender'], variable)
#
# variable = 'Strong'
# plotter(df, ['Gender'], variable)
#
# variable = 'Consonants'
# plotter(df, ['Gender'], variable)
#
# variable = 'Vowels'
# plotter(df, ['Gender'], variable)
#
# variable = 'Length'
# plotter(df, ['Gender'], variable)
#
# variable = 'Last letter vowel'
# plotter(df, ['Gender'], variable)
#
# variable = 'Con ratio'
# plotter(df, ['Gender'], variable)


"""Separating input from output variables"""
x = df[['Last letter vowel', 'Last letter', 'Last two letters']]
y = df['Gender']


"""Splitting of test-/train data and testing with different models"""
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.33, train_size=0.67,
#                                                                             random_state=88, shuffle=True)
#
# knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='auto', p=5)
# knn.fit(x_train, y_train)
# cross_val_score = sklearn.model_selection.cross_val_score(knn, x_train, y_train, cv=100)
# print("KNN Accuray: {}%".format(float('%.3f' % (cross_val_score.mean()*100))))


"""GridSearch for different Parameters"""
svc_c = [0.01, 0.1, 1, 5, 10, 50, 100, 250, 500, 1000, 2500]
svc_gamma = [100, 50, 10, 5, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
svc_kernel = ['rbf', 'linear', 'sigmoid']

gs = GridSearch.GridSearch(grid_size=len(svc_c), cv=10, features=x, target=y, cross_validated=True)
grid = gs.svc_clf_gs(c_list=svc_c, gamma_list=svc_gamma, kernel=svc_kernel[0])
print(grid)
gs.heatmap(y_axis=svc_c, x_axis=svc_gamma, grid_values=grid, cmap_color='Wistia')


"""Chosen models for prediction, which is saved as pickles"""
# # Accuracy value: ~0.96
# # Kernel: rbf, gamma: 0.0001, C: 100
#
# classifier = sklearn.svm.SVC(C=100, gamma=0.0001, kernel='rbf').fit(x_train, y_train)
# pickle.dump(classifier, open('E:/Schule/2. Halbjahr/Wahlpflicht/Projekt2/Model/finalized_model.sav', 'wb'))

# # Accuracy value: ~0.81
# # Kernel: rbf, gamma: 0.001, C: 1000
#
# classifier = sklearn.svm.SVC(C=1000, gamma=0.001, kernel='rbf').fit(x_train, y_train)
# pickle.dump(classifier, open('E:/Schule/2. Halbjahr/Wahlpflicht/Projekt2/Model/finalized_model_without_rating.sav', 'wb'))



