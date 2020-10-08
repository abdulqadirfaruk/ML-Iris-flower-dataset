from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# declare/name data headers
attributes = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=attributes)  # load data & specify custom header names
# dimension of dataset
print(dataset.shape)
# peek dataset
print(dataset.head(20))
# statistical summary
print(dataset.describe())
# class distribution - instances of each class
print(dataset.groupby('class').size())
# univariate plot to analyse each attribute - box and whisker plot
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()
dataset.hist()
pyplot.show()

# multvariate plots - analyze interactions betwn the attributes- scatter plot
scatter_matrix(dataset)
pyplot.show()

# creating a validation dataset-split data, 80%(train,evaluate & choose among models), 20% for validating the accuracy
array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)
# Spot Check Algorithms
models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC(gamma='auto'))]
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# compare the algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('ALGORITHM COMPARISON')
pyplot.show()

# make predictions on the dataset using the model with best accuracy
model = SVC(gamma='auto')
model.fit(x_train, y_train)  # fit linear model (training values, target iris-classes)
predictions = model.predict(x_test)  # predict iris-classes using iris values(x_test)

# EVALUATE PREDICTIONS
accuracy = accuracy_score(y_test, predictions) * 100  # by comparing true y_test and predicted y_test
print('Prediction accuracy: ' + str("%.4f" % accuracy) + '%')
acc = accuracy_score(y_test, predictions, normalize=False)
print('Number of correctly classified samples: ' + str(acc) + ' of ' + str(y_test.size))
print('Confusion matrix: \n' + str(confusion_matrix(y_test, predictions)))
print('Classification report: \n' + str(classification_report(y_test, predictions)))

# print predicted y_test values
print('Predicted iris-classes:')
for x in predictions:
    print(x)
