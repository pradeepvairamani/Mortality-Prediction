from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *


def get_data_from_svmlight():
    data = load_svmlight_file("SVMfile/part-00000")
    return data[0], data[1]


def logistic_regression(X_train, Y_train, X_test):
    logistic = LogisticRegression()
    logistic.fit(X_train, Y_train)
    return logistic.predict(X_test)


def svm(X_train, Y_train, X_test):
    supportvectormachine = LinearSVC()
    supportvectormachine.fit(X_train, Y_train)
    return supportvectormachine.predict(X_test)


def decision_tree(X_train, Y_train, X_test):
    decisiontree = DecisionTreeClassifier(max_depth=5)
    decisiontree.fit(X_train, Y_train)
    return decisiontree.predict(X_test)


def classification_metrics(Y_pred, Y_true):
    return accuracy_score(Y_true, Y_pred), roc_auc_score(Y_true, Y_pred), precision_score(Y_true, Y_pred), recall_score(Y_true, Y_pred), f1_score(Y_true, Y_pred)


def display_metrics(classifierName, Y_pred, Y_true):
    print ("Classifier: " + classifierName)
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred, Y_true)
    print ("Accuracy: " + str((acc)))
    print ("Precision: " + str((precision)))
    print ("Recall: " + str((recall)))
    print ("F1-score: " + str((f1score)))


def main():
    X_train, Y_train = get_data_from_svmlight()
    X_test, Y_test = get_data_from_svmlight()

    display_metrics("Logistic Regression", logistic_regression(X_train, Y_train, X_test), Y_test)
    display_metrics("SVM", svm(X_train, Y_train, X_test), Y_test)
    display_metrics("Decision Tree", decision_tree(X_train, Y_train, X_test), Y_test)

if __name__ == "__main__":
    main()
