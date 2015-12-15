import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

DECISIONTREE_CLASSIFIER = 1
KNEIGHBORS_CLASSIFIER = 2
SVC_CLASSIFIER = 4

def prediction(csvPath, clfType, NUMTrain):
    
    #############################################################
    # exploring the data
    
    # Read student data
    student_data = pd.read_csv(csvPath)
    print "Student data read successfully!"
    # Note: The last column 'passed' is the target/label, all other are feature columns

    n_students = len(student_data)
    n_features = student_data.columns.size - 1
    n_passed = len(student_data[student_data['passed'] == 'yes'])
    n_failed = len(student_data[student_data['passed'] == 'no'])
    grad_rate = (n_passed / (n_students * 1.0))  * 100
    print "Total number of students: {}".format(n_students)
    print "Number of students who passed: {}".format(n_passed)
    print "Number of students who failed: {}".format(n_failed)
    print "Number of features: {}".format(n_features)
    print "Graduation rate of the class: {:.2f}%".format(grad_rate)

    #############################################################
    # Preparing the data
    # Extract feature (X) and target (y) columns
    feature_cols = list(student_data.columns[:-1])  # all columns but last are features
    target_col = student_data.columns[-1]  # last column is the target/label
    print "Feature column(s):-\n{}".format(feature_cols)
    print "Target column: {}".format(target_col)

    X_all = student_data[feature_cols]  # feature values for all students
    y_all = student_data[target_col]  # corresponding targets/labels
    print "\nFeature values:-"
    print X_all.head()  # print the first 5 rows

    X_all = preprocess_features(X_all)
    y_all = y_all.replace(['yes', 'no'], [1,0])
    print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))


    # First, decide how many training vs test samples you want
    num_all = student_data.shape[0]  # same as len(student_data)
    num_train = NUMTrain  # about 75% of the data
    num_test = num_all - 300

    # TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
    # Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = num_test, train_size = num_train, random_state=42)

    print "Training set: {} samples".format(X_train.shape[0])
    print "Test set: {} samples".format(X_test.shape[0])
    # Note: If you need a validation set, extract it from within training data

    # TODO: Choose a model, import it and instantiate an object
    mkscorer = make_scorer(f1_score,greater_is_better=True)
    cv = 5

    if(clfType & DECISIONTREE_CLASSIFIER):
        print "DecisionTree Classifier"
        clf = DecisionTreeClassifier()

        parameter = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
        
        clf = GridSearchCV(clf, parameter, scoring=mkscorer, cv=cv)
        
        train_predict(clf,X_train, y_train, X_test, y_test)

        


    if(clfType & KNEIGHBORS_CLASSIFIER):
        # TODO: Run the helper function above for desired subsets of training data
        # Note: Keep the test set constant
        # TODO: Train and predict using two other models
        print "KNeighbors Classifier"
        clf = KNeighborsClassifier()
        parameter = {'n_neighbors':[3,4,5,6,7,8], 'p':[1,2]}
        clf = GridSearchCV(clf, parameter, scoring=mkscorer, cv=cv)
        train_predict(clf,X_train, y_train, X_test, y_test)
    
    if(clfType & SVC_CLASSIFIER):
        print "SVM"

        clf = SVC()
        parameter = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1,2,3,4,5,6,7,8,9,10]}
        clf = GridSearchCV(clf, parameter, scoring=mkscorer, cv=cv)
        train_predict(clf,X_train, y_train, X_test, y_test)
        # TODO: Fine-tune your model and report the best F1 score


# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

# Train a model
def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()  
    
    print clf.best_params_  
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

# Predict on training set and compute F1 score
def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred)#, pos_label='yes')

# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))



def main():
    prediction("student-data.csv", KNEIGHBORS_CLASSIFIER | DECISIONTREE_CLASSIFIER | SVC_CLASSIFIER, 300)


if __name__ == "__main__":
    main()