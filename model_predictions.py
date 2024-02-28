from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn import metrics



def evaluate_features(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7, stratify=y)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    #plot_importance(model)

    # Fit model using each importance as a threshold

    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:
        #select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        auc = metrics.roc_auc_score(y_test,  predictions)
        print("Thresh=%.8f, n=%d, Accuracy: %.2f%% , AUC: %.3f" % (thresh, select_X_train.shape[1], accuracy*100.0, auc))


#Thresh=0.00750565, n=38, Accuracy: 83.83% , AUC: 0.691 from evaluate_features(X,y)
def run_model(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7, stratify=y)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    X_train.to_csv('x_features.csv')

    #threshold selected from evaluate features function
    selection = SelectFromModel(model, threshold=0.00750565, prefit=True).set_output(transform = 'pandas')
    select_X_train = selection.transform(X_train)
    select_X_train.to_csv('x_selected_features.csv')
        # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
        # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    auc = metrics.roc_auc_score(y_test,  predictions)
    print(" n=%d, Accuracy: %.2f%% , AUC: %.3f" % ( select_X_train.shape[1], accuracy*100.0, auc))
    print(metrics.classification_report(y_test, predictions))

    return model