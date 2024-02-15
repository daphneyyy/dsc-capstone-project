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
    print(model.feature_importances_)
    plot_importance(model)
    # Fit model using each importance as a threshold

    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:
    # select features using threshold
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
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%% , AUC: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0, auc*100.0))

