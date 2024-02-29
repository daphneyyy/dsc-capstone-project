from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn import metrics
from sklearn.model_selection import cross_validate
import shap

#helper to avoid using variables that could discriminate against certain groups
def exclude_columns_with_substrings(df, substrings):
        # Initialize a list to store column names to exclude
        columns_to_exclude = []
        
        # Iterate through the column names
        for col in df.columns:
            # Check if any substring is present in the column name
            if any(substring in col for substring in substrings):
                columns_to_exclude.append(col)
        
        # Exclude columns containing the specified substrings
        df_filtered = df.drop(columns=columns_to_exclude)
        
        return df_filtered

def evaluate_features(X, y):

    
    X_new = exclude_columns_with_substrings(X, ['HEALTHCARE_MEDICAL', 'OTHER_BENEFITS', 'CHILD_DEPENDENTS' ])


    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.33, random_state=7, stratify=y)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    # Get feature importances
    thresholds = sort(model.feature_importances_)
    
    best_auc = 0
    best_acc = 0
    best_thresh = 0
    
    #first 40 are all trivially small thresholds that do not change ths features used
    for thresh in thresholds[40:]:
        # Select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        
        # Create a new XGBClassifier with selected features
        selection_model = XGBClassifier()
        
        # Define evaluation metrics
        scoring = {'auc': 'roc_auc', 'accuracy': 'accuracy'}
        
        # Evaluate the model using cross-validation
        scores = cross_validate(selection_model, select_X_train, y_train, cv=5, scoring=scoring)
        
        # Compute mean AUC and accuracy
        mean_auc = scores['test_auc'].mean()
        mean_acc = scores['test_accuracy'].mean()
        
        # Check if this threshold gives a higher AUC
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_acc = mean_acc
            best_thresh = thresh
    
    # Print the best threshold, AUC, and accuracy
    print("Best Threshold=%.8f, Best AUC=%.3f, Best Accuracy=%.2f%%" % (best_thresh, best_auc, best_acc*100))
    return best_thresh

def get_top_reasons(shap_values, feature_names, num_reasons=3):
    top_reasons = []
    for sv in shap_values:
        enumerated_list = list(enumerate(sv))
        sorted_values_indices = sorted(enumerated_list, key=lambda x: x[1], reverse=True)
        top_num_reasons = sorted_values_indices[:num_reasons]
        reasons = [(feature_names[idx], value) for idx, value in top_num_reasons]
        top_reasons.append(reasons)
    return top_reasons
    
def shap_importance(X, selection, selection_model):
    X_new = exclude_columns_with_substrings(X, ['HEALTHCARE_MEDICAL', 'OTHER_BENEFITS', 'CHILD_DEPENDENTS' ])
    selected_X = selection.transform(X_new)
    selected_X.columns=X_new.columns[selection.get_support()]

    explainer = shap.TreeExplainer(selection_model)
    shap_values = explainer.shap_values(selected_X)

    top_reasons_per_consumer = get_top_reasons(shap_values, selected_X.columns)
    return top_reasons_per_consumer

def run_model(X,y, best_thresh):

    X_new = exclude_columns_with_substrings(X, ['HEALTHCARE_MEDICAL', 'OTHER_BENEFITS', 'CHILD_DEPENDENTS' ])

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.33, random_state=7, stratify=y)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    X_train.to_csv('x_features.csv')

    #threshold selected from evaluate features function
    selection = SelectFromModel(model, threshold=best_thresh, prefit=True).set_output(transform = 'pandas')
    select_X_train = selection.transform(X_train)
    select_X_train.columns=X_train.columns[selection.get_support()] 
    select_X_train.to_csv('x_selected_features.csv')
        # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
        # eval model
    select_X_test = selection.transform(X_test)
    select_X_test.columns=X_test.columns[selection.get_support()]
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    auc = metrics.roc_auc_score(y_test,  predictions)
    print(" n=%d, Accuracy: %.2f%% , AUC: %.3f" % ( select_X_train.shape[1], accuracy*100.0, auc))
    print(metrics.classification_report(y_test, predictions))
    
    reasons = shap_importance(X_test, selection, selection_model)
    
    return reasons, selection_model