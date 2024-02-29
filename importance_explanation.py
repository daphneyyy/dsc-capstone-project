from model_predictions import exclude_columns_with_substrings
import shap

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
    holdout = selection.transform(X_new)
    holdout.columns=X_new.columns[selection.get_support()]

    explainer = shap.TreeExplainer(selection_model)
    shap_values = explainer.shap_values(holdout)

    top_reasons_per_consumer = get_top_reasons(shap_values, holdout.columns)
    return top_reasons_per_consumer

