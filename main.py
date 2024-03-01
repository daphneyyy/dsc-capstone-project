from load_data import *
from feature_creation import *
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from model_predictions import *
# from model import *

def main():
    # Training the model


    print("====== Creating training features. ======")

    consTrain, acctTrain, inflowsTrain, outflowsTrain = load_training_data()

    XTrain, yTrain, cat_percent_model, cat_income_model = create_features(consTrain, acctTrain, inflowsTrain, outflowsTrain)

    print("Number of features: ", XTrain.shape[1] )
    print("====== Features created. ======")


    print("====== Evaluating training features. ======")
    best_thresh = evaluate_features(XTrain,yTrain)

    selection_model, selection = train_model(XTrain,yTrain, best_thresh)
    print("====== Model training complete ======")

    # Run the model
    print("====== Running the model ======")
    consTest, acctTest, inflowsTest, outflowsTest = load_holdout_data()
    XTest = create_features(consTest, acctTest, inflowsTest, outflowsTest, False, cat_percent_model, cat_income_model)
    predictions, reasons = run_model(selection_model , selection, XTest)
    print("====== Predictions complete ======")
    
if __name__ == "__main__":
    main()