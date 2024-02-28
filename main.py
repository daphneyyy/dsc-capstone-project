from feature_creation import *
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from model_predictions import run_model, evaluate_features
# from model import *

def main():
    # Create the features
    print("====== Creating features. ======")
    X, y = create_features()
    print(X.head())
    print("Number of features: ", X.shape[1] )
    print("====== Features created. ======")
    # Train the model
    run_model(X,y)
    # Test the model
    # test_model()
    
if __name__ == "__main__":
    main()