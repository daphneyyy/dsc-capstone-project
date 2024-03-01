import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from IncomeEstimation.income_processing import process_data
from IncomeEstimation.income_model import run_model
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")

def cat_percent_income(incomes,inflow, outflows, cons):
        
    # Total outflow amount by consumer, account type, and category
    outflows_cat_amount = outflows.groupby(['prism_consumer_id', 'category_description'])['amount'].sum().reset_index()
  


    #income or inflow
    inflows_consumer_amount = inflow.groupby(["prism_consumer_id"])['amount'].sum().reset_index()
    income_final = pd.merge(incomes,inflows_consumer_amount,on=["prism_consumer_id"], suffixes=('_income', '_inflow'))
    income_final.loc[income_final['amount_income'] == 0, 'amount_income'] = income_final.loc[income_final['amount_income'] == 0, 'amount_inflow']
    income_final.drop('amount_inflow',axis=1,inplace = True)
    

    
    # Calculate percentage of spending by category for each consumer
    percentage_df = pd.merge(income_final, outflows_cat_amount, on=['prism_consumer_id'], suffixes=('_income', '_outflows'), how='left')
    percentage_df['category_description'].fillna('UNCATEGORIZED', inplace=True)
    percentage_df['percentage'] = percentage_df['amount'] / percentage_df['amount_income']


    cat_percentage = percentage_df.pivot(index='prism_consumer_id', columns='category_description', values='percentage')
    cat_percentage.reset_index(inplace=True)
    cat_percentage.fillna(0, inplace=True)
    

        
    X = cat_percentage.set_index('prism_consumer_id')
    y = (cons.sort_values(by='prism_consumer_id').reset_index(drop=True))['FPF_TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=7, stratify=y)
    
    print(X.columns)
    
    model = LogisticRegression().fit(X_train, y_train)
    predictions = model.predict(X)
    
    predictions_df = pd.DataFrame(predictions, index=X.index, columns=['Predictions'])
    predictions_df.reset_index(inplace=True)

    return predictions_df, model

def cat_percent_income_testing(incomes, inflow, outflows, model):

    outflows_cat_amount = outflows.groupby(['prism_consumer_id', 'category_description'])['amount'].sum().reset_index()

    #income or inflow
    inflows_consumer_amount = inflow.groupby(["prism_consumer_id"])['amount'].sum().reset_index()
    income_final = pd.merge(incomes,inflows_consumer_amount,on=["prism_consumer_id"], suffixes=('_income', '_inflow'))
    income_final.loc[income_final['amount_income'] == 0, 'amount_income'] = income_final.loc[income_final['amount_income'] == 0, 'amount_inflow']
    income_final.drop('amount_inflow',axis=1,inplace = True)
    
 
    # Calculate percentage of spending by category for each consumer
    percentage_df = pd.merge(income_final, outflows_cat_amount, on=['prism_consumer_id'], suffixes=('_income', '_outflows'), how='left')
    percentage_df['category_description'].fillna('UNCATEGORIZED', inplace=True)
    percentage_df['percentage'] = percentage_df['amount'] / percentage_df['amount_income']


    cat_percentage = percentage_df.pivot(index='prism_consumer_id', columns='category_description', values='percentage')
    cat_percentage.reset_index(inplace=True)
    cat_percentage.fillna(0, inplace=True)
        
    X = cat_percentage.set_index('prism_consumer_id')

    print(X.columns)

    predictions = model.predict(X)
    
    predictions_df = pd.DataFrame(predictions, index=X.index, columns=['Predictions'])
    predictions_df.reset_index(inplace=True)
    
    return predictions_df

def income_estimate(inflow, outflow, cons, trainBool = True, model = None):
    #inflow_clean, determined_transactions, undetermined_transactions = process_data(inflow)
    #complete_income = run_model(inflow_clean, determined_transactions, undetermined_transactions)

    if trainBool:
        complete_income = pd.read_csv('IncomeEstimation/income_estimates.csv')
        predictions, model = cat_percent_income(complete_income, inflow, outflow, cons)
        return complete_income, predictions, model
    else:
        complete_income = pd.read_csv('IncomeEstimation/income_holdout_estimates.csv')
        predictions = cat_percent_income_testing(complete_income, inflow, outflow, model)
    
        return  complete_income, predictions