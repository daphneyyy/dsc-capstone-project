import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from IncomeEstimation.income_processing import process_data
from IncomeEstimation.income_model import run_model


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


    cat_percentage = percentage_df.pivot(index='prism_consumer_id', columns='category_description', values='percentage').add_suffix('_income_percent')
    cat_percentage.reset_index(inplace=True)
    cat_percentage.fillna(0, inplace=True)

    X = cat_percentage.drop(columns=['prism_consumer_id'])
    y = (cons.sort_values(by='prism_consumer_id').reset_index(drop=True))['FPF_TARGET']
    coefficients_perc = LogisticRegression().fit(X, y).coef_[0]

    importance = np.abs(coefficients_perc)
    df_importance = pd.DataFrame({feature: importance_value for feature, importance_value in zip(X.columns, importance)}, index=[0]).transpose()
    df_importance.sort_values(0, ascending=False, inplace=True)
    df_importance.columns = ['importance']
    important_category = df_importance[df_importance['importance'] > 0.1].index.to_list()
    
    return cat_percentage[['prism_consumer_id'] + important_category]


def income_estimate(inflow, outflow, cons):
    inflow_clean, determined_transactions, undetermined_transactions = process_data(inflow)
    #complete_income = run_model(inflow_clean, determined_transactions, undetermined_transactions)
    complete_income = pd.read_csv('IncomeEstimation/income_estimates.csv')
    income_percentages = cat_percent_income(complete_income, inflow, outflow, cons)
    
    return  complete_income, income_percentages
