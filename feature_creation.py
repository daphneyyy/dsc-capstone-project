import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from income_estimation import income_estimate

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_data():
    cons = pd.read_parquet('q2_consDF_final.pqt')
    acct = pd.read_parquet('q2_acctDF_final.pqt')
    inflows = pd.read_parquet('q2_inflows_final.pqt')
    outflows = pd.concat([
        pd.read_parquet('q2_outflows_1sthalf_final.pqt'),
        pd.read_parquet('q2_outflows_2ndhalf_final.pqt')
    ])
    inflows = inflows.rename(columns={"memo_clean": "memo"})
    outflows = outflows.rename(columns={"memo_clean": "memo"})

    # Map account types to inflows and outflows
    acct_types = acct.set_index('prism_account_id')['account_type'].to_dict()
    inflows['acct_type'] = inflows['prism_account_id'].apply(lambda x: acct_types[x])
    outflows['acct_type'] = outflows['prism_account_id'].apply(lambda x: acct_types[x])
    
    return cons, acct, inflows, outflows

def cat_percent(inflows, outflows, cons):
    # Total inflow amount by consumer and account type
    inflows_acc_amount = inflows.groupby(['prism_consumer_id'])['amount'].sum().reset_index()
    
    # Total outflow amount by consumer, account type, and category
    outflows_cat_amount = outflows.groupby(['prism_consumer_id', 'category_description'])['amount'].sum().reset_index()
    
    # Calculate percentage of spending by category for each consumer
    percentage_df = pd.merge(inflows_acc_amount, outflows_cat_amount, on=['prism_consumer_id'], suffixes=('_inflows', '_outflows'), how='left')
    percentage_df['category_description'].fillna('UNCATEGORIZED', inplace=True)
    percentage_df['amount_outflows'].fillna(0, inplace=True)
    percentage_df['percentage'] = percentage_df['amount_outflows'] / percentage_df['amount_inflows']
    cat_percentage = percentage_df.pivot_table(index='prism_consumer_id', columns='category_description', values='percentage', aggfunc='first', fill_value=0)
    cat_percentage.reset_index(inplace=True)
    
    X = cat_percentage.drop(columns=['prism_consumer_id'])
    y = (cons.sort_values(by='prism_consumer_id').reset_index(drop=True))['FPF_TARGET']
    coefficients_perc = LogisticRegression().fit(X, y).coef_[0]

    importance = np.abs(coefficients_perc)
    df_importance = pd.DataFrame({feature: importance_value for feature, importance_value in zip(X.columns, importance)}, index=[0]).transpose()
    df_importance.sort_values(0, ascending=False, inplace=True)
    df_importance.columns = ['importance']
    important_category = df_importance[df_importance['importance'] > 0.1].index.to_list()
    
    return cat_percentage[['prism_consumer_id'] + important_category], important_category


def account_count(inflows):
    # Count of accounts by type for each consumer
    acct_count = inflows.groupby(['prism_consumer_id', 'acct_type', 'prism_account_id']).size().groupby(['prism_consumer_id', 'acct_type']).count().reset_index(name='count')
    acct_count_flat = acct_count.pivot_table(index='prism_consumer_id', columns='acct_type', values='count', aggfunc='first', fill_value=0)
    acct_count_flat.reset_index(inplace=True)
    
    return acct_count_flat

# Helper method for cumsum_standardize
def linear_model(df):
    y = df[['amount_standardized']].values
    X = df[['date_delta']].values
    return LinearRegression().fit(X, y).coef_[0][0]

# Helper method for cumsum_standardize
def std_amount(x):
    std_val = x.std()
    
    # Check for division by zero
    if std_val == 0:
        return 0  
    return (x - x.mean()) / std_val

def balance_cumsum_std(inflow, outflow, acct):
    outflow['amount'] *= -1
    all_transactions = pd.concat([inflow,outflow])

    # Extract month from date
    all_transactions['month'] = pd.to_datetime(all_transactions['posted_date']).dt.strftime('%Y-%m')
    
    #cumulative sum by month and account ID
    def cumulative_sum(all_transactions):      
        transactions = all_transactions.groupby(['prism_consumer_id', 'prism_account_id', 'month'])[['amount']].sum().reset_index()
        transactions['cumulative_sum'] = transactions.groupby(['prism_consumer_id', 'prism_account_id'])['amount'].cumsum()
        return transactions
    
    transactions = cumulative_sum(all_transactions)
    
    #final balance from transactions only
    max_months = transactions.groupby(['prism_consumer_id', 'prism_account_id'])['month'].max()
    calculated_balance = transactions[transactions.apply(lambda row: (row['prism_consumer_id'], row['prism_account_id']) in max_months.index and row['month'] == max_months.loc[(row['prism_consumer_id'], row['prism_account_id'])], axis=1)]
    
    #starting accounts balance
    modded_balance = pd.merge(calculated_balance, acct, on = ['prism_consumer_id', 'prism_account_id'] )
    modded_balance['mod_balance'] = modded_balance['balance'] - modded_balance['cumulative_sum']
    
    modded_balance = modded_balance[['prism_consumer_id', 'prism_account_id', 'balance_date', 'mod_balance']]

    #recalculate cumulative sum with starting balance
    min_months = transactions.groupby(['prism_consumer_id', 'prism_account_id'])['month'].min().reset_index()
    modded_balance = pd.merge(modded_balance.drop(columns=['balance_date']), min_months, on = ['prism_consumer_id', 'prism_account_id'])
    temp_transactions = pd.concat([modded_balance, all_transactions])
    complete_balance = cumulative_sum(temp_transactions)
    
    acct_types = acct.set_index('prism_account_id')['account_type'].to_dict()
    complete_balance['acct_type'] = complete_balance['prism_account_id'].apply(lambda x: acct_types[x])
    
        #standardize balances by acct_type per user
    complete_balance['amount_standardized'] = complete_balance.groupby(['prism_consumer_id','acct_type'])['cumulative_sum'].transform(std_amount)
    complete_balance.fillna(0, inplace=True)
    std_balance = complete_balance[['prism_consumer_id', 'prism_account_id','month', 'acct_type','cumulative_sum', 'amount_standardized']]

    # Calculate date delta
    cumsum_std = std_balance#.drop(columns=['cumulative_sum'])
    cumsum_std['date'] = pd.to_datetime(cumsum_std['month'])
    cumsum_std['date_delta'] = (cumsum_std['date'] - cumsum_std.groupby(
        ['prism_consumer_id', 'acct_type']
        )['date'].transform('min')).dt.days

    # Calculate linear model coefficients of cumulative sum over time for each consumer and account type
    coefficients_std = cumsum_std.groupby(['prism_consumer_id','acct_type']).apply(linear_model).to_frame().reset_index()
    coefficients_std.columns = ['prism_consumer_id','acct_type','coefficient']
    coefficients_std_flat = coefficients_std.pivot_table(index='prism_consumer_id', columns='acct_type', values='coefficient', aggfunc='first', fill_value=0)
    coefficients_std_flat.reset_index(inplace=True)
    
    return coefficients_std_flat

# Calculate cumulative sum of standardized amount
def balance_diff_std(inflows, outflows):
    # Merge inflows and outflows
    outflows['amount'] *= -1
    all_transactions = pd.concat([inflows,outflows])

    # Extract month from date
    all_transactions['month'] = pd.to_datetime(all_transactions['posted_date']).dt.strftime('%Y-%m')

    # Standardize amount
    transactions_by_month = all_transactions.groupby(['prism_consumer_id','acct_type','month'])['amount'].sum().reset_index()
    transactions_by_month['amount_standardized'] = transactions_by_month.groupby(['prism_consumer_id','acct_type'])['amount'].transform(std_amount)
    transactions_by_month.fillna(0, inplace=True)

    # Calculate date delta
    transactions_by_month['date'] = pd.to_datetime(transactions_by_month['month'])
    transactions_by_month['date_delta'] = (transactions_by_month['date'] - transactions_by_month.groupby(
        ['prism_consumer_id', 'acct_type']
        )['date'].transform('min')).dt.days

    # Calculate linear model coefficients of cumulative sum over time for each consumer and account type

    coefficients_std = transactions_by_month.groupby(['prism_consumer_id','acct_type']).apply(linear_model).to_frame().reset_index()
    coefficients_std.columns = ['prism_consumer_id','acct_type','coefficient']
    coefficients_std_flat = coefficients_std.pivot_table(index='prism_consumer_id', columns='acct_type', values='coefficient', fill_value=0)
    coefficients_std_flat.reset_index(inplace=True)
    
    return coefficients_std_flat

# Create features
def create_features():
    cons, acct, inflows, outflows = load_data()
    
    # Calculate percentage of spending by category for each consumer
    # Also gives the important outflow categories
    cat_percentage, important_categories = cat_percent(inflows, outflows, cons)

    # Count of accounts by type for each consumer
    acct_count_flat = account_count(inflows)

    # Standardize and calculate cumulative sum of inflows and outflows
    coefficients_std_flat = balance_diff_std(inflows, outflows)

    income, income_percentage = income_estimate(inflows, outflows, cons)

    # Merge all features
    cnt_and_perc = pd.merge(acct_count_flat, cat_percentage, on='prism_consumer_id', how='outer')
    cnt_and_perc = cnt_and_perc.fillna(0)
    cnt_perc_coeff = pd.merge(cnt_and_perc, coefficients_std_flat, on=['prism_consumer_id'], how='outer', suffixes=('_cnt', '_coeff'))
    
    ##new income percentages
    cnt_both_perc_coeff = pd.merge(cnt_perc_coeff, income_percentage, on = 'prism_consumer_id', how = 'outer')

    #new balance funcs
    balance_std_coeff = balance_cumsum_std(inflows,outflows,acct)

    cnt_percs_coeff_balance = pd.merge(cnt_both_perc_coeff, balance_std_coeff, on = 'prism_consumer_id', how = 'outer')


    # Get target variable and drop unnecessary columns
    final_df = pd.merge(cnt_percs_coeff_balance, cons, on=['prism_consumer_id'])
    final_df.drop(columns=['APPROVED','evaluation_date'], inplace=True)
    final_df['FPF_TARGET'] = final_df['FPF_TARGET'].astype(int)

    X = final_df.drop(columns=['FPF_TARGET'])
    y = final_df['FPF_TARGET']
    
    return X, y