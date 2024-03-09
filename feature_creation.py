import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from IncomeEstimation.income_estimation import income_estimate
from income_features import inflow_features
from sklearn.model_selection import train_test_split

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


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
    
    X = cat_percentage.set_index('prism_consumer_id')
    y = (cons.sort_values(by='prism_consumer_id').reset_index(drop=True))['FPF_TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=7, stratify=y)
    
    model = LogisticRegression().fit(X_train, y_train)

    predicted_probabilities = model.predict_proba(X)
    probabilities_class_1 = predicted_probabilities[:, 1]
    
    probs_df = pd.DataFrame(probabilities_class_1, index=X.index, columns=['Predictions'])
    probs_df.reset_index(inplace=True)

    return probs_df, model

def cat_percent_testing(inflows, outflows, model):
    # total inflow amount by consumer and account type
    inflows_acc_amount = inflows.groupby(['prism_consumer_id'])['amount'].sum().reset_index()
    
    # total outflow amount by consumer, account type, and category
    outflows_cat_amount = outflows.groupby(['prism_consumer_id', 'category_description'])['amount'].sum().reset_index()
    
    # calculate percentage of spending by category for each consumer
    percentage_df = pd.merge(inflows_acc_amount, outflows_cat_amount, on=['prism_consumer_id'], suffixes=('_inflows', '_outflows'), how='left')
    percentage_df['category_description'].fillna('UNCATEGORIZED', inplace=True)
    percentage_df['amount_outflows'].fillna(0, inplace=True)
    percentage_df['percentage'] = percentage_df['amount_outflows'] / percentage_df['amount_inflows']
    cat_percentage = percentage_df.pivot_table(index='prism_consumer_id', columns='category_description', values='percentage', aggfunc='first', fill_value=0)
    cat_percentage.reset_index(inplace=True)
    
    X = cat_percentage.set_index('prism_consumer_id')

    
    predicted_probabilities = model.predict_proba(X)
    probabilities_class_1 = predicted_probabilities[:, 1]

    
    probs_df = pd.DataFrame(probabilities_class_1, index=X.index, columns=['Predictions'])
    probs_df.reset_index(inplace=True)

    return probs_df



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
    outflows_negate = outflow.copy()
    outflows_negate['amount'] *= -1
    

    all_transactions = pd.concat([inflow,outflows_negate])

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
    
    return std_balance, coefficients_std_flat

def moving_avg(std_balance):
    
    std_balance = std_balance.sort_values(by = ['prism_consumer_id', 'prism_account_id', 'month'], ascending = True)
    #function to define month span we can user for most users
    def explore_time_span(transactions):
        grouped_data = transactions.groupby(['prism_consumer_id', 'acct_type'])

       
        def calculate_month_span(group):
            min_date = group['posted_date'].min()  
            max_date = group['posted_date'].max()  

        
            span_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1

            return span_months

  
        month_span = grouped_data.apply(calculate_month_span)

        print(month_span.describe())
    
    ###moving average code
    def calculate_sma(data, window):
        return data.rolling(window=window, min_periods=1).mean()

    def calculate_ema(data, span):
        return data.ewm(span=span, adjust=False).mean()

    
    sma_window = 2  
    ema_span = 2   

    std_balance['sma'] = std_balance.groupby(['prism_consumer_id', 'acct_type'])['amount_standardized'].transform(lambda x: calculate_sma(x, sma_window))

    std_balance['ema'] = std_balance.groupby(['prism_consumer_id', 'acct_type'])['amount_standardized'].transform(lambda x: calculate_ema(x, ema_span))

    moving_averages = std_balance[['prism_consumer_id', 'prism_account_id','month','acct_type','sma', 'ema']]

    
    def convert_to_month_identifier(df):

        df = df.sort_values('month', ascending=False)

        # dictionary to map month dates to month identifiers
        month_identifier_mapping = {}
        month_count = 1
        for month in df['month'].unique():
            month_identifier_mapping[month] = f'month{month_count}'
            month_count += 1

        df['month'] = df['month'].map(month_identifier_mapping)
        return df

    # by user and account type conversion
    converted_dfs = []
    for (user_id, acct_type), user_acct_df in moving_averages.groupby(['prism_consumer_id', 'acct_type']):
        converted_df = convert_to_month_identifier(user_acct_df)
        converted_dfs.append(converted_df)
    result_df = pd.concat(converted_dfs)
    
    #from code below, we decided we would look at the most recent seven months of data for users
    #explore_time_span(std_balance)
    result_df_top_months = result_df[result_df['month'].isin(['month1','month2', 'month3', 'month4', 'month5', 'month6', 'month7'])]
    
    top_results = result_df_top_months[result_df_top_months['acct_type'].isin(['CHECKING', 'SAVINGS', 'CREDIT CARD'])]
    pivoted_top = top_results.pivot_table(index='prism_consumer_id', columns = ['acct_type','month'], values = ['sma', 'ema'])
    new_columns = [f"{col[1].lower()}_{col[2]}_{col[0].upper()}" for col in pivoted_top.columns]
    pivoted_top.columns = new_columns
    pivoted_top = pivoted_top.fillna(0)
    
    return pivoted_top


# Calculate trend in standardized differences in balance
def balance_diff_std(inflows, outflows):
    outflows_negate = outflows.copy()
    outflows_negate['amount'] *= -1

    all_transactions = pd.concat([inflows,outflows_negate])

    all_transactions['month'] = pd.to_datetime(all_transactions['posted_date']).dt.strftime('%Y-%m')

    # standardize amount
    transactions_by_month = all_transactions.groupby(['prism_consumer_id','acct_type','month'])['amount'].sum().reset_index()
    transactions_by_month['amount_standardized'] = transactions_by_month.groupby(['prism_consumer_id','acct_type'])['amount'].transform(std_amount)
    transactions_by_month.fillna(0, inplace=True)

    #we want to regress across time, so we calcualte time delta
    transactions_by_month['date'] = pd.to_datetime(transactions_by_month['month'])
    transactions_by_month['date_delta'] = (transactions_by_month['date'] - transactions_by_month.groupby(
        ['prism_consumer_id', 'acct_type']
        )['date'].transform('min')).dt.days

    #  find linear model coefficients of cumulative sum over time for each consumer and account type
    coefficients_std = transactions_by_month.groupby(['prism_consumer_id','acct_type']).apply(linear_model).to_frame().reset_index()
    coefficients_std.columns = ['prism_consumer_id','acct_type','coefficient']
    coefficients_std_flat = coefficients_std.pivot_table(index='prism_consumer_id', columns='acct_type', values='coefficient', fill_value=0)
    coefficients_std_flat.reset_index(inplace=True)
    
    return coefficients_std_flat

#helper function to differentiate columns from various variable creation
def rename_columns(df, suffix):
    df = df.set_index('prism_consumer_id')
    df = df.add_suffix(suffix)
    return df.reset_index()

# Create features
def create_features(cons, acct, inflows, outflows, trainBool = True, cat_income_model = None, cat_percent_model = None):

    if trainBool:
        income, cat_income_predictions, cat_income_model  = income_estimate(inflows, outflows, cons)

    else:
        income, cat_income_predictions  = income_estimate(inflows, outflows, cons, trainBool, cat_income_model)

    cat_income_predictions = rename_columns(cat_income_predictions, '_cat_income_proba')

    
    # Calculate percentage of spending by category for each consumer

    if trainBool:
        cat_percent_predictions, cat_percent_model = cat_percent(inflows, outflows, cons)
    else:
        cat_percent_predictions = cat_percent_testing(inflows, outflows, cat_percent_model)

    cat_percent_predictions = rename_columns(cat_percent_predictions, '_cat_proba')

    # Count of accounts by type for each consumer
    acct_count_flat = account_count(inflows)
    acct_count_flat = rename_columns(acct_count_flat, '_acct_count')
    

    # Standardize and calculate cumulative sum of inflows and outflows
    coefficients_std_flat = balance_diff_std(inflows, outflows)
    coefficients_std_flat = rename_columns(coefficients_std_flat, '_balance_std_diff_regress_coeff')

    inf_features = inflow_features(inflows, outflows, income)
    
    #new balance funcs
    balance_std_df, balance_std_coeff = balance_cumsum_std(inflows,outflows,acct)
    balance_std_coeff = rename_columns(balance_std_coeff, '_balance_std_regress_coeff')

    mvg_avgs = moving_avg(balance_std_df)

    # Merge all features
    cnt_and_perc = pd.merge(acct_count_flat, cat_percent_predictions, on='prism_consumer_id', how='outer')
    cnt_and_perc = cnt_and_perc.fillna(0)
    cnt_perc_coeff = pd.merge(cnt_and_perc, coefficients_std_flat, on='prism_consumer_id', how='outer')
    
    cnt_both_perc_coeff = pd.merge(cnt_perc_coeff, cat_income_predictions, on = 'prism_consumer_id', how = 'outer')

    cnt_percs_coeff_balance = pd.merge(cnt_both_perc_coeff, balance_std_coeff, on = 'prism_consumer_id', how = 'outer')

    with_mvg_avgs = pd.merge(mvg_avgs, cnt_percs_coeff_balance, on = 'prism_consumer_id', how = 'outer')
    all_features = pd.merge(with_mvg_avgs, inf_features, on = 'prism_consumer_id', how = 'outer')

    if trainBool:
        # Get target variable and drop unnecessary columns
        final_df = pd.merge(all_features, cons, on='prism_consumer_id')
        final_df.drop(columns=['APPROVED','evaluation_date'], inplace=True)
        final_df['FPF_TARGET'] = final_df['FPF_TARGET'].astype(int)
        final_df = final_df.set_index('prism_consumer_id')

        X = final_df.drop(columns=['FPF_TARGET'])
        y = final_df['FPF_TARGET']

        return X,y, cat_percent_model, cat_income_model
    
    else:
        
        return all_features