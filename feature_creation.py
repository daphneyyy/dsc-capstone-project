import pandas as pd
from sklearn.linear_model import LinearRegression

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
    
    # Map account types to inflows and outflows
    acct_types = acct.set_index('prism_account_id')['account_type'].to_dict()
    inflows['acct_type'] = inflows['prism_account_id'].apply(lambda x: acct_types[x])
    outflows['acct_type'] = outflows['prism_account_id'].apply(lambda x: acct_types[x])
    
    return cons, acct, inflows, outflows

def percent_spent(inflows, outflows):
    # Total inflow amount by consumer and account type
    inflows_acc_amount = inflows.groupby(['prism_consumer_id', 'acct_type'])['amount'].sum().reset_index()
    
    # Total outflow amount by consumer, account type, and category
    outflows_cat_amount = outflows.groupby(['prism_consumer_id', 'acct_type', 'category_description'])['amount'].sum().reset_index()
    
    # Calculate percentage of spending by category for each consumer
    percentage_df = pd.merge(inflows_acc_amount, outflows_cat_amount, on=['prism_consumer_id', 'acct_type'], suffixes=('_inflows', '_outflows'))
    percentage_df['percentage'] = percentage_df['amount_outflows'] / percentage_df['amount_inflows']
    cat_percentage = percentage_df.pivot_table(index='prism_consumer_id', columns='category_description', values='percentage', aggfunc='first', fill_value=0)
    cat_percentage.reset_index(inplace=True)
    
    return cat_percentage

def account_count(inflows):
    # Count of accounts by type for each consumer
    acct_count = inflows.groupby(['prism_consumer_id', 'acct_type', 'prism_account_id']).size().groupby(['prism_consumer_id', 'acct_type']).count().reset_index(name='count')
    acct_count_flat = acct_count.pivot_table(index='prism_consumer_id', columns='acct_type', values='count', aggfunc='first', fill_value=0)
    acct_count_flat.reset_index(inplace=True)
    
    return acct_count_flat

# Helper method for cumsum_standardize
def linear_model(df):
    y = df[['cumulative_sum']].values
    X = df[['date_delta']].values
    return LinearRegression().fit(X, y).coef_[0][0]

# Helper method for cumsum_standardize
def std_amount(x):
    std_val = x.std()
    
    # Check for division by zero
    if std_val == 0:
        return 0  
    return (x - x.mean()) / std_val


# Calculate cumulative sum of standardized amount
def cumsum_standardize(inflows, outflows):
    # Merge inflows and outflows
    outflows['amount'] *= -1
    all_transactions = pd.concat([inflows,outflows])

    # Extract month from date
    all_transactions['month'] = pd.to_datetime(all_transactions['posted_date']).dt.strftime('%Y-%m')

    # Standardize amount
    transactions_by_month = all_transactions.groupby(['prism_consumer_id','acct_type','category_description','month'])['amount'].sum().reset_index()
    transactions_by_month['amount_standardized'] = transactions_by_month.groupby(['prism_consumer_id','acct_type','category_description'])['amount'].transform(std_amount)
    transactions_by_month.fillna(0, inplace=True)

    # Calculate cumulative sum of standardized amount
    transactions_std = transactions_by_month.groupby(['prism_consumer_id', 'acct_type', 'month'])[['amount_standardized']].sum().reset_index()
    transactions_std['cumulative_sum'] = transactions_std.groupby(['prism_consumer_id', 'acct_type'])['amount_standardized'].cumsum()

    # Calculate date delta
    cumsum_std = transactions_std.drop(columns=['amount_standardized'])
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

# Create features
def create_features():
    cons, acct, inflows, outflows = load_data()
    
    # Calculate percentage of spending by category for each consumer
    cat_percentage = percent_spent(inflows, outflows)

    # Count of accounts by type for each consumer
    acct_count_flat = account_count(inflows)

    # Standardize and calculate cumulative sum of inflows and outflows
    coefficients_std_flat = cumsum_standardize(inflows, outflows)

    # Merge all features
    cnt_and_perc = pd.merge(acct_count_flat, cat_percentage, on='prism_consumer_id', how='outer')
    cnt_and_perc = cnt_and_perc.fillna(0)
    cnt_perc_coeff = pd.merge(cnt_and_perc, coefficients_std_flat, on=['prism_consumer_id'], how='outer', suffixes=('_cnt', '_coeff'))
    
    # Get target variable and drop unnecessary columns
    final_df = pd.merge(cnt_perc_coeff, cons, on=['prism_consumer_id'])
    final_df.drop(columns=['APPROVED','evaluation_date'], inplace=True)
    final_df['FPF_TARGET'] = final_df['FPF_TARGET'].astype(int)

    X = final_df.drop(columns=['FPF_TARGET'])
    y = final_df['FPF_TARGET']
    
    return X, y