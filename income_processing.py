import re

'''
Function changes digits to X's to standardize memos that would otherwise hold the same information
Also standardizes text case

Parameter:
 df: intial inflow dataframe
'''
def remove_digits(df):
    df['memo'] = df['memo'].str.lower().apply(lambda x: re.sub(r'\b(\w*\d+\w*)\b', 'X', x))
    df['category_description'] = df['category_description'].str.lower()
    return df

'''
Function grabs transactions that have already been determined paycheck
Parameter:
 df: standardized inflow dataframe
'''
def determined_income(df):
    paycheck_inflow = df[df['category_description'].isin(['paycheck', 'paycheck_placeholder'])]
    paycheck_inflow['category_description'] = paycheck_inflow['category_description'].str.lower()
    ##age not required for this dataframe, dummy var placed instead
    paycheck_inflow['age'] = -1
    return paycheck_inflow

'''
Function grabs transactions that could possibly be income
Parameter:
 df: standardized inflow dataframe 
'''
def undetermined_income(df):
    rel_inflow = df[df['category_description'].isin(['deposit', 'external_transfer', 'investment_income', 'unemployment_benefits', 'miscellaneous'])]
    return rel_inflow
'''
Helper Function to find the transactions for a given user for a given category
Parameters:
  user: user of interest string 
  category: category of interest string
  df: undetermined transactions dataframe
'''
def user_transactions_by_category(user, category, df):
    trans = df[(df['prism_consumer_id'] == user) & (df['category_description'] == category)].sort_values(by='posted_date')
    return trans

'''
Helper Function to find the earliest transaction for a given user for a given category
to be used after user_transactions_by_category
Parameters:
 trans: dataframe of transactions given
'''
def earliest_by_category(trans):
    earliest= trans.iloc[0]['posted_date']
    return earliest


'''
#Function iterates over users in inflow dataframe of relevant categories
#for each user, we check each category and find the earliest transaction
# populates user dictionary with dictionary of earliest transaction by category
Parameter:
 df - standardized dataframe of undetermined transactions
'''

def populate_date_dictionary(df):
    earliestByUserByCategory  = dict()
    for user in df['prism_consumer_id'].unique():
        earliestByUserByCategory[user] = dict()
        for cat in ['deposit', 'external_transfer', 'investment_income', 'unemployment_benefits', 'miscellaneous']:
            curr_category_transactions = user_transactions_by_category(user, cat,df)
            if len(curr_category_transactions) == 0:
                continue
            else:
                earliestByUserByCategory[user][cat] = earliest_by_category(curr_category_transactions)
    return earliestByUserByCategory
            

'''
#Function calculates the age of a transaction by earlies transaction within same category
#Parameter:
# transac: singlular transaction of interest
# date_dictionary: dictionary with earliest transaction for a given category already populated
'''
def age(transac, date_dictionary):
    cat = transac['category_description']
    first_date = date_dictionary[transac['prism_consumer_id']][cat]
    curr = transac['posted_date']
    return (curr - first_date).days


'''
Function creates age column for each transaction based on the earliest transaction for that user for the same category
#Parameter:
    df: standardized dataframe of undetermined transactions
'''
def create_age_column(df, date_dictionary):
    df['age'] = df.apply(age, axis=1, args=(date_dictionary,))
    return df

##Final processing 
'''
Function applies all preprocessing
Parameters:
  df: initial dataframe of transaction data
'''
def process_data(df):
    df_copy = df.copy()
    standardized_df = remove_digits(df_copy)
    paycheck_trans = determined_income(standardized_df)
    undetermined_trans = undetermined_income(standardized_df)
    earliestByUserByCategory = populate_date_dictionary(undetermined_trans)
    undetermined_trans_with_age = create_age_column(undetermined_trans, earliestByUserByCategory)
    return standardized_df, paycheck_trans, undetermined_trans_with_age
    
    




    
