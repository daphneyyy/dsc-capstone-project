import pandas as pd
import numpy as np
import statistics as stat
# import matplotlib.pyplot as plt

'''
#Function calculates average recurrence of transactions metric

#Parameter:
# trans: transaction dataframe for all transactions grouped together (same min)
'''
def mod_avg_transaction_recurrence(trans):
    ages = trans.sort_values(by = 'age')['age']
    differences = np.diff(ages)
    return sum(differences)/len(differences)

'''
#Function returns all groups of values with values within 10% of each other or median of group 
#Parameter:
# lst: list of amounts for transactions with same memo but not similar enough amounts 
'''
def group_values(lst):
    groups = []
    
    for value in lst:
        added_to_existing_group = False

        for group in groups:
            median = stat.mean(group)
            # Check if the current value is within 10% of any value in the group or the median
            if any(0.9 * x <= value <= 1.1 * x for x in group) or (median and 0.9 * median <= value <= 1.1 * median):
                group.append(value)
                added_to_existing_group = True
                break

        if not added_to_existing_group:
            # Start a new group with the current value
            groups.append([value])

    # Include groups with at least one element
    groups = [group for group in groups if len(group) >= 1]

    return groups



'''
helper function for determine_income, determines whether transactions are recurrent enough

Parameters:
 trans: dataframe of associated transactions'''
def regular_time(trans):
    metric = mod_avg_transaction_recurrence(trans)
        
    #highly frequent transactions should have a higher # of repetitions to ensure large span of time (more likely income than random deposits)
    #sparsely frequent transactios can have fewer repetitions but still represent a large span of time

    if (metric >=7) & (metric <=16):
        recurr = (len(trans) >= 4) #placeholder (equivalent of min 28 days span of time)
            
    elif (metric <= (31 + 3)) & (metric > 16):
        recurr = (len(trans) >= 2) #placeholder (equivalent of min 33 days of time)
    else:
        recurr = False

    return recurr

'''
Function that creates an empty column to populate in model with booelan, True if transaction should be considered income, False otherwise
Parameter:
 df: standardized dataframe with undetermined transactions
'''
def create_valid_income_flag(df):
    df['recurring_flag'] = None
    return df

'''
Function will determine which undetermined transactions should be considered income
Parameter:
 userID: user of interest to determine income
 orig_df: undetermined transactions
'''
def determine_income(userID, orig_df):
    df = orig_df
    user_trans = df[df['prism_consumer_id']==userID]
    #memo dependent repetition
    repeat_memos = user_trans.groupby("memo")[['age']].count()
    memo_list = repeat_memos[repeat_memos['age']>1].index
    

    for memo in memo_list:
        repeat = user_trans[user_trans['memo'] == memo]
        amounts = list(repeat['amount'])
        
        #check whether all values are close enough (ie confident same transactions)
        med_val = stat.median(amounts)
        
        
        #if amounts close enough, check recurrence is often enough to be the same source
        if all(abs(med_val - x) / x <= 0.1 for x in amounts): 
            
            recurr = regular_time(repeat)

            # Update the 'recurring_flag' column in the original dataframe (rel_inflow) based on the condition
            df.loc[(df['prism_consumer_id'] == userID) & (df['memo'] == memo), 'recurring_flag'] = recurr


        #if amount isnt close enough, investigate each subgroup (grouped by amount +/- 10%)
        else:
            transaction_groups = group_values(amounts)
            for group in transaction_groups:
                if len(group)>=2:
                    repeat_subset = repeat[repeat['amount'].isin(list(set(group)))]
                    indxs = repeat_subset.index
                    recurr = regular_time(repeat_subset)
                    
                    # Update the 'recurring_flag' column in the original dataframe (rel_inflow) based on the condition
                    df.loc[(df['prism_consumer_id'] == userID) & (df['memo'] == memo) & df.index.isin(indxs),'recurring_flag'] = recurr
        
    
    ##same amount, regardless of memo
    #for each category in transactions
    #groupby amount 
    #check recurrence 
    for cat in ['deposit', 'external_transfer', 'investment_income', 'unemployment_benefits', 'miscellaneous']:
        user_cat_trans = user_trans[user_trans['category_description'] == cat] 
        repeat_amount = user_cat_trans.groupby("amount")[['age']].count()
        amount_list = repeat_amount[repeat_amount['age']>1].index
        
        for val in amount_list:
            repeat = user_trans[user_trans['amount'] == val]
            
            recurr = regular_time(repeat)

            condition = (df['prism_consumer_id'] == userID) & (df['amount'] == val) & (df['recurring_flag'].isin([False, None]))

            df.loc[condition, 'recurring_flag'] = recurr

    return df


'''
Function applies all model steps to determine whether transactions are income or not
Parameters:
  df: undetermined dataframe with empty recurring flag column
'''
def model_by_user(df):
    new_df = df
    for u in df.groupby('prism_consumer_id').count().index:
        new_df= determine_income(u, new_df)
    return new_df
    
'''
Function to make dataframe that includes all transactions that should be counted as income
'''
def complete_income_estimate(determined_df, model_trained_undetermined_df):
    determined_df['recurring_flag'] = True
    flagged_true= model_trained_undetermined_df[model_trained_undetermined_df['recurring_flag'] == True]
    income = pd.concat([determined_df, flagged_true])
    results = income.groupby('prism_consumer_id')['amount'].sum()
    return results

'''
Function to populate incomes for users that weren't seen
    df: inflow dataframe
    income_estimate: series with income estimates for users with qualifying transactions

'''
def all_users_income(df, income_estimate):
    missing_set = set(df['prism_consumer_id'].unique()) - set(income_estimate.index)
    for ID in missing_set:
        income_estimate[ID] = 0

    return income_estimate

'''
Function estimates the yearly income for each user

Parameters:
    df: inflow dataframe with all transaction types
    income_estimate: series with all estimated incomes
'''
def estimate_yearly_income(df, income_estimate):
    def time_frame(userID, df):
        trans = df[df['prism_consumer_id']==userID].sort_values(by='posted_date')['posted_date']
        if len(trans) ==1:
            return 0
        else:
            first_date = trans.iloc[0]
            last_date = trans.iloc[-1]
            return (last_date - first_date).days
    
    annual_income = pd.Series(dtype = float)
    for userID in income_estimate.index:
        
        income = income_estimate.loc[userID]
        
        if income != 0:
        
            time_span = time_frame(userID, df)
            if time_span != 0:
                estimated_annual_income = (income / time_span) * 365
            else:
                estimated_annual_income = income
            
            annual_income.loc[userID] = estimated_annual_income
        else:
            annual_income.loc[userID] = 0
          
    return annual_income


# '''
# Function creates figures 
# Parameter:
#     estimated_incomes: series of estimated incomes for users
# '''
# def create_figure(estimated_incomes, title):

#     plt.hist(estimated_incomes, bins='auto', alpha=0.7, color='blue', edgecolor='black')

#     plt.xlabel('Estimated Income')
#     plt.ylabel('Frequency')
#     plt.title(title)
#     plt.savefig(f"{title}.png")


#Run entire model processing

'''
Function runs all steps to create and run model, returns series of estimated income for users
'''
def run_model(inflow,  determined_transactions, undetermined_transactions):
    with_flag_df = create_valid_income_flag(undetermined_transactions)
    model_trained_undetermined_transactions= model_by_user(with_flag_df)
    income = complete_income_estimate(determined_transactions, model_trained_undetermined_transactions)
    complete_income = all_users_income(inflow, income)
    # complete_income.to_csv('income_estimates.csv')

    return complete_income