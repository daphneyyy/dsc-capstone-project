import pandas as pd

def inflow_over_outlow_features (inflows, outflows):

    outflows_consumer_amount = outflows.groupby(["prism_consumer_id"])['amount'].sum().reset_index()
    inflows_cat_amount = inflows.groupby(["prism_consumer_id", "category_description"])['amount'].sum().reset_index()

    # create percentage column
    percent_out_df = pd.merge(outflows_consumer_amount, inflows_cat_amount, on=["prism_consumer_id"], suffixes=('_total_outflows', '_inflow_per_cat'), how='right')
    percent_out_df['category_description'].fillna('UNCATEGORIZED', inplace=True)
    percent_out_df['amount_inflow_per_cat'].fillna(0, inplace=True)
    percent_out_df['amount_total_outflows'].fillna(0, inplace=True)
    percent_out_df['percentage'] = percent_out_df['amount_inflow_per_cat'] / percent_out_df['amount_total_outflows']
    percent_out_df['percentage'] = percent_out_df['percentage'].where(percent_out_df['amount_total_outflows'] != 0, 0)

    # using a pivot table to format output
    cat_percent_outflow = percent_out_df.pivot_table(index='prism_consumer_id', columns='category_description', values='percentage', fill_value=0).add_suffix('_inflow_over_outflow')
    cat_percent_outflow.reset_index(inplace=True)
    cat_percent_outflow.drop('UNEMPLOYMENT_BENEFITS_inflow_over_outflow',axis=1,inplace=True)

    return cat_percent_outflow

def inflow_over_inflow_features (inflows):
    
    # inflow categories and denominator processed
    inflows_consumer_amount = inflows.groupby(["prism_consumer_id"])['amount'].sum().reset_index()
    inflows_cat_amount = inflows.groupby(["prism_consumer_id", "category_description"])['amount'].sum().reset_index()

    # create percentage column
    percent_in_df = pd.merge(inflows_consumer_amount, inflows_cat_amount, on=["prism_consumer_id"], suffixes=('_total_inflows', '_inflow_per_cat'))
    percent_in_df['percentage'] = percent_in_df['amount_inflow_per_cat'] / percent_in_df['amount_total_inflows']

    # using a pivot table to format output
    cat_percent_inflow = percent_in_df.pivot_table(index='prism_consumer_id', columns='category_description', values='percentage', fill_value=0).add_suffix('_inflow_over_inflow')
    cat_percent_inflow.reset_index(inplace=True)
    cat_percent_inflow.drop('UNEMPLOYMENT_BENEFITS_inflow_over_inflow',axis=1,inplace=True)

    return cat_percent_inflow

def inflow_over_income_features (inflows, income):

    # inflow categories and denominator processed
    inflows_consumer_amount = inflows.groupby(["prism_consumer_id"])['amount'].sum().reset_index()
    income_final = pd.merge(income,inflows_consumer_amount,on=["prism_consumer_id"], suffixes=('_income', '_inflow'))
    income_final.loc[income_final['amount_income'] == 0, 'amount_income'] = income_final.loc[income_final['amount_income'] == 0, 'amount_inflow']
    income_final.drop('amount_inflow',axis=1,inplace = True)
    inflows_cat_amount = inflows.groupby(["prism_consumer_id", "category_description"])['amount'].sum().reset_index()
    
    # create percentage column
    percent_income_df = pd.merge(income_final, inflows_cat_amount, on=["prism_consumer_id"], suffixes=('_income', '_inflow_per_cat'))
    percent_income_df['percentage'] = percent_income_df['amount'] / percent_income_df['amount_income']
    
    # using a pivot table to format output
    cat_percent_income = percent_income_df.pivot_table(index='prism_consumer_id', columns='category_description', values='percentage', fill_value=0).add_suffix('_inflow_over_income')
    cat_percent_income.reset_index(inplace=True)
    cat_percent_income.drop('UNEMPLOYMENT_BENEFITS_inflow_over_income',axis=1,inplace=True)

    return cat_percent_income

def outflow_over_income_features (inflows, outflows, income):

    # inflow categories and denominator processed
    inflows_consumer_amount = inflows.groupby(["prism_consumer_id"])['amount'].sum().reset_index()
    income_final = pd.merge(income,inflows_consumer_amount,on=["prism_consumer_id"], suffixes=('_income', '_inflow'))
    income_final.loc[income_final['amount_income'] == 0, 'amount_income'] = income_final.loc[income_final['amount_income'] == 0, 'amount_inflow']
    income_final.drop('amount_inflow',axis=1,inplace = True)
    outflows_cat_amount = outflows.groupby(["prism_consumer_id", "category_description"])['amount'].sum().reset_index()

    # create percentage column
    percent_income_df = pd.merge(income_final, outflows_cat_amount, on=["prism_consumer_id"], suffixes=('_income', '_outflow_per_cat'), how='left')
    percent_income_df['category_description'].fillna('UNCATEGORIZED', inplace=True)
    percent_income_df['amount'].fillna(0, inplace=True)
    percent_income_df['percentage'] = percent_income_df['amount'] / percent_income_df['amount_income']

    # using a pivot table to format output
    cat_percent_income = percent_income_df.pivot_table(index='prism_consumer_id', columns='category_description', values='percentage', fill_value=0).add_suffix('_outflow_over_income')
    cat_percent_income.reset_index(inplace=True)

    return cat_percent_income

def inflow_features(inflows,outflows,income):
    
    iof = inflow_over_outlow_features(inflows, outflows) 
    
    iif = inflow_over_inflow_features(inflows)
    
    icf = inflow_over_income_features(inflows,income)
    
    ocf = outflow_over_income_features(inflows,outflows,income)
    
    # merge all df by prism_consumer_id
    features = pd.merge(iof, iif, on="prism_consumer_id")
    features = pd.merge(features, icf, on="prism_consumer_id")
    features = pd.merge(features, ocf, on="prism_consumer_id")
    
    return features
    
    
    