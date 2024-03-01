import pandas as pd

def load_training_data():
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

def load_holdout_data():

    cons = pd.read_parquet('q2_consDF_HOLDOUT_notags_final.pqt')
    acct = pd.read_parquet('q2_acctDF_HOLDOUT_final.pqt')


    inflows = pd.read_parquet('q2_inflows_HOLDOUT_final.pqt')
    outflows = pd.concat([
        pd.read_parquet('q2_outflows_HOLDOUT_1sthalf_final.pqt'),
        pd.read_parquet('q2_outflows_HOLDOUT_2ndhalf_final.pqt')
    ])
    inflows = inflows.rename(columns={"memo_clean": "memo"})
    outflows = outflows.rename(columns={"memo_clean": "memo"})

    # Map account types to inflows and outflows
    acct_types = acct.set_index('prism_account_id')['account_type'].to_dict()
    inflows['acct_type'] = inflows['prism_account_id'].apply(lambda x: acct_types[x])
    outflows['acct_type'] = outflows['prism_account_id'].apply(lambda x: acct_types[x])
    
    return cons, acct, inflows, outflows