from income_processing import process_data
from income_model import run_model

import warnings
warnings.filterwarnings("ignore")

def income_estimate(inflow, outflow, cons):
    inflow_clean, determined_transactions, undetermined_transactions = process_data(inflow)
    complete_income = run_model(inflow_clean, determined_transactions, undetermined_transactions)
    
    return complete_income
