import pandas as pd
import numpy as np
from scipy.stats import skewnorm, beta
import random
from datetime import datetime, timedelta

# ==================== CONFIGURATION ====================
np.random.seed(42)
random.seed(42)

NUM_SAMPLES = 50000  # 50,000 rows
OUTPUT_FILE = "synthetic_credit_dataset_80features.csv"

# ==================== HELPER FUNCTIONS ====================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_skewed_normal(mean, std, skew, size, low=None, high=None):
    """Generate skewed normal distribution with bounds"""
    a = skew
    data = skewnorm.rvs(a, loc=mean, scale=std, size=size)
    if low is not None:
        data = np.maximum(data, low)
    if high is not None:
        data = np.minimum(data, high)
    return data

def generate_beta(alpha, beta_param, size, low=0, high=1):
    """Generate beta distribution scaled to [low, high]"""
    data = np.random.beta(alpha, beta_param, size)
    return low + (high - low) * data

# ==================== GENERATE BASE DEMOGRAPHICS ====================
print("Generating base demographics...")

# Age: Normal distribution, clip between 21-75
age = np.random.normal(35, 10, NUM_SAMPLES)
age = np.clip(age, 21, 65).astype(int)

# Gender
gender_choices = ['Male', 'Female', 'Other']
gender_probs = [0.55, 0.44, 0.01]
gender = np.random.choice(gender_choices, NUM_SAMPLES, p=gender_probs)

# Marital Status (correlated with age)
marital_status = []
for a in age:
    if a < 25:
        probs = [0.2, 0.75, 0.04, 0.01]  # Married, Single, Divorced, Widowed
    elif a < 35:
        probs = [0.5, 0.4, 0.08, 0.02]
    elif a < 50:
        probs = [0.7, 0.1, 0.15, 0.05]
    else:
        probs = [0.65, 0.05, 0.15, 0.15]
    marital_status.append(np.random.choice(['Married', 'Single', 'Divorced', 'Widowed'], p=probs))
marital_status = np.array(marital_status)

# Dependents (correlated with marital status and age)
dependents = []
for i in range(NUM_SAMPLES):
    if marital_status[i] == 'Married':
        if age[i] < 30:
            probs = [0.3, 0.4, 0.25, 0.05]
        elif age[i] < 40:
            probs = [0.1, 0.2, 0.5, 0.2]
        else:
            probs = [0.2, 0.3, 0.3, 0.2]
    else:
        probs = [0.8, 0.15, 0.04, 0.01]
    dependents.append(np.random.choice([0, 1, 2, 3], p=probs))
dependents = np.array(dependents)

# Education
education_choices = ['Under Graduate', 'Graduate', 'Post Graduate', 'PhD']
education_probs = [0.3, 0.45, 0.2, 0.05]
education = np.random.choice(education_choices, NUM_SAMPLES, p=education_probs)

# ==================== EMPLOYMENT ====================
print("Generating employment data...")

# Employment Type (correlated with age)
employment_type = []
for a in age:
    if a < 25:
        probs = [0.5, 0.1, 0.3, 0.08, 0.02]  # Salaried, SelfEmp, Gig, Unemp, Retired
    elif a < 35:
        probs = [0.7, 0.2, 0.08, 0.02, 0.0]
    elif a < 55:
        probs = [0.65, 0.25, 0.05, 0.03, 0.02]
    else:
        probs = [0.3, 0.2, 0.0, 0.05, 0.45]
    employment_type.append(np.random.choice(
        ['Salaried', 'SelfEmployed', 'Gig', 'Unemployed', 'Retired'], 
        p=probs
    ))
employment_type = np.array(employment_type)

# Years in current job (0-40, correlated with age and employment type)
years_in_current_job = []
for i in range(NUM_SAMPLES):
    if employment_type[i] in ['Unemployed', 'Retired']:
        years_in_current_job.append(0)
    else:
        # Max possible: age - 18 - education years
        max_years = age[i] - 18 - (4 if education[i] == 'Under Graduate' else 
                                   5 if education[i] == 'Graduate' else
                                   7 if education[i] == 'Post Graduate' else 10)
        max_years = max(1, max_years)
        years = np.random.exponential(5)
        years = min(int(years), max_years)
        years_in_current_job.append(max(1, years))
years_in_current_job = np.array(years_in_current_job)

# Income verification status
income_verification_status = np.random.choice(
    ['Verified', 'Pending', 'Not Verified'], 
    NUM_SAMPLES, 
    p=[0.7, 0.2, 0.1]
)

# ==================== INCOME & FINANCIALS ====================
print("Generating income and financial data...")

# Monthly income (log-normal)
monthly_income = np.random.lognormal(mean=10.7, sigma=0.8, size=NUM_SAMPLES)  # ~45k median
monthly_income = np.clip(monthly_income, 10000, 500000).astype(int)

# Adjust income by employment type
for i in range(NUM_SAMPLES):
    if employment_type[i] == 'SelfEmployed':
        monthly_income[i] = int(monthly_income[i] * np.random.uniform(0.8, 2.5))
    elif employment_type[i] == 'Gig':
        monthly_income[i] = int(monthly_income[i] * np.random.uniform(0.5, 1.2))
    elif employment_type[i] == 'Unemployed':
        monthly_income[i] = np.random.randint(0, 15000)
    elif employment_type[i] == 'Retired':
        monthly_income[i] = int(monthly_income[i] * np.random.uniform(0.4, 0.8))

# Co-applicant income (40% have co-applicant)
has_coapplicant = np.random.random(NUM_SAMPLES) < 0.4
co_applicant_income = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if has_coapplicant[i]:
        co_applicant_income[i] = int(monthly_income[i] * np.random.uniform(0.3, 1.2))
    else:
        co_applicant_income[i] = 0

# Household income
household_income = monthly_income + co_applicant_income

# Existing EMI obligations (% of income)
existing_emi_obligations = (monthly_income * np.random.beta(1, 3, NUM_SAMPLES) * 0.5).astype(int)

# Monthly expenses (30-80% of income)
expense_ratio = np.random.beta(5, 5, NUM_SAMPLES) * 0.5 + 0.3  # 0.3 to 0.8
total_monthly_expenses = (monthly_income * expense_ratio).astype(int)

# Savings balance (0 to 24 months of income)
savings_months = np.random.exponential(6, NUM_SAMPLES)
savings_months = np.clip(savings_months, 0, 24)
savings_balance = (monthly_income * savings_months).astype(int)

# ==================== CREDIT HISTORY ====================
print("Generating credit history...")

# Credit score (300-900)
credit_score_base = np.random.normal(650, 120, NUM_SAMPLES)
credit_score_base = np.clip(credit_score_base, 300, 900).astype(int)

# Adjust credit score by age, income, etc.
credit_score = credit_score_base.copy()
for i in range(NUM_SAMPLES):
    # Older = slightly higher score (more history)
    credit_score[i] += int((age[i] - 30) * 2) if age[i] > 30 else 0
    # Higher income = slightly higher score
    credit_score[i] += int((monthly_income[i] - 30000) / 5000) if monthly_income[i] > 30000 else 0
    # Cap at 900
    credit_score[i] = min(900, max(300, credit_score[i]))

# Credit history length (years)
credit_history_length = np.maximum(0, age - 18 - np.random.randint(0, 5, NUM_SAMPLES))
credit_history_length = np.clip(credit_history_length, 0, 30).astype(int)

# Past defaults count (70% have 0, 20% have 1, etc.)
past_defaults_count = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if credit_score[i] < 550:
        probs = [0.4, 0.3, 0.2, 0.1]
    elif credit_score[i] < 700:
        probs = [0.7, 0.2, 0.07, 0.03]
    else:
        probs = [0.9, 0.08, 0.02, 0.0]
    past_defaults_count[i] = np.random.choice([0, 1, 2, 3], p=probs)

# Active loans count
active_loans_count = np.random.poisson(lam=2, size=NUM_SAMPLES)
active_loans_count = np.clip(active_loans_count, 0, 8).astype(int)

# Credit utilization ratio (0-1.2)
credit_utilization_ratio = generate_beta(1, 3, NUM_SAMPLES, 0, 1.2)

# ==================== LOAN DETAILS ====================
print("Generating loan application details...")

# Loan purpose
purpose_choices = ['Home', 'Car', 'Education', 'Personal', 'Business', 'Debt Consolidation']
purpose_probs = [0.25, 0.15, 0.1, 0.3, 0.15, 0.05]
loan_purpose = np.random.choice(purpose_choices, NUM_SAMPLES, p=purpose_probs)

# Loan amount (depends on purpose and income)
loan_amount_requested = []
for i in range(NUM_SAMPLES):
    if loan_purpose[i] == 'Home':
        base = household_income[i] * np.random.uniform(3, 8)
        amount = np.random.lognormal(mean=13.5, sigma=0.8)  # ~20-30 lakhs
    elif loan_purpose[i] == 'Car':
        amount = np.random.lognormal(mean=12.2, sigma=0.5)  # ~5-10 lakhs
    elif loan_purpose[i] == 'Education':
        amount = np.random.lognormal(mean=11.5, sigma=0.6)  # ~2-5 lakhs
    elif loan_purpose[i] == 'Business':
        amount = np.random.lognormal(mean=12.8, sigma=1.0)  # ~3-15 lakhs
    else:  # Personal or Debt Consolidation
        amount = np.random.lognormal(mean=11.0, sigma=0.7)  # ~1-3 lakhs
    
    # Scale by income
    amount = amount * (household_income[i] / 50000)
    amount = np.clip(amount, 50000, 5000000).astype(int)
    loan_amount_requested.append(amount)
loan_amount_requested = np.array(loan_amount_requested)

# Loan term (months) - depends on purpose
loan_term_months = []
for i in range(NUM_SAMPLES):
    if loan_purpose[i] == 'Home':
        term = np.random.choice([180, 240, 300, 360], p=[0.2, 0.3, 0.3, 0.2])
    elif loan_purpose[i] == 'Car':
        term = np.random.choice([36, 48, 60, 72], p=[0.2, 0.3, 0.4, 0.1])
    elif loan_purpose[i] == 'Education':
        term = np.random.choice([24, 36, 48, 60], p=[0.1, 0.3, 0.4, 0.2])
    else:
        term = np.random.choice([12, 24, 36, 48, 60], p=[0.1, 0.2, 0.4, 0.2, 0.1])
    loan_term_months.append(term)
loan_term_months = np.array(loan_term_months)

# Interest rate (inverse relation to credit score)
interest_rate = 18 - (credit_score - 300) / 600 * 12 + np.random.normal(0, 2, NUM_SAMPLES)
interest_rate = np.clip(interest_rate, 7, 24).round(2)

# Monthly EMI (PMT formula)
monthly_emi = []
for i in range(NUM_SAMPLES):
    r = interest_rate[i] / 100 / 12
    n = loan_term_months[i]
    pv = loan_amount_requested[i]
    if r == 0:
        emi = pv / n
    else:
        emi = pv * r * (1 + r)**n / ((1 + r)**n - 1)
    monthly_emi.append(int(emi))
monthly_emi = np.array(monthly_emi)

# Debt to Income Ratio
debt_to_income_ratio = (existing_emi_obligations + monthly_emi) / monthly_income
debt_to_income_ratio = np.clip(debt_to_income_ratio, 0, 1.5).round(3)

# Loan to Value Ratio (depends on purpose)
loan_to_value_ratio = []
for i in range(NUM_SAMPLES):
    if loan_purpose[i] == 'Home':
        ltv = np.random.uniform(0.7, 0.9)
    elif loan_purpose[i] == 'Car':
        ltv = np.random.uniform(0.8, 0.95)
    else:
        ltv = np.random.uniform(0.5, 1.0)
    # Better credit = higher LTV allowed
    ltv = min(0.95, ltv * (1 + (credit_score[i] - 600) / 1000))
    loan_to_value_ratio.append(round(ltv, 2))
loan_to_value_ratio = np.array(loan_to_value_ratio)

# ==================== BEHAVIORAL & OTHER PARAMETERS ====================
print("Generating behavioral and contextual parameters...")

# Payment punctuality score (0-100, correlated with credit score)
payment_punctuality_score = credit_score / 9 + np.random.normal(0, 5, NUM_SAMPLES)
payment_punctuality_score = np.clip(payment_punctuality_score, 0, 100).astype(int)

# Bank account balance volatility (0-100)
balance_volatility = 100 - payment_punctuality_score / 2 + np.random.normal(0, 10, NUM_SAMPLES)
balance_volatility = np.clip(balance_volatility, 0, 100).astype(int)

# Recent credit inquiries (past 6 months)
recent_inquiries = np.random.poisson(lam=2, size=NUM_SAMPLES)
recent_inquiries = np.clip(recent_inquiries, 0, 10).astype(int)

# Max days past due ever
max_dpd = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if past_defaults_count[i] > 0:
        max_dpd[i] = np.random.choice([30, 60, 90, 120, 180], p=[0.4, 0.3, 0.15, 0.1, 0.05])
    else:
        max_dpd[i] = np.random.choice([0, 15, 30], p=[0.7, 0.2, 0.1])

# Income growth rate (% over 3 years)
income_growth_rate = np.random.normal(8, 10, NUM_SAMPLES)
income_growth_rate = np.clip(income_growth_rate, -5, 25).round(1)

# Job change frequency (in last 5 years)
job_change_freq = np.random.poisson(lam=1, size=NUM_SAMPLES)
job_change_freq = np.clip(job_change_freq, 0, 5).astype(int)

# Number of bank accounts
num_bank_accounts = np.random.poisson(lam=3, size=NUM_SAMPLES)
num_bank_accounts = np.clip(num_bank_accounts, 1, 8).astype(int)

# Digital payment frequency (0-100)
digital_payment_freq = 100 - (age - 20) * 1.5 + np.random.normal(0, 10, NUM_SAMPLES)
digital_payment_freq = np.clip(digital_payment_freq, 0, 100).astype(int)

# Application channel
channel_choices = ['Online', 'Branch', 'Agent', 'Mobile App']
channel_probs = [0.5, 0.3, 0.15, 0.05]
application_channel = np.random.choice(channel_choices, NUM_SAMPLES, p=channel_probs)

# Region economic risk score (1-10, 1=low risk)
region_risk = np.random.randint(1, 11, NUM_SAMPLES)

# Phone number vintage (days)
phone_vintage = np.random.exponential(scale=700, size=NUM_SAMPLES)
phone_vintage = np.clip(phone_vintage, 1, 5000).astype(int)

# Property area
property_area_choices = ['Urban', 'Semi-Urban', 'Rural']
property_area_probs = [0.45, 0.35, 0.2]
property_area = np.random.choice(property_area_choices, NUM_SAMPLES, p=property_area_probs)

# Home ownership status
home_ownership = []
for i in range(NUM_SAMPLES):
    if age[i] < 30:
        probs = [0.2, 0.2, 0.5, 0.1]
    elif age[i] < 45:
        probs = [0.4, 0.3, 0.25, 0.05]
    else:
        probs = [0.6, 0.25, 0.1, 0.05]
    home_ownership.append(np.random.choice(
        ['Own', 'Mortgage', 'Rent', 'With Parents'], 
        p=probs
    ))
home_ownership = np.array(home_ownership)

# ==================== OPTIONAL PARAMETERS ====================
print("Generating optional parameters...")

# GST filing consistency (only for self-employed)
gst_filing_consistency = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if employment_type[i] == 'SelfEmployed':
        gst_filing_consistency[i] = np.random.randint(40, 101) if np.random.random() < 0.6 else np.random.randint(0, 40)

# Utility bill payment score
utility_bill_score = payment_punctuality_score + np.random.normal(0, 5, NUM_SAMPLES)
utility_bill_score = np.clip(utility_bill_score, 0, 100).astype(int)

# Rent payment consistency (only for renters)
rent_consistency = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if home_ownership[i] == 'Rent':
        rent_consistency[i] = payment_punctuality_score[i] + np.random.randint(-10, 11)
        rent_consistency[i] = np.clip(rent_consistency[i], 0, 100).astype(int)

# Insurance premium history
insurance_premium_history = np.zeros(NUM_SAMPLES)
has_insurance = np.random.random(NUM_SAMPLES) < 0.5
for i in range(NUM_SAMPLES):
    if has_insurance[i]:
        insurance_premium_history[i] = payment_punctuality_score[i] + np.random.randint(-5, 6)
        insurance_premium_history[i] = np.clip(insurance_premium_history[i], 0, 100).astype(int)

# Investment balance (only 40% have investments)
investment_balance = np.zeros(NUM_SAMPLES)
has_investments = np.random.random(NUM_SAMPLES) < 0.4
for i in range(NUM_SAMPLES):
    if has_investments[i]:
        investment_balance[i] = int(monthly_income[i] * np.random.uniform(3, 48))

# Business vintage (only for self-employed)
business_vintage = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if employment_type[i] == 'SelfEmployed':
        business_vintage[i] = min(years_in_current_job[i], np.random.randint(1, 31))

# Family dependency ratio
family_dependency_ratio = dependents / np.maximum(1, (marital_status == 'Married').astype(int) + 1)

# Cost of living index (100 = average)
cost_of_living = np.ones(NUM_SAMPLES) * 100
for i in range(NUM_SAMPLES):
    if property_area[i] == 'Urban':
        cost_of_living[i] = np.random.randint(110, 151)
    elif property_area[i] == 'Semi-Urban':
        cost_of_living[i] = np.random.randint(90, 121)
    else:
        cost_of_living[i] = np.random.randint(70, 101)

# Device risk score (0=clean)
device_risk = np.zeros(NUM_SAMPLES)
risky_devices = np.random.random(NUM_SAMPLES) < 0.05
device_risk[risky_devices] = np.random.randint(30, 101, size=np.sum(risky_devices))

# Application completion time (seconds)
app_completion_time = np.random.normal(480, 120, NUM_SAMPLES)
app_completion_time = np.clip(app_completion_time, 120, 1200).astype(int)

# Form edit count
form_edit_count = np.random.poisson(lam=3, size=NUM_SAMPLES)
form_edit_count = np.clip(form_edit_count, 0, 15).astype(int)

# Email domain risk flag
email_domain_risk = (np.random.random(NUM_SAMPLES) < 0.05).astype(int)

# ==================== EXTENDED PARAMETERS (28) ====================
print("Generating extended parameters...")

# Occupation type (simplified)
occupation_types = ['IT Professional', 'Teacher', 'Doctor', 'Engineer', 'Business Owner', 
                   'Government Employee', 'Banking Professional', 'Sales', 'Laborer', 'Other']
occupation_probs = [0.15, 0.1, 0.05, 0.1, 0.15, 0.1, 0.05, 0.1, 0.1, 0.1]
occupation = np.random.choice(occupation_types, NUM_SAMPLES, p=occupation_probs)

# Has credit card flag
has_credit_card = (np.random.random(NUM_SAMPLES) < 0.6).astype(int)

# Days since phone change (from phone vintage)
days_since_phone_change = phone_vintage

# Loan type (simplified)
loan_type = loan_purpose.copy()  # Use purpose as proxy

# Interest rate spread
interest_rate_spread = np.random.uniform(1, 8, NUM_SAMPLES).round(2)

# Annuity amount (same as EMI)
annuity_amount = monthly_emi

# Total assets value
total_assets = (monthly_income * np.random.uniform(12, 240, NUM_SAMPLES)).astype(int)

# Outstanding debt excluding this
outstanding_debt = (existing_emi_obligations * loan_term_months * 0.7).astype(int)

# Credit card utilization (only if has card)
cc_utilization = np.zeros(NUM_SAMPLES)
cc_utilization[has_credit_card == 1] = generate_beta(1, 3, np.sum(has_credit_card == 1), 0, 1.2)

# Recent inquiries 12 months
recent_inquiries_12m = recent_inquiries + np.random.poisson(lam=1, size=NUM_SAMPLES)
recent_inquiries_12m = np.clip(recent_inquiries_12m, 0, 15).astype(int)

# Count 30+ DPD
count_30plus_dpd = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if past_defaults_count[i] > 0:
        count_30plus_dpd[i] = np.random.poisson(lam=2)
    else:
        count_30plus_dpd[i] = np.random.poisson(lam=0.2)
count_30plus_dpd = np.clip(count_30plus_dpd, 0, 10).astype(int)

# Past default flag
past_default_flag = (past_defaults_count > 0).astype(int)

# Credit mix diversity (1-5)
credit_mix = np.random.poisson(lam=2, size=NUM_SAMPLES) + 1
credit_mix = np.clip(credit_mix, 1, 5).astype(int)

# Employer sector risk
sector_risk_map = {
    'IT Professional': 3, 'Teacher': 2, 'Doctor': 2, 'Engineer': 3,
    'Business Owner': 7, 'Government Employee': 1, 'Banking Professional': 4,
    'Sales': 6, 'Laborer': 8, 'Other': 5
}
employer_sector_risk = np.array([sector_risk_map[occ] for occ in occupation])

# Total work experience
total_work_exp = np.maximum(0, age - 22 - (education == 'Under Graduate').astype(int) * 0).astype(int)

# Income volatility index
income_volatility = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if employment_type[i] == 'Gig':
        income_volatility[i] = np.random.randint(50, 100)
    elif employment_type[i] == 'SelfEmployed':
        income_volatility[i] = np.random.randint(30, 80)
    elif employment_type[i] == 'Salaried':
        income_volatility[i] = np.random.randint(0, 30)
    else:
        income_volatility[i] = np.random.randint(0, 20)

# Salary credit regularity
salary_regularity = np.zeros(NUM_SAMPLES)
salary_regularity[employment_type == 'Salaried'] = np.random.randint(70, 101, np.sum(employment_type == 'Salaried'))
salary_regularity[employment_type != 'Salaried'] = np.random.randint(0, 71, np.sum(employment_type != 'Salaried'))

# EMI payment punctuality (same as payment punctuality)
emi_punctuality = payment_punctuality_score

# Savings rate
savings_rate = ((monthly_income - total_monthly_expenses - existing_emi_obligations) / monthly_income)
savings_rate = np.clip(savings_rate, 0, 0.5).round(3)

# Bounce frequency
bounce_freq = np.random.poisson(lam=0.2, size=NUM_SAMPLES)
bounce_freq = np.clip(bounce_freq, 0, 10).astype(int)

# UPI transaction consistency
upi_consistency = digital_payment_freq + np.random.randint(-10, 11, NUM_SAMPLES)
upi_consistency = np.clip(upi_consistency, 0, 100).astype(int)

# Wallet top-up regularity
wallet_regularity = digital_payment_freq + np.random.randint(-20, 21, NUM_SAMPLES)
wallet_regularity = np.clip(wallet_regularity, 0, 100).astype(int)

# Property ownership status (from home ownership)
property_ownership = home_ownership

# Region default rate index (2-15%)
region_default_rate = region_risk * 1.5 + np.random.uniform(0, 2, NUM_SAMPLES)
region_default_rate = np.clip(region_default_rate, 2, 15).round(2)

# Urban/rural indicator (from property area)
urban_rural = property_area

# Local employment stability index (1-10)
local_employment_stability = 10 - region_risk / 2 + np.random.randint(-1, 2, NUM_SAMPLES)
local_employment_stability = np.clip(local_employment_stability, 1, 10).astype(int)

# External risk score 1 (0-1)
external_risk_1 = 1 - (credit_score - 300) / 600 + np.random.normal(0, 0.1, NUM_SAMPLES)
external_risk_1 = np.clip(external_risk_1, 0, 1).round(3)

# Fraud flag (2% fraud rate)
fraud_flag = (np.random.random(NUM_SAMPLES) < 0.02).astype(int)

# ==================== TARGET VARIABLES ====================
print("Generating target variables (loan approval & default)...")

# Loan approval probability (sigmoid of features)
approval_score = (
    + 0.8 * (credit_score / 900)
    + 0.5 * np.minimum(monthly_income / 500000, 1)
    - 0.4 * debt_to_income_ratio
    - 0.3 * np.minimum(past_defaults_count, 3) / 3
    + 0.3 * (employment_type == 'Salaried').astype(int)
    - 0.2 * np.minimum(loan_amount_requested / 5000000, 1)
    + np.random.normal(0, 0.1, NUM_SAMPLES)
)
approval_prob = sigmoid(approval_score)
loan_approved = (approval_prob > 0.5).astype(int)

# Adjust to get ~65-70% approval
current_approval_rate = loan_approved.mean()
if current_approval_rate < 0.65:
    # Lower threshold
    loan_approved = (approval_prob > 0.45).astype(int)
elif current_approval_rate > 0.7:
    # Raise threshold
    loan_approved = (approval_prob > 0.55).astype(int)

# Default status (only for approved loans)
default_status = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if loan_approved[i] == 1:
        default_score = (
            - 0.9 * (credit_score[i] / 900)
            + 0.6 * debt_to_income_ratio[i]
            + 0.5 * np.minimum(past_defaults_count[i], 3) / 3
            + 0.3 * (income_volatility[i] / 100)
            - 0.2 * np.minimum(savings_balance[i] / 500000, 1)
            + 0.2 * loan_to_value_ratio[i]
            + np.random.normal(0, 0.15)
        )
        default_prob = sigmoid(default_score)
        default_status[i] = (default_prob > 0.5).astype(int)

# ==================== CREATE DATAFRAME ====================
print("Creating final dataframe...")

df = pd.DataFrame({
    # MANDATORY (25)
    'applicant_age': age,
    'gender': gender,
    'marital_status': marital_status,
    'dependents_count': dependents,
    'education_level': education,
    'loan_amount_requested': loan_amount_requested,
    'loan_purpose': loan_purpose,
    'loan_term_months': loan_term_months,
    'interest_rate': interest_rate,
    'monthly_income': monthly_income,
    'co_applicant_income': co_applicant_income,
    'existing_emi_obligations': existing_emi_obligations,
    'total_monthly_expenses': total_monthly_expenses,
    'savings_balance': savings_balance,
    'debt_to_income_ratio': debt_to_income_ratio,
    'credit_score': credit_score,
    'credit_history_length_years': credit_history_length,
    'past_defaults_count': past_defaults_count,
    'active_loans_count': active_loans_count,
    'credit_utilization_ratio': credit_utilization_ratio,
    'employment_type': employment_type,
    'years_in_current_job': years_in_current_job,
    'income_verification_status': income_verification_status,
    
    # IMPORTANT (15)
    'property_area': property_area,
    'home_ownership_status': home_ownership,
    'loan_to_value_ratio': loan_to_value_ratio,
    'monthly_emi_amount': monthly_emi,
    'payment_punctuality_score': payment_punctuality_score,
    'bank_account_balance_volatility': balance_volatility,
    'recent_credit_inquiries_6months': recent_inquiries,
    'max_days_past_due_ever': max_dpd,
    'income_growth_rate_3years': income_growth_rate,
    'job_change_frequency': job_change_freq,
    'number_of_bank_accounts': num_bank_accounts,
    'digital_payment_frequency': digital_payment_freq,
    'application_channel': application_channel,
    'region_economic_risk_score': region_risk,
    'phone_number_vintage_days': phone_vintage,
    
    # OPTIONAL (12)
    'gst_filing_consistency': gst_filing_consistency,
    'utility_bill_payment_score': utility_bill_score,
    'rent_payment_consistency': rent_consistency,
    'insurance_premium_history': insurance_premium_history,
    'investment_balance': investment_balance,
    'business_vintage_years': business_vintage,
    'family_dependency_ratio': family_dependency_ratio,
    'cost_of_living_index': cost_of_living,
    'device_risk_score': device_risk,
    'application_completion_time_seconds': app_completion_time,
    'form_edit_count': form_edit_count,
    'email_domain_risk_flag': email_domain_risk,
    
    # EXTENDED (28 selected)
    'occupation_type': occupation,
    'has_credit_card_flag': has_credit_card,
    'days_since_phone_change': days_since_phone_change,
    'loan_type': loan_type,
    'interest_rate_spread': interest_rate_spread,
    'annuity_amount': annuity_amount,
    'total_assets_value': total_assets,
    'outstanding_debt_excluding_this': outstanding_debt,
    'credit_card_utilization': cc_utilization,
    'recent_credit_inquiries_12months': recent_inquiries_12m,
    'count_30plus_dpd': count_30plus_dpd,
    'past_default_flag': past_default_flag,
    'credit_mix_diversity': credit_mix,
    'employer_sector_risk_score': employer_sector_risk,
    'total_work_experience_years': total_work_exp,
    'income_volatility_index': income_volatility,
    'salary_credit_regularity': salary_regularity,
    'emi_payment_punctuality_score': emi_punctuality,
    'savings_rate': savings_rate,
    'bounce_frequency_12months': bounce_freq,
    'upi_transaction_consistency': upi_consistency,
    'wallet_topup_regularity': wallet_regularity,
    'property_ownership_status': property_ownership,
    'region_default_rate_index': region_default_rate,
    'urban_rural_indicator': urban_rural,
    'local_employment_stability_index': local_employment_stability,
    'external_risk_score_1': external_risk_1,
    'fraud_flag': fraud_flag,
    
    # TARGETS (2)
    'loan_approved': loan_approved,
    'default_status': default_status
})

# ==================== VALIDATION & STATISTICS ====================
print("\n" + "="*60)
print("DATASET GENERATION COMPLETE")
print("="*60)
print(f"Total rows: {len(df):,}")
print(f"Total features: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n--- TARGET DISTRIBUTIONS ---")
print(f"Loan Approval Rate: {df['loan_approved'].mean()*100:.2f}%")
approved_defaults = df[df['loan_approved']==1]['default_status'].mean()*100
print(f"Default Rate (among approved): {approved_defaults:.2f}%")
print(f"Fraud Rate: {df['fraud_flag'].mean()*100:.2f}%")

print("\n--- CREDIT SCORE DISTRIBUTION ---")
print(df['credit_score'].describe())
print("\nCredit Score Bands:")
print(f"  Poor (300-550): {((df['credit_score'] >= 300) & (df['credit_score'] < 550)).mean()*100:.1f}%")
print(f"  Average (550-700): {((df['credit_score'] >= 550) & (df['credit_score'] < 700)).mean()*100:.1f}%")
print(f"  Good (700-800): {((df['credit_score'] >= 700) & (df['credit_score'] < 800)).mean()*100:.1f}%")
print(f"  Excellent (800-900): {(df['credit_score'] >= 800).mean()*100:.1f}%")

print("\n--- INCOME DISTRIBUTION ---")
print(f"Mean Monthly Income: ₹{df['monthly_income'].mean():,.0f}")
print(f"Median Monthly Income: ₹{df['monthly_income'].median():,.0f}")
print(f"Min: ₹{df['monthly_income'].min():,} | Max: ₹{df['monthly_income'].max():,}")

print("\n--- LOAN AMOUNT DISTRIBUTION ---")
print(f"Mean Loan Amount: ₹{df['loan_amount_requested'].mean():,.0f}")
print(f"Median Loan Amount: ₹{df['loan_amount_requested'].median():,.0f}")
print(f"Min: ₹{df['loan_amount_requested'].min():,} | Max: ₹{df['loan_amount_requested'].max():,}")

# print("\n--- CORRELATION WITH TARGET ---")
# print("Loan Approval Correlations:")
# top_corr = df.corrwith(df['loan_approved']).abs().sort_values(ascending=False).head(10)
# for feat, corr in top_corr.items():
#     if feat not in ['loan_approved', 'default_status']:
#         print(f"  {feat}: {corr:.3f}")



# ==================== SAVE TO CSV ====================
print(f"\nSaving to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Dataset saved successfully!")

# Show sample rows
print("\n--- FIRST 5 ROWS SAMPLE ---")
print(df.head().to_string())

print("\n" + "="*60)
print("🎯 READY FOR MODEL TRAINING!")
print("="*60)