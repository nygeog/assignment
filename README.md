## View the Notebook in nbviewer

https://nbviewer.jupyter.org/github/nygeog/assignment/blob/master/report.ipynb

# Dependencies

Using Python 3.7

    pip install ipython
    pip install pandas
    pip install matplotlib
    pip install -U scikit-learn
    pip install jupyter notebook


# Part 1 Data Exploration and Evaluation
- [ ] Clean columns, carry forward

        ['loan_amnt’, 'funded_amnt’, 'term’, 'int_rate’, 'grade’, 'annual_inc’, 'issue_d’, 'dti’, 'revol_bal’, 'total_pymnt’, 'loan_status’]
      
- [ ] Perform any necessary cleaning and aggregations to explore and better
  understand the dataset.
  - [ ] Describe the data.describe()

- [ ] Describe any assumptions you made to handle null variables and outliers.
    - [ ] Remove outliers for Annual Income
    - [ ] Remove outliers for DTI
    - [ ] Total credit revolving balance (revol_bal)

- [ ] Describe the distributions of the features.
    - [ ] Include two data visualizations and 
    - [ ] Two summary statistics to support these findings.
  
# Part 2 Business Analysis
- [ ] Assume a 36 month investment period for each loan, and exclude loans with less than 36 months of data available.

- [ ] What percentage of loans has been fully paid?
    * **0.7562866221941765**

- [ ] When bucketed by year of origination and grade, which cohort has the highest rate of defaults? Here you may assume that any loan which was not fully paid had “defaulted”.

- [ ] When bucketed by year of origination and grade, what annualized rate of
      return have these loans generated on average?
      
    * For simplicity, use the following approximation:
      `Annualized rate of return = (total_pymnt / funded_amnt) ^ (1/3) - 1`


# Part 3 Modeling
- [ ] Build a logistic regression model to predict loan defaults

- [ ] Assume that

    - [ ] (i) You are given the ability to invest in each loan independently
    
    - [ ] (ii) You invest immediately following loan origination and hold to maturity (36 months)
    
    - [ ] (iii) All loan fields that would be known upon origination are made available to you.

- [ ] Was the model effective? 
    - [ ] Explain how you validated your model and describe how you measure the performance of the model.
    
    
## Fields to use Definitions. 

* **loan_amnt** - The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
* **funded_amnt** - The total amount committed to that loan at that point in time.
* **term** - The number of payments on the loan. Values are in months and can be either 36 or 60.
* **int_rate** - Interest Rate on the loan
* **grade** - LC assigned loan grade
* **annual_inc** - The self-reported annual income provided by the borrower during registration.
* **issue_d** - The month which the loan was funded
* **dti** - A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
* **revol_bal** - Total credit revolving balance
* **total_pymnt** - Payments received to date for total amount funded
* **loan_status** - Current status of the loan
