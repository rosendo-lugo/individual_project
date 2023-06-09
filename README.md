# American Bankruptcy Analysis


https://images.unsplash.com/photo-1567427017947-545c5f8d16ad?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80


## Data Information
A novel dataset for bankruptcy prediction related to American public companies listed on the New York Stock Exchange and NASDAQ is provided. The dataset comprises accounting data from 8,262 distinct companies recorded during the period spanning from 1999 to 2018.

According to the Security Exchange Commission (SEC), a company in the American market is deemed bankrupt under two circumstances. Firstly, if the firm's management files for Chapter 11 of the Bankruptcy Code, indicating an intention to "reorganize" its business. In this case, the company's management continues to oversee day-to-day operations, but significant business decisions necessitate approval from a bankruptcy court. Secondly, if the firm's management files for Chapter 7 of the Bankruptcy Code, indicating a complete cessation of operations and the company going out of business entirely.

In this dataset, the fiscal year prior to the filing of bankruptcy under either Chapter 11 or Chapter 7 is labeled as "Bankruptcy" (1) for the subsequent year. Conversely, if the company does not experience these bankruptcy events, it is considered to be operating normally (0). The dataset is complete, without any missing values, synthetic entries, or imputed added values.

The resulting dataset comprises a total of 78,682 observations of firm-year combinations.


## Project Summary
This project aims to analyze the financial indicators of American companies to predict their bankruptcy status. The project includes data cleaning, exploratory data analysis (EDA), hypothesis testing, and predictive modeling. The goal is to build a predictive model that can accurately identify companies at risk of bankruptcy, which could be a valuable tool for investors, creditors, and policymakers.

## Project Planning
Goal:
The goal of this project is to identify financial indicators that can predict a company's bankruptcy status.

## Initial Hypotheses
- Hypothesis 1: There is a significant difference in total assets between bankrupt and non-bankrupt companies.

- Hypothesis 2: There is a significant difference in total liabilities between bankrupt and non-bankrupt companies.

- Hypothesis 3: Bankruptcy status varies with the size of total assets.

- Hypothesis 4: The number of bankruptcies has changed over the years.

## Project Planning Initial Thoughts
The project will start with data cleaning and EDA to understand the data better. This will be followed by hypothesis testing to answer the questions posed during the EDA phase. Finally, a predictive model will be built to predict the bankruptcy status of a company based on its financial indicators.

## Deliverables
>• Final clean, interactive Python notebook

>• README.md file with project summary, planning, and key findings

>• Project Summary document with key findings, tested hypotheses, and takeaways


## Data Dictionary

| Variable Name            | Description                                                                                                     |
|--------------------------|--------------------------------------------------------------------------------------------------------|
| X1                       | Current assets - All the assets of a company that are expected to be sold or used as a result of standard business operations over the next year. |
| X2                       | Cost of goods sold - The total amount a company paid as a cost directly related to the sale of products.                                                        |
| X3                       | Depreciation and amortization - Depreciation refers to the loss of value of a tangible fixed asset over time (such as property, machinery, buildings, and plant). Amortization refers to the loss of value of intangible assets over time.  |
| X4                       | EBITDA - Earnings before interest, taxes, depreciation, and amortization. It is a measure of a company's overall financial performance, serving as an alternative to net income. |
| X5                       | Inventory - The accounting of items and raw materials that a company either uses in production or sells.                                                       |
| X6                       | Net Income - The overall profitability of a company after all expenses and costs have been deducted from total revenue.                                        |
| X7                       | Total Receivables - The balance of money due to a firm for goods or services delivered or used but not yet paid for by customers.                                |
| X8                       | Market value - The price of an asset in a marketplace. In this dataset, it refers to the market capitalization since companies are publicly traded in the stock market. |
| X9                       | Net sales - The sum of a company's gross sales minus its returns, allowances, and discounts.                                                                       |
| X10                      | Total assets - All the assets, or items of value, a business owns.                                                                                                 |
| X11                      | Total Long-term debt - A company's loans and other liabilities that will not become due within one year of the balance sheet date.                               |
| X12                      | EBIT - Earnings before interest and taxes.                                                                                                                         |
| X13                      | Gross Profit - The profit a business makes after subtracting all the costs that are related to manufacturing and selling its products or services.               |
| X14                      | Total Current Liabilities - The sum of accounts payable, accrued liabilities, and taxes such as Bonds payable at the end of the year, salaries, and commissions remaining.       |
| X15                      | Retained Earnings - The amount of profit a company has left over after paying all its direct costs, indirect costs, income taxes, and its dividends to shareholders. |
| X16                      | Total Revenue - The amount of income that a business has made from all sales before subtracting expenses. It may include interest and dividends from investments.     |
| X17                      | Total Liabilities - The combined debts and obligations that the company owes to outside parties.                                                                   |
| X18                      | Total Operating Expenses - The expenses a business incurs through its normal business operations.                                                                   |



## Key Findings
### Univariate
- The dataset contains 18 financial indicators for each company, including total assets, total liabilities, net income, and others.
- The dataset is imbalanced with a higher number of non-bankrupt companies compared to bankrupt companies.

### Bivariate
- There is a significant difference in total assets and total liabilities between bankrupt and non-bankrupt companies.
- Bankruptcy status varies with the size of total assets.

### Multivariate
- The number of bankruptcies has changed over the years.
- There are financial indicators that are significantly different between the companies that are alive and those that are bankrupt.


### Tested Hypotheses
- There is a significant difference in total assets between bankrupt and non-bankrupt companies. Result: Rejected Null Hypothesis.
- There is a significant difference in total liabilities between bankrupt and non-bankrupt companies. Result: Rejected Null Hypothesis.
- Bankruptcy status varies with the size of total assets. Result: Rejected Null Hypothesis.
- The number of bankruptcies has changed over the years. Result: Rejected Null Hypothesis.


### Takeaways
- Total assets and total liabilities are significant indicators of a company's bankruptcy status.
- The size of total assets is associated with bankruptcy status.
- The number of bankruptcies has changed over the years, indicating a possible trend or pattern.


## REPO REPLICATION

#### In order to get started reproducing this project, you'll need to set up a proper environment.

- 1.
    a. use the link provided in side the final_individual_project.ipynb. 
    b. If link doesn't not work downloading the American Bankruptcy dataset from the below link.
https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset

- 2. If you downloaded the file you may need to unzip the downloaded file to recover the ***american_bankruptcy.csv*** file.

- 3. Prep your repo:

>• Create a new repository on GitHub to house this project.

>• Clone it into your local machine by copying the SSH link from GitHub and running **'git clone <SSH link'>** in your terminal.

- 4. Create a **.gitignore** file in your local repository and include any files you don't want to be tracked by Git or shared on the internet. This can include your newly downloaded **.csv** files. You can create and edit this file by running **'code .gitignore'** in your terminal (if you're using Visual Studio Code as your text editor) if you open it from the terminal run 'open .gitignore'.

- 5. Create a **README.md** file to begin noting the steps taken so far. You can create and edit this file by running code **README.md** in your terminal.

- 6. Transfer your **'american_bankruptcy.csv'** file into your newly established local repository.

- 7. Create a Jupyter Lab environment to continue working in. You can do this by running **'jupyter lab'** in your terminal.

- 8. In Jupyter Lab, create a new Jupyter Notebook to begin the data pipeline.

***Remember to regularly commit and push your changes to GitHub to ensure your work is saved and accessible from the remote repository.***