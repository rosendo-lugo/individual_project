# standard imports
import pandas as pd
import numpy as np

# visualized your data
import matplotlib.pyplot as plt
import seaborn as sns

# Stats
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind, levene
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE



# ----------------------------------------------------------------------------------
def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

# ----------------------------------------------------------------------------------
def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols


'''
*------------------*
|                  |
|     EXPLORE      |
|                  |
*------------------*
'''

 # ----------------------------------------------------------------------------------  
def exploring_countplot(df):
    # Checking the distribution of 'status_label' column
    plt.figure(figsize=(8, 6))
    sns.countplot(x='status_label', data=df)
    plt.title('Distribution of Company Status')
    plt.show()

    # Checking the distribution of 'year' column
    plt.figure(figsize=(10, 6))
    sns.countplot(x='year', data=df)
    plt.title('Distribution of Year')
    plt.xticks(rotation=45)
    plt.show()
# -------------------------THESE ARE THE MAIN FUNCTIONS------------------------------
def explore_univariate(tr, cat_vars, quant_vars):
    for var in cat_vars:
        explore_univariate_categorical(tr, var)
        # print('_________________________________________________________________')
    for col in quant_vars:
        p, descriptive_stats = explore_univariate_quant(tr, col)
        print('______________________________________________________________________________________')
        plt.show(p)
        print(descriptive_stats)
# ----------------------------------------------------------------------------------       
def explore_bivariate(tr, target, cat_vars, quant_vars):
    for cat in cat_vars:
        if cat in tr.columns:
            explore_bivariate_categorical(tr, target, cat)
        else:
            print(f"Warning: Column '{cat}' not found in the DataFrame.")
    for quant in quant_vars:
        if quant in tr.columns:
            explore_bivariate_quant(tr, target, quant)
        else:
            print(f"Warning: Column '{quant}' not found in the DataFrame.")
# ----------------------------------------------------------------------------------
def explore_multivariate(tr, target, cat_vars, quant_vars):
    '''
    '''
    plot_swarm_grid_with_color(tr, target, cat_vars, quant_vars)
    plt.show()
    violin = plot_violin_grid_with_color(tr, target, cat_vars, quant_vars)
    plt.show()
    pair = sns.pairplot(data=tr, vars=quant_vars, hue=target)
    plt.show()
    plot_all_continuous_vars(tr, target, quant_vars)
    plt.show()    


# ----------------THE MAIN FUNCTIONS READ THE BELOW FUNCTIONS-----------------------
# -----------------------------Univariate-------------------------------------------
### Univariate

def explore_univariate_categorical(tr, cat_var):
    '''
    takes in a dataframe and a categorical variable and returns
    a frequency table and barplot of the frequencies. 
    '''
    frequency_table = freq_table(tr, cat_var)
    plt.figure(figsize=(2,2))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()
    print(frequency_table)

# ----------------------------------------------------------------------------------    
def explore_univariate_quant(tr, quant_var):
    '''
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    descriptive_stats = tr[quant_var].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(tr[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(tr[quant_var])
    p = plt.title(quant_var)
    return p, descriptive_stats

# ----------------------------------------------------------------------------------    
def freq_table(tr, cat_var):
    '''
    for a given categorical variable, compute the frequency count and percent split
    and return a dataframe of those values along with the different classes. 
    '''
    class_labels = list(tr[cat_var].unique())

    frequency_table = (
        pd.DataFrame({cat_var: class_labels,
                      'Count': tr[cat_var].value_counts(normalize=False), 
                      'Percent': round(tr[cat_var].value_counts(normalize=True)*100,2)}
                    )
    )
    return frequency_table

# ----------------------------------------------------------------------------------  
# -----------------------------Bivariate--------------------------------------------
#### Bivariate

#*************************
def explore_bivariate_categorical(tr, target, cat_var):
    if cat_var in tr.columns:
        print(cat_var, "\n_____________________\n")
        ct = pd.crosstab(tr[cat_var], tr[target], margins=True)
        chi2_summary, observed, expected = run_chi2(tr, cat_var, target)
        p = plot_cat_by_target(tr, target, cat_var)
        
        print(chi2_summary)
        print("\nobserved:\n", ct)
        print("\nexpected:\n", expected)
        plt.show(p)
        print("\n_____________________\n")
    else:
        print(f"Warning: Column '{cat_var}' not found in the DataFrame.")
    
# ---------------------------------------------------------------------------------- 
#*************************
def explore_bivariate_quant(tr, target, quant_var):
    if quant_var in tr.columns:
        print(quant_var, "\n____________________\n")
        descriptive_stats = tr.groupby(target)[quant_var].describe()
        average = tr[quant_var].mean()
        mann_whitney = compare_means(tr, target, quant_var)
        plt.figure(figsize=(4, 4))
        boxen = plot_boxen(tr, target, quant_var)
        swarm = plot_swarm(tr, target, quant_var)
        plt.show()
        print(descriptive_stats, "\n")
        print("\nMann-Whitney Test:\n", mann_whitney)
        print("\n____________________\n")
    else:
        print(f"Warning: Column '{quant_var}' not found in the DataFrame.")

# ----------------------------------------------------------------------------------  
# ------------------------Bivariate Categorical-------------------------------------    
## Bivariate Categorical

def run_chi2(tr, cat_var, target):
    observed = pd.crosstab(tr[cat_var], tr[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected
# ---------------------------------------------------------------------------------- 
def plot_cat_by_target(tr, target, cat_var):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(cat_var, target, data=tr, alpha=.8, color='lightseagreen')
    overall_rate = tr[target].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p
# ---------------------------------------------------------------------------------- 
def correlation_matrix(tr):
    list_col = list(tr.columns)
    data = tr[list_col[3:]]
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Pearson Correlation Matrix')

    plt.show()
# ----------------------------------------------------------------------------------     
def bivariate_boxplot(tr, numerical_cols, categorical_cols):
    # Bivariate exploration: Plot each variable against your target

    # For a categorical target variable, your target can be on the x-axis, and numeric variables on the y
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='status_label', y=col, data=tr)
        plt.title(f'{col} vs status_label')
        plt.show()

    # For a numeric target variable, your target can be on the y-axis, and independent variables on the x-axis
    for col in categorical_cols:
        if df[col].nunique() < 10:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=col, y='status_label', data=tr)
            plt.title(f'status_label vs {col}')
            plt.xticks(rotation=45)
            plt.show()
# ----------------------------------------------------------------------------------  
# ------------------------Bivariate Quant------------------------------------------- 
## Bivariate Quant
# This function was used during the individual project
def plot_swarm(tr, target, quant_var):
    average = tr[quant_var].mean()
    p = sns.swarmplot(data=tr, x=target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p
# ---------------------------------------------------------------------------------- 
def plot_boxen(tr, target, quant_var):
    average = tr[quant_var].mean()
    p = sns.boxenplot(data=tr, x=target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

# ---------------------------------------------------------------------------------- 
# alt_hyp = ‘two-sided’, ‘less’, ‘greater’

def compare_means(tr, target, quant_var, alt_hyp='two-sided'):
    x = tr[tr[target]==0][quant_var]
    y = tr[tr[target]==1][quant_var]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)

# ----------------------------------------------------------------------------------  
# -------------------------Multivariate---------------------------------------------  
### Multivariate
# This function was used during the individual project
def plot_all_continuous_vars(tr, target, quant_vars):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing target. 
    '''
    my_vars = [item for sublist in [quant_vars[2:], [target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = tr[my_vars].melt(id_vars=target, var_name="measurement")
    plt.figure(figsize=(48,20))
    p = sns.boxenplot(x="measurement", y="value", hue=target, data=melt)
    p.set(yscale="log", xlabel='')
    plt.xticks(rotation=45)
    plt.show() 

# ----------------------------------------------------------------------------------
# This function was used during the individual project
def subplot_all_continuous_vars(tr, target, quant_vars):
    my_vars = [item for sublist in [quant_vars[2:], [target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = tr[my_vars].melt(id_vars=target, var_name="measurement")

    # Determine the number of rows and columns for your subplots
    n = len(my_vars) - 1  # Subtract 1 because 'status_label' is not a quantitative variable
    ncols = 3  # You can adjust this as needed
    nrows = n // ncols if n % ncols == 0 else n // ncols + 1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4*nrows))

    for i, ax in enumerate(axs.flatten()):
        if i < n:
            sns.boxenplot(x="measurement", y="value", hue=target, 
                          data=melt[melt['measurement'] == my_vars[i]], ax=ax)
            ax.set(yscale="log", xlabel='')
            ax.set_title(my_vars[i])  # Set title to the variable name
        else:
            ax.remove()  # Remove extra subplots

    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------------
# This function was used during the individual project    
def multivariate_scatterplot(tr, quant_vars):
    plt.figure(figsize=(18,54))

    # Number of rows and columns for subplot layout
    nrows = int(np.ceil(len(quant_vars) / 2))
    ncols = 2

    for i, col in enumerate(quant_vars):
        plt.subplot(nrows, ncols, i+1)
        sns.scatterplot(x=col, y='total_assets', hue='status_label', data=tr)
        plt.title(col)

    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------------
# This function was used during the individual project      
def multivariate_violinplot(tr, categorical_cols, numerical_cols):
    for cat in categorical_cols[1:]:
        if tr[cat].nunique() < 10:
            for num in numerical_cols:
                plt.figure(figsize=(8, 6))
                sns.violinplot(x=cat, y=num, hue='status_label', data=tr)
                plt.title(f'{num} vs {cat} by status_label')
                plt.xticks(rotation=45)
        plt.show()
# ----------------------------------------------------------------------------------  
def plot_violin_grid_with_color(tr, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.violinplot(x=cat, y=quant, data=tr, split=True, 
                           ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()
        
# ----------------------------------------------------------------------------------      
def plot_swarm_grid_with_color(tr, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.swarmplot(x=cat, y=quant, data=tr, ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()


# ----------------------------------------------------------------------------------  
# all variables are continuous and I want to compare them to quality, a continuous variable
# pairplot showed roughly normal dist for all but residual_sugar, but I have > 500 observations; so normal
# BUT, a levene test showed equal_variance to be false for all, so use spearmanr
# - Spearmanr for the stat test to check for correlation 
# - H0: the feature-target pair is not correlated, Ha: they are correlated

# will check for equal variance with levene test 
# - from documentation: "small p-value suggests that the populations do not have equal variances"
# - H0: feature-target pair have equal variance, Ha: not equal variance
# - a p < .05 suggests not equal variance and we will set that value to False in the SpearmansR test

def get_spearmanr_regplots(df, target):
    """
    This function will
    - accept a dataframe with continuous variables and the target column, also continuous
    - accept a string which is the name of the target column
    - prints regplots of each feature vs the target
    - returns a dataframe with results of levene test for equal variance and spearmanr 
      test for correlation on each feature vs target
    """
    
#     list_col = list(df.columns)
#     data = df[list_col[3:]]
    # target = 'quality'
    list_col = list(df.drop(columns = [target]).columns)
    features = df[list_col[2:]]
    results_list = []

    for col in features:
        lt, lp = levene(df[col], df[target])
        if lp < .05: 
            eq_var = False
        else: 
            eq_var = True

        t, p = spearmanr(df[col], df[target])

        sns.regplot(df[col], df[target], scatter_kws = {"color": "black", "alpha": 0.5}
                    , line_kws = {"color": "red"})
        if p < .05:
            plt.title('p < alpha: suggest H_a (correlated)')
            p_string = 'REJECT H0, suggests H_a (correlated)'
        else:
            plt.title('p >= alpha: cannot reject H_0 (not correlated)')
            p_string = 'CANNOT reject H0 (no correlation)'
        plt.show()

        results_list.append([col, lp, eq_var, t, p, p_string])

    results_df = pd.DataFrame(results_list, columns=['column','levene p', 'equal_var', 'spearmanr t'
                                                     , 'spearmanr p', 'hypothesis result'])
    return results_df


# ---------------------------------------------------------------------------------- 
# Question 1: How has the number of bankruptcies changed over the years?
def bankruptcies_per_year(df):
    bankruptcies_per_year = df[df['status_label'] == 0]['year'].value_counts().sort_index()
    bankruptcies_per_year.plot(kind='line')
    plt.title('Number of Bankruptcies Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Bankruptcies')
    plt.show()
    
# ----------------------------------------------------------------------------------     
def chi2_test(df):
    # Create a cross-tabulation of status and year
    observed = pd.crosstab(df['status_label'], df['year'])

    # Run the Chi-Square test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    # Print the results
    print(f'Chi-square Statistic: {chi2}')
    print(f'p-value: {p}')
    return chi2, p, degf, expected
# ----------------------------------------------------------------------------------     
def eval_results(p, alpha, group1='status_label', group2='year'):
    '''
    this function will take in the p-value, alpha, and a name for the 2 variables 
    you are comparing (group 1 and group 2)
    '''
    alpha = 0.05
    
    if p < alpha:
        print(f'There exists some relationship between {group1} and the {group2}. (p-value: {p})')
    else:
        print(f'There is not a significant relationship between {group1} and {group2}. (p-value: {p})')
        
# -----------------------------Question 2-------------------------------------------           
def get_bankrupt_non_bankrupt(df):
    # Filter the data for bankrupt and non-bankrupt companies
    bankrupt = df[df['status_label'] == 0]
    non_bankrupt = df[df['status_label'] == 1]
    
    return bankrupt, non_bankrupt

# ----------------------------------------------------------------------------------
def boxplot(bankrupt, non_bankrupt):   
    # Plot total assets and total liabilities for bankrupt and non-bankrupt companies
    plt.figure(figsize=(14,6))

    plt.subplot(121)
    sns.boxplot(x='total_assets_size', y='total_assets', data=bankrupt)
    plt.title('Total Assets for Bankrupt Companies')

    plt.subplot(122)
    sns.boxplot(x='total_assets_size', y='total_liabilities', data=bankrupt)
    plt.title('Total Liabilities for Bankrupt Companies')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14,6))

    plt.subplot(121)
    sns.boxplot(x='total_assets_size', y='total_assets', data=non_bankrupt)
    plt.title('Total Assets for Non-Bankrupt Companies')

    plt.subplot(122)
    sns.boxplot(x='total_assets_size', y='total_liabilities', data=non_bankrupt)
    plt.title('Total Liabilities for Non-Bankrupt Companies')

    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------------

def violin_plot(bankrupt, non_bankrupt):
    # Plot total assets and total liabilities for bankrupt and non-bankrupt companies using violin plots
    plt.figure(figsize=(14,6))

    plt.subplot(121)
    sns.violinplot(x='total_assets_size', y='total_assets', data=bankrupt)
    plt.title('Total Assets for Bankrupt Companies')

    plt.subplot(122)
    sns.violinplot(x='total_assets_size', y='total_liabilities', data=bankrupt)
    plt.title('Total Liabilities for Bankrupt Companies')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14,6))

    plt.subplot(121)
    sns.violinplot(x='total_assets_size', y='total_assets', data=non_bankrupt)
    plt.title('Total Assets for Non-Bankrupt Companies')

    plt.subplot(122)
    sns.violinplot(x='total_assets_size', y='total_liabilities', data=non_bankrupt)
    plt.title('Total Liabilities for Non-Bankrupt Companies')

    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------------
def ttest(df):
    # Analyze the total assets and total liabilities compared between bankrupt and non-bankrupt companies

    bankrupt_assets = df[df['status_label'] == 0]['total_assets']
    not_bankrupt_assets = df[df['status_label'] == 1]['total_assets']

    t_stat_assets, p_assets = ttest_ind(bankrupt_assets, not_bankrupt_assets, equal_var=False)

    bankrupt_liabilities = df[df['status_label'] == 0]['total_liabilities']
    not_bankrupt_liabilities = df[df['status_label'] == 1]['total_liabilities']

    t_stat_liabilities, p_liabilities = ttest_ind(bankrupt_liabilities, not_bankrupt_liabilities, equal_var=False)
    
    print(f'Total Assets: t-statistic: {t_stat_assets}, p-value: {p_assets}')
    print(f'Total liabilities: t-statistic: {t_stat_liabilities}, p-value: {p_liabilities}')

    return t_stat_assets, p_assets, t_stat_liabilities, p_liabilities

# ----------------------------------------------------------------------------------
    
def validate_p_value(p_value, alpha=0.05):
    '''
    This function takes a p_value and an alpha level (default is 0.05),
    and prints a statement about the hypothesis based on the p_value.
    '''
    if p_value < alpha:
        print(f"We reject the null hypothesis, p-value: {p_value}")
    else:
        print(f"We fail to reject the null hypothesis, p-value: {p_value}")
# -----------------------------Question 3-----------------------------------------

def financial_indicators(df, financial_indicators):
    # Set the figure size
    plt.figure(figsize=(20, 10))

    # Loop through each financial indicator
    for i, indicator in enumerate(financial_indicators):
        plt.subplot(3, 6, i+1)
        sns.boxplot(x='status_label', y=indicator, data=df)
        plt.title(indicator)
        plt.tight_layout()

    # Show the plot
    plt.show()
    
# ----------------------------------------------------------------------------------
def financial_indicator_ttest(df, financial_indicators):
    # Initialize an empty dataframe to store the t-test results
    t_test_results = pd.DataFrame(columns=['financial_indicator', 't_stat', 'p_value'])

    # Perform t-test for each financial indicator
    for indicator in financial_indicators:
        alive = df[df['status_label'] == 1][indicator]
        bankrupt = df[df['status_label'] == 0][indicator]
        t_stat, p_value = ttest_ind(alive, bankrupt, equal_var=False)
        t_test_results = t_test_results.append({'financial_indicator': indicator, 't_stat': t_stat, 'p_value': p_value}, ignore_index=True)
        
    # Add a new column to the t_test_results dataframe indicating if the null hypothesis was rejected
    t_test_results['reject_null'] = t_test_results['p_value'] < 0.05

    # Display the t-test results
    return t_test_results

# -------------------------------Question 4-----------------------------------------
def count_plot(df):
    # Plot the distribution of bankruptcy status with respect to the size of total assets
    plt.figure(figsize=(10, 6))
    sns.countplot(x='total_assets_size', hue='status_label', data=df)
    plt.title('Bankruptcy Status by Total Assets Size')
    plt.xlabel('Total Assets Size')
    plt.ylabel('Count')
    plt.show()
    
# ----------------------------------------------------------------------------------
def chi2_square(df):
    # Create a contingency table
    contingency_table = pd.crosstab(df['total_assets_size'], df['status_label'])

    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the Chi-square test result
    return chi2, p
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------