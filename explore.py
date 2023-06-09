# standard imports
import pandas as pd
import numpy as np

# visualized your data
import matplotlib.pyplot as plt
import seaborn as sns

# Stats
from scipy import stats
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