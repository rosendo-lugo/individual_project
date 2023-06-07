import pandas as pd
import numpy as np
import os

# visualized your data
import matplotlib.pyplot as plt
import seaborn as sns


# Stats
from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind, levene
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector



'''
*------------------*
|                  |
|     MODELING     |
|                  |
*------------------*
'''

# ----------------------------------------------------------------------------------
def metrics_reg(y_tr, yhat):
    """
    y_tr = y_train
    yhat = y_pred
    send in y_train, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y_tr, yhat, squared=False)
    r2 = r2_score(y_tr, yhat)
    return rmse, r2


# ----------------------------------------------------------------------------------
def rfe(X_tr, X_val, y_tr, k=3):
    '''
    # X_tr = X_train_scaled
    # X_val = X_validate_scaled
    # y_tr = y_train
    # k = the number of features to select
    '''
    
    # make a model object to use in RFE process.
    # The model is here to give us metrics on feature importance and model score
    # allowing us to recursively reduce the number of features to reach our desired space
    model = LinearRegression()
    
    # MAKE the thing
    rfe = RFE(model, n_features_to_select=k)

    # FIT the thing
    rfe.fit(X_tr, y_tr)
    
    X_tr_rfe = pd.DataFrame(rfe.transform(X_tr),index=X_tr.index,
                                          columns = X_tr.columns[rfe.support_])
    
    X_val_rfe = pd.DataFrame(rfe.transform(X_val),index=X_val.index,
                                      columns = X_val.columns[rfe.support_])
    
    top_k_rfe = X_tr.columns[rfe.get_support()]
    
    return top_k_rfe, X_tr_rfe, X_val_rfe

# ----------------------------------------------------------------------------------
def select_kbest(X_tr, y_tr,k):
    '''
    # X_tr = X_train
    # y_tr = y_train
    # k = the number of features to select (we are only sending two as of right now)
    '''
    
    # MAKE the thing
    kbest = SelectKBest(f_regression, k=k)

    # FIT the thing
    kbest.fit(X_tr, y_tr)
    
    # Create a DATAFRAME
    kbest_results = pd.DataFrame(
                dict(pvalues=kbest.pvalues_, feature_scores=kbest.scores_),
                index = X_tr.columns)
    
    # we can apply this mask to the columns in our original dataframe
    top_k = X_tr.columns[kbest.get_support()]
    
    return top_k
# ----------------------------------------------------------------------------------
def get_models_dataframe(baseline, tr, y_tr, y_val, y_ts, X_tr_sc, X_val_sc, X_ts_sc):
    baseline_array = np.repeat(baseline, len(tr))
    rmse_tr, r2 = metrics_reg(y_tr, baseline_array)
    rmse_val = ''
    metrics_df = pd.DataFrame(data=[{'model': 'baseline','rmse train':rmse_tr ,'rmse validate': rmse_val, 'R2 validate': r2}])

    # OLS + RFE
    top_k_rfe, X_tr_rfe, X_val_rfe = rfe(X_tr_sc, X_val_sc, y_tr, 3) # use the 3 best features of RFE
    lr_rfe = LinearRegression()
    lr_rfe.fit(X_tr_rfe, y_tr)

    pred_lr_rfe_tr = lr_rfe.predict(X_tr_rfe)
    rmse_tr, r2_tr = metrics_reg(y_tr, pred_lr_rfe_tr)

    pred_lr_rfe_val = lr_rfe.predict(X_val_rfe)
    rmse_val, r2_val = metrics_reg(y_val, pred_lr_rfe_val)
    metrics_df.loc[1] = ['ols+RFE', rmse_tr, rmse_val, r2_val]

    # OLS
    lr = LinearRegression()
    lr.fit(X_tr_sc, y_tr)

    pred_lr_tr = lr.predict(X_tr_sc)
    rmse_tr, r2 = metrics_reg(y_tr, pred_lr_tr)

    pred_lr_val = lr.predict(X_val_sc)
    rmse_val, r2 = metrics_reg(y_val, pred_lr_val)
    metrics_df.loc[2] = ['ols', rmse_tr, rmse_val, r2]

    # LARS
    lars = LassoLars(alpha=.5) # default is 1.  .5
    lars.fit(X_tr_sc, y_tr)

    pred_lars_tr = lars.predict(X_tr_sc)
    rmse_tr, r2 = metrics_reg(y_tr, pred_lars_tr)

    pred_lars_val = lars.predict(X_val_sc)
    rmse_val, r2 = metrics_reg(y_val, pred_lars_val)
    metrics_df.loc[3] = ['lars', rmse_tr, rmse_val, r2]

    # # Polynomial
    degrees = 2
    # for degree in degrees:
    pf = PolynomialFeatures(degree=degrees)
    X_tr_degree = pf.fit_transform(X_tr_sc)
    X_val_degree = pf.transform(X_val_sc)
    X_ts_degree = pf.transform(X_ts_sc)

    pr = LinearRegression()
    pr.fit(X_tr_degree, y_tr)

    pred_pr_tr = pr.predict(X_tr_degree)
    rmse_tr, r2 = metrics_reg(y_tr, pred_pr_tr)

    pred_pr_val = pr.predict(X_val_degree)
    rmse_val, r2 = metrics_reg(y_val, pred_pr_val)
    metrics_df.loc[4] = [f"poly_{degrees}D",rmse_tr, rmse_val, r2]


    # GLM
    # power = 0: Normal Distribution
    # power = 1: Poisson Distribution
    # power = 2: Gamma Distribution
    # power = 3: Inverse Gaussian Distribution
    glm = TweedieRegressor(power=0, alpha=.5) # default 1
    glm.fit(X_tr_sc, y_tr)

    pred_glm_tr = glm.predict(X_tr_sc)
    rmse_tr, r2 = metrics_reg(y_tr, pred_glm_tr)

    pred_glm_val = glm.predict(X_val_sc)
    rmse_val, r2 = metrics_reg(y_val, pred_glm_val)
    metrics_df.loc[5] = ['glm', rmse_tr, rmse_val, r2]

    return metrics_df, pred_lr_rfe_tr, pred_lr_tr, pred_lars_tr, pred_pr_tr, pred_glm_tr
# ----------------------------------------------------------------------------------
def test_best_model(X_ts_sc, y_ts, X_tr_sc, y_tr):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_tr_2d = pf.fit_transform(X_tr_sc)
    # transform & X_test_scaled
    X_ts_2d = pf.transform(X_ts_sc)

    #make it
    pr = LinearRegression()
    #fit it
    pr.fit(X_tr_2d, y_tr)

    #use it
    pred_ts = pr.predict(X_ts_2d)
    rmse, r2 = metrics_reg(y_ts, pred_ts)
    result = {'model Poly_2D': 'test', 'rmse': rmse, 'r2': r2}
    result = pd.DataFrame([result])
    return result

# ----------------------------------------------------------------------------------
def predict_vs_actual_graph(baseline, y_tr, pred_lr_tr, pred_pr_tr, pred_glm_tr):
    plt.scatter(pred_lr_tr, y_tr, label='linear regression')
    plt.scatter(pred_pr_tr, y_tr, label='polynominal 2deg')
    plt.scatter(pred_glm_tr, y_tr, label='glm')
    plt.plot(y_tr, y_tr, label='Perfect line of regression', color='grey')

    plt.axhline(baseline, ls=':', color='grey')
    plt.annotate("Baseline", (65, 81))

    plt.title("Where are predictions more extreme? More modest?")
    plt.ylabel("Actual Quality")
    plt.xlabel("Predicted Quality")
    plt.legend()

    plt.show()
    
# ----------------------------------------------------------------------------------
def residual_scatter(y_tr, pred_lr_tr, pred_pr_tr, pred_glm_tr):
    plt.axhline(label="No Error")

    plt.scatter(y_tr, pred_lr_tr - y_tr, alpha=.5, color="red", label="LinearRegression")
    plt.scatter(y_tr, pred_glm_tr - y_tr, alpha=.5, color="yellow", label="TweedieRegressor")
    plt.scatter(y_tr, pred_pr_tr - y_tr, alpha=.5, color="green", label="Polynomial 2deg ")

    plt.legend()
    plt.title("Do the size of errors change as the actual value changes?")
    plt.xlabel("Actual Quality")
    plt.ylabel("Residual: Predicted Quality - Actual Quality")

    plt.show()

# ----------------------------------------------------------------------------------
def distribution_actual_vs_predict(y_tr, pred_lr_tr, pred_pr_tr, pred_glm_tr):
    plt.hist(y_tr, color='blue', alpha=.5, label="Actual")
    plt.hist(pred_lr_tr, color='red', alpha=.5, label="LinearRegression")
    plt.hist(pred_glm_tr, color='yellow', alpha=.5, label="TweedieRegressor")
    plt.hist(pred_pr_tr, color='green', alpha=.5, label="Polynomial 2Deg")

    plt.xlabel("Quality")
    plt.ylabel("Number of Wines")
    plt.title("Comparing the Distribution of Actual to Predicted Quality")
    plt.legend()
    plt.show()