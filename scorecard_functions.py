import scorecardpy as sc
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import itertools
from warnings import filterwarnings
filterwarnings('ignore')
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,scale
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier 
from sklearn import neighbors
from sklearn.svm import SVR
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,precision_recall_curve,f1_score,auc
from sklearn import metrics



#df = pd.read_excel("default_data.xls")

def feature_engineering(data):
    data_copy=data.copy()
    #Extracting limit bal ratio from PAY_AMT variable for standartization
    for i in range(1,7):
        data_copy["LIMIT_BAL_PAY_AMT_"+str(i)]=data_copy["PAY_AMT"+str(i)]/data_copy.LIMIT_BAL
    #Extracting limit bal ratio from BILL_AMT variable for standartization
    for i in range(1,7):
        data_copy["LIMIT_BAL_BILL_AMT_"+str(i)]=data_copy["BILL_AMT"+str(i)]/data_copy.LIMIT_BAL
    #sum of LIMIT_BAL_PAY_AMT
    cols_pay = []
    for i in range(1,7): cols_pay.append("LIMIT_BAL_PAY_AMT_"+str(i))
    data_copy['LIMIT_BAL_PAY_AMT_TOT'] = data_copy[cols_pay].sum(axis=1)
    #sum of LIMIT_BAL_BILL_AMT
    cols_bill = []
    for i in range(1,7): cols_bill.append("LIMIT_BAL_BILL_AMT_"+str(i))
    data_copy['LIMIT_BAL_BILL_AMT_TOT'] = data_copy[cols_bill].sum(axis=1)
    #sum of PAY_0 to PAY_6 with every combination
    cols_pay= ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    for i in range(2,7):
        combinations=list(itertools.combinations(cols_pay, i))
        for a,b in zip(combinations,range(0,len(combinations))):
            data_copy['PAY_SUM'+str(i)+str(b)] = data_copy[list(a)].sum(axis=1)
    #combine SEX and EDUCATION
    data_copy["SEX_EDUCATION"] = data_copy["SEX"].astype(str) + data_copy["EDUCATION"].astype(str)
    data_copy['SEX_EDUCATION']=data_copy['SEX_EDUCATION'].astype('category').cat.codes
    #combine SEX and MARRIAGE
    data_copy["SEX_MARRIAGE"] = data_copy["SEX"].astype(str) + data_copy["MARRIAGE"].astype(str)
    data_copy['SEX_MARRIAGE']= data_copy['SEX_MARRIAGE'].astype('category').cat.codes
    #combine EDUCATION and MARRIAGE
    data_copy["EDUCATION_MARRIAGE"] = data_copy["EDUCATION"].astype(str) + data_copy["MARRIAGE"].astype(str)
    data_copy['EDUCATION_MARRIAGE']=data_copy['EDUCATION_MARRIAGE'].astype('category').cat.codes
    #combine SEX, EDUCATION and MARRIAGE
    data_copy["SEX_EDUCATION_MARRIAGE"] = data_copy["SEX"].astype(str)+data_copy["EDUCATION"].astype(str) + data_copy["MARRIAGE"].astype(str)
    data_copy['SEX_EDUCATION_MARRIAGE']=data_copy['SEX_EDUCATION_MARRIAGE'].astype('category').cat.codes
    #Create BILL MINUS PAY variable
    for i in range(1,7):
        data_copy["BILL_MINUS_PAY_"+str(i)] = data_copy["BILL_AMT"+str(i)]-data_copy["PAY_AMT"+str(i)]
    #drop old variables
    data_copy.drop(['SEX','EDUCATION','MARRIAGE'],inplace=True,axis=1)
    #one hot encoding for the below columns
    #data_copy=pd.get_dummies(data_copy, columns = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',])

    return(data_copy)

#df_engineering=feature_engineering(df)

def outlier(data):
    unique=pd.DataFrame(data.nunique(),columns=["Unique"]).rename_axis('Column').reset_index()
    for i in unique.loc[unique.Unique>45].Column:
        q1, q3= np.percentile(data[i],[2,98])
        iqr = q3 - q1
        lower_bound = q1 -(1.5 * iqr)
        upper_bound = q3 +(1.5 * iqr)
        data=data.loc[(data[i] > lower_bound) & (data[i] < upper_bound)]
    return(data)

def outlier_limit_bal(data):
    q1, q3= np.percentile(data["LIMIT_BAL"],[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr)
    upper_bound = q3 +(1.5 * iqr)
    df_engineering=data.loc[(data["LIMIT_BAL"] > lower_bound) & (data["LIMIT_BAL"] < upper_bound)]
    return(df_engineering)

#df_engineering=outlier_limit_bal(df_engineering)
#df_engineering=outlier(df_engineering)

def binning_split(data):
    from optbinning import BinningProcess
    X = data.loc[:,data.columns != 'default']
    variable_names=X.columns.tolist()
    y=data.default.tolist()
    binning_rules = BinningProcess(variable_names).fit(X,y)
    X=binning_rules.transform(X)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y, test_size=0.25,random_state=42)
    return(y_train_bin,X_train_bin,y_test_bin,X_test_bin)

#y_train_bin,X_train_bin,y_test_bin,X_test_bin=binning_split(df_engineering) #if you want to bin the data use this function

def scorecard():

    return()
#auto_sc_algo=LogisticRegression()
def auto_scorecard(algorithm,data,target="default"):
    from optbinning import BinningProcess
    from optbinning import Scorecard
    from optbinning.scorecard import ScorecardMonitoring,plot_ks,plot_auc_roc
    X = data.loc[:,data.columns != target]
    variable_names=X.columns.tolist()
    y=data[target].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
    variable_names=X.columns.tolist()
    binning_process = BinningProcess(variable_names)
    estimator = algorithm
    scorecard = Scorecard(binning_process=binning_process, estimator=estimator,scaling_method="min_max",scaling_method_params={"min": 0, "max": 800},
                      reverse_scorecard=False,verbose=True).fit(X, y)
                      
    scorecard_table=scorecard.table(style="detailed")
    return(scorecard_table)


def base_model():
    lr = LogisticRegression().fit(X_train_bin, y_train_bin)
    #create a variable with all zeros
    all_zero_ytest = pd.DataFrame(0, index=range(len(y_test_bin)), columns=range(1))
    predicted_test=pd.Series(lr.predict(X_test_bin),name="predicted_test").reset_index(drop=True)
    predicted_train=pd.Series(lr.predict(X_train_bin),name="predicted_train").reset_index(drop=True)
    y_prob = pd.Series(lr.predict_proba(X_test_bin)[:, 1],name="predicted_prob").reset_index(drop=True)
    y_prob_train = pd.Series(lr.predict_proba(X_train_bin)[:, 1],name="predicted_prob").reset_index(drop=True)
    auc_test = roc_auc_score(y_test_bin, y_prob)
    auc_train = roc_auc_score(y_train_bin, y_prob_train)
    print("Train Data Accuracy is: ",accuracy_score(y_train_bin, predicted_train))
    print("Test Data Accuracy is: ",accuracy_score(y_test_bin, predicted_test))
    print("Test Data All Zero (Base) Accuracy is: ",accuracy_score(y_test_bin, all_zero_ytest))
    print("Terain AUC: ",auc_train)
    print("Test AUC: ",auc_test)
    return()

#algorithm_RFECV=GradientBoostingClassifier()
def feature_elimination_RFECV(algorithm):
    return()
