from sklearn import preprocessing
from functools import reduce
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier ,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB 

np.random.seed(100)
    # print(np.random.random())
for i in range(100):
        print(np.random.random())

RANDOM_SEED =  0

def metri(true_labels_cv,predictions,m):
    ACC = []
    Re = []
    Pr = []
    F1= []
    MCC = []
    AUC = []
    ALL = []
    #print(confusion_matrix(true_labels_cv, predictions))
    #print ('ACC: %.4f' % metrics.accuracy_score(true_labels_cv, predictions))
    #print ('Recall: %.4f' % metrics.recall_score(true_labels_cv, predictions,pos_label=1))
    #print ('Precesion: %.4f' %metrics.precision_score(true_labels_cv, predictions))
    #print ('F1-score: %.4f' %metrics.f1_score(true_labels_cv, predictions))
    rig=[1]
    test_y_1=[+1 if x in rig else -1 for x in true_labels_cv]
    y_pred_1=[+1 if x in rig else -1 for x in predictions]
    #print ('MCC: %.4f' %metrics.matthews_corrcoef(test_y_1,y_pred_1))
    #print ('AUC: %.4f' % metrics.roc_auc_score(true_labels_cv, predictions))
    ACC.append(metrics.accuracy_score(true_labels_cv, predictions))
    Re.append(metrics.recall_score(true_labels_cv, predictions,pos_label=1))
    Pr.append(metrics.precision_score(true_labels_cv, predictions))
    F1.append(metrics.f1_score(true_labels_cv, predictions))
    MCC.append(metrics.matthews_corrcoef(test_y_1,y_pred_1))
    #AUC.append(metrics.roc_auc_score(true_labels_cv, predictions))
    #ALL = ACC+Re+Pr+F1+MCC+AUC
    if m==1:
        return ACC
    if m==2:
        return Re
    if m==3:
        return Pr
    if m==4:
        return F1
    if m==5:
        return MCC
    else:
        ALL = ACC+Re+Pr+F1+MCC
        return ALL




xtrain= pd.read_csv('1471_20.csv')
xtest= pd.read_csv('159_20.csv')

columns_name = list(xtrain.columns[0:])    
print("---------------------------columns name---------------------------------")
print(columns_name)

#label of training set
df = pd.read_csv('1471labels.csv',header=None)
y_train = []
y_train=df.iloc[:,0]
#label of independent set
df1 = pd.read_csv('159labels.csv',header=None)
y_test = []
y_test=df1.iloc[:,0]




triple = []
triple=[list(range(1,40)),list(range(40,624)),list(range(624,664)),list(range(664,689)),list(range(689,714)),
        list(range(714,734)),list(range(734,1134)),list(range(1134,9134)),list(range(9134,9534)),list(range(9534,10534)),
        list(range(10534,10554)),list(range(10554,10954)),list(range(10954,10994)),list(range(10994,11234)),list(range(11234,13634)),
        list(range(13634,13907)),list(range(13907,14250)),list(range(14250,14280)),list(range(14280,14302)),list(range(14302,14365)),
        list(range(1,14365))
        ]


K=[]
for i in range(1,2):
        E=[]
        W=[]
        E=list(combinations(triple,i))
        #print(E)
        #print(len(E))
        for i in range(len(E)):
            W=list(E[i])
            S=reduce(lambda x, y:np.hstack((x,y)),W)
            K.append(S)

Featu=["ss3-","ss8-","RSA-","Diso-","PBS-","Mono-","di-","th-","pssm","smoth-pssm",
        "aac_pssm","rpm-pssm","pse-pssm","dp_pssm-","CKSAAP-","CTD-","CTraid-","SC-PseAAC","PC-PseAAC","other",
        "ALL"]
lidtF=[]
for i in range(1,2):
        lidtF1=[]
        lidtF2=[]
        lidtF1=list(combinations(Featu,i))
        for i in range(len(lidtF1)):
            lidtF2=list(lidtF1[i])
            lidtF3=list(reduce(lambda x, y:np.hstack((x,y)),lidtF2))
            lidtF.append(lidtF3)

n = 0
for i in K:
    print(str(len(i))+str(lidtF[n]))
    n = n+1
    X_train=xtrain.iloc[:,i]
    X_test = xtest.iloc[:,i]
    
    print(X_train.columns[0:])
    print(X_test.columns[0:])

    kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
    xgb = XGBClassifier(njobs = -1)

    varesult = []

    for train, test in kfold.split(X_train, y_train):
            train_new = np.asarray(X_train.iloc[train])
            test_new = np.asarray(X_train.iloc[test])
            ytrain_new = np.asarray(y_train.iloc[train])
            ytest_new = np.asarray(y_train.iloc[test])

            min_max_l = preprocessing.MinMaxScaler()
            min_max_l.fit(train_new)
            train_new = min_max_l.transform(train_new)
            test_new = min_max_l.transform(test_new) 
                
            xgb.fit(train_new, ytrain_new)
            splitresult = xgb.predict(test_new)
            ALL8 = metri(ytest_new,splitresult,6)
            varesult.append(ALL8)       
            #each validation
            #print("-----result-------------------------------------------------")
            #print(ALL8)
    #average of 5-fold validation
    print("----va-result-------------------------------------------------")
    b = np.array(varesult)
    b_1 = np.mean(b,axis=0)     
    print(b_1)
print("---------------DDDDDDDDDDDDDDDDDDDDDDDDDDDONE------------------------------------")

