from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier ,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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

#####输入已经过特征选择后的特征向量集

X_train= pd.read_csv('1471_reliefF_1100.csv')

featuresname = list(X_train.columns[1:])  #la colonne 0 est le quote_conversionflag  
#print(featuresname)
X_train=X_train[featuresname]
#print(X_train)

df = pd.read_csv('1471labels.csv',header=None)
y_train = []
y_train=df.iloc[:,0]

#设置交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)


######################################################################################################################

#####    XGB


n_estimators_options = list(range(100, 1100, 100))
learning_rate_options=[0.001,0.01,0.1,0.2,0.3]

xgb_val_results = []
xgb_para1 = []
xgb_para2 = []

for n_estimators_size in n_estimators_options:
        for learning_rate_size in learning_rate_options:

            xgb_para1.append(n_estimators_size)
            xgb_para2.append(learning_rate_size)

            xgb = XGBClassifier(n_estimators=n_estimators_size, learning_rate = learning_rate_size,n_jobs=-1)
        
            teresult = []
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
                    ALL8 = metri(ytest_new,splitresult,1)
                    teresult.append(ALL8)       
                    #print("-----result-------------------------------------------------")
                    #print(ALL8)
                
            b = np.array(teresult)
            b_1 = np.mean(b,axis=0)     
            xgb_val_results.append(b_1.tolist())

            print("----va-result-------------------------------------------------")
            print(b_1)

output = pd.DataFrame()
output["n_estimators_size"] = xgb_para1
output["learning_rate_size"] = xgb_para2

output["val_results"] = xgb_val_results
output.to_csv("xgb_hypara_1.csv")

print("-DDDDDDDDDDDDDONE---------------------------------------------")


######################################################################################################################

#####    RandomForestClassifier
#设置参数
n_estimators_options = list(range(100, 1100, 100))

max_features_options = [ 'sqrt', 'log2']

rf_val_results = []
rf_para1 = []
rf_para2 = []

for n_estimators_size in n_estimators_options:
    for max_features_size in max_features_options:
        
        rf_para1.append(n_estimators_size)
        rf_para2.append(max_features_size)

        rf = RandomForestClassifier(n_estimators=n_estimators_size, max_features = max_features_size, random_state = RANDOM_SEED,n_jobs=-1)
        teresult = []
        for train, test in kfold.split(X_train, y_train):
                train_new = np.asarray(X_train.iloc[train])
                test_new = np.asarray(X_train.iloc[test])
                ytrain_new = np.asarray(y_train.iloc[train])
                ytest_new = np.asarray(y_train.iloc[test])

                min_max_l = preprocessing.MinMaxScaler()
                min_max_l.fit(train_new)
                train_new = min_max_l.transform(train_new)
                test_new = min_max_l.transform(test_new) 
                
                rf.fit(train_new, ytrain_new)
                splitresult = rf.predict(test_new)
                ALL8 = metri(ytest_new,splitresult,1)
                teresult.append(ALL8)       

                #print("-----result-------------------------------------------------")
                #print(ALL8)
                
        b = np.array(teresult)
        b_1 = np.mean(b,axis=0)     


        rf_val_results.append(b_1.tolist())


        print("----va-result-------------------------------------------------")

        print(b_1)

output = pd.DataFrame()
output["n_estimators_size"] = rf_para1
output["max_features_size"] = rf_para2
output["val_results"] = rf_val_results
output.to_csv("RF_hypara_1.csv")

print("-DDDDDDDDDDDDDONE---------------------------------------------")


######################################################################################################################

#####    ERT

n_estimators_options = list(range(100, 1100, 100))
#max_depth_options = list(range(10, 110, 10))
max_features_options = [ 'sqrt', 'log2']

ert_val_results = []
ert_para1 = []
ert_para2 = []

for n_estimators_size in n_estimators_options:
    for max_features_size in max_features_options:
        
        ert_para1.append(n_estimators_size)
        ert_para2.append(max_features_size)

        ert = ExtraTreesClassifier(n_estimators=n_estimators_size, max_features = max_features_size, random_state = RANDOM_SEED,n_jobs=-1)  
        teresult = []
        for train, test in kfold.split(X_train, y_train):



                train_new = np.asarray(X_train.iloc[train])
                test_new = np.asarray(X_train.iloc[test])
                ytrain_new = np.asarray(y_train.iloc[train])
                ytest_new = np.asarray(y_train.iloc[test])

                min_max_l = preprocessing.MinMaxScaler()
                min_max_l.fit(train_new)
                train_new = min_max_l.transform(train_new)
                test_new = min_max_l.transform(test_new) 
                
                ert.fit(train_new, ytrain_new)
                splitresult = ert.predict(test_new)
                ALL8 = metri(ytest_new,splitresult,1)
                teresult.append(ALL8)       

                #print("-----result-------------------------------------------------")
                #print(ALL8)
                
        b = np.array(teresult)
        b_1 = np.mean(b,axis=0)     


        ert_val_results.append(b_1.tolist())


        print("----va-result-------------------------------------------------")

        #print(b)

        print(b_1)

output = pd.DataFrame()
output["n_estimators_size"] = ert_para1
output["max_features_size"] = ert_para2
output["val_results"] = ert_val_results
output.to_csv("ert_hypara_1.csv")

print("-DDDDDDDDDDDDDONE---------------------------------------------")


######################################################################################################################

#####    SVM

C_options = [pow(2,-6), pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),1,pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6)]
gamma_options = C_options

svm_val_results = []
svm_para1 = []
svm_para2 = []

for gamma_size in gamma_options:
    for C_size in C_options:
        
        svm_para1.append(gamma_size)
        svm_para2.append(C_size)

        svm = SVC(gamma=gamma_size, C = C_size, random_state = RANDOM_SEED)
        
        teresult = []
        for train, test in kfold.split(X_train, y_train):

                train_new = np.asarray(X_train.iloc[train])
                test_new = np.asarray(X_train.iloc[test])
                ytrain_new = np.asarray(y_train.iloc[train])
                ytest_new = np.asarray(y_train.iloc[test])

                min_max_l = preprocessing.MinMaxScaler()
                min_max_l.fit(train_new)
                train_new = min_max_l.transform(train_new)
                test_new = min_max_l.transform(test_new) 
                
                svm.fit(train_new, ytrain_new)
                splitresult = svm.predict(test_new)
                ALL8 = metri(ytest_new,splitresult,1)
                teresult.append(ALL8)       
                #每一次交叉验证结果 
                #print("-----result-------------------------------------------------")
                #print(ALL8)
                
        b = np.array(teresult)
        b_1 = np.mean(b,axis=0)     

        # 记录当前的参数
        svm_val_results.append(b_1.tolist())

        print("----va-result-------------------------------------------------")

        print(b_1)

output = pd.DataFrame()
output["gamma_size"] = svm_para1
output["C_size"] = svm_para2
output["val_results"] = svm_val_results
output.to_csv("svm_hypara.csv")

print("-DDDDDDDDDDDDDONE---------------------------------------------")




######################################################################################################################

#####    LR

solver_options = ['newton-cg','lbfgs','liblinear','sag']
multi_class_options = ['auto', 'ovr']

LR_val_results = []
LR_para1 = []
LR_para2 = []

for multi_class_size in multi_class_options:
    for solver_size in solver_options:
        
        LR_para1.append(multi_class_size)
        LR_para2.append(solver_size)

        LR = LogisticRegression(multi_class=multi_class_size, solver = solver_size, random_state = RANDOM_SEED,n_jobs=4)
        
        teresult = []
        for train, test in kfold.split(X_train, y_train):
                train_new = np.asarray(X_train.iloc[train])
                test_new = np.asarray(X_train.iloc[test])
                ytrain_new = np.asarray(y_train.iloc[train])
                ytest_new = np.asarray(y_train.iloc[test])

                min_max_l = preprocessing.MinMaxScaler()
                min_max_l.fit(train_new)
                train_new = min_max_l.transform(train_new)
                test_new = min_max_l.transform(test_new) 
                
                LR.fit(train_new, ytrain_new)
                splitresult = LR.predict(test_new)
                ALL8 = metri(ytest_new,splitresult,1)
                teresult.append(ALL8)       
                #print("-----result-------------------------------------------------")
                #print(ALL8)
                
        b = np.array(teresult)
        b_1 = np.mean(b,axis=0)     

        LR_val_results.append(b_1.tolist())


        print("----va-result-------------------------------------------------")

        print(b_1)

output = pd.DataFrame()
output["multi_class_size"] = LR_para1
output["solver_size"] = LR_para2
output["val_results"] = LR_val_results
output.to_csv("LR_hypara.csv")

print("-DDDDDDDDDDDDDONE---------------------------------------------")



######################################################################################################################

#####    KNN


k_options = list(range(1,41))

knn_val_results = []
knn_para = []

for k in k_options:

        knn_para.append(k)
        knn = KNeighborsClassifier(n_neighbors =k,n_jobs=-1)

        teresult = []
        for train, test in kfold.split(X_train, y_train):

                train_new = np.asarray(X_train.iloc[train])
                test_new = np.asarray(X_train.iloc[test])
                ytrain_new = np.asarray(y_train.iloc[train])
                ytest_new = np.asarray(y_train.iloc[test])

                min_max_l = preprocessing.MinMaxScaler()
                min_max_l.fit(train_new)
                train_new = min_max_l.transform(train_new)
                test_new = min_max_l.transform(test_new) 
                
                knn.fit(train_new, ytrain_new)
                splitresult = knn.predict(test_new)
                ALL8 = metri(ytest_new,splitresult,1)
                teresult.append(ALL8)       

                #print("-----result-------------------------------------------------")
                #print(ALL8)
                
        b = np.array(teresult)
        b_1 = np.mean(b,axis=0)     

        knn_val_results.append(b_1.tolist())


        print("----va-result-------------------------------------------------")

        print(b_1)

output = pd.DataFrame()
output["k"] = knn_para
output["val_results"] = knn_val_results
output.to_csv("knn_hypara_onlyva.csv")

print("-------KNN------DDDDDDDDDDDDDONE---------------------------------------------")

print("--KNN--BEST-------------------------------------------------")
print(max(knn_val_results))


######################################################################################################################

#####    ML

layer1_options = [16,32,48,64]
layer2_options = layer1_options



ml_val_results = []
ml_para1 = []
ml_para2 = []

for layer1_size in layer1_options:
    for layer2_size in layer2_options:

            ml_para1.append(layer1_size)
            ml_para2.append(layer2_size)

            ml = MLPClassifier(hidden_layer_sizes=(layer1_size,layer2_size), max_iter=1000, random_state = RANDOM_SEED)

        
            teresult = []
            for train, test in kfold.split(X_train, y_train):



                    train_new = np.asarray(X_train.iloc[train])
                    test_new = np.asarray(X_train.iloc[test])
                    ytrain_new = np.asarray(y_train.iloc[train])
                    ytest_new = np.asarray(y_train.iloc[test])

                    min_max_l = preprocessing.MinMaxScaler()
                    min_max_l.fit(train_new)
                    train_new = min_max_l.transform(train_new)
                    test_new = min_max_l.transform(test_new) 
                
                    ml.fit(train_new, ytrain_new)
                    splitresult = ml.predict(test_new)
                    ALL8 = metri(ytest_new,splitresult,1)
                    teresult.append(ALL8)       

                
            b = np.array(teresult)
            b_1 = np.mean(b,axis=0)     

            # 记录当前的参数
            ml_val_results.append(b_1.tolist())

            print("----va-result-------------------------------------------------")

            print(b_1)

output = pd.DataFrame()
output["layer1_size"] = ml_para1
output["layer2_size"] = ml_para2
output["val_results"] = ml_val_results
output.to_csv("ml_hypara_onlyva.csv")

print("-ML----------DDDDDDDDDDDDDONE---------------------------------------------")



#####    GBM



n_estimators_options = list(range(100, 1100, 100))
learning_rate_options=[0.001,0.01,0.1,0.2,0.3]


gbm_val_results = []
gbm_para1 = []
gbm_para2 = []

for n_estimators_size in n_estimators_options:
        for learning_rate_size in learning_rate_options:

            gbm_para1.append(n_estimators_size)
            gbm_para2.append(learning_rate_size)

            gbm = GradientBoostingClassifier(n_estimators=n_estimators_size,  learning_rate = learning_rate_size, random_state = RANDOM_SEED)
        
            teresult = []
            for train, test in kfold.split(X_train, y_train):
                    train_new = np.asarray(X_train.iloc[train])
                    test_new = np.asarray(X_train.iloc[test])
                    ytrain_new = np.asarray(y_train.iloc[train])
                    ytest_new = np.asarray(y_train.iloc[test])

                    min_max_l = preprocessing.MinMaxScaler()
                    min_max_l.fit(train_new)
                    train_new = min_max_l.transform(train_new)
                    test_new = min_max_l.transform(test_new) 
                
                    gbm.fit(train_new, ytrain_new)
                    splitresult = gbm.predict(test_new)
                    ALL8 = metri(ytest_new,splitresult,1)
                    teresult.append(ALL8)       
                    #print("-----result-------------------------------------------------")
                    #print(ALL8)
                
            b = np.array(teresult)
            b_1 = np.mean(b,axis=0)     


            gbm_val_results.append(b_1.tolist())

            print("----va-result-------------------------------------------------")

            print(b_1)

output = pd.DataFrame()
output["n_estimators_size"] = gbm_para1
output["learning_rate_size"] = gbm_para2
output["val_results"] = gbm_val_results
output.to_csv("gbm_hypara_1.csv")

print("-DDDDDDDDDDDDDONE---------------------------------------------")

