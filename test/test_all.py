from sklearn.model_selection import GridSearchCV
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
from sklearn.metrics import confusion_matrix
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
    print(confusion_matrix(true_labels_cv, predictions))
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
X_test= pd.read_csv('159_reliefF_1100.csv')

#特征名称
featuresname = list(X_train.columns[1:])  #la colonne 0 est le quote_conversionflag  
print(featuresname)

X_train=X_train[featuresname]
X_test=X_test[featuresname]

print(X_train)
print(X_test)

#引入训练集类别
df = pd.read_csv('1471labels.csv',header=None)
y_train = []
y_train=df.iloc[:,0]
#引入测试集类别
df1 = pd.read_csv('159labels.csv',header=None)
y_test = []
y_test=df1.iloc[:,0]


clf1 = RandomForestClassifier(n_estimators=300, max_features = 'sqrt', random_state = RANDOM_SEED,n_jobs=-1)
clf2 = GaussianNB()
clf3 = XGBClassifier(n_estimators=700,  learning_rate = 0.1,n_jobs=-1)
clf4 = LogisticRegression(multi_class='auto', solver = 'liblinear', random_state = RANDOM_SEED,n_jobs=-1)
clf5 = GradientBoostingClassifier(n_estimators=300,  learning_rate = 0.2, random_state = RANDOM_SEED)
clf6 = SVC(gamma=0.03125, C = 4, random_state = RANDOM_SEED)
clf7 = ExtraTreesClassifier(n_estimators=900, max_features = 'sqrt', random_state = RANDOM_SEED,n_jobs=-1)
clf8 = KNeighborsClassifier(n_neighbors =2,n_jobs=-1)
clf9 = MLPClassifier(hidden_layer_sizes=(48,16), max_iter=1000, random_state = RANDOM_SEED)


min_max = preprocessing.MinMaxScaler()
min_max.fit(X_train)
trainplus = min_max.transform(X_train)
testplus = min_max.transform(X_test)

for clf, label in zip([clf1, clf2, clf3,clf4,clf5,clf6,clf7,clf8,clf9], 
                      ['RandomForest','GaussianNB',' XGBoost','LogisticRegression','GradientBoosting','SVM','ExtraTrees','KNN','MLP']):

        model=clf.fit(trainplus, y_train)
        y_pred = model.predict(testplus)
        #print(y_pred)
        ALL9 = metri(y_test,y_pred,6)
        #result.append(ALL9)    
        print("----------------------------------------prediction-result-----------------------"+label+"------------------------------------------------")
        print(ALL9)
