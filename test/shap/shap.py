#from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier ,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB 
import shap

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


#设置交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)



xgb_test = XGBClassifier(n_estimators=700, learning_rate = 0.1,n_jobs=-1)
     


min_max = preprocessing.MinMaxScaler()
min_max.fit(X_train)
trainplus = min_max.transform(X_train)
testplus = min_max.transform(X_test)

X_train = pd.DataFrame(trainplus,columns = featuresname )
testplus = pd.DataFrame(testplus,columns = featuresname )

xgb_test.fit(X_train, y_train)
y_pred = xgb_test.predict(testplus)
ALL9 = metri(y_test,y_pred,6)
xgb_pred_results.append(ALL9)        
print("----prediction-result-------------------------------------------------")
print(ALL9)





shap.initjs()
explainer = shap.TreeExplainer(xgb_test)
shap_values = explainer.shap_values(X_train)
#shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])

#shap_values = shap.TreeExplainer(xgb_test,feature_perturbation = "tree_path_dependent").shap_values(X_train)

#plt.figure()
shap.summary_plot(shap_values, X_train,max_display=30)
shap.summary_plot(shap_values, X_train, plot_type="bar",max_display=30)        
#result.append(ALL9)    


top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
for i in range(5):
    shap.dependence_plot(top_inds[i], shap_values, X_train)

'''
import matplotlib
# feature_importances DataFrame
feature_importances3 = pd.DataFrame({'Feature_name':X_train.columns,
                                    'Importances':xgb_test.feature_importances_})

# reset index
feature_importances3 = feature_importances3.set_index('Feature_name')

# sort by importances
feature_importances3 = feature_importances3.sort_values(by = 'Importances')


ax = feature_importances3.iloc[-50:].plot(kind='barh',figsize = (15,20),color = 'g',
                          align='center',
                                   title='top 50 Feature Importances')
ax.figure.savefig("2_top50_FeatureImportances.png")

feature_importances3.to_csv("2_FeatureImportances.csv")

'''
shap_interaction_values = shap.TreeExplainer(xgb_test).shap_interaction_values(X_train[X_train.columns])
#shap_interaction_values = shap.TreeExplainer(xgb_test,feature_perturbation = "tree_path_dependent").shap_interaction_values(X_train[cols])
shap.summary_plot(shap_interaction_values, X_train[X_train.columns], max_display=10)
'''
shap_interaction_values = shap.TreeExplainer(xgb_test).shap_interaction_values(X_train[X_train.columns])
#shap_interaction_values = shap.TreeExplainer(xgb_test,feature_perturbation = "tree_path_dependent").shap_interaction_values(X_train[cols])
shap.summary_plot(shap_interaction_values, X_train[X_train.columns], max_display=15)

shap_interaction_values = shap.TreeExplainer(xgb_test).shap_interaction_values(X_train[X_train.columns])
#shap_interaction_values = shap.TreeExplainer(xgb_test,feature_perturbation = "tree_path_dependent").shap_interaction_values(X_train[cols])
shap.summary_plot(shap_interaction_values, X_train[X_train.columns], max_display=20)
'''

