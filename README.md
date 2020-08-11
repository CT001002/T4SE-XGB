# T4SE-XGB	

### T4SE-XGB: interpretable sequence-based prediction of type IV secreted effectors using eXtreme gradient boosting algorithm

#### Intruduction

Type IV secreted effectors (T4SEs) can be translocated into the cytosol of host cells via type IV secretion system (T4SS) to achieve destruction of host cells. However, experimental approaches to identify T4SEs are time- and resource-consuming, and the existing computational tools based on machine learning techniques have some limitations such as the lack of interpretability in the prediction models. In this present study, we proposed a new model, T4SE-XGB, which uses the eXtreme gradient boosting (XGBoost) algorithm for accurate identification of type IV effectors based on optimal features extracted from protein sequences. After trying 20 different features, the best result achieved when all features were fed into XGBoost by the 5-fold cross validation compared with different machine learning methods. Then, the ReliefF algorithm was adopted to optimize feature vectors and got final 1100 features for our dataset which obviously improved the model performance. T4SE-XGB exhibited highest predictive performance on the independent test set and the results showed T4SE-XGB clearly outperforms other recent prediction tools. What’s more, the SHAP method was used to interpret the contribution of features to model predictions. The identification of key features can contribute to an improved understanding of multifactorial contributors to host-pathogen interactions and bacterial pathogenesis. In addition to type IV effector prediction, we believe that the proposed framework composed of model construction and model interpretation can provide more instructive guidance for further research of developing novel computational methods and mechanism exploration of biological problems.

#### Requirements

This method developed with Python 3.7. The latest version of T4SE-XGB requires is specified in requirements.txt. 

#### Reference

T4SE-XGB: interpretable sequence-based prediction of type IV secreted effectors using eXtreme gradient boosting algorithm