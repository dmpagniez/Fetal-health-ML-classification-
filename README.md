# Fetal-health-ML-classification

In this project the objective is to use an obstetric dataset containing Cardiotocogram information from different fetuses to make a Machine Learning model that is able to classify the health of a fetus as Normal, Suspect or Pathological. The ultimate goal of this project is the early detection of fetal malformations incompatible with life in order to try to prevent avoidable maternal deaths.

The content of the notebook includes EDA, Feature engineering, and different ML algorithms (SVM-C, Random Forest, KNN) tested with different approaches: 
  - unbalanced data respect to the target variable leaving the X as it is. 
  - unbalanced data respect to the target variable doing dimensionality reduction of the variables includes on the X through principal component analysis (PCA).
  - oversampling through synthetic minority oversampling technique (SMOTE) to balance the target variable.
  - oversampling through synthetic minority oversampling technique (SMOTE) to balance the target variable doing dimensionality reduction of the variables includes on the X through principal component analysis (PCA). 

# Acknowledgements

Ayres de Campos et al. (2000) SisPorto 2.0 A Program for Automated Analysis of Cardiotocograms. J Matern Fetal Med 5:311-318 
Link ->. https://onlinelibrary.wiley.com/doi/10.1002/1520-6661(200009/10)9:5%3C311::AID-MFM12%3E3.0.CO;2-9

# Metrics

ML Classification (Unbalanced Target)

SVM-C X_pca	

     Classification report 
          precision    recall  f1-score  support
     1.0   0.936660  0.964427  0.950341    506.0
     2.0   0.666667  0.625000  0.645161     80.0
     3.0   0.947368  0.750000  0.837209     48.0

     Summary of the Metrics 
           Accuracy  Precision    Recall        f1
    Train  0.888445   0.837484  0.759595  0.789763
    test   0.905363   0.850232  0.779809  0.810904

SVM-C actual X

     Classification report 
         precision    recall  f1-score  support
    1.0   0.854167  0.972332  0.909427    506.0
    2.0   0.652174  0.375000  0.476190     80.0
    3.0   1.000000  0.250000  0.400000     48.0

     Summary of the Metrics
           Accuracy  Precision    Recall        f1
    Train  0.809349   0.772462  0.506250  0.554433
    test   0.842271   0.835447  0.532444  0.595206

Random forest X_pca
 
     Classification report 
          precision    recall  f1-score  support
    1.0   0.937864  0.954545  0.946131    506.0
    2.0   0.653333  0.612500  0.632258     80.0
    3.0   0.909091  0.833333  0.869565     48.0

     Summary of the Metrics 

           Accuracy  Precision    Recall        f1
    Train  0.908053   0.868086  0.792390  0.822580
    test   0.902208   0.833429  0.800126  0.815985

Random forest acutal X

      Classification report 
          precision    recall  f1-score  support
    1.0   0.964981  0.980237  0.972549    506.0
    2.0   0.851351  0.787500  0.818182     80.0
    3.0   0.956522  0.916667  0.936170     48.0

     Summary of the Metrics 
           Accuracy  Precision    Recall        f1
    Train  0.940499   0.920322  0.866249  0.887940
    test   0.951104   0.924285  0.894801  0.908967

KNN X_pca

      Classification report 
          precision    recall  f1-score  support
    1.0   0.915110  0.981785  0.947276    549.0
    2.0   0.774648  0.561224  0.650888     98.0
    3.0   0.933333  0.724138  0.815534     58.0

      Summary of the Metrics 
           Accuracy  Precision    Recall        f1
    Train  0.887057   0.831719  0.739023  0.773470
    test   0.902128   0.874364  0.755716  0.804566

KNN actual X

      Classification report 
          precision    recall  f1-score  support
    1.0   0.936283  0.963570  0.949731    549.0
    2.0   0.750000  0.612245  0.674157     98.0
    3.0   0.833333  0.862069  0.847458     58.0

       Summary of the Metrics 
           Accuracy  Precision    Recall        f1
    Train  0.901277   0.837362  0.809658  0.818521
    test   0.906383   0.839872  0.812628  0.823782



ML Classification (SMOTE over-sampling)

SVM-C X_pca

      Classification report 
         precision    recall  f1-score  support
    1.0   0.934537  0.846626  0.888412    489.0
    2.0   0.773519  0.919255  0.840114    483.0
    3.0   0.959140  0.874510  0.914872    510.0

      Summary of the Metrics 
           Accuracy  Precision   Recall        f1
    Train  0.879061   0.889701  0.87887  0.880727
    test   0.879892   0.889065  0.88013  0.881132

SVM-C actual X

      Classification report 
         precision    recall  f1-score  support
    1.0   0.653846  0.660532  0.657172    489.0
    2.0   0.728752  0.834369  0.777992    483.0
    3.0   0.829885  0.707843  0.764021    510.0

      Summary of the Metrics  		   
           Accuracy  Precision    Recall        f1
    Train  0.723672   0.729764  0.723199  0.722759
    test   0.733468   0.737494  0.734248  0.733062

Random forest X_pca

      Classification report 
         precision    recall  f1-score  support
    1.0   0.952586  0.903885  0.927597    489.0
    2.0   0.871893  0.944099  0.906561    483.0
    3.0   0.971717  0.943137  0.957214    510.0

      Summary of the Metrics 
           Accuracy  Precision    Recall        f1
    Train  0.932582   0.935354  0.932701  0.933154
    test   0.930499   0.932065  0.930374  0.930457

Random forest acutal X

      Classification report 
         precision    recall  f1-score  support
    1.0   0.974843  0.950920  0.962733    489.0
    2.0   0.941767  0.971014  0.956167    483.0
    3.0   0.994083  0.988235  0.991150    510.0

      Summary of the Metrics 
           Accuracy  Precision    Recall        f1
    Train  0.966145   0.966870  0.966257  0.966265
    test   0.970310   0.970231  0.970057  0.970017

KNN X_pca

       Classification report 
         precision    recall  f1-score  support
    1.0   0.948244  0.934426  0.941284    549.0
    2.0   0.927536  0.932605  0.930064    549.0
    3.0   0.972875  0.981752  0.977293    548.0

       Summary of the Metrics 
           Accuracy  Precision    Recall        f1
    Train  0.937427   0.938451  0.937434  0.937465
    test   0.949575   0.949552  0.949594  0.949547

KNN actual X

        Classification report 
         precision    recall  f1-score  support
    1.0   0.968635  0.956284  0.962420    549.0
    2.0   0.956204  0.954463  0.955333    549.0
    3.0   0.978417  0.992701  0.985507    548.0

        Summary of the Metrics 
           Accuracy  Precision    Recall        f1
    Train  0.958384   0.959368  0.958390  0.958434
    test   0.967801   0.967752  0.967816  0.967753
