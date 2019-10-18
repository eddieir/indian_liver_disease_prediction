# indian_liver_disease_prediction

Liver-diesease-prediction
Business Problem
Problem Context

Patients with Liver disease have been continuously increasing because of excessive consumption of alcohol, inhale of harmful gases, intake of contaminated food, pickles and drugs. This dataset was used to evaluate prediction algorithms in an effort to reduce burden on doctors.
Content

This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease). This data set contains 441 male patient records and 142 female patient records.

Any patient whose age exceeded 89 is listed as being of age "90".
Features:

    Age of the patient
    Gender of the patient
    Total Bilirubin
    Direct Bilirubin
    Alkaline Phosphotase
    Alamine Aminotransferase
    Aspartate Aminotransferase
    Total Protiens
    Albumin
    Albumin and Globulin Ratio
    Dataset: field used to split the data into two sets (patient with liver disease, or no disease)

Mapping business problem to ML problem
Type of Machine Learning Problem

It is a binary classification problem, where given the above set of features, we need to predict if a given patient has liver disease or not
Evaluation Metric (KPI)

Since this is binary classification problem, we use the following metrics:

    Confusion matrix - For getting a better clarity of the no of correct/incorrect predictions by the model
    ROC-AUC - It considers the rank of the output probabilities and intuitively measures the likelihood that model can distinguish between a positive point and a negative point. (Note: ROC-AUC is typically used for binary classification only). We will use AUC to select the best model.

Before going into the model definition we need to have the clear idea about the data that we are dealing with so The first figure which we could see at below describes the destribution of numerical features ![distribution_of_numerical_features](https://user-images.githubusercontent.com/23243761/67105262-ba212b80-f1c8-11e9-93bc-505e2a41786e.png)

    
For this project we have done the analysis based on the following methods:
1.  Logistic Regression
2.  Random Forest
3.  XGBoost
4.  KNN
5.  Descision Trees
6.  SVC
7.  Gradient Boosting
8.  Neural Networks
