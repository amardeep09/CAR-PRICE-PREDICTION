# CAR-PRICE-PREDICTION
Determining whether the listed price of a used car is a challenging task, due to the many factors that drive a used vehicle’s price on the market. The focus of this project is developing machine learning models that can accurately predict the price of a used car based on its features, in order to make informed purchases. I have implemented and evaluated various learning methods on a dataset consisting of the sale prices of different makes and models from car dekho website. The results show that ExtraTreesRegressor and RandomizedSearchCV with Random search of parameters, using 3 fold cross validation yield the best results and this model is implemented as a web app using flask.

# DATASET
For this project, I am using the dataset on used car sales from Car Dekho website, available on Kaggle.. The features available in this dataset are Year, Selling Price, Present Price, Kms Driven, Fuel Type and Seller Type.

# DATA PREPROCESSING
There was no null value in the dataset. It had some categorical variable like Fuel type with types Petrol, Desiel and CNG etc. I have converted the Make, Model and State into one-hot vectors. I replaced the string representing the Fuel type and Dealer type with a boolean example- if Petrol has value 1 and Desiel has value 0 then it represent the fuel type is petrol. If both petrol and Desiel have value 0 then it represent fuel type is CNG. Below table shows this pre-processing step. Certain features such as name dropped during training as these were unique to each vehicle, thus adding no value to training process.

<p align ="centre"><img src ="https://user-images.githubusercontent.com/49056493/126598550-292a0c98-d4e3-48c3-9dc9-ce1820f34d8d.png"></p>

<h3 align ="centre"> <b>Fig: Before one hot encoding</b> </h3>

<p align ="centre"><img src ="https://user-images.githubusercontent.com/49056493/126598638-2f1f5315-f5d3-4124-932f-a783b9948051.png"></p>

<h3 align ="centre"> <b>Fig: After one hot encoding</b> </h3>


# Methodology

<p align ="centre"><img src ="https://user-images.githubusercontent.com/49056493/126599069-e6fbe3fb-22a6-44e8-82c6-64d703811386.png"></p>
<h2 align ="centre"> Fig: Top 5 important features selected </h2>


I have utilized several methods, including ensemble learning techniques, with a 70% - 30% split for the training and test data. For most of the model implementations, the open-source Scikit-Learn package was used.

# 1. Linear Regression
Linear Regression was chosen as the first model due to its simplicity and comparatively small training time. The features, without any feature mapping, were used directly as the feature vectors. No regularization was used since the results clearly showed low variance.
# 2. Extra tree regressor
Extra-trees differ from classic decision trees in the way they are built. When looking for the best split to separate the samples of a node into two groups, random splits are drawn for each of the max_features randomly selected features and the best split among those is chosen. When max_features is set 1, this amounts to building a totally random decision tree.
# 3. RandomizedSearchCV
Randomized search on hyper parameters. RandomizedSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used. The parameters of the estimator used to apply these methods are optimized
