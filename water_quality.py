import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.svm import SVC
water_df = pd.read_csv("datasets/water_potability.csv")


plt.figure(figsize=(12,8))
sns.heatmap(water_df.isnull())

sns.countplot(x="Potability",data=water_df)

# visualization dataset for outliers

fig, ax = plt.subplots(ncols=5, nrows=2, figsize = (20,10))

ax = ax.flatten()
index = 0

for col,values in water_df.items():
    sns.boxplot(y=col,data=water_df, ax=ax[index])

    index += 1


sns.pairplot(water_df)

fig = px.pie(water_df, names="Potability",hole=0.4, template = "plotly_dark")
fig.show()

fig = px.scatter(water_df, x ="ph", y="Sulfate",color="Potability", template ="plotly_dark")
fig.show()

fig = px.scatter(water_df, x ="Organic_carbon", y="Hardness",color="Potability", template ="plotly_dark")
fig.show()

water_df.isnull().mean().plot.bar(figsize = (10,6))
plt.xlabel("Features")
plt.ylabel("Percentage of missing values")

water_df["ph"] = water_df["ph"].fillna(water_df["ph"].mean())
water_df["Sulfate"] = water_df["Sulfate"].fillna(water_df["Sulfate"].mean())
water_df["Trihalomethanes"] = water_df["Trihalomethanes"].fillna(water_df["Trihalomethanes"].mean())

water_df.isnull().sum()

sns.heatmap(water_df.isnull())

#Data Preparations for Training

x = water_df.drop("Potability", axis=1)
y = water_df["Potability"]

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

x_train.shape, x_test.shape

#Logistic Regression

# object of LR

model_lr = LogisticRegression()

# Training Model

model_lr.fit(x_train,y_train)

# Making Prediction

pred_lr = model_lr.predict(x_test)

pred_lr

#accuracy score - Başarı skoru

accuracy_score_lr =accuracy_score(y_test,pred_lr)
accuracy_score_lr
# 5884146341463414

# Decision Tree Classifier
# creating the model object

model_dt = DecisionTreeClassifier(max_depth=4)

#Training of decision tree

model_dt.fit(x_train,y_train)

# Making prediction using Decision Tree

pred_dt = model_dt.predict((x_test))

accuracy_score_dt = accuracy_score(y_test,pred_dt)
accuracy_score_dt

#0.6326219512195121

# confusion matrix

cm2 = confusion_matrix(y_test, pred_dt)
cm2


# Random Forest Classifier

# creating model object
model_rf = RandomForestClassifier()

# training model rf
model_rf.fit(x_train, y_train)

# Making Prediction

pred_rf = model_rf.predict(x_test)
# creating model objectre
accuracy_score_rf = accuracy_score(y_test,pred_rf)
accuracy_score_rf
# accuracy_score_rf * 100
#0.6417682926829268


cm3 = confusion_matrix(y_test,pred_rf)
cm3


# Knn K-Neighbours Classifiers

# Creating Model Object

model_knn = KNeighborsClassifier()

for i in range(4,11):
    model_knn = KNeighborsClassifier(n_neighbors=i)
    model_knn.fit(x_train,y_train)
    pred_knn = model_knn.predict(x_test)
    accuracy_score_knn =accuracy_score(y_test,pred_knn)
    print(i, accuracy_score_knn)

# 4 0.6036585365853658
# 5 0.6097560975609756
# 6 0.6173780487804879
# 7 0.6234756097560976
# 8 0.6341463414634146
# 9 0.6387195121951219
# 10 0.6341463414634146

model_knn = KNeighborsClassifier(n_neighbors=11)
model_knn.fit(x_train,y_train)
pred_knn = model_knn.predict(x_test)
accuracy_score_knn =accuracy_score(y_test,pred_knn)
print(i, accuracy_score_knn)

# 0.635670731707317

#AdaBoostClassifier

# Making object of Model
model_ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.03)
# training the model
model_ada.fit(x_train,y_train)

# Making prediction
pred_ada = model_ada.predict(x_test)

# accuracy check
accuracy_score_ada = accuracy_score(y_test,pred_ada)
accuracy_score_ada

# 0.5990853658536586

# XGBooST

# create model

model_xgb = XGBClassifier(n_estimators=200,learning_rate=0.03)

# training model

model_xgb.fit(x_train,y_train)

# prediction

pred_xgb = model_xgb.predict(x_test)

# accuracy

accuracy_score_xgb =accuracy_score(y_test,pred_xgb)
accuracy_score_xgb
# 0.6478658536585366



# Light Gbm
# creating the model object

model_lgb = lgb.LGBMClassifier()

#Training of decision tree

model_lgb.fit(x_train,y_train)

# Making prediction using Decision Tree

pred_lgb = model_lgb.predict((x_test))

accuracy_score_lgb = accuracy_score(y_test,pred_lgb)
accuracy_score_lgb
#  0.6387195121951219



# Accuracy Visualization on Ml Algorithms

models = pd.DataFrame({
    "Model":["Logistic Regression",
             "Decision Tree",
             "Random Forest",
             "KNN",
             "AdaBoost",
             "XGBoost",
             "LightGBM"],
    "Accuracy Score":[accuracy_score_lr, accuracy_score_dt,accuracy_score_rf,accuracy_score_knn,accuracy_score_ada,
                       accuracy_score_xgb,accuracy_score_lgb]
})

models

sns.barplot(x="Accuracy Score",y="Model",data=models)
models.sort_values(by="Accuracy Score", ascending=False)

