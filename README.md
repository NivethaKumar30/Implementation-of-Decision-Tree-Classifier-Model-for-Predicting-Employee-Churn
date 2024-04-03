# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NIVETHA . K 
RegisterNumber:  212222230102

```
```
import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier,plot_tree
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plot_tree(dt, feature_names=x.columns, class_names=['not left', 'left'], filled=True)
plt.show()
```
## Output:

DATASET:

![Screenshot 2024-04-03 161008](https://github.com/NivethaKumar30/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559844/ffb913a7-9260-4483-99a0-38886f321d27)


ACCURACY AND dt predict()


![image](https://github.com/NivethaKumar30/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559844/63a3faae-7174-4fa8-9255-cab5738aa458)


![image](https://github.com/NivethaKumar30/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559844/82d373f0-c78a-49c8-9f82-02139a851e20)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
