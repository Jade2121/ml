import pandas as pd
import os
import numpy as np

path = os.getcwd()+'\\Carseats.csv'
f = open(path, encoding='utf-8')
df_sales = pd.read_csv(f)
df_sales.Urban = df_sales.Urban.replace(to_replace=['No', 'Yes'], value=[0, 1])
df_sales.US = df_sales.US.replace(to_replace=['No', 'Yes'], value=[0, 1])
df_sales.ShelveLoc = df_sales.ShelveLoc.replace(to_replace=['Bad', 'Medium', 'Good'], value=[1, 2, 3])
df_sales

y = df_sales['Sales'].values
X = df_sales[['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'ShelveLoc', 'Age', 'Education', 'Urban', 'US']].values


# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# le = LabelEncoder()
# y = le.fit_transform(y) # 把label转换为0和1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,  random_state=1) 


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
# 生成500个决策树，详细的参数建议参考官方文档
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)

# 度量单个决策树的准确性
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
# Output：Decision tree train/test accuracies 1.000/0.854

# 度量bagging分类器的准确性
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))
# Output：Bagging train/test accuracies 1.000/0.896