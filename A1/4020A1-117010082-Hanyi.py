from scipy import io
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import KFold

# Read data
def read_spam_data():
    raw_data = io.loadmat('spamData.mat')
    
    # turn Xtrain, Xtest, ytrain, ytest into dataframe seperately
    Xtrain = pd.DataFrame([[row.flat[0] for row in line] for line in raw_data['Xtrain']])
    Xtest = pd.DataFrame([[row.flat[0] for row in line] for line in raw_data['Xtest']])
    ytrain = pd.DataFrame([line.flat[0] for line in raw_data['ytrain']], columns=['label'])
    ytest = pd.DataFrame([line.flat[0] for line in raw_data['ytest']], columns=['label'])

    return Xtrain, Xtest, ytrain, ytest

Xtrain, Xtest, ytrain, ytest = read_spam_data()

# generate preprocessed data
Xtrain_standard = pd.DataFrame(preprocessing.scale(Xtrain, axis=0))
Xtest_standard = pd.DataFrame(preprocessing.scale(Xtest, axis=0))
train_data_stnd = pd.concat([Xtrain_standard, ytrain], axis=1)
test_data_stnd = pd.concat([Xtest_standard, ytest], axis=1)

data_stnd = [train_data_stnd, test_data_stnd]
data = [data_stnd]

# Then for each version of the data,implement spam classiﬁcation using logistic regression.

def LogisticRegression_fit(X_train, Y_train, lamda):
    # gradient descent method

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    # gradient of loss function L(w, b)
    def L_w_b(X, Y, w, b):
        lw = (np.array([0]*57)).reshape(57,1)
        lb = 0
        for i in range(len(Y)):
            z = np.dot(X.iloc[i,:].values.reshape(1,57),w)+b
            #z = (np.dot(X.iloc[i,:].values.reshape(1,57),w)+b)[0][0]*Y.iloc[i]
            p = sigmoid(z)
            lw = lw-(Y.iloc[i]-p)*(X.iloc[i,:].values.reshape(57,1))
            lb = lb-(Y.iloc[i]-p)
            #lw = lw-int((1-p)*Y.iloc[i])*(X.iloc[i,:].values.reshape(57,1))
            #lb = lb-int((1-p)*Y.iloc[i])
        lw = lw+lamda*w
        lb = lb+lamda*b
        return (lw, lb)
    
    learning_rate = 0.001 
    n_iter = 2000
    w = np.zeros((X_train.shape[1],1))
    b = 0
    
    for i in range(n_iter):
        gradient_w, gradient_b = L_w_b(X_train, Y_train, w, b)
        w_new = w - learning_rate * gradient_w
        b_new = b - learning_rate * gradient_b
        
        if np.linalg.norm(w_new-w, ord =1) + abs(b_new-b) < 0.01:
            break
        w = w_new
        b = b_new
        
    return (w, b)

def LogisticRegression_accuracy(X, Y, w, b):
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    prediction = sigmoid(np.dot(X, w)+b) >= 0.5
    accuracy = np.sum(prediction == Y.values.reshape(-1,1))*1/X.values.shape[0]
    return accuracy

# Use 5-fold cross validation to choose the strength of the l2 regularizer
#lamdas=np.linspace(0.01,2,num=10) # regularizer candidates
#lamda_best=[] # the best regularizer 
#for train_data in [train_data_stnd]:
#    scores=[]
#    for lamda in lamdas:
#        accs=[]
#        kf = KFold(n_splits=5)
#        for train_index, test_index in kf.split(train_data):
#            w, b = LogisticRegression_fit(train_data.iloc[train_index,0:57], train_data.iloc[train_index,57], lamda)
#            acc = LogisticRegression_accuracy(train_data.iloc[test_index,0:57], train_data.iloc[test_index,57], w, b)
#            accs.append(acc)
#        scores.append(np.mean(accs))
#    score_best=np.max(scores)
#    lamda_best.append(lamdas[scores.index(score_best)])

# The best regularizer that calculated
lamda_best = [0.23111]

# At last, report the mean error rate on the training and test sets

# For logistic regression:
train_error_lr = [] # error rate using the best regularizer for 3 trainning data
test_error_lr = [] # error rate using the best regularizer for 3 testing data

for i in range(1):
    w, b = LogisticRegression_fit(data[i][0].iloc[:,0:57], data[i][0].iloc[:,57], lamda=lamda_best[i])
    train_error_lr.append(1-LogisticRegression_accuracy(data[i][0].iloc[:,0:57], data[i][0].iloc[:,57], w, b))
    test_error_lr.append(1-LogisticRegression_accuracy(data[i][1].iloc[:,0:57], data[i][1].iloc[:,57], w, b))

print("---logistic regression---")
print("method train test")
print("stnd  ", '{:.3f} {:.3f}'.format(train_error_lr[0], test_error_lr[0]))