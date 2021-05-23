
import numpy as np
import matplotlib.pyplot as plt
import math
import collections
import time
 
def multiGaussian(x,mu,sigma):
    return 1/((2*np.pi)*pow(np.linalg.det(sigma),0.5))*np.exp(-0.5*(x-mu).dot(np.linalg.pinv(sigma)).dot((x-mu).T))
 
def computeGamma(X,mu,sigma,alpha,multiGaussian):
    n_samples=X.shape[0]
    n_clusters=len(alpha)
    gamma=np.zeros((n_samples,n_clusters))
    p=np.zeros(n_clusters)
    g=np.zeros(n_clusters)
    for i in range(n_samples):
        for j in range(n_clusters):
            p[j]=multiGaussian(X[i],mu[j],sigma[j])
            g[j]=alpha[j]*p[j]
        for k in range(n_clusters):
            gamma[i,k]=g[k]/np.sum(g)
    return gamma
 
class MyGMM():
    def __init__(self,n_clusters,ITER=50):
        self.n_clusters=n_clusters
        self.ITER=ITER
        self.mu=0
        self.sigma=0
        self.alpha=0
      
    def fit(self,data):
        start = time.time()
        n_samples=data.shape[0]
        n_features=data.shape[1]

        alpha=np.ones(self.n_clusters)/self.n_clusters
        
        mu=data[np.random.choice(range(n_samples),self.n_clusters)]
        
        sigma=np.full((self.n_clusters,n_features,n_features),np.diag(np.full(n_features,0.1)))
        for i in range(self.ITER):
            gamma=computeGamma(data,mu,sigma,alpha,multiGaussian)
            alpha=np.sum(gamma,axis=0)/n_samples
            for i in range(self.n_clusters):
                mu[i]=np.sum(data*gamma[:,i].reshape((n_samples,1)),axis=0)/np.sum(gamma,axis=0)[i]
                sigma[i]=0
                for j in range(n_samples):
                    sigma[i]+=(data[j].reshape((1,n_features))-mu[i]).T.dot((data[j]-mu[i]).reshape((1,n_features)))*gamma[j,i]
                sigma[i]=sigma[i]/np.sum(gamma,axis=0)[i]
        self.mu=mu
        self.sigma=sigma
        self.alpha=alpha
        end = time.time()
        return end-start
        
    def predict(self,data):
        start = time.time()
        pred=computeGamma(data,self.mu,self.sigma,self.alpha,multiGaussian)
        cluster_results=np.argmax(pred,axis=1)
        end = time.time()
        return cluster_results, start-end

def reuslt_tuning(labels, results):
    orders = []
    for i in range(len(results)):
        results[i] += 10
    for i in range(3):
        clusters_labels = [0,0,0]
        for j in range(i*70,(i+1)*70):
            if (results[j] == 10): clusters_labels[0]+=1
            elif(results[j] == 11): clusters_labels[1]+=1
            else: clusters_labels[2]+=1
        most_label = clusters_labels.index(max(clusters_labels))
        if (most_label == 0): orders.append(10)
        elif (most_label == 1): orders.append(11)
        else: orders.append(12)
    for i in range(len(results)):
        if (results[i] == orders[0]): results[i] = 1
        elif (results[i] == orders[1]): results[i] = 2
        else: results[i] = 3
    return results

def purity(result, label):
    # 计算纯度
    total_num = len(label)
    cluster_counter = collections.Counter(result)
    original_counter = collections.Counter(label)

    t = []
    for k in cluster_counter:
        p_k = []
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j: # 求交集
                    count += 1
            p_k.append(count)
        temp_t = max(p_k)
        t.append(temp_t)
    
    return sum(t)/total_num

def NMI(result, label):
    # 标准化互信息
    total_num = len(label)
    cluster_counter = collections.Counter(result)
    original_counter = collections.Counter(label)
    
    # 计算互信息量
    MI = 0
    eps = 1.4e-45 # 取一个很小的值来避免log 0
    for k in cluster_counter:
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j:
                    count += 1
            p_k = 1.0*cluster_counter[k] / total_num
            p_j = 1.0*original_counter[j] / total_num
            p_kj = 1.0*count / total_num
            MI += p_kj * math.log(p_kj /(p_k * p_j) + eps, 2)

    # 标准化互信息量
    H_k = 0
    for k in cluster_counter:
        H_k -= (1.0*cluster_counter[k] / total_num) * math.log(1.0*cluster_counter[k] / total_num+eps, 2)
    H_j = 0
    for j in original_counter:
        H_j -= (1.0*original_counter[j] / total_num) * math.log(1.0*original_counter[j] / total_num+eps, 2)
        
    return 2.0 * MI / (H_k + H_j)

def rand_index(result, label):
    total_num = len(label)
    TP = TN = FP = FN = 0
    for i in range(total_num):
        for j in range(i + 1, total_num):
            if label[i] == label[j] and result[i] == result[j]:
                TP += 1
            elif label[i] != label[j] and result[i] != result[j]:
                TN += 1
            elif label[i] != label[j] and result[i] == result[j]:
                FP += 1
            elif label[i] == label[j] and result[i] != result[j]:
                FN += 1
    return 1.0*(TP + TN)/(TP + FP + FN + TN)

def model_evaluation(results, labels):
    return [purity(results,labels), NMI(results,labels), rand_index(results,labels)]

if __name__ == '__main__':
    # data loading
    dataset = []
    labels = []
    data_file = open('seeds_dataset.txt','r')
    lines = data_file.readlines()
    for i in lines:
        row = i.split()
        row = [float(j) for j in row]
        data = row[:-1]
        label = int(row[-1])
        dataset.append(data)
        labels.append(label) # 70 for 1, 70 for 2, 70 for 3
    dataset = np.array(dataset)

    purity_all = []
    NMI_all = []
    RI_all = []
    iter_nums = []
    running_time = []
    for i in range(20):
        GMM = MyGMM(3)
        current_running_time = GMM.fit(dataset)
        current_results, predicting_time = GMM.predict(dataset)
        current_running_time = predicting_time+current_running_time
        reuslt_tuning(label,current_results)
        [current_purity,current_NMI,current_RI] = model_evaluation(current_results,labels)
        purity_all.append(current_purity)
        NMI_all.append(current_NMI)
        RI_all.append(current_RI)
        running_time.append(current_running_time)
        print([current_purity,current_NMI,current_RI])

    # plot the graph

    plt.subplot(221)
    plt.plot([i for i in range(1,len(iter_nums)+1)],iter_nums,label = 'Number of Iterations')
    plt.legend()
    plt.title("Iterations")
    plt.ylabel('Number')
    plt.xlabel('The ith Clustering')

    plt.subplot(222)
    plt.plot([i for i in range(1,len(running_time)+1)],running_time,label = 'Time Until Convergence')
    plt.legend()
    plt.title("Running Time")
    plt.ylabel('Time (s)')
    plt.xlabel('The ith Clustering')

    plt.subplot(223)
    plt.plot([i for i in range(1,len(purity_all)+1)],purity_all,label = 'Purity',color='red')
    plt.plot([i for i in range(1,len(NMI_all)+1)],NMI_all,label = 'NMI', color='blue')
    plt.plot([i for i in range(1,len(RI_all)+1)],RI_all,label = 'Rand Index',color='black')
    plt.legend()
    plt.title("Performances")
    plt.ylabel('Value')
    plt.xlabel('The ith Clustering')

    plt.show()
