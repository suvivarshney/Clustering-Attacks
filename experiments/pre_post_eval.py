from gan_attack import GAN_attack
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
import pandas as pd

def acc(y_true, y_pred):
    y_true, y_pred = pd.Series(list(y_true)), pd.Series(list(y_pred))
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def get_Y(labels, n_samples, n_clusters):
    Y = np.zeros([n_samples, n_clusters])
    for idx, label in enumerate(labels):
        Y[idx, label] = 1
    return Y

def calculate_spillover_metric(labels_org, labels_adv, n_samples, n_clusters): #The higher the value the more efficient the attack
    Y_org, Y_adv = get_Y(labels_org, n_samples, n_clusters), get_Y(labels_adv, n_samples, n_clusters)
    dist = np.linalg.norm(np.matmul(Y_org,Y_org.T) - np.matmul(Y_adv,Y_adv.T))
    return dist

def calculate_nmi_drop(labels_org, labels_adv, ground_truth): #The higher the difference the better the attack
    NMI_org = normalized_mutual_info_score(labels_org, ground_truth, average_method='arithmetic')
    NMI_adv = normalized_mutual_info_score(labels_adv, ground_truth, average_method='arithmetic')
    return NMI_org - NMI_adv

def calculate_acc_drop(labels_org, labels_adv, ground_truth): #The higher the difference the better the attack
    ACC_org = acc(labels_org, ground_truth)
    ACC_adv = acc(labels_adv, ground_truth)
    return ACC_org - ACC_adv

def kmeans_pre_post(X_org, X_adv, n_clusters, random_state):
    kme_org = KMeans(n_clusters=n_clusters, random_state=random_state)
    kme_org.fit(X_org)
    labels_org = kme_org.labels_
    kme_adv = KMeans(n_clusters=n_clusters, random_state=random_state)
    kme_adv.fit(X_adv)
    labels_adv = kme_adv.labels_
    return labels_org, labels_adv

def ward_pre_post(X_org, X_adv, n_clusters):
    ward_org = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    ward_org.fit(X_org)
    labels_org = ward_org.labels_
    ward_adv = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    ward_adv.fit(X_adv)
    labels_adv = ward_adv.labels_
    return labels_org, labels_adv

def spectral_pre_post(X_org, X_adv, n_clusters, random_state):
    sc_org = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
    sc_org.fit(X_org)
    labels_org = sc_org.labels_
    sc_adv = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
    sc_adv.fit(X_adv)
    labels_adv = sc_adv.labels_
    return labels_org, labels_adv


if __name__ == "__main__":
    attack = GAN_attack('MNIST', 0.1, 'cuda', 42) 

    _ = attack.generate_samples(length=2000, clamp=0.1)
 
    X_adv = attack.xadv
    print(X_adv.shape)
 
    X_org = attack.original
    print(X_org.shape)

    ground_truth = attack.original_labels
    print(ground_truth.shape)

    n_samples = X_org.shape[0]
    n_clusters = len(np.unique(ground_truth))

    print("###############################")

    print("K-Means Attack Results ->")
    labels_org, labels_adv = kmeans_pre_post(X_org, X_adv, n_clusters, 42)
    print("Spillover: {}".format(calculate_spillover_metric(labels_org, labels_adv, n_samples, n_clusters)))
    print("NMI Decrease: {}".format(calculate_nmi_drop(labels_org, labels_adv, ground_truth))) # Can also use ACC and ARI drop functions similarly
    print("###############################")


    print("Ward's Attack Results ->")
    labels_org, labels_adv = ward_pre_post(X_org, X_adv, n_clusters)
    print("Spillover: {}".format(calculate_spillover_metric(labels_org, labels_adv, n_samples, n_clusters)))
    print("NMI Decrease: {}".format(calculate_nmi_drop(labels_org, labels_adv, ground_truth))) # Can also use ACC and ARI drop functions similarly
    print("###############################")


    # Spectral clustering is giving memory errors, as dataset might be too large. Can revisit later.
    #print("Spectral Clustering Attack Results ->") 
    #labels_org, labels_adv = spectral_pre_post(X_org, X_adv, n_clusters, 42)
    #print("Spillover: {}".format(calculate_spillover_metric(labels_org, labels_adv, n_samples, n_clusters)))
    #print("NMI Decrease: {}".format(calculate_nmi_drop(labels_org, labels_adv, ground_truth))) # Can also use ACC and ARI drop functions similarly
    #print("###############################")
