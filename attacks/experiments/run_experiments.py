import sys

sys.path.extend(["./"])

import datetime, warnings
from tqdm import tqdm
from src.threat.clustering.constrained_poisoning import ConstrainedAdvPoisoningGlobal

from experiments.utilities import *
from pre_post_eval import calculate_nmi_drop
from pre_post_eval import calculate_acc_drop
from pre_post_eval import kmeans_pre_post
from pre_post_eval import ward_pre_post
from pre_post_eval import spectral_pre_post

def run(
    X,
    Y,
    model,
    seeds,
    dt_range,
    s_range,
    box=(0, 1),
    lb=1 / 255,
    measure="AMI",
    mutation_rate=0.05,
    outpath="AdvMNISTPoisoning_seed/",
):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    n, m, k = X.shape
    yhat = model.fit_predict(X)
    yhat = relabel(Y, yhat)
    filename = model.to_string()

    with open(outpath + filename + ".csv", "w+") as writer:

        writer.write(
            "model;delta;perc_samples;num_samples;totOverN;"
            "ARI;AMI;NMI;"
            "l0;l2;linf;"
            "peps;neps;"
            "ARI_hat;AMI_hat;NMI_hat;seed"
        )
        writer.write("\n")
        for seed in seeds:
            for dt in tqdm(dt_range, total=len(dt_range)):
                for s in s_range:
                    set_seed(seed)
                    # n_points = int(nc * s)
                    start = datetime.datetime.now()
                    print(
                        "\n====================== Starting time:",
                        start,
                        "model:",
                        model.to_string(),
                        "==============",
                    )

                    T = ConstrainedAdvPoisoningGlobal(
                        delta=dt,
                        s=s,
                        clst_model=model,
                        lb=lb,
                        G=110,
                        domain_cons=box,
                        objective=measure,
                        mode="guided",
                        link="centroids",
                        mutation_rate=mutation_rate,
                        crossover_rate=0.85,
                        zero_rate=0.001,
                    )
                    Xadv, leps, ts_idx, direction = T.forward(X, yhat)
                    pertub =[]
                    for i in range(X.shape[0]):
                        if((X[i]-Xadv[i]).sum()!=0):
                            pertub.append(i)
                    idx = random.choices(pertub,k=int(s*len(pertub)))
                    X_adv = X.detach().clone()
                    X_adv[idx,:,:] = Xadv[idx,:,:]
                    end = datetime.datetime.now()
                    yadv = model.fit_predict(X_adv)
                    #yhat_,yadv = kmeans_pre_post(X, Xadv, 2, 42)
                    #yhat_ = relabel(Y, yhat_)
                    ARI, AMI, NMI = attack_scores(Y, yadv)
                    ARI_hat, AMI_hat, NMI_hat = attack_scores(yhat, yadv)
                    NMI_drop=calculate_nmi_drop(yhat,yadv,Y)
                    ACC_drop=calculate_acc_drop(yhat,yadv,Y)
                    l0, l2, linf = power_noise(X, X_adv)

                    eps = X_adv[ts_idx] - X[ts_idx]
                    peps = eps[eps > 0.0].mean()
                    neps = eps[eps < 0.0].mean()

                    n_points = len(ts_idx)
                    totOverN = len(ts_idx) / n

                    writer.write(
                        "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(
                            model.to_string(),
                            dt,
                            s,
                            n_points,
                            totOverN,
                            ARI,
                            AMI,
                            NMI,
                            l0,
                            l2,
                            linf,
                            peps,
                            neps,
                            ARI_hat,
                            AMI_hat,
                            NMI_hat,
                            seed,
                        )
                    )

                    print(
                        "delta:{}  s:{}  AMI_hat:{}  pmean_eps:{} AMI:{} NMI_drop:{} ACC_drop:{} dt:{} model:{};".format(
                            dt, s, AMI_hat, peps, AMI, NMI_drop, ACC_drop, dt, model.to_string()
                        )
                    )
                    print(
                        "======================== Time:",
                        end - start,
                        "=================================================",
                    )
                    writer.write("\n")
                   
        writer.close()
    return X,Y,X_adv,yadv,ts_idx


def test_robustness(
    X,
    Y,
    dt_range,
    s_range,
    models,
    path,
    lb=1 / 255,
    box=(0, 1),
    measure="AMI",
    mutation_rate=0.05,
    seeds=[42],
):
    # deltas = np.linspace(start=0.05, num=20, stop=1)
    # n_samples = np.linspace(start=0.01, num=20, stop=0.6)
    #seeds = [42]#[444, 314, 4, 410, 8089]
    models = [ClusteringWrapper3Dto2D(m) for m in models]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for model in models:
            X,Y,X_adv,yadv,ts_idx=run(
                X,
                Y,
                model,
                seeds,
                dt_range=dt_range,
                s_range=s_range,
                lb=lb,
                outpath=path,
                box=box,
                measure=measure,
                mutation_rate=mutation_rate,
            )
    return X,Y,X_adv,yadv,ts_idx