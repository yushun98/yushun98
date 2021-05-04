import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
#LM22计算得
mean=4.577141586995454      #全局期望
sd=1.8499410627124944       #全局标准差

#基因表达谱模拟，N_gene：基因数 K：细胞类数   n_sample：抽样数          diffrate：差异比率
def cells_gene_imitate(N_gene,K,n_sample,diffrate=0.8,mean=mean,sd=sd):
    X=np.random.normal(mean,sd,(N_gene,K,n_sample))
    dirichlet= np.random.dirichlet([1]*K,1)[0]*int(N_gene*diffrate)
    start=0
    for i in range(K):
        for j in range(int(round(dirichlet[i]))):
            j=j+start
            mean_power=np.random.chisquare(10)
            sd_power=np.random.chisquare(3)
            X[j,i]=np.random.normal(mean*mean_power,sd*sd_power,n_sample)
        start=start+int(round(dirichlet[i]))
    return X

# 什么情况好，什么情况一样好，横轴可能的原因，纵轴准确度，随原因变化，有差异位点多少增加，变化趋势，其他但可能原因，方差大小，作图。
# 相关性，误差，之类


#更新beta
def update(beta,X_mean,sigma_data,Y):
    W_data=np.dot(np.dot(sigma_data,np.diag(beta)),beta)

    mod_wls = sm.WLS(Y, X_mean, weights=1/W_data)
    res_wls = mod_wls.fit()

    beta=res_wls.params.tolist()
    return beta

#估计beta
def beta_estimate(n,n_beta,X_mean,sigma_data,Y):
    K=n_beta
    beta=np.array([1/K]*K)
    params=[beta]
    H=np.array(np.ones(K))
    for i in range(n):
        beta=update(beta,X_mean,sigma_data,Y) #加权最小二乘
        W_data=np.dot(np.dot(sigma_data,np.diag(beta)),beta)
        W_adverse=np.diag(1/W_data)
        XWX_adverse=np.linalg.inv(np.dot(np.dot(X_mean.T,W_adverse),X_mean))
        #beta=beta+np.dot(XWX_adverse,H)/np.dot(np.dot(H,XWX_adverse),H)*(1-np.dot(H,beta))#带约束:np.dot(H,beta)==1
        params.append(beta)

    return beta

#产生样本数据
def sampledata_create(N_gene,n_beta,n_sample):
    cells_gene = cells_gene_imitate(N_gene, n_beta, n_sample=n_sample + 1, mean=mean, sd=sd)
    cells_gene_x = cells_gene[:, :, 0:n_sample]
    cells_gene_to_y = np.squeeze(cells_gene[:, :, n_sample])
    nz=n_sample  #样本数
    X_mean=np.mean(cells_gene_x,2)
    sigma_data=(np.var(cells_gene_x,2)*nz/(nz-1))
    return X_mean,sigma_data,cells_gene_to_y

#多次比例混合 求估计
# def guina(cishu,N_gene,n_beta,cells_gene_to_y,X_mean,sigma_data):
#     t1 = 0 ; t2 = 0 ; x1 = 0 ; x2 = 0 ; x3 = 0
#     for i in range(cishu):
#         beta = list(np.squeeze(np.random.dirichlet([1]*n_beta,1))) # 模拟beta 生成Y
#         y = np.dot(cells_gene_to_y,beta)
#         Y=np.squeeze(y)
#
#         mod_ols = sm.OLS(Y, X_mean)
#         res_ols = mod_ols.fit()
#         beta_ols=res_ols.params.tolist()
#
#         beta_wls=beta_estimate(5,n_beta,X_mean=X_mean,sigma_data=sigma_data,Y=Y)
#
#         beta_ols=np.array(beta_ols)/sum(beta_ols)
#         beta_wls=np.array(beta_wls)
#
#         loss1_ols = np.sum(np.abs(beta-beta_ols)) #相对误差
#         loss1_wls = np.sum(np.abs(beta-beta_wls))
#
#         loss2_ols = np.dot((beta-beta_ols),(beta-beta_ols).T)
#         loss2_wls = np.dot((beta-beta_wls),(beta-beta_wls).T)
#
#         res = [[0]*(n_beta+2), [0]*(n_beta+2), [0]*(n_beta+2), [""]*(n_beta+2)]
#
#         res[0][:n_beta] = beta
#         res[1][:n_beta] = beta_ols
#         res[2][:n_beta] = beta_wls
#         res[1][-1] = loss2_ols
#         res[2][-1] = loss2_wls
#         res[1][-2] = loss1_ols
#         res[2][-2] = loss1_wls
#
#         if loss2_ols < loss2_wls:
#             res[0][-1] = "False"
#         else:
#             res[0][-1] = "True"
#             t2 += 1
#
#         if loss1_ols < loss1_wls:
#             res[0][-2] = "False"
#         elif loss1_ols/loss1_wls > 2:
#             res[0][-2] = "*"*3
#             t1 += 1
#             x3 += 1
#         elif loss1_ols/loss1_wls > 1.5:
#             res[0][-2] = "*"*2
#             t1 += 1
#             x2 += 1
#         else:
#             res[0][-2] = "*"*1
#             t1 += 1
#             x1 += 1
#
#         if i == 0:
#             a = res
#         else:
#             a = a + res
#
#     result = pd.DataFrame(a)
#     result.iloc[-1,-1] = t2/cishu
#     result.iloc[-1,-2] = t1/cishu
#     result.iloc[-1,-4] = x3/cishu
#     result.iloc[-1,-3] = x2/cishu
#     result.iloc[-1,-5] = "***、**、l1、欧式"
#     col = [""]*(n_beta+2)
#     for i in range(n_beta):
#         col[i] = "Beta {}".format(i+1)
#     col[-1] = "loss"
#     result.columns = col
#     return result

#多次比例混合 求估计

def guina(cishu,N_gene,n_beta):
    t1 = 0 ; t2 = 0 ; x1 = 0 ; x2 = 0 ; x3 = 0
    for i in range(cishu):
        X_mean, sigma_data, cells_gene_to_y = sampledata_create(N_gene=N_gene, n_beta=n_beta, n_sample=20)
        beta = list(np.squeeze(np.random.dirichlet([1]*n_beta,1))) # 模拟beta 生成Y
        y = np.dot(cells_gene_to_y,beta)
        Y=np.squeeze(y)

        mod_ols = sm.OLS(Y, X_mean)
        res_ols = mod_ols.fit()
        beta_ols=res_ols.params.tolist()

        beta_wls=beta_estimate(5,n_beta,X_mean=X_mean,sigma_data=sigma_data,Y=Y)

        beta_ols=np.array(beta_ols)#/sum(beta_ols)
        beta_wls=np.array(beta_wls)

        loss1_ols = np.sum(np.abs(beta-beta_ols)) #相对误差
        loss1_wls = np.sum(np.abs(beta-beta_wls))

        loss2_ols = np.dot((beta-beta_ols),(beta-beta_ols).T)
        loss2_wls = np.dot((beta-beta_wls),(beta-beta_wls).T)

        res = [[0]*(n_beta+2), [0]*(n_beta+2), [0]*(n_beta+2), [""]*(n_beta+2)]

        res[0][:n_beta] = beta
        res[1][:n_beta] = beta_ols
        res[2][:n_beta] = beta_wls
        res[1][-1] = loss2_ols
        res[2][-1] = loss2_wls
        res[1][-2] = loss1_ols
        res[2][-2] = loss1_wls

        if loss2_ols < loss2_wls:
            res[0][-1] = "False"
        else:
            res[0][-1] = "True"
            t2 += 1

        if loss1_ols < loss1_wls:
            res[0][-2] = "False"
        elif loss1_ols/loss1_wls > 2:
            res[0][-2] = "*"*3
            t1 += 1
            x3 += 1
        elif loss1_ols/loss1_wls > 1.5:
            res[0][-2] = "*"*2
            t1 += 1
            x2 += 1
        else:
            res[0][-2] = "*"*1
            t1 += 1
            x1 += 1

        if i == 0:
            a = res
        else:
            a = a + res

    result = pd.DataFrame(a)
    result.iloc[-1,-1] = t2/cishu
    result.iloc[-1,-2] = t1/cishu
    result.iloc[-1,-4] = x3/cishu
    result.iloc[-1,-3] = x2/cishu
    result.iloc[-1,-5] = "***、**、l1、欧式"
    col = [""]*(n_beta+2)
    for i in range(n_beta):
        col[i] = "Beta {}".format(i+1)
    col[-1] = "loss"
    result.columns = col
    return result

def main(cishu ,N_gene,n_beta):
    print("l1为相对误差的指标，三*说明很好，二*次之，true仅仅说明好一点")
    result=guina(cishu,N_gene,n_beta)
    return result

# -*- coding: utf-8 -*-
if __name__ == '__main__':
    try:
        result=main(cishu=1000,N_gene=600,n_beta=7)
        print(result)
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt me! ;-) See you!\n")
        sys.exit(0)