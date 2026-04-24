# -*- coding: utf-8 -*-
"""
Created on 2026/04/24 14:13:00

@File    :   model_msvr.py
@Author  :   Ronghong Ji
"""

# %%
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


"""
Multi-output Support Vector Regression
"""
# Copyright (C) 2020 Xinze Zhang, Kaishuai Xu, Siyue Yang, Yukun Bao
# <xinze@hust.edu.cn>, <xu.kaishuai@gmail.com>, <siyue_yang@hust.edu.cn>, <yukunbao@hust.edu.cn>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the Apache.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details.

class MSVR():
    def __init__(self, kernel='rbf', degree=3, gamma=None, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1):
        super(MSVR, self).__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.Beta = None
        self.NSV = None
        self.xTrain = None

    def fit(self, x, y):
        self.xTrain = x.copy()
        C = self.C
        epsi = self.epsilon
        tol = self.tol

        n_m = np.shape(x)[0]  # num of samples
        n_d = np.shape(x)[1]  # input data dimensionality
        n_k = np.shape(y)[1]  # output data dimensionality (output variables)

        # H = kernelmatrix(ker, x, x, par)
        H = pairwise_kernels(x, x, metric=self.kernel, filter_params=True,
                             degree=self.degree, gamma=self.gamma, coef0=self.coef0)

        self.Beta = np.zeros((n_m, n_k))

        #E = prediction error per output (n_m * n_k)
        E = y - np.dot(H, self.Beta)
        #RSE
        u = np.sqrt(np.sum(E**2, 1, keepdims=True))

        #RMSE
        RMSE = []
        RMSE_0 = np.sqrt(np.mean(u**2))
        RMSE.append(RMSE_0)

        #points for which prediction error is larger than epsilon
        i1 = np.where(u > epsi)[0]

        #set initial values of alphas a (n_m * 1)
        a = 2 * C * (u - epsi) / u

        #L (n_m * 1)
        L = np.zeros(u.shape)

        # we modify only entries for which  u > epsi. with the sq slack
        L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2

        #Lp is the quantity to minimize (sq norm of parameters + slacks)
        Lp = []
        BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
        Lp_0 = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
        Lp.append(Lp_0)

        eta = 1
        k = 1
        hacer = 1
        val = 1

        while(hacer):
            Beta_a = self.Beta.copy()
            E_a = E.copy()
            u_a = u.copy()
            i1_a = i1.copy()

            M1 = H[i1][:, i1] + \
                np.diagflat(1/a[i1]) + 1e-10 * np.eye(len(a[i1]))

            #compute betas
            #       sal1 = np.dot(np.linalg.pinv(M1),y[i1])  #求逆or广义逆（M-P逆）无法保证M1一定是可逆的？
            sal1 = np.dot(np.linalg.inv(M1), y[i1])

            eta = 1
            self.Beta = np.zeros(self.Beta.shape)
            self.Beta[i1] = sal1.copy()

            #error
            E = y - np.dot(H, self.Beta)
            #RSE
            u = np.sqrt(np.sum(E**2, 1)).reshape(n_m, 1)
            i1 = np.where(u >= epsi)[0]

            L = np.zeros(u.shape)
            L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2

            #%recompute the loss function
            BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
            Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
            Lp.append(Lp_k)

            #Loop where we keep alphas and modify betas
            while(Lp[k] > Lp[k-1]):
                eta = eta/10
                i1 = i1_a.copy()

                self.Beta = np.zeros(self.Beta.shape)
                #%the new betas are a combination of the current (sal1)
                #and of the previous iteration (Beta_a)
                self.Beta[i1] = eta*sal1 + (1-eta)*Beta_a[i1]

                E = y - np.dot(H, self.Beta)
                u = np.sqrt(np.sum(E**2, 1)).reshape(n_m, 1)

                i1 = np.where(u >= epsi)[0]

                L = np.zeros(u.shape)
                L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2
                BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
                Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
                Lp[k] = Lp_k

                #stopping criterion 1
                if(eta < 1e-16):
                    Lp[k] = Lp[k-1] - 1e-15
                    self.Beta = Beta_a.copy()

                    u = u_a.copy()
                    i1 = i1_a.copy()

                    hacer = 0

            #here we modify the alphas and keep betas
            a_a = a.copy()
            a = 2 * C * (u - epsi) / u

            RMSE_k = np.sqrt(np.mean(u**2))
            RMSE.append(RMSE_k)

            if((Lp[k-1]-Lp[k])/Lp[k-1] < tol):
                hacer = 0

            k = k + 1

            #stopping criterion #algorithm does not converge. (val = -1)
            if(len(i1) == 0):
                hacer = 0
                self.Beta = np.zeros(self.Beta.shape)
                val = -1

        self.NSV = len(i1)

    def predict(self, x):
        H = pairwise_kernels(x, self.xTrain, metric=self.kernel, filter_params=True,
                             degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        yPred = np.dot(H, self.Beta)
        return yPred

# 定义评估指标函数，分别计算 CL 和 V 的指标
def calculate_regression_metrics(y_true, y_pred):
    metrics = {}
    for i, column in enumerate(y_true.columns):
        metrics[column] = {
            'RMSE': np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i])),
            'R2': r2_score(y_true.iloc[:, i], y_pred[:, i]),
            'MAE': mean_absolute_error(y_true.iloc[:, i], y_pred[:, i]),
        }
    return metrics

# 十折交叉验证部分，返回 CL 和 V 的回归指标
def cross_val_regression_metrics(model, X, y, cv):
    metrics_list = {col: [] for col in y.columns}
    for train_idx, val_idx in cv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train_fold.values, y_train_fold.values)
        val_predictions = model.predict(X_val_fold.values)
        fold_metrics = calculate_regression_metrics(y_val_fold, val_predictions)
        for col in y.columns:
            metrics_list[col].append(fold_metrics[col])
    
    # 计算平均值和标准差
    final_metrics = {}
    for col in y.columns:
        final_metrics[col] = {
            'RMSE': (np.mean([m['RMSE'] for m in metrics_list[col]]), np.std([m['RMSE'] for m in metrics_list[col]])),
            'R2': (np.mean([m['R2'] for m in metrics_list[col]]), np.std([m['R2'] for m in metrics_list[col]])),
            'MAE': (np.mean([m['MAE'] for m in metrics_list[col]]), np.std([m['MAE'] for m in metrics_list[col]])),
        }
    return final_metrics

# %%
seed = 486

# 加载数据
df = pd.read_excel(r'../data/df_rf_log_allfeature.xlsx')

# 特征列，去除 'CL' 和 'V' 列
X = df.drop(columns=['CL', 'V'])
y = df[['CL', 'V']]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 网格搜索超参数
# init里规定了默认参数
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # 'poly'
    'C': [0.001, 0.003, 0.01, 0.1, 1.0], # 0.003,
    'epsilon': [0.01, 0.1, 0.3, 0.5, 0.7, 1], # 0.1
    'coef0': [-1, 0, 1], #[-1, 0, 1]
    'degree': [4, 5, 6, 7, 8], # 3,
}

# 自定义的 MSVR 包装器以便与 GridSearchCV 兼容
class MSVRWrapper:
    def __init__(self, kernel='poly', C=1.0, epsilon=0.1, degree=3, coef0=0):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.degree = degree
        self.coef0 = coef0
        self.model = MSVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, degree=self.degree, coef0=self.coef0)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"kernel": self.kernel, "C": self.C, "epsilon": self.epsilon, "degree": self.degree, "coef0": self.coef0}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            setattr(self.model, key, value)  # Update the model's parameters as well
        return self

# 创建 GridSearchCV 对象
msvr_wrapper = MSVRWrapper()
grid_search = GridSearchCV(msvr_wrapper, param_grid, cv=KFold(n_splits=10, shuffle=True, random_state=seed),
                           scoring='r2', n_jobs=-1)

# 执行网格搜索
grid_search.fit(X_train.values, y_train.values)

# 获取最佳参数
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 使用最佳参数训练最终模型
best_msvr_model = MSVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon'], degree=best_params['degree'], coef0=best_params['coef0'])
best_msvr_model.fit(X_train.values, y_train.values)

# 计算十折交叉验证的回归指标
msvr_cv_metrics = cross_val_regression_metrics(best_msvr_model, X_train, y_train, KFold(n_splits=10, shuffle=True, random_state=seed))

# 进行预测
msvr_y_pred = best_msvr_model.predict(X_test.values)

# 计算测试集上的回归指标
msvr_test_metrics = calculate_regression_metrics(y_test, msvr_y_pred)

# 合并十折交叉验证和测试集的回归指标
metrics = {}
for col in y_train.columns:
    metrics[f'Ten_fold_CV_RMSE_{col}'] = f"{msvr_cv_metrics[col]['RMSE'][0]:.2f} ± {msvr_cv_metrics[col]['RMSE'][1]:.2f}"
    metrics[f'Ten_fold_CV_R2_{col}'] = f"{msvr_cv_metrics[col]['R2'][0]:.2f} ± {msvr_cv_metrics[col]['R2'][1]:.2f}"
    metrics[f'Ten_fold_CV_MAE_{col}'] = f"{msvr_cv_metrics[col]['MAE'][0]:.2f} ± {msvr_cv_metrics[col]['MAE'][1]:.2f}"

# 添加测试集指标
for col in y_test.columns:
    metrics[f'Test_RMSE_{col}'] = f"{msvr_test_metrics[col]['RMSE']:.2f}"
    metrics[f'Test_R2_{col}'] = f"{msvr_test_metrics[col]['R2']:.2f}"
    metrics[f'Test_MAE_{col}'] = f"{msvr_test_metrics[col]['MAE']:.2f}"

# 保存到 XLSX 文件
metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_excel(r'../output/ML输出/MSVR_GridSearch_Regression.xlsx', index=False)
metrics_df


# %%
best_params

# %%
# 保存最佳参数到 Excel 文件
best_params_df = pd.DataFrame(best_params, index=[0])
best_params_df.to_excel(r'../output/param/MSVR_best_params.xlsx', index=False)
best_params_df

# %%
import pandas as pd

# 如果 y_test 是 pandas DataFrame 或 Series，可以重置索引以避免问题
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)  # 如果是 Series 或 DataFrame
msvr_y_pred_df = pd.DataFrame(msvr_y_pred).reset_index(drop=True)  # 同样处理预测值

# 确保列名称正确
y_test_df.columns = ['True Values cl', 'True Values v']
msvr_y_pred_df.columns = ['Predicted Values cl', 'Predicted Values v']

# 将两者合并到一个 DataFrame 中
output_df = pd.concat([y_test_df, msvr_y_pred_df], axis=1)

# 保存到 Excel 文件
output_df.to_excel(r'../output/pred/MSVR_predictions.xlsx', index=False)

print("ok!")


# %%
import numpy as np
import pandas as pd

def calculate_accuracy_within_percentage(y_true, y_pred, percentage):
    # 计算范围
    lower_bound = y_true * (1 - percentage / 100)
    upper_bound = y_true * (1 + percentage / 100)
    
    # 检查预测值是否在范围内
    within_range = (y_pred >= lower_bound) & (y_pred <= upper_bound)
    
    return np.mean(within_range)  # 返回在范围内的比例

# 计算在不同百分比误差范围内的准确率
accuracies = {}
for pct in [10, 20, 30, 40, 50]:
    accuracies[f"Accuracy_within_{pct}%_CL"] = calculate_accuracy_within_percentage(y_test['CL'].values, msvr_y_pred[:, 0], pct)
    accuracies[f"Accuracy_within_{pct}%_V"] = calculate_accuracy_within_percentage(y_test['V'].values, msvr_y_pred[:, 1], pct)

# 创建一个新的 DataFrame 用于存储准确率
accuracy_df = pd.DataFrame({
    '±10%': [accuracies['Accuracy_within_10%_CL'], accuracies['Accuracy_within_10%_V']],
    '±20%': [accuracies['Accuracy_within_20%_CL'], accuracies['Accuracy_within_20%_V']],
    '±30%': [accuracies['Accuracy_within_30%_CL'], accuracies['Accuracy_within_30%_V']],
    '±40%': [accuracies['Accuracy_within_40%_CL'], accuracies['Accuracy_within_40%_V']],
    '±50%': [accuracies['Accuracy_within_50%_CL'], accuracies['Accuracy_within_50%_V']],
}, index=['CL', 'V'])

# 显示准确率 DataFrame
print(accuracy_df)

# 保存到 XLSX 文件
accuracy_df.to_excel(r'../output/accuracy/MSVR_acc.xlsx', index=False)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 绘制 CL 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test['CL'], msvr_y_pred[:, 0], color='black', s=15)
plt.plot([y_test['CL'].min(), y_test['CL'].max()], [y_test['CL'].min(), y_test['CL'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('MSVR Scatter Plot for CL', fontsize=20)

# 计算 R² 并添加到图中
r2_cl = r2_score(y_test['CL'], msvr_y_pred[:, 0])
plt.text(0.8, 0.3, f"R² = {r2_cl:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# 绘制 V 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test['V'], msvr_y_pred[:, 1], color='black', s=15)
plt.plot([y_test['V'].min(), y_test['V'].max()], [y_test['V'].min(), y_test['V'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('MSVR Scatter Plot for V', fontsize=20)

# 计算 R² 并添加到图中
r2_v = r2_score(y_test['V'], msvr_y_pred[:, 1])
plt.text(0.8, 0.3, f"R² = {r2_v:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# %%
import matplotlib.pyplot as plt
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 绘制拟合散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test['CL'], msvr_y_pred[:, 0], color='black', s=15)
plt.plot([y_test['CL'].min(), y_test['CL'].max()], [y_test['CL'].min(), y_test['CL'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('MSVR Fitted Scatter Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test['V'], msvr_y_pred[:, 1], color='black', s=15)
plt.plot([y_test['V'].min(), y_test['V'].max()], [y_test['V'].min(), y_test['V'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('MSVR Fitted Scatter Plot for V', fontsize=20)
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
residuals_cl = y_test['CL'] - msvr_y_pred[:, 0]
plt.scatter(y_test['CL'], residuals_cl, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('MSVR Residual Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
residuals_v = y_test['V'] - msvr_y_pred[:, 1]
plt.scatter(y_test['V'], residuals_v, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('MSVR Residual Plot for V', fontsize=20)
plt.show()


# %%
import shap
import matplotlib.pyplot as plt
import numpy as np

# 使用 SHAP 的 KernelExplainer 来解释 MSVR 模型
explainer = shap.KernelExplainer(best_msvr_model.predict, X_test.values)

# 计算 SHAP 值，返回形状为 (样本数, 特征数, 输出变量数)
shap_values = explainer.shap_values(X_test.values)

# 为 CL 和 V 分别绘制 SHAP summary plot
output_columns = ['CL', 'V']

for i, col in enumerate(output_columns):      
    shap_values_output = shap_values[:, :, i]  
    plt.figure()
    shap.summary_plot(shap_values_output, X_test, feature_names=X_test.columns, show=False)
    plt.xlabel("SHAP value")
    plt.title(f'SHAP Summary Plot for {output_columns}')
    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import shap
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

X_test_exp = X_test.copy()

# 获取数值型特征（排除 'cts'）
numeric_columns = ['Weight', 'eGFR']

# 对数值型特征进行指数变换
X_test_exp[numeric_columns] = np.exp(X_test[numeric_columns])

# 分别计算 CL 和 V 的 SHAP 值并绘制依赖图
for i, column in enumerate(y.columns):
    print(f"Calculating SHAP values for {column}...")
    explainer = shap.KernelExplainer(best_msvr_model.predict, X_test.values)
    shap_values = explainer.shap_values(X_test.values)
    shap_values_output = shap_values[:, :, i]

    # 为每个特征绘制当前目标变量的 SHAP 依赖图
    for j, feature in enumerate(X_test_exp.columns):
        plt.figure()
        shap.dependence_plot(feature, shap_values_output, X_test_exp, show=False)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.title(f'SHAP Dependence Plot for {feature} ({column})', fontsize=16)
        plt.xlabel(f"{feature}", fontsize=14)
        plt.ylabel("SHAP value", fontsize=14)
        plt.tight_layout()
        plt.show()



