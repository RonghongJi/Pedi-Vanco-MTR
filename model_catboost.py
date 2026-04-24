# -*- coding: utf-8 -*-
"""
Created on 2026/04/24 13:50:50

@File    :   model_catboost.py
@Author  :   Ronghong Ji
"""

# %%
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
import numpy as np

seed = 486

# 加载数据
df = pd.read_excel(r'../data/df_rf_log_allfeature.xlsx')

# 特征列，去除 'CL' 和 'V' 列
X = df.drop(columns=['CL', 'V'])
y = df[['CL', 'V']]

categarical_cols =  ['CTS', 'Gender', 'ICU', 'PMA_class']

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 逐个优化参数
def grid_search_cv_single_param(param_grid, X, y, fixed_params):
    model = CatBoostRegressor(random_seed=seed, **fixed_params, verbose=0)
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1)
    grid_search.fit(X, y['CL'])  # 仅使用 'CL' 作为评分
    return grid_search.best_params_

param_grids = {
    'learning_rate': [0.005, 0.008, 0.01, 0.03, 0.05, 0.1, 0.3],
    'depth': [3, 4, 5, 6, 7],
    'l2_leaf_reg': [3],
    'subsample': [0.8, 0.9, 1.0],
    'rsm': [0.8, 0.9, 1.0],
}

# 初始化最佳参数字典
best_params = {}

# 逐个优化参数
for param, grid in param_grids.items():
    print(f"Optimizing {param}...")
    param_grid = {param: grid}
    best_param = grid_search_cv_single_param(param_grid, X_train, y_train, fixed_params={})
    best_params.update(best_param)
    print(f"Best {param}: {best_param}")

# 使用逐个优化后的最佳参数训练模型
catboost_best_model = CatBoostRegressor(**best_params, 
                                        random_seed=seed, 
                                        verbose=0,
                                        iterations=5000,
                                        early_stopping_rounds=100)
multi_output_model = MultiOutputRegressor(catboost_best_model)

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
        model.fit(X_train_fold, y_train_fold, cat_features=categarical_cols)
        val_predictions = model.predict(X_val_fold)
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

# 定义十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# 计算十折交叉验证的回归指标
catboost_cv_metrics = cross_val_regression_metrics(multi_output_model, X_train, y_train, kf)

# 训练最终模型并预测
multi_output_model.fit(X_train, y_train, cat_features=categarical_cols)
catboost_y_pred = multi_output_model.predict(X_test)

# 计算测试集上的回归指标
catboost_test_metrics = calculate_regression_metrics(y_test, catboost_y_pred)

# 合并十折交叉验证和测试集的回归指标
metrics = {}
for col in y_train.columns:
    metrics[f'Ten_fold_CV_RMSE_{col}'] = f"{catboost_cv_metrics[col]['RMSE'][0]:.2f} ± {catboost_cv_metrics[col]['RMSE'][1]:.2f}"
    metrics[f'Ten_fold_CV_R2_{col}'] = f"{catboost_cv_metrics[col]['R2'][0]:.2f} ± {catboost_cv_metrics[col]['R2'][1]:.2f}"
    metrics[f'Ten_fold_CV_MAE_{col}'] = f"{catboost_cv_metrics[col]['MAE'][0]:.2f} ± {catboost_cv_metrics[col]['MAE'][1]:.2f}"

# 添加测试集指标
for col in y_test.columns:
    metrics[f'Test_RMSE_{col}'] = f"{catboost_test_metrics[col]['RMSE']:.2f}"
    metrics[f'Test_R2_{col}'] = f"{catboost_test_metrics[col]['R2']:.2f}"
    metrics[f'Test_MAE_{col}'] = f"{catboost_test_metrics[col]['MAE']:.2f}"

# 保存到 XLSX 文件
metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_excel(r'../output/ML输出/CatBoost_GridSearch_Regression.xlsx', index=False)
metrics_df

# %%
import pandas as pd

# 如果 y_test 是 pandas DataFrame 或 Series，可以重置索引以避免问题
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)  # 如果是 Series 或 DataFrame
catboost_y_pred_df = pd.DataFrame(catboost_y_pred).reset_index(drop=True)  # 同样处理预测值

# 确保列名称正确
y_test_df.columns = ['True Values cl', 'True Values v']
catboost_y_pred_df.columns = ['Predicted Values cl', 'Predicted Values v']

# 将两者合并到一个 DataFrame 中
output_df = pd.concat([y_test_df, catboost_y_pred_df], axis=1)

# 保存到 Excel 文件
output_df.to_excel(r'../output/pred/CatBoost_predictions.xlsx', index=False)

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
    accuracies[f"Accuracy_within_{pct}%_CL"] = calculate_accuracy_within_percentage(y_test['CL'].values, catboost_y_pred[:, 0], pct)
    accuracies[f"Accuracy_within_{pct}%_V"] = calculate_accuracy_within_percentage(y_test['V'].values, catboost_y_pred[:, 1], pct)

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
accuracy_df.to_excel(r'../output/accuracy/CatBoost_acc.xlsx', index=False)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 绘制 CL 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test['CL'], catboost_y_pred[:, 0], color='black', s=15)
plt.plot([y_test['CL'].min(), y_test['CL'].max()], [y_test['CL'].min(), y_test['CL'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('CatBoost Scatter Plot for CL', fontsize=20)

# 计算 R² 并添加到图中
r2_cl = r2_score(y_test['CL'], catboost_y_pred[:, 0])
plt.text(0.8, 0.3, f"R² = {r2_cl:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# 绘制 V 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test['V'], catboost_y_pred[:, 1], color='black', s=15)
plt.plot([y_test['V'].min(), y_test['V'].max()], [y_test['V'].min(), y_test['V'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('CatBoost Scatter Plot for V', fontsize=20)

# 计算 R² 并添加到图中
r2_v = r2_score(y_test['V'], catboost_y_pred[:, 1])
plt.text(0.8, 0.3, f"R² = {r2_v:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# %%
import matplotlib.pyplot as plt
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 绘制拟合散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test['CL'], catboost_y_pred[:, 0], color='black', s=15)
plt.plot([y_test['CL'].min(), y_test['CL'].max()], [y_test['CL'].min(), y_test['CL'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('CatBoost Fitted Scatter Plot for CL', fontsize=20)
plt.savefig("../pic/CatBoost_Fitted_Scatter_Plot_CL.tif", dpi=1000)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test['V'], catboost_y_pred[:, 1], color='black', s=15)
plt.plot([y_test['V'].min(), y_test['V'].max()], [y_test['V'].min(), y_test['V'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('CatBoost Fitted Scatter Plot for V', fontsize=20)
plt.savefig("../pic/CatBoost_Fitted_Scatter_Plot_V.tif", dpi=1000)

plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
residuals_cl = y_test['CL'] - catboost_y_pred[:, 0]
plt.scatter(y_test['CL'], residuals_cl, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('CatBoost Residual Plot for CL', fontsize=20)
plt.savefig("../pic/CatBoost_Residual_Plot_CL.tif", dpi=1000)

plt.show()

plt.figure(figsize=(10, 6))
residuals_v = y_test['V'] - catboost_y_pred[:, 1]
plt.scatter(y_test['V'], residuals_v, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('CatBoost Residual Plot for V', fontsize=20)
plt.savefig("../pic/CatBoost_Residual_Plot_V.tif", dpi=1000)

plt.show()


# %% [markdown]
# 变量重要性得分

# %%
import pandas as pd

# 获取每个特征的重要性得分
cl_importances = multi_output_model.estimators_[0].get_feature_importance()
v_importances = multi_output_model.estimators_[1].get_feature_importance()
feature_names = X.columns

# 将 CL 特征重要性得分与特征名称结合成 DataFrame
cl_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_CL': cl_importances
}).sort_values(by='Importance_CL', ascending=False).reset_index(drop=True)

# 将 V 特征重要性得分与特征名称结合成 DataFrame
v_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_V': v_importances
}).sort_values(by='Importance_V', ascending=False).reset_index(drop=True)

# 保存为 Excel 文件
with pd.ExcelWriter(r'../output/ML输出/feature_importance.xlsx') as writer:
    cl_importance_df.to_excel(writer, sheet_name='Importance_CL', index=False)
    v_importance_df.to_excel(writer, sheet_name='Importance_V', index=False)

# 输出查看
cl_importance_df, v_importance_df


# %% [markdown]
# shap图

# %%
import shap
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.grid'] = False

# 分别计算 CL 和 V 的 SHAP 值
for i, column in enumerate(y.columns):
    print(f"Calculating SHAP values for {column}...")
    explainer = shap.Explainer(multi_output_model.estimators_[i])
    shap_values = explainer(X_test)

    # 绘制 SHAP 总体影响图 (summary plot)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    plt.xlabel("SHAP value")
    plt.title(f'SHAP Summary Plot for {column}')
    plt.tight_layout()
    plt.savefig(f'../output/ML输出/SHAP_Summary_{column}.png')
    plt.savefig(f'../pic/SHAP_Summary_{column}.tif', dpi=1000)
    # plt.close()
    plt.show()


# %%
df

# %% [markdown]
# 依赖图

# %%
import numpy as np
import shap
import matplotlib.pyplot as plt

# 假设 'cts' 是分类变量，定义 cts 列不进行指数变换
X_test_exp = X_test.copy()  # 保持原始数据副本

# 获取数值型特征（排除 'cts'）
numeric_columns = X_test.drop(columns=['CTS', 'Gender', 'ICU', 'PMA_class']).select_dtypes(include=[np.number]).columns

# 对数值型特征进行指数变换
X_test_exp[numeric_columns] = np.exp(X_test[numeric_columns])

# 分别计算 CL 和 V 的 SHAP 值并绘制依赖图
for i, column in enumerate(y.columns):
    print(f"Calculating SHAP values for {column}...")
    explainer = shap.Explainer(multi_output_model.estimators_[i])
    shap_values = explainer(X_test)

    # 为每个特征绘制当前目标变量的 SHAP 依赖图
    for feature in X_test_exp.columns:
        plt.figure()
        # 只绘制单一变量的 SHAP 值
        shap.dependence_plot(feature, shap_values.values, X_test_exp, show=False)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.title(f'SHAP Dependence Plot for {feature} ({column})', fontsize=16)
        plt.xlabel(f"{feature}", fontsize=14)
        plt.ylabel("SHAP value", fontsize=14)
        plt.tight_layout()
        plt.savefig(f'../pic/SHAP_Dependence_{column}_{feature}.tif', dpi=1000)
        
        plt.show()


# %% [markdown]
# 亚组分析

# %% [markdown]
# pma<=44 & pma>44

# %%
X

# %%
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

pma_lower_than_44_idx = X_test_reset[X_test_reset['PMA_class']==0].index
y_test_pma_lower_than_44 = y_test_reset.loc[pma_lower_than_44_idx]
y_pred_pma_lower_than_44 = catboost_y_pred[pma_lower_than_44_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_pma_lower_than_44['CL'], y_pred_pma_lower_than_44[:, 0]))
r2_cl = r2_score(y_test_pma_lower_than_44['CL'], y_pred_pma_lower_than_44[:, 0])
mae_cl = mean_absolute_error(y_test_pma_lower_than_44['CL'], y_pred_pma_lower_than_44[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_pma_lower_than_44['V'], y_pred_pma_lower_than_44[:, 1]))
r2_v = r2_score(y_test_pma_lower_than_44['V'], y_pred_pma_lower_than_44[:, 1])
mae_v = mean_absolute_error(y_test_pma_lower_than_44['V'], y_pred_pma_lower_than_44[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/PMA<44_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/PMA<44_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_pma_lower_than_44['CL']
prediction['Predicted Values cl'] = y_pred_pma_lower_than_44[:, 0]
prediction['True Values v'] = y_test_pma_lower_than_44['V']
prediction['Predicted Values v'] = y_pred_pma_lower_than_44[:, 1]
prediction.to_excel(r"../output/ML输出/PMA<44_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
len(pma_lower_than_44_idx)

# %%
df_cl

# %%
df_v

# %%
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

pma_higher_than_44_idx = X_test_reset[X_test_reset['PMA_class']==1].index
y_test_pma_higher_than_44 = y_test_reset.loc[pma_higher_than_44_idx]
y_pred_pma_higher_than_44 = catboost_y_pred[pma_higher_than_44_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_pma_higher_than_44['CL'], y_pred_pma_higher_than_44[:, 0]))
r2_cl = r2_score(y_test_pma_higher_than_44['CL'], y_pred_pma_higher_than_44[:, 0])
mae_cl = mean_absolute_error(y_test_pma_higher_than_44['CL'], y_pred_pma_higher_than_44[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_pma_higher_than_44['V'], y_pred_pma_higher_than_44[:, 1]))
r2_v = r2_score(y_test_pma_higher_than_44['V'], y_pred_pma_higher_than_44[:, 1])
mae_v = mean_absolute_error(y_test_pma_higher_than_44['V'], y_pred_pma_higher_than_44[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/PMA≥44_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/PMA≥44_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_pma_higher_than_44['CL']
prediction['Predicted Values cl'] = y_pred_pma_higher_than_44[:, 0]
prediction['True Values v'] = y_test_pma_higher_than_44['V']
prediction['Predicted Values v'] = y_pred_pma_higher_than_44[:, 1]
prediction.to_excel(r"../output/ML输出/PMA≥44_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
len(pma_higher_than_44_idx)

# %%
df_cl

# %%
df_v

# %% [markdown]
# 早产儿（PMA<37）

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' < 1 且 'PMA' >= 44 的样本的索引
PMA_lower_37_idx = raw_test_reset[raw_test_reset['PMA'] < 37].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_PMA_lower_37 = y_test_reset.loc[PMA_lower_37_idx]

# 使用模型对这些数据进行预测
y_pred_PMA_lower_37 = catboost_y_pred[PMA_lower_37_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_PMA_lower_37['CL'], y_pred_PMA_lower_37[:, 0]))
r2_cl = r2_score(y_test_PMA_lower_37['CL'], y_pred_PMA_lower_37[:, 0])
mae_cl = mean_absolute_error(y_test_PMA_lower_37['CL'], y_pred_PMA_lower_37[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_PMA_lower_37['V'], y_pred_PMA_lower_37[:, 1]))
r2_v = r2_score(y_test_PMA_lower_37['V'], y_pred_PMA_lower_37[:, 1])
mae_v = mean_absolute_error(y_test_PMA_lower_37['V'], y_pred_PMA_lower_37[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/PMA<37_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/PMA<37_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_PMA_lower_37['CL']
prediction['Predicted Values cl'] = y_pred_PMA_lower_37[:, 0]
prediction['True Values v'] = y_test_PMA_lower_37['V']
prediction['Predicted Values v'] = y_pred_PMA_lower_37[:, 1]
# prediction.to_excel(r"../output/ML输出/PMA<37_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
len(PMA_lower_37_idx)

# %% [markdown]
# 新生儿（37-44）

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' < 1 且 'PMA' >= 44 的样本的索引
PMA_37_44_idx = raw_test_reset[(raw_test_reset['PMA'] >= 37) & (raw_test_reset['PMA'] < 44)].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_PMA_37_44 = y_test_reset.loc[PMA_37_44_idx]

# 使用模型对这些数据进行预测
y_pred_PMA_37_44 = catboost_y_pred[PMA_37_44_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_PMA_37_44['CL'], y_pred_PMA_37_44[:, 0]))
r2_cl = r2_score(y_test_PMA_37_44['CL'], y_pred_PMA_37_44[:, 0])
mae_cl = mean_absolute_error(y_test_PMA_37_44['CL'], y_pred_PMA_37_44[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_PMA_37_44['V'], y_pred_PMA_37_44[:, 1]))
r2_v = r2_score(y_test_PMA_37_44['V'], y_pred_PMA_37_44[:, 1])
mae_v = mean_absolute_error(y_test_PMA_37_44['V'], y_pred_PMA_37_44[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/PMA37-44_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/PMA37-44_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_PMA_37_44['CL']
prediction['Predicted Values cl'] = y_pred_PMA_37_44[:, 0]
prediction['True Values v'] = y_test_PMA_37_44['V']
prediction['Predicted Values v'] = y_pred_PMA_37_44[:, 1]
prediction.to_excel(r"../output/ML输出/PMA37-44_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
len(PMA_37_44_idx)

# %% [markdown]
# 婴儿（1个月-1岁）

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' < 1 且 'PMA' >= 44 的样本的索引
age_lower_1_idx = raw_test_reset[(raw_test_reset['Age'] < 1) & (raw_test_reset['PMA']>= 44 )].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_age_lower_1 = y_test_reset.loc[age_lower_1_idx]

# 使用模型对这些数据进行预测
y_pred_age_lower_1 = catboost_y_pred[age_lower_1_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_age_lower_1['CL'], y_pred_age_lower_1[:, 0]))
r2_cl = r2_score(y_test_age_lower_1['CL'], y_pred_age_lower_1[:, 0])
mae_cl = mean_absolute_error(y_test_age_lower_1['CL'], y_pred_age_lower_1[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_age_lower_1['V'], y_pred_age_lower_1[:, 1]))
r2_v = r2_score(y_test_age_lower_1['V'], y_pred_age_lower_1[:, 1])
mae_v = mean_absolute_error(y_test_age_lower_1['V'], y_pred_age_lower_1[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/PMA44-Age1_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/PMA44-Age1_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_age_lower_1['CL']
prediction['Predicted Values cl'] = y_pred_age_lower_1[:, 0]
prediction['True Values v'] = y_test_age_lower_1['V']
prediction['Predicted Values v'] = y_pred_age_lower_1[:, 1]
prediction.to_excel(r"../output/ML输出/PMA44-Age1_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
len(age_lower_1_idx)

# %% [markdown]
# 1岁-3岁

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' < 4 且 'Age' >= 1 的样本的索引
age_1_3_idx = raw_test_reset[(raw_test_reset['Age'] < 4) & (raw_test_reset['Age']>= 1 )].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_age_1_3 = y_test_reset.loc[age_1_3_idx]

# 使用模型对这些数据进行预测
y_pred_age_1_3 = catboost_y_pred[age_1_3_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_age_1_3['CL'], y_pred_age_1_3[:, 0]))
r2_cl = r2_score(y_test_age_1_3['CL'], y_pred_age_1_3[:, 0])
mae_cl = mean_absolute_error(y_test_age_1_3['CL'], y_pred_age_1_3[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_age_1_3['V'], y_pred_age_1_3[:, 1]))
r2_v = r2_score(y_test_age_1_3['V'], y_pred_age_1_3[:, 1])
mae_v = mean_absolute_error(y_test_age_1_3['V'], y_pred_age_1_3[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/Age1-3_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/Age1-3_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_age_1_3['CL']
prediction['Predicted Values cl'] = y_pred_age_1_3[:, 0]
prediction['True Values v'] = y_test_age_1_3['V']
prediction['Predicted Values v'] = y_pred_age_1_3[:, 1]
prediction.to_excel(r"../output/ML输出/Age1-3_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
len(age_1_3_idx)

# %% [markdown]
# 4岁-12岁

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' < 12 且 'Age' >= 4 的样本的索引
age_4_12_idx = raw_test_reset[(raw_test_reset['Age'] < 12) & (raw_test_reset['Age']>= 4 )].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_age_4_12 = y_test_reset.loc[age_4_12_idx]

# 使用模型对这些数据进行预测
y_pred_age_4_12 = catboost_y_pred[age_4_12_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_age_4_12['CL'], y_pred_age_4_12[:, 0]))
r2_cl = r2_score(y_test_age_4_12['CL'], y_pred_age_4_12[:, 0])
mae_cl = mean_absolute_error(y_test_age_4_12['CL'], y_pred_age_4_12[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_age_4_12['V'], y_pred_age_4_12[:, 1]))
r2_v = r2_score(y_test_age_4_12['V'], y_pred_age_4_12[:, 1])
mae_v = mean_absolute_error(y_test_age_4_12['V'], y_pred_age_4_12[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/Age4-12_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/Age4-12_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_age_4_12['CL']
prediction['Predicted Values cl'] = y_pred_age_4_12[:, 0]
prediction['True Values v'] = y_test_age_4_12['V']
prediction['Predicted Values v'] = y_pred_age_4_12[:, 1]
# prediction.to_excel(r"../output/ML输出/Age4-12_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
len(age_4_12_idx)

# %% [markdown]
# 12岁以上

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' >= 12 且 'Age' >= 4 的样本的索引
age_more_12_idx = raw_test_reset[raw_test_reset['Age'] >= 12].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_age_more_12 = y_test_reset.loc[age_more_12_idx]

# 使用模型对这些数据进行预测
y_pred_age_more_12 = catboost_y_pred[age_more_12_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_age_more_12['CL'], y_pred_age_more_12[:, 0]))
r2_cl = r2_score(y_test_age_more_12['CL'], y_pred_age_more_12[:, 0])
mae_cl = mean_absolute_error(y_test_age_more_12['CL'], y_pred_age_more_12[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_age_more_12['V'], y_pred_age_more_12[:, 1]))
r2_v = r2_score(y_test_age_more_12['V'], y_pred_age_more_12[:, 1])
mae_v = mean_absolute_error(y_test_age_more_12['V'], y_pred_age_more_12[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/Age>=12_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/Age>=12_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_age_more_12['CL']
prediction['Predicted Values cl'] = y_pred_age_more_12[:, 0]
prediction['True Values v'] = y_test_age_more_12['V']
prediction['Predicted Values v'] = y_pred_age_more_12[:, 1]
prediction.to_excel(r"../output/ML输出/Age>=12_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
len(age_more_12_idx)

# %% [markdown]
# eGFR<60

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' >= 12 且 'Age' >= 4 的样本的索引
egfr_lower_60_idx = raw_test_reset[raw_test_reset['eGFR'] < 60].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_egfr_lower_60 = y_test_reset.loc[egfr_lower_60_idx]

# 使用模型对这些数据进行预测
y_pred_egfr_lower_60 = catboost_y_pred[egfr_lower_60_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_egfr_lower_60['CL'], y_pred_egfr_lower_60[:, 0]))
r2_cl = r2_score(y_test_egfr_lower_60['CL'], y_pred_egfr_lower_60[:, 0])
mae_cl = mean_absolute_error(y_test_egfr_lower_60['CL'], y_pred_egfr_lower_60[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_egfr_lower_60['V'], y_pred_egfr_lower_60[:, 1]))
r2_v = r2_score(y_test_egfr_lower_60['V'], y_pred_egfr_lower_60[:, 1])
mae_v = mean_absolute_error(y_test_egfr_lower_60['V'], y_pred_egfr_lower_60[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/eGFR<60_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/eGFR<60_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_egfr_lower_60['CL']
prediction['Predicted Values cl'] = y_pred_egfr_lower_60[:, 0]
prediction['True Values v'] = y_test_egfr_lower_60['V']
prediction['Predicted Values v'] = y_pred_egfr_lower_60[:, 1]
prediction.to_excel(r"../output/ML输出/eGFR<60_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
df_cl, df_v

# %% [markdown]
# eGFR<30

# %%
X_test

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' >= 12 且 'Age' >= 4 的样本的索引
egfr_lower_30_idx = raw_test_reset[raw_test_reset['eGFR'] < 30].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_egfr_lower_30 = y_test_reset.loc[egfr_lower_30_idx]

# 使用模型对这些数据进行预测
y_pred_egfr_lower_30 = catboost_y_pred[egfr_lower_30_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_egfr_lower_30['CL'], y_pred_egfr_lower_30[:, 0]))
r2_cl = r2_score(y_test_egfr_lower_30['CL'], y_pred_egfr_lower_30[:, 0])
mae_cl = mean_absolute_error(y_test_egfr_lower_30['CL'], y_pred_egfr_lower_30[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_egfr_lower_30['V'], y_pred_egfr_lower_30[:, 1]))
r2_v = r2_score(y_test_egfr_lower_30['V'], y_pred_egfr_lower_30[:, 1])
mae_v = mean_absolute_error(y_test_egfr_lower_30['V'], y_pred_egfr_lower_30[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/eGFR<30_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/eGFR<30_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_egfr_lower_30['CL']
prediction['Predicted Values cl'] = y_pred_egfr_lower_30[:, 0]
prediction['True Values v'] = y_test_egfr_lower_30['V']
prediction['Predicted Values v'] = y_pred_egfr_lower_30[:, 1]
prediction.to_excel(r"../output/ML输出/eGFR<30_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
df_cl, df_v

# %%
len(egfr_lower_30_idx)

# %% [markdown]
# eGFR 30-60

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' >= 12 且 'Age' >= 4 的样本的索引
egfr_30_60_idx = raw_test_reset[(raw_test_reset['eGFR'] >= 30) & (raw_test_reset['eGFR'] < 60)].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_egfr_30_60 = y_test_reset.loc[egfr_30_60_idx]

# 使用模型对这些数据进行预测
y_pred_egfr_30_60 = catboost_y_pred[egfr_30_60_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_egfr_30_60['CL'], y_pred_egfr_30_60[:, 0]))
r2_cl = r2_score(y_test_egfr_30_60['CL'], y_pred_egfr_30_60[:, 0])
mae_cl = mean_absolute_error(y_test_egfr_30_60['CL'], y_pred_egfr_30_60[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_egfr_30_60['V'], y_pred_egfr_30_60[:, 1]))
r2_v = r2_score(y_test_egfr_30_60['V'], y_pred_egfr_30_60[:, 1])
mae_v = mean_absolute_error(y_test_egfr_30_60['V'], y_pred_egfr_30_60[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/eGFR30-60_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/eGFR30-60_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_egfr_30_60['CL']
prediction['Predicted Values cl'] = y_pred_egfr_30_60[:, 0]
prediction['True Values v'] = y_test_egfr_30_60['V']
prediction['Predicted Values v'] = y_pred_egfr_30_60[:, 1]
prediction.to_excel(r"../output/ML输出/eGFR30-60_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
df_cl, df_v

# %%
len(egfr_30_60_idx)

# %% [markdown]
# eGFR 60-90

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' >= 12 且 'Age' >= 4 的样本的索引
egfr_60_90_idx = raw_test_reset[(raw_test_reset['eGFR'] >= 60) & (raw_test_reset['eGFR'] < 90)].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_egfr_60_90 = y_test_reset.loc[egfr_60_90_idx]

# 使用模型对这些数据进行预测
y_pred_egfr_60_90 = catboost_y_pred[egfr_60_90_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_egfr_60_90['CL'], y_pred_egfr_60_90[:, 0]))
r2_cl = r2_score(y_test_egfr_60_90['CL'], y_pred_egfr_60_90[:, 0])
mae_cl = mean_absolute_error(y_test_egfr_60_90['CL'], y_pred_egfr_60_90[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_egfr_60_90['V'], y_pred_egfr_60_90[:, 1]))
r2_v = r2_score(y_test_egfr_60_90['V'], y_pred_egfr_60_90[:, 1])
mae_v = mean_absolute_error(y_test_egfr_60_90['V'], y_pred_egfr_60_90[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/eGFR60-90_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/eGFR60-90_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_egfr_60_90['CL']
prediction['Predicted Values cl'] = y_pred_egfr_60_90[:, 0]
prediction['True Values v'] = y_test_egfr_60_90['V']
prediction['Predicted Values v'] = y_pred_egfr_60_90[:, 1]
prediction.to_excel(r"../output/ML输出/eGFR60-90_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
df_cl, df_v

# %%
len(egfr_60_90_idx)

# %% [markdown]
# eGFR 90-120

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' >= 12 且 'Age' >= 4 的样本的索引
egfr_90_120_idx = raw_test_reset[(raw_test_reset['eGFR'] >= 90) & (raw_test_reset['eGFR'] < 120)].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_egfr_90_120 = y_test_reset.loc[egfr_90_120_idx]

# 使用模型对这些数据进行预测
y_pred_egfr_90_120 = catboost_y_pred[egfr_90_120_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_egfr_90_120['CL'], y_pred_egfr_90_120[:, 0]))
r2_cl = r2_score(y_test_egfr_90_120['CL'], y_pred_egfr_90_120[:, 0])
mae_cl = mean_absolute_error(y_test_egfr_90_120['CL'], y_pred_egfr_90_120[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_egfr_90_120['V'], y_pred_egfr_90_120[:, 1]))
r2_v = r2_score(y_test_egfr_90_120['V'], y_pred_egfr_90_120[:, 1])
mae_v = mean_absolute_error(y_test_egfr_90_120['V'], y_pred_egfr_90_120[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/eGFR90-120_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/eGFR90-120_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_egfr_90_120['CL']
prediction['Predicted Values cl'] = y_pred_egfr_90_120[:, 0]
prediction['True Values v'] = y_test_egfr_90_120['V']
prediction['Predicted Values v'] = y_pred_egfr_90_120[:, 1]
prediction.to_excel(r"../output/ML输出/eGFR90-120_CatBoost_subgrouppredictions.xlsx", index=False)

# %%
df_cl, df_v

# %%
len(egfr_90_120_idx)

# %% [markdown]
# eGFR >120

# %%
# 读取原数据集
df_raw = pd.read_excel(r'../data/df_final_en.xlsx')

# 将数据集拆分为训练集和测试集
raw_train, raw_test= train_test_split(df_raw, test_size=0.2, random_state=seed)
# 对 raw_test 和 y_test 重置索引
raw_test_reset = raw_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 从 raw_test 中选出 'Age' >= 12 且 'Age' >= 4 的样本的索引
egfr_more_120_idx = raw_test_reset[raw_test_reset['eGFR'] >= 120].index

# 获取对应的 y_test 数据（现在可以直接使用 reset 后的索引）
y_test_egfr_more_120 = y_test_reset.loc[egfr_more_120_idx]

# 使用模型对这些数据进行预测
y_pred_egfr_more_120 = catboost_y_pred[egfr_more_120_idx]

# 计算 R2 值
rmse_cl = np.sqrt(mean_squared_error(y_test_egfr_more_120['CL'], y_pred_egfr_more_120[:, 0]))
r2_cl = r2_score(y_test_egfr_more_120['CL'], y_pred_egfr_more_120[:, 0])
mae_cl = mean_absolute_error(y_test_egfr_more_120['CL'], y_pred_egfr_more_120[:, 0])

rmse_v = np.sqrt(mean_squared_error(y_test_egfr_more_120['V'], y_pred_egfr_more_120[:, 1]))
r2_v = r2_score(y_test_egfr_more_120['V'], y_pred_egfr_more_120[:, 1])
mae_v = mean_absolute_error(y_test_egfr_more_120['V'], y_pred_egfr_more_120[:, 1])

df_cl = [{
    "RMSE": f"{rmse_cl:.2f}",
    "R2": f"{r2_cl:.2f}",
    "MAE": f"{mae_cl:.2f}",
}]

df_v = [{
    "RMSE": f"{rmse_v:.2f}",
    "R2": f"{r2_v:.2f}",
    "MAE": f"{mae_v:.2f}",
}]

pd.DataFrame(df_cl).to_excel(r"../output/ML输出/eGFR>=120_subgroupregression_cl.xlsx", index=False)
pd.DataFrame(df_v).to_excel(r"../output/ML输出/eGFR>=120_subgroupregression_v.xlsx", index=False)

prediction = pd.DataFrame()
prediction['True Values cl'] = y_test_egfr_more_120['CL']
prediction['Predicted Values cl'] = y_pred_egfr_more_120[:, 0]
prediction['True Values v'] = y_test_egfr_more_120['V']
prediction['Predicted Values v'] = y_pred_egfr_more_120[:, 1]
prediction.to_excel(r"../output/ML输出/eGFR>=120_CatBoost_subgrouppredictions.xlsx", index=False)
