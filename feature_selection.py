# %%
# -*- coding: utf-8 -*-
"""
Created on 2026/04/24 13:486:27

@File    :   feature_selection.py
@Author  :   Ronghong Ji
"""


# %%
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler


# 加载数据
df = pd.read_excel(r'../data/df_final_en.xlsx')

# 特征列，去除 'CL' 和 'V' 列
X = df.drop(columns=['CL', 'V', 'DataSource', 'patient_id'])
y = df[['CL', 'V']]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=486)

# 逐个优化参数
def grid_search_cv_single_param(param_grid, X, y, fixed_params):
    model = xgb.XGBRegressor(random_state=486, **fixed_params)
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1)
    grid_search.fit(X, y['CL'])  # 仅使用 'CL' 作为评分
    return grid_search.best_params_

param_grids = {
    'n_estimators': [100, 150, 200, 250, 300],
    'learning_rate': [0.03, 0.05, 0.1, 0.3],
    'max_depth': [3, 4, 5, 6]
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

best_params_df = pd.DataFrame([best_params])
xgb_best_model = xgb.XGBRegressor(**best_params, random_state=486)

# %%
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_excel(r'../data/df_final_en.xlsx')
data = data.drop(columns=['patient_id', 'DataSource'])

# 划分X和y
X = data.drop(columns=['CL', 'V'])
y = data[['CL', 'V']]  # 同时选择 CL 和 V 作为目标变量

# 将数据集拆分为训练集和测试集
train_x, X_test, train_y, y_test = train_test_split(X, y, test_size=0.2, random_state=486)

num_features = train_x.shape[1]

# 初始化存储选择的特征和得分的列表
selected_feat_1r=[]
selected_feat1_1r=[]
scores_1r = []

# 特征选择循环，选择 1 到 train_x 中的所有特征
for i in range(1, num_features + 1):
    print(f"选择 {i} 个特征:")
    
    # Sequential Feature Selector 使用 MultiOutputRegressor 包裹 XGBRegressor
    sfs = SFS(MultiOutputRegressor(xgb_best_model),
              k_features=i, 
              forward=True, 
              floating=False, 
              verbose=2,
              scoring='r2',
              cv=3,
              n_jobs=-1)
    
    # 拟合模型
    sfs = sfs.fit(train_x, train_y)

    # 记录选择的特征
    selected_feat_1r = train_x.columns[list(sfs.k_feature_idx_)]
    selected_feat1_1r.append(selected_feat_1r)
    
    # 记录当前选择的特征组合的 r2 分数
    scores_1r.append(round(sfs.k_score_, 2))


# %%
# 找到 r2 分数中的最大值及其对应的索引
best_index_1r = scores_1r.index(max(scores_1r))

# 找到最优的特征组合
best_features_1r = selected_feat1_1r[best_index_1r]

# 输出最优特征组合及其对应的 r2 值
print("最优特征组合：", best_features_1r)
print("对应的 R2 分数：", scores_1r[best_index_1r])

# %%
len(best_features_1r)

# %%
import matplotlib.pyplot as plt

# 设置字体为新罗马
plt.rc('font', family='Times New Roman')

# 绘制 AUC 随着特征数量变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_x.columns)+1), scores_1r, marker='o', linestyle='--', color='#4682B4')  # 使用钢蓝色
plt.xlabel('Variable Nums', fontsize=18)
plt.ylabel(r'$R^2$', fontsize=18)  # 使用 r 前缀和 LaTeX 语法表示 R²
# plt.title('R² vs. Number of Features', fontsize=20)

# 去掉负值并从0开始
plt.xlim(0, 28)  # x轴从0开始
plt.ylim(0.5, 1)  # y轴从0.5到1

# 设置坐标轴样式
plt.grid(True)
# 设置 xy 轴的黑线
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# 设置刻度字体大小
plt.xticks(range(0, 30, 2), fontsize=16)  # 每隔2设置一个x轴刻度
plt.yticks(fontsize=16)

# # 添加数据标签
# for i, score in enumerate(scores_1r):
#     plt.text(i + 1, score, f'{score:.2f}', fontsize=12, ha='center', va='bottom')
plt.savefig("../pic/特征筛选图.tif", dpi=1000)
plt.show()



