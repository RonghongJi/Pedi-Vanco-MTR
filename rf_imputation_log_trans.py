# -*- coding: utf-8 -*-
"""
Created on 2026/04/24 13:48:03

@File    :   rf_imputation_log_trans.py
@Author  :   Ronghong Ji
"""

# %% [markdown]
# 随机森林填补

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 区分分类变量和数值型变量
def categorize_columns(df, threshold=10):
    categorical_columns = []
    continuous_columns = []
    
    for col in df.columns:
        # 如果唯一值的数量少于阈值，则视为分类变量
        if df[col].nunique() < threshold:
            categorical_columns.append(col)
        else:
            continuous_columns.append(col)
    
    return categorical_columns, continuous_columns

# 将分类变量转换为 'category' 类型
def convert_to_category(df, categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype('category')  # 或者使用 'object' 类型 df[col].astype('object')
    return df

# %%
df_forward = pd.read_excel(r'../output/df_forward.xlsx')

# %%
# 根据唯一值的数量来区分分类变量和数值变量
categorical_cols, continuous_cols = categorize_columns(df_forward, threshold=10)
df_convert = convert_to_category(df_forward, categorical_cols)

# %%
def random_forest_imputation(df, categorical_cols, continuous_cols):
    df_filled = df.copy()
    
    # 使用列表合并而不是相加
    all_cols = df.columns
    
    for col in all_cols:
        print(f"Processing column: {col}")
        
        # 检查列是否存在于数据框中
        if col not in df_filled.columns:
            print(f"Column {col} not found in the dataframe. Skipping.")
            continue
        
        # 训练数据
        train_data = df_filled[df_filled[col].notna()]
        X_train = train_data.drop(columns=[col])
        y_train = train_data[col]
        
        # 测试数据
        test_data = df_filled[df_filled[col].isna()]
        X_test = test_data.drop(columns=[col])
        
        if X_test.shape[0] == 0:
            print(f"No missing values to predict for column: {col}")
            continue
        
        # 对于分类变量，使用LabelEncoder
        if col in categorical_cols:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            model = RandomForestClassifier(n_estimators=100, random_state=486)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=486)
        
        # 创建和训练模型
        model.fit(X_train, y_train)
        
        # 预测缺失值
        if col in categorical_cols:
            predicted_values = le.inverse_transform(model.predict(X_test))
        else:
            predicted_values = model.predict(X_test)
        
        # 填充缺失值
        df_filled.loc[df_filled[col].isna(), col] = predicted_values
    
    return df_filled


df_rf = random_forest_imputation(df_convert, categorical_cols, continuous_cols)
df_rf.to_excel(r'../data/df_rf.xlsx',index=False)

# %%
df_rf.isnull().sum()

# %%
df_rf

# %% [markdown]
# 对连续变量取log

# %%
import numpy as np

# 创建新的数据集
df_log = df_rf.copy()

# list不能用drop，用remove
continuous_cols.remove('CL')  # 删除 'CL'
continuous_cols.remove('V')   # 删除 'V'

# 对连续变量取对数，避免负值或零值
for col in continuous_cols:
    # 添加一个小常数以避免对数运算中的负无穷
    df_log[col] = np.log(df_log[col] + 1e-5)


# 输出新的数据集
print("新的数据集：")
print(df_log.head())

df_log.to_excel(r'../data/df_rf_log_allfeature.xlsx', index=False)


