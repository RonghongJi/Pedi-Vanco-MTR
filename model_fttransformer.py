# -*- coding: utf-8 -*-
"""
Created on 2026/04/24 13:55:34

@File    :   model_fttransformer.py
@Author  :   Ronghong Ji
"""

# %%
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# feedforward and attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tab_transformer_pytorch import FTTransformer
import random

seed = 486

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 数据加载
df = pd.read_excel(r'../data/df_rf_log_allfeature.xlsx')
X = df.drop(columns=['CL', 'V'])
y = df[['CL', 'V']].values

cat_col = ['CTS', 'Gender', 'ICU', 'PMA_class']


# 将数据分为连续特征和分类特征
X_continuous = X.drop(columns=cat_col).values  # 连续特征
X_categ = X[cat_col].values  # 分类特征

# 将数据划分为训练集和测试集
X_train_continuous, X_test_continuous, X_train_categ, X_test_categ, y_train, y_test = train_test_split(
    X_continuous, X_categ, y, test_size=0.2, random_state=seed
)
# 将分类特征转换为适当的形状
X_train_categ = X_train_categ.reshape(-1, len(cat_col))  # 如果分类变量只有一列，确保其形状是 (batch_size, 1)
X_test_categ = X_test_categ.reshape(-1, len(cat_col))

# 获取每个分类特征的类别数
categories = [len(np.unique(X[cat])) for cat in cat_col]

# FTTransformer的交叉验证和评估
def cross_val_regression_metrics(model, X_continuous, X_categ, y, num_epochs=100, cv_splits=10):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    metrics_list = {col: [] for col in ['CL', 'V']}

    for train_idx, val_idx in kf.split(X_continuous):
        X_train_continuous_fold, X_val_continuous_fold = X_continuous[train_idx], X_continuous[val_idx]
        X_train_categ_fold, X_val_categ_fold = X_categ[train_idx], X_categ[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # 数据转换为张量
        train_tensor_continuous = torch.tensor(X_train_continuous_fold, dtype=torch.float32)
        train_tensor_categ = torch.tensor(X_train_categ_fold, dtype=torch.long)
        target_tensor = torch.tensor(y_train_fold, dtype=torch.float32)
        
        val_tensor_continuous = torch.tensor(X_val_continuous_fold, dtype=torch.float32)
        val_tensor_categ = torch.tensor(X_val_categ_fold, dtype=torch.long)
        val_target_tensor = torch.tensor(y_val_fold, dtype=torch.float32)

        # 定义模型
        model = FTTransformer(
            categories=categories,     # 分类特征的类别数
            num_continuous=X_continuous.shape[1],      # 连续特征的数量
            dim=16,
            dim_out=2,                                 # 输出维度为2（CL和V）
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1
        )

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # 模型训练
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad() # 清空模型参数的梯度
            outputs = model(train_tensor_categ, train_tensor_continuous)  # 训练模型
            loss = criterion(outputs, target_tensor)
            loss.backward() # 根据当前的损失值（通常是一个标量）计算模型参数的梯度，并反向传播
            optimizer.step() # 根据优化器的配置和计算得到的梯度来更新模型参数

        # 验证集上预测
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_tensor_categ, val_tensor_continuous).numpy()
            fold_metrics = calculate_regression_metrics(y_val_fold, val_predictions)
            for col in ['CL', 'V']:
                metrics_list[col].append(fold_metrics[col])

    # 计算平均值和标准差
    final_metrics = {}
    for col in ['CL', 'V']:
        final_metrics[col] = {
            'RMSE': (np.mean([m['RMSE'] for m in metrics_list[col]]), np.std([m['RMSE'] for m in metrics_list[col]])),
            'R2': (np.mean([m['R2'] for m in metrics_list[col]]), np.std([m['R2'] for m in metrics_list[col]])),
            'MAE': (np.mean([m['MAE'] for m in metrics_list[col]]), np.std([m['MAE'] for m in metrics_list[col]])),
        }
    return final_metrics

# 定义回归指标计算函数
def calculate_regression_metrics(y_true, y_pred):
    metrics = {}
    for i, column in enumerate(['CL', 'V']):
        metrics[column] = {
            'RMSE': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'R2': r2_score(y_true[:, i], y_pred[:, i]),
            'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
        }
    return metrics

# 运行十折交叉验证
cv_metrics = cross_val_regression_metrics(
    FTTransformer,
    X_train_continuous,
    X_train_categ,
    y_train,
    num_epochs=100,
    cv_splits=10
)

# 使用训练集和最佳模型参数在测试集上评估
final_model = FTTransformer(
    categories=categories,  # 分类特征的类别数
    num_continuous=X_continuous.shape[1],  # 连续特征的数量
    dim=16,
    dim_out=2,  # 输出维度为2
    depth=6,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1
)

train_tensor_continuous = torch.tensor(X_train_continuous, dtype=torch.float32)
train_tensor_categ = torch.tensor(X_train_categ, dtype=torch.long)
target_tensor = torch.tensor(y_train, dtype=torch.float32)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_tensor_categ, train_tensor_continuous, target_tensor),
    batch_size=32,
    shuffle=True
)

criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.01)

# 训练最终模型
for epoch in range(100):
    for batch_X_categ, batch_X_cont, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = final_model(batch_X_categ, batch_X_cont)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# 在测试集上进行评估
final_model.eval()
with torch.no_grad():
    test_tensor_continuous = torch.tensor(X_test_continuous, dtype=torch.float32)
    test_tensor_categ = torch.tensor(X_test_categ, dtype=torch.long)
    y_pred = final_model(test_tensor_categ, test_tensor_continuous).numpy()

test_metrics = calculate_regression_metrics(y_test, y_pred)

# 汇总结果
metrics = {}
for col in ['CL', 'V']:
    metrics[f'Ten_fold_CV_RMSE_{col}'] = f"{cv_metrics[col]['RMSE'][0]:.2f} ± {cv_metrics[col]['RMSE'][1]:.2f}"
    metrics[f'Ten_fold_CV_R2_{col}'] = f"{cv_metrics[col]['R2'][0]:.2f} ± {cv_metrics[col]['R2'][1]:.2f}"
    metrics[f'Ten_fold_CV_MAE_{col}'] = f"{cv_metrics[col]['MAE'][0]:.2f} ± {cv_metrics[col]['MAE'][1]:.2f}"

for col in ['CL', 'V']:
    metrics[f'Test_RMSE_{col}'] = f"{test_metrics[col]['RMSE']:.2f}"
    metrics[f'Test_R2_{col}'] = f"{test_metrics[col]['R2']:.2f}"
    metrics[f'Test_MAE_{col}'] = f"{test_metrics[col]['MAE']:.2f}"

# 保存到Excel
metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_excel(r'../output/ML输出/FTTransformer_Regression.xlsx', index=False)

metrics_df
# %%
import pandas as pd

# 如果 y_test 是 pandas DataFrame 或 Series，可以重置索引以避免问题
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)  # 如果是 Series 或 DataFrame
y_pred_df = pd.DataFrame(y_pred).reset_index(drop=True)  # 同样处理预测值

# 确保列名称正确
y_test_df.columns = ['True Values cl', 'True Values v']
y_pred_df.columns = ['Predicted Values cl', 'Predicted Values v']

# 将两者合并到一个 DataFrame 中
output_df = pd.concat([y_test_df, y_pred_df], axis=1)

# 保存到 Excel 文件
output_df.to_excel(r'../output/pred/FTTransformer_predictions.xlsx', index=False)

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
    accuracies[f"Accuracy_within_{pct}%_CL"] = calculate_accuracy_within_percentage(y_test[:,0], y_pred[:, 0], pct)
    accuracies[f"Accuracy_within_{pct}%_V"] = calculate_accuracy_within_percentage(y_test[:,1], y_pred[:, 1], pct)

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
accuracy_df.to_excel(r'../output/accuracy/FTTransformer_acc.xlsx', index=False)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 绘制 CL 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,0], y_pred[:, 0], color='black', s=15)
plt.plot([y_test[:,0].min(), y_test[:,0].max()], [y_test[:,0].min(), y_test[:,0].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('FTTransformer Scatter Plot for CL', fontsize=20)

# 计算 R² 并添加到图中
r2_cl = r2_score(y_test[:,0], y_pred[:, 0])
plt.text(0.8, 0.3, f"R² = {r2_cl:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# 绘制 V 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,1], y_pred[:, 1], color='black', s=15)
plt.plot([y_test[:,1].min(), y_test[:,1].max()], [y_test[:,1].min(), y_test[:,1].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('FTTransformer Scatter Plot for V', fontsize=20)

# 计算 R² 并添加到图中
r2_v = r2_score(y_test[:,1], y_pred[:, 1])
plt.text(0.8, 0.3, f"R² = {r2_v:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()



# %%
import matplotlib.pyplot as plt
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 绘制拟合散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,0], y_pred[:, 0], color='black', s=15)
plt.plot([y_test[:,0].min(), y_test[:,0].max()], [y_test[:,0].min(), y_test[:,0].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('FTTransformer Fitted Scatter Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,1], y_pred[:, 1], color='black', s=15)
plt.plot([y_test[:,1].min(), y_test[:,1].max()], [y_test[:,1].min(), y_test[:,1].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('FTTransformer Fitted Scatter Plot for V', fontsize=20)
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
residuals_cl = y_test[:,0] - y_pred[:, 0]
plt.scatter(y_test[:,0], residuals_cl, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('FTTransformer Residual Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
residuals_v = y_test[:,1] - y_pred[:, 1]
plt.scatter(y_test[:,1], residuals_v, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('FTTransformer Residual Plot for V', fontsize=20)
plt.show()



