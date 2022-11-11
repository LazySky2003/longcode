# -*- coding: utf-8 -*-
# 库导入
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 数据读取
train_data = pd.read_csv('data/dataTrain.csv')
test_data = pd.read_csv('data/dataA.csv')
submission = pd.read_csv('data/submit_example_A.csv')
data_nolabel = pd.read_csv('data/dataNoLabel.csv')
print(f'train_data.shape = {train_data.shape}')
print(f'test_data.shape  = {test_data.shape}')

# 特征构造：自己DIY的特征
train_data['f47'] = train_data['f1'] * 10 + train_data['f2']
test_data['f47'] = test_data['f1'] * 10 + test_data['f2']
# 特征构造：网上的特征
# 暴力Feature 位置类特征          (将几个不同的特征进行组合，新的特征也许能够更好的表征数据，优先考虑强特征维度)
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]]+1)

# 暴力Feature 通话类特征
com_f = ['f43', 'f44', 'f45', 'f46']
for df in [train_data, test_data]:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]]+1)

# ID类特征数值化 （必须先将类别标签映射为数值，然后才能用于建模算法：标签的编码）
cat_columns = ['f3']
data = pd.concat([train_data, test_data])

for col in cat_columns:
    lb = LabelEncoder()  # 获取一个LabelEncoder
    lb.fit(data[col])  # 训练LabelEncoder
    train_data[col] = lb.transform(train_data[col])  # 使用训练好的LabelEncoder对原数据进行编码
    test_data[col] = lb.transform(test_data[col])
print(train_data[col])  # 根据打印结果可以知道， LabelEncoder将：high编码为0，low编码为1，mid编码为2

# 特征构造：最后构造出训练集和测试集
num_columns = [ col for col in train_data.columns if col not in ['id', 'label', 'f3']]
feature_columns = num_columns + cat_columns
target = 'label'

train = train_data[feature_columns]
label = train_data[target]
test = test_data[feature_columns]

# 模型训练：常用的交叉验证模型框架  在样本量不充足的情况下，为了充分利用数据集对算法效果进行测试，将数据集A随机分为k个包，每次将其中一个包作为测试集，剩下k-1个包作为训练集进行训练
def model_train(model, model_name, kfold=5):  #
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)  # n_splits
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        model.fit(x_train,y_train)  # sklearn里的封装好的各种算法使用前都要fit，fit相对于整个代码而言，为后续API服务。

        y_pred = model.predict_proba(x_test)[:,1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test,y_pred)  # roc_auc_score() ROC 曲线下面积AUC
        print("- KFold = %d, val_auc = %.4f" % (k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold

# # 数据清洗
# gbc = GradientBoostingClassifier()
# gbc_test_preds = model_train(gbc, "GradientBoostingClassifier", 60)

# 剔除干扰数据
train = train[:50000]
label = label[:50000]
# print(train)
# print(label)

#模型融合：选取6个树模型作为基础模型
gbc = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=5
)
hgbc = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5
)
xgbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
gbm = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    num_leaves=2 ** 6,
    max_depth=8,
    colsample_bytree=0.8,
    subsample_freq=1,
    max_bin=255,
    learning_rate=0.05,
    n_estimators=100,
    metrics='auc'
)
cbc = CatBoostClassifier(
    iterations=210,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=1,
    loss_function='Logloss',
    verbose=0
)

# 通过StackingClassifier将6个模型进行Stack，Stack模型用LogisticRegression
estimators = [
    ('gbc', gbc),
    ('hgbc', hgbc),
    ('xgbc', xgbc),
    ('gbm', gbm),
    ('cbc', cbc)
]
clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# 特征筛选：
# 先将训练数据划分成训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(
    train, label, stratify=label, random_state=2022)

# 然后用组合模型进行训练和验证    跟gbc模型（原模型）对比如何？ 特征筛选后又如何？
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('特征筛选前的auc = %.8f' % auc)

# 循环遍历特征，对验证集中的特征进行mask
ff = []
for col in feature_columns:
    x_test = X_test.copy()
    x_test[col] = 0
    auc1 = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    if auc1 < auc:
        ff.append(col)
    print('%5s | %.8f | %.8f' % (col, auc1, auc1 - auc))

# 这里选取所有差值为负的特征，对比特征筛选后的特征提升
clf.fit(X_train[ff], y_train)
y_pred = clf.predict_proba(X_test[ff])[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('特征筛选后的auc = %.8f' % auc)

# 模型训练
train = train[ff]
test = test[ff]

clf_test_preds = model_train(clf, "StackingClassifier", 10)

submission['label'] = clf_test_preds
submission.to_csv('submission.csv', index=False)
