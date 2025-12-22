import pandas as pd
import numpy as np
import random

# 1. 设置随机种子，保证结果可复现
np.random.seed(42)

# 2. 生成 30 个项目 ID (例如 P_001 到 P_030)
num_projects = 30
project_ids = [f"P_{i:03d}" for i in range(1, num_projects + 1)]

# 3. 生成预算成本 (Cost Baseline)
# 假设预算在 50万 到 500万 之间
cost_baseline = np.random.randint(500000, 5000000, size=num_projects)

# 4. 生成实际成本 (Cost Actual)
# 为了模拟真实情况，我们设定实际成本在预算的 80% 到 130% 之间波动
# (即：有的项目省钱，有的项目超支)
fluctuation = np.random.uniform(0.8, 1.3, size=num_projects)
cost_actual = (cost_baseline * fluctuation).astype(int)

# 5. 创建 DataFrame
df = pd.DataFrame({
    'project_id': project_ids,
    'cost_baseline': cost_baseline,
    'cost_actual': cost_actual
})

# 6. 计算 COR (Cost Overrun Rate)
# 保留 4 位小数
df['COR'] = round((df['cost_actual'] - df['cost_baseline']) / df['cost_baseline'], 4)

# 7. 查看前几行数据
print("数据预览：")
print(df.head())

# 8. 导出为 CSV
df.to_csv('labels.csv', index=False, encoding='utf-8')
print("\n文件已保存为 labels.csv")