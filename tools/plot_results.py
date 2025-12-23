import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# 设置学术画图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)

def parse_actions(x):
    if pd.isna(x) or x == '': return []
    try:
        # 兼容多种格式的字符串解析
        if ',' in str(x) and not str(x).startswith('['):
            items = ast.literal_eval("[" + str(x) + "]")
        else:
            items = [ast.literal_eval(str(x))]
        return [str(i).strip() for i in items]
    except:
        return [i.strip().replace("'", "").replace('"', "") for i in str(x).split(',')]

# 1. 预算-收益分析
df = pd.read_csv('policy05_budget_sweep.csv')
plt.figure(figsize=(7, 5))
df['abs_delta'] = df['delta'].abs()
sns.lineplot(data=df[df['budget']>0], x='budget', y='abs_delta', marker='o', color='#2b6cb0', errorbar=('ci', 95))
plt.title('Impact of Budget on Risk Mitigation ($\Delta$COR)', fontweight='bold')
plt.xlabel('Budget Allocation ($)')
plt.ylabel('Reduction in Predicted COR')
plt.savefig('fig1_budget_impact.png', dpi=300, bbox_inches='tight')

# 2. 动作分布情况
action_rows = []
for _, row in df.iterrows():
    for a in parse_actions(row['selected']):
        action_rows.append({'budget': row['budget'], 'action': a})
action_df = pd.DataFrame(action_rows)
if not action_df.empty:
    pivot_df = action_df.groupby(['budget', 'action']).size().unstack(fill_value=0)
    pivot_df.drop(0, errors='ignore').plot(kind='bar', stacked=True, colormap='viridis', figsize=(9, 6), edgecolor='white')
    plt.title('Optimal Action Mix at Different Budget Levels', fontweight='bold')
    plt.xlabel('Budget ($)')
    plt.ylabel('Selection Frequency')
    plt.legend(title='Action ID', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=0)
    plt.savefig('fig2_action_distribution.png', dpi=300, bbox_inches='tight')

# 3. 失效原因饼图
fail_df = pd.read_csv('fail_reason_counts.csv', index_name=0)
fail_df.columns = ['reason', 'count']
plt.figure(figsize=(6, 6))
plt.pie(fail_df[fail_df['count']>0]['count'], labels=fail_df[fail_df['count']>0]['reason'],
        autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'), shadow=False)
plt.title('Analysis of Decision Failure Reasons', fontweight='bold')
plt.savefig('fig3_failure_analysis.png', dpi=300, bbox_inches='tight')

# 4. GNN 拟合度
labels_df = pd.read_csv('labels.csv')
merged = df[['project_id', 'pred_base']].drop_duplicates().merge(labels_df[['project_id', 'COR']], on='project_id')
plt.figure(figsize=(7, 6))
sns.regplot(data=merged, x='COR', y='pred_base', scatter_kws={'alpha':0.5}, line_kws={'color':'#e53e3e'})
plt.title('Model Fidelity: GNN Predictions vs. Ground Truth', fontweight='bold')
plt.xlabel('Observed Actual COR')
plt.ylabel('GNN Predicted Baseline COR')
plt.savefig('fig4_prediction_fidelity.png', dpi=300, bbox_inches='tight')

print("All figures saved to current directory.")