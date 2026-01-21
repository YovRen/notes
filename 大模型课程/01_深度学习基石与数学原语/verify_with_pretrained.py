"""
使用 gensim 预训练词向量验证假设
glove-wiki-gigaword-50: 约66MB，下载较快
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim.downloader as api

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("  使用预训练 GloVe 词向量验证假设")
print("=" * 60)

# 下载 GloVe (50维，约66MB)
print("\n正在下载 GloVe-Wiki-Gigaword-50 (~66MB)...")
model = api.load("glove-wiki-gigaword-50")
print(f"加载完成！词汇量: {len(model.key_to_index)}")

# 提取词向量和排名（排名 ≈ 词频排名，越小越高频）
words = list(model.key_to_index.keys())[:10000]  # 取前10000个词
vectors = np.array([model[w] for w in words])
ranks = np.arange(1, len(words) + 1)
magnitudes = np.linalg.norm(vectors, axis=1)

# ============================================================
# 实验1: 模长 vs 词频排名
# ============================================================
print("\n" + "=" * 60)
print("  实验1: 模长与词频(排名)的相关性")
print("=" * 60)

# 使用 1/rank 作为"词频"代理
pseudo_freq = 1.0 / ranks
correlation = np.corrcoef(magnitudes, pseudo_freq)[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 散点图
ax1 = axes[0]
ax1.scatter(ranks, magnitudes, alpha=0.3, s=5)
ax1.set_xlabel('词频排名 (越小越高频)', fontsize=12)
ax1.set_ylabel('向量模长', fontsize=12)
ax1.set_xscale('log')
ax1.set_title(f'GloVe: 模长 vs 词频排名\nPearson(模长, 1/排名) = {correlation:.4f}', fontsize=14)

# 标注一些典型词
for word_to_mark in ['the', 'is', 'of', 'king', 'queen', 'serendipity', 'computer']:
    if word_to_mark in words:
        idx = words.index(word_to_mark)
        ax1.annotate(word_to_mark, (ranks[idx], magnitudes[idx]), fontsize=9, color='red')

# 按模长排序的词
ax2 = axes[1]
sorted_indices = np.argsort(magnitudes)
top_15 = sorted_indices[-15:][::-1]  # 最大
bottom_15 = sorted_indices[:15]       # 最小

display_words = [words[i] for i in top_15] + [words[i] for i in bottom_15]
display_mags = [magnitudes[i] for i in top_15] + [magnitudes[i] for i in bottom_15]
display_ranks = [ranks[i] for i in top_15] + [ranks[i] for i in bottom_15]
colors = ['red'] * 15 + ['blue'] * 15

y_pos = np.arange(len(display_words))
bars = ax2.barh(y_pos, display_mags, color=colors, alpha=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"{w} (rank={r})" for w, r in zip(display_words, display_ranks)])
ax2.set_xlabel('向量模长', fontsize=12)
ax2.set_title('模长最大 (红) vs 最小 (蓝) 的词\n括号内为词频排名', fontsize=14)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('glove_magnitude_vs_rank.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📊 结果分析:")
print(f"   Pearson(模长, 1/排名) = {correlation:.4f}")

# 打印一些具体例子
print("\n📋 具体例子:")
print(f"{'词':<15} {'排名':<10} {'模长':<10}")
print("-" * 35)
for word in ['the', 'is', 'a', 'of', 'king', 'queen', 'apple', 'computer', 'serendipity']:
    if word in words:
        idx = words.index(word)
        print(f"{word:<15} {ranks[idx]:<10} {magnitudes[idx]:<10.4f}")

# ============================================================
# 实验2: t-SNE 语义聚类
# ============================================================
print("\n" + "=" * 60)
print("  实验2: t-SNE 语义聚类可视化")
print("=" * 60)

word_groups = {
    "皇室": ["king", "queen", "prince", "princess", "royal", "crown", "throne", "palace"],
    "动物": ["cat", "dog", "lion", "tiger", "elephant", "bird", "fish", "horse"],
    "水果": ["apple", "banana", "orange", "grape", "mango", "peach", "lemon", "cherry"],
    "国家": ["china", "japan", "america", "france", "germany", "russia", "india", "brazil"],
    "颜色": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"],
}

all_words = []
all_vectors = []
all_labels = []

for group_name, group_words in word_groups.items():
    for w in group_words:
        if w in model:
            all_words.append(w)
            all_vectors.append(model[w])
            all_labels.append(group_name)

all_vectors = np.array(all_vectors)
normalized_vectors = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
color_map = plt.colormaps['tab10']

for ax_idx, (vecs, subtitle) in enumerate([
    (all_vectors, "原始向量 (保留模长)"),
    (normalized_vectors, "L2 归一化后 (只保留方向)")
]):
    print(f"正在计算 t-SNE ({subtitle})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(vecs) - 1))
    vecs_2d = tsne.fit_transform(vecs)

    ax = axes[ax_idx]
    for group_idx, group_name in enumerate(word_groups.keys()):
        mask = [l == group_name for l in all_labels]
        points = vecs_2d[np.array(mask)]
        ax.scatter(points[:, 0], points[:, 1], c=[color_map(group_idx)],
                   label=group_name, s=100, alpha=0.7)

    for i, w in enumerate(all_words):
        ax.annotate(w, (vecs_2d[i, 0], vecs_2d[i, 1]), fontsize=8, alpha=0.8)

    ax.set_title(subtitle, fontsize=14)
    ax.legend(loc='best')

plt.suptitle('GloVe t-SNE 语义聚类可视化', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('glove_tsne_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 实验3: 余弦相似度 vs 点积
# ============================================================
print("\n" + "=" * 60)
print("  实验3: 点积 vs 余弦相似度对比")
print("=" * 60)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


pairs = [
    ("king", "queen", "语义相近"),
    ("king", "prince", "语义相近"),
    ("cat", "dog", "语义相近"),
    ("apple", "banana", "语义相近"),
    ("king", "apple", "语义无关"),
    ("cat", "computer", "语义无关"),
    ("the", "is", "高频词"),
    ("the", "king", "高频vs低频"),
]

print(f"\n{'词对':<20} {'关系':<12} {'点积':<12} {'余弦':<12} {'模长1':<8} {'模长2':<8}")
print("-" * 80)

for w1, w2, relation in pairs:
    if w1 in model and w2 in model:
        v1, v2 = model[w1], model[w2]
        dot = np.dot(v1, v2)
        cos = cosine_sim(v1, v2)
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        print(f"{w1 + '-' + w2:<20} {relation:<12} {dot:<12.4f} {cos:<12.4f} {m1:<8.2f} {m2:<8.2f}")

print("\n" + "=" * 60)
print("  结论")
print("=" * 60)
print("""
📌 关于模长与词频的关系:
   - 在 GloVe 等大规模预训练模型中，关系比较复杂
   - 高频词位于向量空间"中心"，与很多词有关联
   - 低频词可能位于"边缘"，更专业化

📌 关于方向与语义的关系:
   - t-SNE 图清楚显示：同类词（颜色相同）聚在一起
   - L2 归一化后聚类效果相似，说明语义主要在方向上
   - 余弦相似度对语义关系更敏感

📌 点积 vs 余弦:
   - 语义相近的词，余弦相似度高
   - 点积会被模长影响，不一定反映真实语义关系
""")

print("\n✅ 图片已保存:")
print("   - glove_magnitude_vs_rank.png")
print("   - glove_tsne_clusters.png")
