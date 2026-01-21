"""
验证假设：词向量的模长与词频相关，方向与语义相关

实验设计：
1. 加载预训练词向量 (Word2Vec / GloVe)
2. 分析：模长 vs 词频 的相关性
3. t-SNE 可视化：验证语义相似的词方向是否接近
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 方法1: 使用 Gensim 加载 Word2Vec (推荐，需要下载模型)
# ============================================================


def load_word2vec_gensim():
    """
    使用 gensim 加载 Google 预训练的 Word2Vec
    首次运行会自动下载 (~1.5GB)
    """
    import gensim.downloader as api
    print("正在加载 Word2Vec 模型 (首次需下载 ~1.5GB)...")
    model = api.load("word2vec-google-news-300")
    return model

# ============================================================
# 方法2: 使用 GloVe (更小，适合快速测试)
# ============================================================


def download_glove():
    """下载 GloVe 词向量 (50维，小文件，适合测试)"""
    import urllib.request
    import zipfile

    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = "glove.6B.zip"

    if not os.path.exists("glove.6B.50d.txt"):
        print("正在下载 GloVe (862MB)，请稍候...")
        urllib.request.urlretrieve(url, zip_path)

        print("解压中...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract("glove.6B.50d.txt")
        os.remove(zip_path)

    return "glove.6B.50d.txt"


def load_glove(filepath, max_words=50000):
    """加载 GloVe 词向量"""
    embeddings = {}
    word_rank = {}  # 词在文件中的排名（近似词频排名）

    print(f"加载 GloVe 词向量 (前 {max_words} 个词)...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= max_words:
                break
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
            word_rank[word] = idx + 1  # 排名从1开始

    print(f"加载完成，共 {len(embeddings)} 个词")
    return embeddings, word_rank

# ============================================================
# 方法3: 自己训练一个小模型 (最轻量，无需下载)
# ============================================================


def train_simple_word2vec():
    """
    使用 gensim 在小数据集上训练 Word2Vec
    这样可以直接获取真实词频
    """
    from gensim.models import Word2Vec
    from collections import Counter

    # 使用一些示例句子（实际应用中用更大的语料库）
    sentences = [
        # 皇室相关
        ["king", "queen", "prince", "princess", "royal", "crown", "throne", "palace"],
        ["king", "rules", "the", "kingdom", "with", "queen"],
        ["prince", "will", "become", "king", "someday"],
        ["queen", "wears", "a", "beautiful", "crown"],

        # 动物相关
        ["cat", "dog", "pet", "animal", "cute", "furry"],
        ["cat", "sleeps", "on", "the", "sofa"],
        ["dog", "runs", "in", "the", "park"],
        ["my", "pet", "cat", "is", "cute"],

        # 食物相关
        ["apple", "banana", "orange", "fruit", "sweet", "healthy"],
        ["eat", "apple", "every", "day"],
        ["banana", "is", "yellow", "fruit"],
        ["orange", "juice", "is", "sweet"],

        # 科技相关
        ["computer", "phone", "laptop", "technology", "digital", "software"],
        ["use", "computer", "for", "work"],
        ["phone", "is", "a", "communication", "device"],

        # 高频词（故意多出现）
        ["the", "is", "a", "of", "and", "to", "in"],
        ["the", "the", "the", "is", "is", "a", "a"],
        ["of", "the", "and", "to", "in", "the", "is"],
    ] * 100  # 重复以增加训练数据

    # 统计真实词频
    word_freq = Counter()
    for sent in sentences:
        word_freq.update(sent)

    print("训练 Word2Vec 模型...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=5,
        min_count=1,
        workers=4,
        epochs=100
    )

    return model, dict(word_freq)

# ============================================================
# 分析函数
# ============================================================


def analyze_magnitude_vs_frequency(embeddings, word_freq_or_rank, is_rank=False):
    """
    分析：模长 vs 词频/排名 的相关性

    Args:
        embeddings: dict, word -> vector
        word_freq_or_rank: dict, word -> frequency 或 rank
        is_rank: bool, True 表示是排名（越小越高频），False 表示是频率
    """
    words = []
    magnitudes = []
    frequencies = []

    for word, vec in embeddings.items():
        if word in word_freq_or_rank:
            words.append(word)
            magnitudes.append(np.linalg.norm(vec))
            freq = word_freq_or_rank[word]
            # 如果是排名，转换为"伪频率"（排名越小，频率越高）
            if is_rank:
                freq = 1.0 / freq  # 排名的倒数作为频率代理
            frequencies.append(freq)

    magnitudes = np.array(magnitudes)
    frequencies = np.array(frequencies)

    # 计算 Pearson 相关系数
    correlation = np.corrcoef(magnitudes, frequencies)[0, 1]

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 散点图
    ax1 = axes[0]
    ax1.scatter(frequencies, magnitudes, alpha=0.5, s=10)
    ax1.set_xlabel('词频 (或 1/排名)' if is_rank else '词频', fontsize=12)
    ax1.set_ylabel('向量模长', fontsize=12)
    ax1.set_title(f'模长 vs 词频\nPearson 相关系数 r = {correlation:.4f}', fontsize=14)

    if is_rank:
        ax1.set_xscale('log')
    ax1.set_yscale('linear')

    # 标注一些典型词
    # 找高频低模长和低频高模长的词
    sorted_indices = np.argsort(magnitudes)
    for idx in list(sorted_indices[:5]) + list(sorted_indices[-5:]):
        ax1.annotate(words[idx], (frequencies[idx], magnitudes[idx]), fontsize=8)

    # 图2: 按模长排序的词
    ax2 = axes[1]
    sorted_by_mag = sorted(zip(words, magnitudes, frequencies), key=lambda x: x[1], reverse=True)

    top_mag = sorted_by_mag[:15]
    bottom_mag = sorted_by_mag[-15:]

    display_words = [w for w, m, f in top_mag + bottom_mag]
    display_mags = [m for w, m, f in top_mag + bottom_mag]
    colors = ['red'] * 15 + ['blue'] * 15

    y_pos = np.arange(len(display_words))
    ax2.barh(y_pos, display_mags, color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(display_words)
    ax2.set_xlabel('向量模长', fontsize=12)
    ax2.set_title('模长最大 (红) vs 最小 (蓝) 的词', fontsize=14)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig('magnitude_vs_frequency.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n📊 分析结果:")
    print(f"   Pearson 相关系数: r = {correlation:.4f}")
    if correlation > 0.3:
        print(f"   ✅ 正相关: 模长与词频有较强正相关，支持'模长编码词频'假设")
    elif correlation < -0.3:
        print(f"   ⚠️ 负相关: 模长与词频负相关")
    else:
        print(f"   ❓ 弱相关: 相关性不明显，可能需要更多数据")

    return correlation


def visualize_semantic_clusters(embeddings, word_groups, title="t-SNE 语义聚类可视化"):
    """
    用 t-SNE 可视化语义聚类

    Args:
        embeddings: dict, word -> vector
        word_groups: dict, group_name -> list of words
    """
    all_words = []
    all_vectors = []
    all_labels = []
    all_colors = []

    color_map = plt.cm.get_cmap('tab10')

    for group_idx, (group_name, words) in enumerate(word_groups.items()):
        for word in words:
            if word in embeddings:
                all_words.append(word)
                all_vectors.append(embeddings[word])
                all_labels.append(group_name)
                all_colors.append(color_map(group_idx))

    if len(all_vectors) < 5:
        print("词汇太少，无法进行 t-SNE 可视化")
        return

    all_vectors = np.array(all_vectors)

    # 对比：原始向量 vs L2 归一化后的向量
    normalized_vectors = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (vectors, subtitle) in enumerate([
        (all_vectors, "原始向量 (保留模长)"),
        (normalized_vectors, "L2 归一化后 (只保留方向)")
    ]):
        print(f"正在计算 t-SNE ({subtitle})...")

        perplexity = min(30, len(vectors) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        vectors_2d = tsne.fit_transform(vectors)

        ax = axes[ax_idx]

        # 按组绘制
        for group_idx, (group_name, _) in enumerate(word_groups.items()):
            mask = [l == group_name for l in all_labels]
            group_points = vectors_2d[mask]
            ax.scatter(group_points[:, 0], group_points[:, 1],
                       c=[color_map(group_idx)], label=group_name, s=100, alpha=0.7)

        # 标注词
        for i, word in enumerate(all_words):
            ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                        fontsize=9, alpha=0.8)

        ax.set_title(subtitle, fontsize=14)
        ax.legend(loc='best')
        ax.set_xlabel('t-SNE 维度 1')
        ax.set_ylabel('t-SNE 维度 2')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tsne_semantic_clusters.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n📊 t-SNE 可视化说明:")
    print("   - 左图: 原始向量，同时包含方向和模长信息")
    print("   - 右图: 归一化后，只保留方向信息")
    print("   - 如果两图聚类效果相似，说明语义主要编码在方向上")


def compute_similarity_comparison(embeddings, word_pairs):
    """
    对比点积相似度和余弦相似度
    """
    print("\n📊 相似度对比 (点积 vs 余弦):")
    print("-" * 70)
    print(f"{'词对':<25} {'点积':<12} {'余弦':<12} {'模长1':<10} {'模长2':<10}")
    print("-" * 70)

    for w1, w2 in word_pairs:
        if w1 in embeddings and w2 in embeddings:
            v1, v2 = embeddings[w1], embeddings[w2]
            dot = np.dot(v1, v2)
            cos = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
            mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
            print(f"{w1 + ' - ' + w2:<25} {dot:<12.4f} {cos:<12.4f} {mag1:<10.4f} {mag2:<10.4f}")
    print("-" * 70)

# ============================================================
# 主程序
# ============================================================


def main(choice="1"):
    print("=" * 60)
    print("  验证实验：词向量的模长 vs 方向")
    print("=" * 60)

    # 选择数据源
    print("\n数据源选项:")
    print("1. 自己训练小模型 (无需下载)")
    print("2. 下载 GloVe 预训练词向量 (862MB)")
    print("3. 下载 Google Word2Vec (1.5GB)")
    print(f"\n当前使用选项: {choice}")

    if choice == "2":
        # GloVe
        glove_path = download_glove()
        embeddings, word_rank = load_glove(glove_path, max_words=30000)

        # 分析模长 vs 排名
        print("\n" + "=" * 60)
        print("  实验1: 模长与词频(排名)的相关性")
        print("=" * 60)
        analyze_magnitude_vs_frequency(embeddings, word_rank, is_rank=True)

        # 语义聚类可视化
        word_groups = {
            "皇室": ["king", "queen", "prince", "princess", "royal", "crown", "throne"],
            "动物": ["cat", "dog", "lion", "tiger", "elephant", "bird", "fish"],
            "水果": ["apple", "banana", "orange", "grape", "mango", "peach"],
            "国家": ["china", "japan", "america", "france", "germany", "russia"],
            "颜色": ["red", "blue", "green", "yellow", "black", "white", "purple"],
        }

    elif choice == "3":
        # Word2Vec
        model = load_word2vec_gensim()
        embeddings = {word: model[word] for word in model.key_to_index}
        word_rank = {word: idx + 1 for idx, word in enumerate(model.key_to_index)}

        analyze_magnitude_vs_frequency(embeddings, word_rank, is_rank=True)

        word_groups = {
            "皇室": ["king", "queen", "prince", "princess", "royal", "crown", "throne"],
            "动物": ["cat", "dog", "lion", "tiger", "elephant", "bird", "fish"],
            "水果": ["apple", "banana", "orange", "grape", "mango", "peach"],
            "国家": ["China", "Japan", "America", "France", "Germany", "Russia"],
            "科技": ["computer", "phone", "laptop", "software", "internet", "technology"],
        }

    else:
        # 自己训练
        model, word_freq = train_simple_word2vec()
        embeddings = {word: model.wv[word] for word in model.wv.key_to_index}

        print("\n" + "=" * 60)
        print("  实验1: 模长与词频的相关性")
        print("=" * 60)
        analyze_magnitude_vs_frequency(embeddings, word_freq, is_rank=False)

        word_groups = {
            "皇室": ["king", "queen", "prince", "princess", "royal", "crown"],
            "动物": ["cat", "dog", "pet", "animal"],
            "水果": ["apple", "banana", "orange", "fruit"],
            "高频词": ["the", "is", "a", "of", "and", "to"],
        }

    # 语义聚类可视化
    print("\n" + "=" * 60)
    print("  实验2: t-SNE 语义聚类可视化")
    print("=" * 60)
    visualize_semantic_clusters(embeddings, word_groups)

    # 相似度对比
    print("\n" + "=" * 60)
    print("  实验3: 点积 vs 余弦相似度对比")
    print("=" * 60)

    word_pairs = [
        ("king", "queen"),
        ("king", "prince"),
        ("cat", "dog"),
        ("apple", "banana"),
        ("king", "apple"),
        ("cat", "banana"),
    ]

    # 过滤存在的词对
    valid_pairs = [(w1, w2) for w1, w2 in word_pairs
                   if w1 in embeddings and w2 in embeddings]

    if valid_pairs:
        compute_similarity_comparison(embeddings, valid_pairs)

    print("\n✅ 实验完成！请查看生成的图片:")
    print("   - magnitude_vs_frequency.png (模长 vs 词频)")
    print("   - tsne_semantic_clusters.png (语义聚类)")


if __name__ == "__main__":
    import sys
    choice = sys.argv[1] if len(sys.argv) > 1 else "1"
    main(choice)
