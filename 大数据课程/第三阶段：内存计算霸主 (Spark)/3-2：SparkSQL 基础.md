你好！欢迎来到 **3-2** 课。

上一节课我们学了 RDD，它是 Spark 的“原子”，功能强大但比较底层。
今天我们要升级了。在现代 Spark 开发中，我们很少直接操作 RDD，而是使用更高级、更像 SQL 的接口——**DataFrame**。

这就好比：RDD 是让你用**汇编语言**盖房子，而 SparkSQL 是让你用**预制板**盖房子，速度快且不容易出错。

---

### 1. 核心进化：DataFrame vs RDD

**RDD 的痛点**：
RDD 是一个装满对象的袋子。Spark 只知道袋子里有东西，但不知道东西长啥样。

* 比如一个 RDD 里存的是 `Person` 对象。你想按“年龄”排序，Spark 必须先把对象反序列化（拆开包装），读出年龄，再排序。这很慢。

**DataFrame 的进化**：
**DataFrame = RDD + Schema（结构信息）**。

#### 📊 生活类比：普通文本 vs Excel 表格

* **RDD 就像一个 `data.txt` 纯文本文件**：
  * 里面密密麻麻写满了字。你想找“第三列大于10的行”，你必须写一段复杂的代码去切分每一行，转换类型，再判断。
* **DataFrame 就像一个 `data.xlsx` Excel 表格**：
  * 它有**表头（Schema）**！
  * Spark 清楚地知道：第一列叫 `Name` (String)，第二列叫 `Age` (Int)。
  * 你想找“年龄大于10”，Spark 不需要拆包，直接去读第二列的数据，效率极高。

---

### 2. 环境与实操：从 Python 到 SQL

在 SparkSQL 中，我们的入口不再是 `sc` (SparkContext)，而是 **`spark` (SparkSession)**。

请在 PySpark 环境中运行以下代码（Docker 容器里输入 `pyspark` 后，`spark` 变量通常已自动创建）：

```python
from pyspark.sql import SparkSession

# 1. 准备数据（就像准备 Excel 里的几行数据）
# 这是一个普通的 Python 列表
local_data = [
    ("Alice", 18, "Beijing"),
    ("Bob", 25, "Shanghai"),
    ("Charlie", 30, "Beijing")
]

# 2. 创建 DataFrame (关键步骤)
# 我们明确告诉 Spark：这三列分别是 Name, Age, City
df = spark.createDataFrame(local_data, ["Name", "Age", "City"])

# 看看长什么样
print("--- 原始表结构 ---")
df.show()
# 输出:
# +-------+---+-------+
# |   Name|Age|   City|
# +-------+---+-------+
# |  Alice| 18|Beijing|
# ...

# 3. 注册临时视图 (TempView)
# 这一步是把 DataFrame 挂载到 SQL 引擎上，起个名字叫 "people_table"
# 这样我们就能用 SQL 语句去查它了！
df.createOrReplaceTempView("people_table")

# 4. 直接写 SQL 查询
# 需求：查找所有在北京的人
sql_result = spark.sql("SELECT * FROM people_table WHERE City = 'Beijing'")

print("--- SQL 查询结果 ---")
sql_result.show()
```

---

### 3. 原理揭秘：Catalyst 优化器（Spark 为什么比手写 RDD 快？）

你可能会问：“SQL 不是要翻译成代码吗？为什么翻译过的反而比我直接手写的代码还快？”

这是因为 SparkSQL 内部住着一个天才军师——**Catalyst 优化器**。

**通俗解释**：
假设你要去城市的另一头（处理数据）：

* **手写 RDD**：就像你自己开车。虽然你认识路，但你不知道哪里堵车。你可能写了一段代码，先加载了 1TB 数据，然后再过滤掉 90%。Spark 只能听你的，傻傻地去加载。
* **SparkSQL (Catalyst)**：就像用了**高德地图/Google Maps**。
  * 你告诉它：“我要去终点（要结果）”。
  * **逻辑优化**：Catalyst 会看一眼地图，发现前面堵车。它会自作主张修改你的执行计划：“**既然你只要 1% 的数据，那我先在读取文件的时候就过滤掉（谓词下推），不要把所有数据都读进内存！**”
  * **物理优化**：它会把很多小的操作合并成一个大的操作，减少 CPU 的调用次数。

**结论**：大部分情况下，SparkSQL 自动优化出来的执行路径，比普通程序员手写 RDD 优化出来的要快得多。

---

### 4. 课后作业：RDD 变身 DataFrame

在实际工作中，我们经常需要把旧的 RDD 代码转换成 DataFrame。请完成以下任务：

**前置代码**：

```python
# 这是一个原始的 RDD，里面是 Tuple
rdd = sc.parallelize([("Jack", 90), ("Rose", 85), ("Tom", 60)])
```

**任务要求**：

1. 请写一行代码，利用 `rdd.toDF(...)` 方法，将上面的 RDD 转换成 DataFrame，列名定为 `["Name", "Score"]`。
2. 对这个 DataFrame 使用 SQL 或 DSL（代码风格），查询出分数大于 80 分的人。

请回复你的代码（哪怕只写关键行也可以）。
