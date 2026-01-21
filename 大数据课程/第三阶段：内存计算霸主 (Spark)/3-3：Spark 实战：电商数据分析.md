你好！欢迎来到 **3-3** 实战课。

上一节课我们学了“怎么把 RDD 变成 DataFrame”，今天我们来点真格的。
在企业面试中，面试官给你一块白板，让你手写 Spark 代码，通常考的就是**DSL 风格（Domain Specific Language）**的链式调用，因为这样写代码看起来最优雅、最像老手。

我们抛弃 SQL 字符串，完全用代码来操作数据。

---

### 1. 场景模拟：电商销售分析

* **数据源**：`orders.csv` (1GB)
* **内容**：

  ```text
  order_id,user_id,item_id,amount,city
  1001,u_001,i_500,99.0,Beijing
  1002,u_002,i_501,200.5,Shanghai
  ...
  ```

* **目标**：不写一句 SQL，用 Python 代码算出结果。

---

### 2. 代码实操：PySpark DSL 风格

请仔细看每一行注释，这是**标准的大数据开发范式**。

```python
from pyspark.sql import SparkSession
# 引入 Spark 的内置函数库，这里面有 sum, col (列), desc (降序) 等神器
from pyspark.sql.functions import col, sum, desc

# 0. 初始化
spark = SparkSession.builder.appName("EcommerceAnalysis").getOrCreate()

# 1. 读取 CSV 文件
# header=True: 告诉 Spark 第一行是表头，不要把它当成数据
# inferSchema=True: 自动推断类型（amount是数字还是字符串？让他自己猜）
df = spark.read.csv("orders.csv", header=True, inferSchema=True)

print("--- 原始数据预览 ---")
df.show(5)

# 2. 需求1：过滤出 amount > 100 的大额订单
# DSL 风格：直接用 .filter() 或 .where()
# col("amount") 表示引用这一列
big_orders_df = df.filter(col("amount") > 100)

# 3. 需求2：统计每个城市的销售总额，并从高到低排序
# 链式调用 (Chain Operation) —— Spark 代码的精髓
result_df = df.groupBy("city") \
    .agg(sum("amount").alias("total_sales")) \
    .orderBy(col("total_sales").desc())

# .agg(...): 聚合操作
# .alias(...): 给计算结果起个别名，不然列名会叫 "sum(amount)" 很难看
# .desc(): 降序排列

print("--- 最终统计结果 ---")
result_df.show()
```

---

### 3. 关键点解析：inferSchema 是把双刃剑

在上面的代码里，我用了 `inferSchema=True`。

* **这是干什么的？**
  它让 Spark **“先读一遍文件”**，去看看每一列长什么样。

  * 看到 "1001" -> 猜它是 Int。
  * 看到 "99.0" -> 猜它是 Double。
  * 看到 "Beijing" -> 猜它是 String。
* **为什么生产环境（公司里）尽量少用？**

  1. **性能极差**：为了猜类型，Spark 必须把这个 1GB（甚至 1TB）的文件先从头到尾读一遍。还没开始算呢，时间就浪费了一半。
  2. **容易猜错**：
     * 比如**邮政编码** `01001`。Spark 可能会把它猜成整数 `1001`，前面的 `0` 就丢了，导致数据错误。
     * 比如**身份证号**。如果身份证号太长，可能会被猜成科学计数法。
* **专家建议**：
  在写正式代码时，我们会**手动定义 Schema**（结构），告诉 Spark 每一列是什么，既快又准。

---

### 4. 课后作业：保存结果 (Write)

我们算出了 `result_df`，现在需要把它保存回 HDFS 或者本地文件系统，格式要求是 **Parquet**（一种在大数据领域比 CSV 性能好得多的列式存储格式），并且如果目录存在则**覆盖**。

**请写出这行代码：**
(提示：对象是 `result_df`，方法是 `write`，需要指定 format 和 mode)。
