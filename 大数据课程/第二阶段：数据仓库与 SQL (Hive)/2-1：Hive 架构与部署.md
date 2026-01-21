欢迎进入第二阶段！🎉

从今天开始，你的角色将发生转变：从需要懂底层原理的“架构师”，转变为专注于数据价值的“数据分析师”。

**Hive** 是大数据生态中最重要的工具之一。如果不夸张地说，**80% 的离线大数据处理工作都是在写 Hive SQL**。

---

### 1. 核心通俗化：Hive 是什么？

**场景**：
Facebook 在 2008 年的时候，每天产生海量数据。但只有几十个 Java 工程师能写 MapReduce 去分析数据，而公司里有几百个分析师只会写 SQL。
分析师想看报表，还得求工程师写代码，效率太低。

**Hive 的诞生**：
Facebook 的工程师想：**能不能做一个“翻译官”，让分析师直接写 SQL，然后这个软件在后台自动把它翻译成 MapReduce 代码去跑？**
于是，Hive 诞生了。

**面试必问：Hive 和 MySQL 的区别？**
这是新人最容易混淆的概念。它们虽然都写 SQL，但本质完全不同：

* **MySQL (OLTP - 联机事务处理)**：
  * **类比**：超市的**收银台**。
  * **特点**：快。处理的是“刚才买了一瓶水”这种**小而快**的增删改查。
  * **数据量**：撑死几百 GB。
* **Hive (OLAP - 联机分析处理)**：
  * **类比**：超市的**年度财务室**。
  * **特点**：慢（可能有延迟）。处理的是“过去 10 年所有门店的销售趋势”这种**海量只读**数据。
  * **数据量**：起步就是 PB 级。
  * **注意**：Hive **极少**用来修改数据（Update/Delete），通常是一次写入，多次读取。

---

### 2. 架构解析：SQL 是怎么变成 MapReduce 的？

Hive 本身不存数据，也不算数据（除了元数据）。它只是一个**客户端工具**。

**架构图解**：

```text
用户 (写 SQL) 
    ⬇️
Hive Driver (驱动器/翻译官)
    ⬇️ 1. 语法解析 & 优化
    ⬇️ 2. 查找 Metastore (元数据) —— 关键！
    ⬇️ 3. 编译成 MapReduce 任务
YARN (资源调度)
    ⬇️
HDFS (读写原始文件)
```

**核心概念：Metastore (元数据存储)**

* **问题**：HDFS 上存的只是一个普通的文本文件（比如 `data.txt`），里面全是逗号分隔的字符串。HDFS 根本不知道哪一列是“年龄”，哪一列是“姓名”。
* **解决**：Hive 需要一个**小本子 (MySQL)** 来记录映射关系：“`data.txt` 的第一列是 ID (Int)，第二列是 Name (String)”。
* 这个记录表结构的小本子，就是 **Metastore**。

---

### 3. 环境与实操：启动 Hive

为了简化环境，我们使用 Apache 官方的 Hive 镜像，它支持“本地模式”（Local Mode），不需要完整的 Hadoop 集群也能跑 SQL 练手。

#### 步骤一：启动 Hive 容器

在你的终端运行：

```bash
# 启动一个包含 Hive 的容器，并进入命令行
# 注意：第一次拉取镜像可能需要一点时间
docker run -it --name hive-learn apache/hive:3.1.3 /bin/bash
```

#### 步骤二：初始化并启动 Hive

进入容器后，执行以下命令：

```bash
# 1. 初始化元数据（就在本地生成一个数据库文件）
schematool -dbType derby -initSchema

# 2. 启动 Hive 客户端
hive
```

当你看到 `hive>` 的提示符时，恭喜你，你已经准备好写 SQL 了！

---

### 4. 代码解析：Create Table 的魔法

在 Hive 里建表，不仅是定义列名，更是**定义如何解析文件**。

**执行这行代码：**

```sql
CREATE TABLE students (
    id INT, 
    name STRING, 
    age INT
) 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ',';
```

**逐行解析：**

1. `CREATE TABLE students (...)`: 和 MySQL 一样，定义表名和列。
2. `ROW FORMAT DELIMITED`: **这是 Hive 特有的！** 告诉 Hive，我要处理的是文本格式。
3. `FIELDS TERMINATED BY ','`: **核心中的核心**。
   * 它告诉 Hive：“当你在 HDFS 上读到一个文本文件时，请用**逗号**把每一行切开。切开的第一部分给 `id`，第二部分给 `name`...”
   * 这就是 Hive **“读时模式 (Schema on Read)”** 的精髓。

---

### 5. 面试与实战视角

**Q: 为什么 Hive 查询这么慢？**

* **实战**：你在 MySQL 查 1 万条数据可能要 0.01 秒。在 Hive 里查 10 条数据可能要 30 秒。
* **原因**：Hive 的底层是要启动 MapReduce 任务的。光是向 YARN 申请资源、启动容器的“冷启动”时间就要十几秒。
* **结论**：不要用 Hive 做实时查询（比如网页后台接口），它是用来做**离线报表**的（比如每天凌晨跑昨天的数）。

**Q: 内部表 vs 外部表？（下节课详解，先预热）**

* 企业里 **90%** 的情况都用**外部表 (External Table)**。因为防止误删数据时把 HDFS 上的源文件也删了。

---

### 6. 课后作业：手动做一次 ETL（抽取-转换-加载）

请在你的 Hive 容器里完成以下全套流程，并将最后一步的查询结果截图或复制给我。

1. **建表**：使用上面给出的 `CREATE TABLE` 语句建立 `students` 表。
2. **造数**：
   * 打开一个新的终端窗口（或者先用 `!bash` 暂时退出 hive cli，或者在容器外）。
   * 创建一个名为 `student_data.txt` 的文件，内容如下：

     ```text
     1,Jack,20
     2,Rose,18
     3,Tom,22
     ```

3. **加载 (Load)**：
   * 使用 Hive 的加载命令把数据导入表中（注意：这里演示从本地路径加载）：

     ```sql
     LOAD DATA LOCAL INPATH '/路径/到/student_data.txt' INTO TABLE students;
     ```

     *(提示：如果你是在容器内部直接创建的文件，路径可能是 `/opt/hive/student_data.txt` 或者你所在的目录)*
4. **查询**：
   * 运行 `SELECT * FROM students;`
   * 运行 `SELECT name, age FROM students WHERE age > 19;`

**等待你的作业结果！** 学会了这个，你就打通了“文件 -> 数据库”的任督二脉。
