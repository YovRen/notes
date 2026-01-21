# 🚀 部署指南 - 多端访问你的文档

## 方式一：本地预览（最快）

### 使用 Python（推荐）
```bash
# 在项目目录下运行
cd AI_Agent_深度架构与数学原理
python -m http.server 3000
```
然后在浏览器打开 http://localhost:3000

### 使用 Node.js
```bash
# 安装 serve（一次性）
npm install -g serve

# 启动
cd AI_Agent_深度架构与数学原理
serve .
```

### 使用 docsify-cli
```bash
# 安装
npm install -g docsify-cli

# 启动（支持热更新）
cd AI_Agent_深度架构与数学原理
docsify serve .
```

---

## 方式二：部署到 GitHub Pages（免费在线访问）

### 步骤 1：创建 GitHub 仓库
1. 登录 GitHub
2. 点击 "New repository"
3. 仓库名建议：`ai-agent-learning`
4. 选择 Public（免费 Pages 需要公开仓库）

### 步骤 2：初始化并推送代码
```bash
cd AI_Agent_深度架构与数学原理

# 初始化 Git
git init

# 添加所有文件
git add .

# 提交
git commit -m "初始化 AI Agent 学习文档"

# 添加远程仓库（替换 YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/ai-agent-learning.git

# 推送
git push -u origin main
```

### 步骤 3：启用 GitHub Pages
1. 进入仓库设置 Settings → Pages
2. Source 选择 "Deploy from a branch"
3. Branch 选择 `main`，文件夹选择 `/ (root)`
4. 点击 Save

### 步骤 4：访问你的站点
几分钟后，你就可以通过以下地址访问：
```
https://YOUR_USERNAME.github.io/ai-agent-learning/
```

---

## 方式三：部署到 Vercel（更快的 CDN）

### 步骤 1：准备
确保代码已推送到 GitHub

### 步骤 2：部署
1. 访问 [vercel.com](https://vercel.com)
2. 使用 GitHub 账号登录
3. 点击 "Import Project"
4. 选择你的仓库
5. 直接点击 "Deploy"（无需任何配置）

### 步骤 3：自定义域名（可选）
Vercel 会自动分配一个域名如 `ai-agent-learning.vercel.app`
你也可以绑定自己的域名

---

## 方式四：部署到 Netlify

类似 Vercel，访问 [netlify.com](https://netlify.com)，导入 GitHub 仓库即可。

---

## 📱 多端访问

部署完成后，你可以通过以下方式访问：

- **电脑浏览器**：直接访问 URL
- **手机浏览器**：访问同一 URL
- **添加到主屏幕**：在手机浏览器中选择"添加到主屏幕"，获得类 App 体验
- **平板**：同样通过浏览器访问

## 🔧 自定义配置

如需修改主题或功能，编辑 `index.html` 中的配置：

```javascript
window.$docsify = {
  name: '你的文档名称',
  // 其他配置...
}
```

## 📚 更多参考

- [Docsify 官方文档](https://docsify.js.org/#/zh-cn/)
- [GitHub Pages 文档](https://docs.github.com/cn/pages)
