import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# --- 1. 定义 LoRA 模块 ---


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # 冻结的原始权重 W_0
        # 在实际应用中，W_0 是预训练模型的原始权重
        # 这里我们模拟一个随机初始化的 W_0
        self.W0 = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # 可训练的 LoRA A 矩阵
        # 通常初始化为小的随机值
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))

        # 可训练的 LoRA B 矩阵
        # 通常初始化为零矩阵，这样在训练开始时，LoRA 模块的贡献为零
        # 确保模型从 W_0 的预训练状态开始
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 缩放因子，通常设置为 alpha / rank
        # 帮助保持 LoRA 模块的贡献与 rank 无关
        self.scaling = self.alpha / self.rank

        # LoRA 模块的激活状态，True表示启用，False表示禁用（即只使用W0）
        self.active = True

        # 记录是否已合并，用于防止重复合并
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果 LoRA 模块被禁用或已合并，只使用原始权重 W_0
        if not self.active or self.merged:
            return F.linear(x, self.W0)

        # 计算原始路径的输出
        original_output = F.linear(x, self.W0)

        # 计算 LoRA 路径的输出: (B @ A) @ x
        # 注意矩阵乘法的顺序，A 和 B 都是nn.Parameter，可以直接相乘
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)

        # 最终输出 = 原始输出 + scaling * LoRA 输出
        return original_output + self.scaling * lora_output

    def merge_weights(self):
        """
        将 LoRA 权重合并到原始权重 W0 中。
        在推理时可以这样做，以消除 LoRA 模块的额外计算开销。
        """
        if self.merged:
            print("警告: 权重已合并，跳过再次合并。")
            return

        print("合并 LoRA 权重到 W0...")
        # 计算 LoRA 增量
        delta_W = self.scaling * (self.lora_B @ self.lora_A)

        # 将增量加到原始权重 W0 上
        self.W0.data += delta_W

        # 移除 LoRA 参数，因为它们已经被合并，不再需要
        del self.lora_A
        del self.lora_B
        self.lora_A = None
        self.lora_B = None

        # 标记为已合并
        self.merged = True
        self.active = False  # 合并后，LoRA路径不再激活

    def enable_lora(self):
        """启用 LoRA 模块（默认行为）"""
        self.active = True
        self.W0.requires_grad = False  # 确保原始权重仍然冻结

    def disable_lora(self):
        """禁用 LoRA 模块，只使用原始权重 W0"""
        self.active = False
        self.W0.requires_grad = False  # 确保原始权重仍然冻结

# --- 2. 模拟一个简单的模型，其中包含 LoRA 层 ---


class SimpleModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        # 使用 LoRALayer 替换传统的线性层
        self.lora_linear = LoRALayer(in_dim, hidden_dim, rank, alpha)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, out_dim)  # 这是一个普通的线性层，未 LoRA 化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lora_linear(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


# --- 3. 训练过程模拟 ---
if __name__ == "__main__":
    # 参数设置
    in_features = 128    # 输入特征维度
    out_features = 64    # 隐藏层输出维度 (也是LoRA层的out_features)
    final_output_dim = 10  # 最终输出维度
    lora_rank = 4        # LoRA 的秩
    lora_alpha = 32      # LoRA 的缩放因子 alpha
    learning_rate = 0.01
    epochs = 100
    batch_size = 32

    # 创建模型
    model = SimpleModel(in_features, out_features, final_output_dim, lora_rank, lora_alpha)

    print(f"原始线性层参数量 (W0): {model.lora_linear.W0.numel()}")
    print(f"LoRA A 矩阵参数量: {model.lora_linear.lora_A.numel()}")
    print(f"LoRA B 矩阵参数量: {model.lora_linear.lora_B.numel()}")
    print(f"LoRA 总参数量 (A + B): {model.lora_linear.lora_A.numel() + model.lora_linear.lora_B.numel()}")

    # 统计可训练参数数量
    # 观察输出，只有 lora_A, lora_B 和 output_layer 的权重/偏置是可训练的
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总可训练参数量: {trainable_params}")
    # 注意: model.lora_linear.W0.requires_grad 应该是 False

    # 优化器只优化 LoRA A, B 矩阵 和 output_layer
    # model.parameters() 会自动排除 requires_grad=False 的参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 模拟数据
    X_train = torch.randn(1000, in_features)
    y_train = torch.randn(1000, final_output_dim)

    # 训练循环
    print("\n--- 开始训练 LoRA 模块 ---")
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 简单的前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    print("\n--- 训练结束 ---")

    # --- 4. 参数合并与推理模拟 ---
    print("\n--- 演示参数合并 ---")

    # 在合并前，测试一次推理
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        initial_inference_output = model(X_train[:1])
    print(f"合并前LoRA层的输出 (示例): {model.lora_linear(X_train[:1]).detach().numpy()}")
    print(f"合并前完整模型的推理输出 (示例): {initial_inference_output.numpy()}")

    # 执行合并
    model.lora_linear.merge_weights()

    # 合并后，再次测试推理。此时 LoRA 模块内的 lora_A 和 lora_B 已经不存在，
    # 并且 forward 函数会直接使用更新后的 W0。
    with torch.no_grad():
        merged_inference_output = model(X_train[:1])
    print(f"合并后LoRA层的输出 (示例): {model.lora_linear(X_train[:1]).detach().numpy()}")
    print(f"合并后完整模型的推理输出 (示例): {merged_inference_output.numpy()}")

    # 验证合并前后输出是否一致 (由于浮点数精度，可能会有微小差异，但应非常接近)
    print(f"合并前后推理输出差异 (L2范数): {torch.norm(initial_inference_output - merged_inference_output).item()}")

    # 验证 LoRA 模块是否真的只用 W0 了
    print(f"LoRA 模块激活状态: {model.lora_linear.active}")
    print(f"LoRA A 矩阵是否存在: {model.lora_linear.lora_A is None}")
    print(f"LoRA B 矩阵是否存在: {model.lora_linear.lora_B is None}")
    print(f"W0 是否依然冻结: {model.lora_linear.W0.requires_grad}")
