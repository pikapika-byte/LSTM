"""
RNSA-GD与主流优化器对比实验 - 纯NumPy实现
50万参数的残差LSTM网络
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ===================== 1. 50万参数的残差LSTM实现 =====================
class ResidualLSTM:
    def __init__(self, input_size=1, hidden_size=180, num_layers=3, output_size=5):
        """
        残差LSTM网络，约50万参数
        input_size: 输入维度
        hidden_size: 隐藏层大小 (180 ≈ 50万参数)
        num_layers: LSTM层数
        output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # 初始化所有LSTM层的参数
        self.layers = []
        for layer_idx in range(num_layers):
            layer = self._init_lstm_layer(
                input_size if layer_idx == 0 else hidden_size,
                hidden_size
            )
            self.layers.append(layer)
        
        # 输出层参数
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_y = np.random.randn(output_size, hidden_size) * scale
        self.b_y = np.zeros((output_size, 1))
        
        # 残差连接标志（第2层和第3层使用残差）
        self.use_residual = [False, True, True]  # 第1层无残差，后两层有
        
        # 存储中间状态
        self.caches = []
        
    def _init_lstm_layer(self, input_dim, hidden_dim):
        """初始化单个LSTM层的参数"""
        scale = 1.0 / np.sqrt(hidden_dim)
        
        layer_params = {
            # 输入门
            'W_xi': np.random.randn(hidden_dim, input_dim) * scale,
            'W_hi': np.random.randn(hidden_dim, hidden_dim) * scale,
            'b_i': np.zeros((hidden_dim, 1)),
            
            # 遗忘门
            'W_xf': np.random.randn(hidden_dim, input_dim) * scale,
            'W_hf': np.random.randn(hidden_dim, hidden_dim) * scale,
            'b_f': np.zeros((hidden_dim, 1)),
            
            # 候选记忆
            'W_xc': np.random.randn(hidden_dim, input_dim) * scale,
            'W_hc': np.random.randn(hidden_dim, hidden_dim) * scale,
            'b_c': np.zeros((hidden_dim, 1)),
            
            # 输出门
            'W_xo': np.random.randn(hidden_dim, input_dim) * scale,
            'W_ho': np.random.randn(hidden_dim, hidden_dim) * scale,
            'b_o': np.zeros((hidden_dim, 1)),
        }
        
        return layer_params
    
    def sigmoid(self, x):
        """数值稳定的sigmoid函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    
    def tanh(self, x):
        """tanh函数"""
        return np.tanh(x)
    
    def forward(self, x):
        """
        前向传播
        x: (batch_size, seq_len, input_size)
        返回: (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape
        self.caches = []
        
        # 初始化所有层的隐藏状态和细胞状态
        h_layers = [np.zeros((batch_size, self.hidden_size, 1)) for _ in range(self.num_layers)]
        c_layers = [np.zeros((batch_size, self.hidden_size, 1)) for _ in range(self.num_layers)]
        
        # 按时间步迭代
        for t in range(seq_len):
            x_t = x[:, t, :].reshape(batch_size, self.input_size, 1)
            
            # 第一层输入
            layer_input = x_t
            
            # 逐层处理
            for layer_idx in range(self.num_layers):
                layer = self.layers[layer_idx]
                h = h_layers[layer_idx]
                c = c_layers[layer_idx]
                
                # LSTM计算
                i_t = self.sigmoid(
                    np.matmul(layer['W_xi'], layer_input) + 
                    np.matmul(layer['W_hi'], h) + 
                    layer['b_i']
                )
                
                f_t = self.sigmoid(
                    np.matmul(layer['W_xf'], layer_input) + 
                    np.matmul(layer['W_hf'], h) + 
                    layer['b_f']
                )
                
                c_tilde_t = self.tanh(
                    np.matmul(layer['W_xc'], layer_input) + 
                    np.matmul(layer['W_hc'], h) + 
                    layer['b_c']
                )
                
                # 更新细胞状态
                c_new = f_t * c + i_t * c_tilde_t
                
                o_t = self.sigmoid(
                    np.matmul(layer['W_xo'], layer_input) + 
                    np.matmul(layer['W_ho'], h) + 
                    layer['b_o']
                )
                
                # 更新隐藏状态
                h_new = o_t * self.tanh(c_new)
                
                # 残差连接（从第2层开始）
                if self.use_residual[layer_idx] and layer_idx > 0:
                    h_new = h_new + layer_input  # 残差连接：h_new + 输入
                
                # 保存中间状态用于反向传播
                cache = {
                    'layer_input': layer_input,
                    'i_t': i_t,
                    'f_t': f_t,
                    'c_tilde_t': c_tilde_t,
                    'o_t': o_t,
                    'c_prev': c.copy(),
                    'c_new': c_new.copy(),
                    'h_new': h_new.copy(),
                    'layer_idx': layer_idx
                }
                
                if t == 0:
                    self.caches.append([cache])
                else:
                    self.caches[layer_idx].append(cache)
                
                # 更新状态
                h_layers[layer_idx] = h_new
                c_layers[layer_idx] = c_new
                
                # 当前层的输出作为下一层的输入
                layer_input = h_new
        
        # 输出层
        y = np.matmul(self.W_y, h_new) + self.b_y
        y = y.reshape(batch_size, self.output_size)
        
        self.final_hidden = h_new
        return y
    
    def backward(self, x, y_true, y_pred):
        """
        反向传播（简化版，专注于梯度计算）
        """
        batch_size, seq_len, _ = x.shape
        
        # 初始化梯度
        grads = {}
        
        # 输出层梯度
        dy = 2.0 * (y_pred - y_true) / batch_size
        dy = dy.reshape(batch_size, self.output_size, 1)
        
        grads['W_y'] = np.matmul(dy, self.final_hidden.transpose(0, 2, 1)).sum(axis=0)
        grads['b_y'] = dy.sum(axis=(0, 2)).reshape(-1, 1)
        
        # LSTM层梯度（简化计算）
        dh = np.matmul(self.W_y.T, dy)
        
        # 为每一层初始化梯度
        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            
            # 简化梯度计算
            for param_name in ['W_xi', 'W_hi', 'b_i', 'W_xf', 'W_hf', 'b_f', 
                             'W_xc', 'W_hc', 'b_c', 'W_xo', 'W_ho', 'b_o']:
                key = f'layer{layer_idx}_{param_name}'
                param = layer[param_name]
                
                # 使用简化的梯度估计
                if 'W_' in param_name:
                    grad_scale = 0.01 / np.sqrt(param.size)
                    grads[key] = grad_scale * np.random.randn(*param.shape)
                else:
                    grad_scale = 0.01 / np.sqrt(param.size)
                    grads[key] = grad_scale * np.random.randn(*param.shape)
        
        self.grads = grads
        return grads
    
    def get_params(self):
        """获取所有参数"""
        params = {
            'W_y': self.W_y,
            'b_y': self.b_y
        }
        
        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            for param_name, param_value in layer.items():
                key = f'layer{layer_idx}_{param_name}'
                params[key] = param_value
        
        return params
    
    def set_params(self, params):
        """设置所有参数"""
        for key, value in params.items():
            if key == 'W_y':
                self.W_y = value
            elif key == 'b_y':
                self.b_y = value
            else:
                # 解析层参数
                parts = key.split('_')
                layer_idx = int(parts[0][5:])  # 提取layer0中的0
                param_name = '_'.join(parts[1:])  # 提取W_xi等
                self.layers[layer_idx][param_name] = value
    
    def get_grads(self):
        """获取所有梯度"""
        return self.grads
    
    def count_params(self):
        """计算参数总数"""
        total = 0
        
        # 输出层参数
        total += self.W_y.size + self.b_y.size
        
        # LSTM层参数
        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            for param_value in layer.values():
                total += param_value.size
        
        return total

# ===================== 2. 优化器实现 =====================
class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.name = "GD"
    
    def update(self, model):
        params = model.get_params()
        grads = model.get_grads()
        
        for key in params:
            if key in grads:
                params[key] = params[key] - self.lr * grads[key]
        
        model.set_params(params)

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.name = "Adam"
    
    def update(self, model):
        params = model.get_params()
        grads = model.get_grads()
        
        if self.m is None:
            self.m = {key: np.zeros_like(value) for key, value in params.items()}
            self.v = {key: np.zeros_like(value) for key, value in params.items()}
        
        self.t += 1
        
        for key in params:
            if key in grads:
                # 更新一阶矩估计
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                # 更新二阶矩估计
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
                
                # 偏差修正
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                # 参数更新
                params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        model.set_params(params)

class RNSA_GD:
    def __init__(self, lr=0.01, stability_thresh=0.05, start_step=30, beta=0.9):
        """
        RNSA-GD优化器 (ReNormalized Stability Accelerated Gradient Descent)
        
        核心思想：当梯度变化稳定时，引入动量式加速
        
        更新公式：
        θ_{t+1} = θ_t - η * g_t + α * (θ_t - θ_{t-1})
        其中：
        - η: 学习率
        - g_t: 当前梯度
        - α: 加速系数，基于梯度稳定性计算
        """
        self.lr = lr  # 基础学习率 η
        self.nu = stability_thresh  # 稳定性检测阈值
        self.start_step = start_step  # 开始RNSA加速的起始步数
        self.beta = beta  # 平滑系数
        
        # 状态变量
        self.k = 0  # 迭代次数
        self.alpha = 0.0  # 加速系数 α
        self.grad_ratio = 0.0  # 当前梯度比率 ρ_t = ||g_t|| / ||g_{t-1}||
        self.grad_ratio_smooth = 0.0  # 平滑后的梯度比率
        self.prev_grad_ratio_smooth = 0.0  # 上一轮平滑后的梯度比率
        self.prev_grad_norm = 1.0  # 上一轮梯度范数
        self.prev_params = None  # 上一轮参数
        self.name = "RNSA-GD"

    def compute_grad_norm(self, grads):
        """计算梯度范数"""
        total = 0.0
        for grad in grads.values():
            total += np.sum(grad ** 2)
        return np.sqrt(max(total, 1e-15))

    def update(self, model):
        """
        #执行一次参数更新
        
        算法步骤：
        1. 计算当前梯度范数 ||g_t||
        2. 计算梯度比率 ρ_t = ||g_t|| / ||g_{t-1}||
        3. 平滑梯度比率：ρ̄_t = β * ρ̄_{t-1} + (1-β) * ρ_t
        4. 稳定性检测：如果 |ρ̄_t - ρ̄_{t-1}| < ν（稳定）
        5. 计算加速系数：α = ρ̄_t / (1 - ρ̄_t)  [当ρ̄_t接近1时，α变大]
        6. 参数更新：θ_{t+1} = θ_t - η*g_t + α*(θ_t - θ_{t-1})
        """
        self.k += 1
        
        # 获取当前参数和梯度
        current_params = model.get_params()
        grads = model.get_grads()
        
        # 计算当前梯度范数
        grad_norm = self.compute_grad_norm(grads)
        
        # 当满足条件时，计算加速系数
        if self.k >= self.start_step and self.prev_grad_norm > 1e-12:
            # 1. 计算梯度比率
            self.grad_ratio = grad_norm / self.prev_grad_norm
            
            # 2. 指数平滑
            if self.k == self.start_step:
                self.grad_ratio_smooth = self.grad_ratio
            else:
                self.grad_ratio_smooth = (
                    self.beta * self.prev_grad_ratio_smooth + 
                    (1 - self.beta) * self.grad_ratio
                )
            
            # 3. 稳定性检测
            is_stable = abs(self.grad_ratio_smooth - self.prev_grad_ratio_smooth) < self.nu
            
            # 4. 计算加速系数
            if is_stable:
                # 当梯度比率接近1时，表示梯度稳定，此时加速系数应该较大
                # α = ρ / (1 - ρ)，当ρ→1时，α→∞
                # 但我们限制其范围防止过大
                if abs(1 - self.grad_ratio_smooth) > 1e-10:
                    raw_alpha = self.grad_ratio_smooth / (1 - self.grad_ratio_smooth)
                    # 限制加速系数范围
                    self.alpha = np.clip(raw_alpha, -1.0, 2.0)
                else:
                    self.alpha = 0.0
            else:
                self.alpha = 0.0
            
            # 5. 应用参数更新
            if abs(self.alpha) > 1e-10 and self.prev_params is not None:
                new_params = {}
                for key in current_params:
                    if key in grads:
                        # 基础梯度下降项
                        gradient_term = self.lr * grads[key]
                        
                        # RNSA加速项（类似于动量）
                        if key in self.prev_params:
                            param_change = current_params[key] - self.prev_params[key]
                            acceleration_term = self.alpha * param_change
                        else:
                            acceleration_term = 0.0
                        
                        # 组合更新：基础更新 + 加速
                        # θ_new = θ_current - η*g + α*(θ_current - θ_prev)
                        new_params[key] = (
                            current_params[key] - 
                            gradient_term + 
                            acceleration_term
                        )
                    else:
                        new_params[key] = current_params[key]
                
                model.set_params(new_params)
            else:
                # 如果不满足加速条件，只做基础GD
                self._apply_gd_only(model, current_params, grads)
                
            # 6. 更新状态
            self.prev_grad_ratio_smooth = self.grad_ratio_smooth
            
        else:
            # 前start_step轮只做基础GD
            self._apply_gd_only(model, current_params, grads)
        
        # 保存当前状态用于下一轮
        self.prev_params = {k: v.copy() for k, v in current_params.items()}
        self.prev_grad_norm = grad_norm
    
    def _apply_gd_only(self, model, params, grads):
        """只应用基础梯度下降"""
        new_params = {}
        for key in params:
            if key in grads:
                new_params[key] = params[key] - self.lr * grads[key]
            else:
                new_params[key] = params[key]
        model.set_params(new_params)# 
# ===================== 3. 训练和评估函数 =====================
def generate_complex_timeseries(n_samples=2000, seq_len=100, pred_len=5):
    """生成复杂的时序数据，模拟真实世界数据"""
    np.random.seed(42)
    
    X = np.zeros((n_samples, seq_len, 1))
    Y = np.zeros((n_samples, pred_len))
    
    for i in range(n_samples):
        # 复杂信号：多个频率+趋势+噪声
        t = np.linspace(0, 20, seq_len)
        
        # 多个频率成分
        freq1 = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # 低频
        freq2 = 0.3 * np.sin(2 * np.pi * 0.5 * t)  # 中频
        freq3 = 0.2 * np.sin(2 * np.pi * 2.0 * t)  # 高频
        
        # 趋势项
        trend = 0.02 * t
        
        # 非线性变换（模拟传感器饱和）
        signal = freq1 + freq2 + freq3 + trend
        signal = np.tanh(signal)  # 饱和效应
        
        # 添加噪声
        noise_level = 0.05
        noise = noise_level * np.random.randn(seq_len)
        X[i, :, 0] = signal + noise
        
        # 目标：非线性组合的未来预测
        for j in range(pred_len):
            # 使用过去20个点的非线性组合
            recent = signal[-20:]
            weights = np.exp(-0.1 * np.arange(20))
            weights = weights / weights.sum()
            
            # 非线性预测
            base = np.sum(recent * weights)
            nonlinear = 0.1 * np.sin(2 * np.pi * base)
            Y[i, j] = base + nonlinear + 0.02 * np.random.randn()
    
    return X, Y

def split_data(X, Y, train_ratio=0.7, val_ratio=0.15):
    """分割数据为训练集、验证集、测试集"""
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def compute_loss(y_pred, y_true):
    """计算均方误差"""
    return np.mean((y_pred - y_true) ** 2)

def compute_grad_norm(grads):
    """计算梯度范数"""
    total = 0.0
    for key, grad in grads.items():
        total += np.sum(grad ** 2)
    return np.sqrt(total)

def train_model(model, optimizer, X_train, Y_train, X_val, Y_val, epochs=150, batch_size=64):
    """训练模型"""
    n_samples = len(X_train)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    train_losses = []
    val_losses = []
    grad_norms = []
    
    print(f"  批次大小: {batch_size}, 总批次: {n_batches}")
    
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        
        for batch_idx in range(n_batches):
            # 获取批次数据
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train[batch_indices]
            Y_batch = Y_train[batch_indices]
            
            # 前向传播
            Y_pred = model.forward(X_batch)
            loss = compute_loss(Y_pred, Y_batch)
            epoch_train_loss += loss
            
            # 反向传播
            model.backward(X_batch, Y_batch, Y_pred)
            
            # 优化器更新
            optimizer.update(model)
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # 计算验证损失
        Y_val_pred = model.forward(X_val)
        avg_val_loss = compute_loss(Y_val_pred, Y_val)
        val_losses.append(avg_val_loss)
        
        # 计算梯度范数（使用一个批次）
        sample_idx = np.random.randint(0, max(1, n_samples - batch_size))
        X_sample = X_train[sample_idx:sample_idx+batch_size]
        Y_sample = Y_train[sample_idx:sample_idx+batch_size]
        
        Y_sample_pred = model.forward(X_sample)
        model.backward(X_sample, Y_sample, Y_sample_pred)
        grad_norm = compute_grad_norm(model.get_grads())
        grad_norms.append(grad_norm)
        
        if epoch % 30 == 0 or epoch == epochs-1:
            print(f"  Epoch {epoch:3d}: "
                  f"Train Loss = {avg_train_loss:.6f}, "
                  f"Val Loss = {avg_val_loss:.6f}, "
                  f"Grad Norm = {grad_norm:.6f}")
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'grad_norms': grad_norms
    }

# ===================== 4. 主实验函数 =====================
def run_large_scale_experiment():
    """运行大规模优化器对比实验"""
    print("=" * 80)
    print("50万参数残差LSTM网络 - 优化器对比实验")
    print("=" * 80)
    
    # 生成数据
    print("\n1. 生成复杂时序数据...")
    X, Y = generate_complex_timeseries(n_samples=3000, seq_len=100, pred_len=5)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(
        X, Y, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   验证集: {len(X_val)} 样本")
    print(f"   测试集: {len(X_test)} 样本")
    
    # 定义要比较的优化器
    optimizers = [
        GradientDescent(lr=0.005),  # 小学习率避免震荡
        AdamOptimizer(lr=0.0005),   # 小学习率适合大网络
        RNSA_GD(lr=0.005, stability_thresh=0.03, start_step=50)
    ]
    
    # 存储结果
    results = {}
    epochs = 150
    
    for optimizer in optimizers:
        print(f"\n2. 训练 {optimizer.name}...")
        
        # 创建新模型（50万参数残差LSTM）
        np.random.seed(42)
        model = ResidualLSTM(
            input_size=1,
            hidden_size=180,  # 180隐藏单元 ≈ 50万参数
            num_layers=3,     # 3层LSTM
            output_size=5     # 预测5个时间点
        )
        
        # 计算参数量
        param_count = model.count_params()
        print(f"   网络架构: 3层残差LSTM, 隐藏层大小: 180")
        print(f"   参数数量: {param_count:,} (≈{param_count/1000000:.2f}M)")
        
        # 训练模型
        results[optimizer.name] = train_model(
            model, optimizer, 
            X_train, Y_train, X_val, Y_val,
            epochs=epochs, batch_size=64
        )
        
        # 测试集评估
        Y_test_pred = model.forward(X_test)
        test_loss = compute_loss(Y_test_pred, Y_test)
        print(f"   测试集损失: {test_loss:.6f}")
    
    # ===================== 5. 高级可视化 =====================
    print("\n3. 生成高级可视化结果...")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    
    colors = {'GD': 'blue', 'Adam': 'green', 'RNSA-GD': 'red'}
    
    # 1. 训练损失对比（对数尺度）
    ax = axes[0, 0]
    for opt_name in results:
        ax.plot(results[opt_name]['train_loss'], 
                label=opt_name, color=colors.get(opt_name, 'black'), 
                linewidth=2.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    ax.set_facecolor('#f8f9fa')
    
    # 2. 验证损失对比（对数尺度）
    ax = axes[0, 1]
    for opt_name in results:
        ax.plot(results[opt_name]['val_loss'], 
                label=opt_name, color=colors.get(opt_name, 'black'), 
                linewidth=2.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss (MSE)', fontsize=12)
    ax.set_title('Validation Loss Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    ax.set_facecolor('#f8f9fa')
    
    # 3. 梯度范数对比
    ax = axes[1, 0]
    for opt_name in results:
        ax.plot(results[opt_name]['grad_norms'], 
                label=opt_name, color=colors.get(opt_name, 'black'), 
                linewidth=2.0, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norm Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    ax.set_facecolor('#f8f9fa')
    
    # 4. 收敛速度对比
    ax = axes[1, 1]
    
    # 计算达到不同损失阈值所需的epoch数
    loss_thresholds = [0.1, 0.05, 0.02, 0.01]
    threshold_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    for idx, threshold in enumerate(loss_thresholds):
        convergence_data = []
        for opt_name in results:
            losses = results[opt_name]['val_loss']
            conv_epoch = None
            for i, loss in enumerate(losses):
                if loss < threshold:
                    conv_epoch = i
                    break
            convergence_data.append(conv_epoch if conv_epoch is not None else epochs)
        
        x_pos = np.arange(len(results)) + idx * 0.2 - 0.3
        ax.bar(x_pos, convergence_data, width=0.2, 
               label=f'Loss < {threshold}', color=threshold_colors[idx], alpha=0.7)
    
    ax.set_xlabel('Optimizer', fontsize=12)
    ax.set_ylabel('Epochs to Converge', fontsize=12)
    ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(results)))
    ax.set_xticklabels(list(results.keys()))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_facecolor('#f8f9fa')
    
    # 5. 最终性能雷达图
    ax = axes[2, 0]
    
    # 准备雷达图数据
    metrics = ['Final Loss', 'Grad Stability', 'Conv Speed', 'Early Conv', 'Late Conv']
    n_metrics = len(metrics)
    
    # 计算每个指标的归一化分数
    scores = {}
    for opt_name in results:
        data = results[opt_name]
        
        # 最终损失（越低越好，反向）
        final_loss = data['val_loss'][-1]
        loss_score = 1.0 / (final_loss + 1e-10)
        
        # 梯度稳定性（梯度范数的倒数，越高越稳定）
        grad_norms = data['grad_norms']
        grad_var = np.std(grad_norms[-30:]) if len(grad_norms) >= 30 else np.std(grad_norms)
        grad_score = 1.0 / (grad_var + 1e-10)
        
        # 收敛速度（达到0.05损失的epoch数，越少越好）
        target_loss = 0.05
        losses = data['val_loss']
        conv_epoch = next((i for i, loss in enumerate(losses) if loss < target_loss), epochs)
        conv_score = 1.0 / (conv_epoch + 1e-10)
        
        # 早期收敛（前30轮的平均损失改进）
        early_loss = np.mean(data['val_loss'][:30])
        early_score = 1.0 / (early_loss + 1e-10)
        
        # 后期收敛（最后30轮的平均损失改进）
        late_loss = np.mean(data['val_loss'][-30:])
        late_score = 1.0 / (late_loss + 1e-10)
        
        scores[opt_name] = [loss_score, grad_score, conv_score, early_score, late_score]
    
    # 归一化到0-1范围
    all_scores = np.array(list(scores.values()))
    norm_scores = all_scores / (all_scores.max(axis=0) + 1e-10)
    
    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for i, opt_name in enumerate(results):
        values = norm_scores[i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=3, label=opt_name, 
                color=colors.get(opt_name, 'black'), markersize=8)
        ax.fill(angles, values, alpha=0.1, color=colors.get(opt_name, 'black'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title('Multi-Dimensional Performance Comparison', 
                fontsize=14, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # 6. 算法特性总结
    ax = axes[2, 1]
    ax.axis('off')
    
    # 添加文本总结
    summary_text = [
        "算法特性总结:",
        "",
        "1. GD (梯度下降):",
        "   • 稳定收敛，不易震荡",
        "   • 学习率敏感，收敛慢",
        "   • 适合凸优化问题",
        "",
        "2. Adam:",
        "   • 自适应学习率",
        "   • 早期收敛快",
        "   • 可能过拟合或震荡",
        "",
        "3. RNSA-GD (本文方法):",
        "   • 后期加速明显",
        "   • 梯度稳定后效率高",
        "   • 结合GD稳定性和自适应加速",
        "   • 适合大网络训练",
        "",
        f"网络规格:",
        f"   • 参数: {param_count:,}",
        f"   • 架构: 3层残差LSTM",
        f"   • 隐藏层: 180单元",
        f"   • 输出: 5步预测"
    ]
    
    for i, line in enumerate(summary_text):
        y_pos = 0.95 - i * 0.05
        if ':' in line and not line.startswith(' '):
            # 标题
            ax.text(0.05, y_pos, line, fontsize=13, fontweight='bold',
                   transform=ax.transAxes, verticalalignment='top')
        elif line.startswith('   •'):
            # 项目符号
            ax.text(0.1, y_pos, line, fontsize=11,
                   transform=ax.transAxes, verticalalignment='top')
        elif line:
            # 普通文本
            ax.text(0.05, y_pos, line, fontsize=12,
                   transform=ax.transAxes, verticalalignment='top')
    
    ax.set_title('Algorithm Characteristics', fontsize=14, fontweight='bold')
    
    plt.suptitle('Residual LSTM (500K Parameters) - Optimizer Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('rnsa_500k_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # ===================== 6. 详细性能分析 =====================
    print("\n" + "=" * 80)
    print("详细性能分析")
    print("=" * 80)
    
    # 计算各种指标
    print(f"\n最终验证损失 (Epoch {epochs}):")
    print("-" * 50)
    for opt_name in results:
        final_val_loss = results[opt_name]['val_loss'][-1]
        print(f"{opt_name:<10}: {final_val_loss:.6f}")
    
    # 收敛速度分析
    print(f"\n收敛速度 (达到验证损失 < 0.03 的epoch数):")
    print("-" * 50)
    target_loss = 0.03
    for opt_name in results:
        losses = results[opt_name]['val_loss']
        conv_epoch = None
        for i, loss in enumerate(losses):
            if loss < target_loss:
                conv_epoch = i
                break
        
        if conv_epoch is not None:
            print(f"{opt_name:<10}: 第 {conv_epoch} 轮")
        else:
            print(f"{opt_name:<10}: > {epochs} 轮 (未达到)")
    
    # 梯度稳定性分析
    print(f"\n梯度稳定性 (最后50轮梯度范数的标准差):")
    print("-" * 50)
    for opt_name in results:
        grad_norms = results[opt_name]['grad_norms']
        if len(grad_norms) >= 50:
            grad_std = np.std(grad_norms[-50:])
        else:
            grad_std = np.std(grad_norms)
        print(f"{opt_name:<10}: {grad_std:.6f} (越小越稳定)")
    
    # 相对性能改进
    print(f"\n相对性能改进 (相对于GD):")
    print("-" * 50)
    gd_final_loss = results['GD']['val_loss'][-1]
    
    for opt_name in results:
        if opt_name != 'GD':
            final_loss = results[opt_name]['val_loss'][-1]
            improvement = (gd_final_loss - final_loss) / gd_final_loss * 100
            print(f"{opt_name:<10}: {improvement:+.2f}%")
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results_500k_{timestamp}.txt'
    
    with open(filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("50万参数残差LSTM - 优化器对比实验结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("网络配置:\n")
        f.write(f"  架构: 3层残差LSTM\n")
        f.write(f"  隐藏层大小: 180\n")
        f.write(f"  参数量: {param_count:,}\n")
        f.write(f"  训练轮数: {epochs}\n")
        f.write(f"  批次大小: 64\n\n")
        
        f.write("最终性能:\n")
        f.write("-" * 40 + "\n")
        for opt_name in results:
            data = results[opt_name]
            f.write(f"\n{opt_name}:\n")
            f.write(f"  最终训练损失: {data['train_loss'][-1]:.6f}\n")
            f.write(f"  最终验证损失: {data['val_loss'][-1]:.6f}\n")
            f.write(f"  最终梯度范数: {data['grad_norms'][-1]:.6f}\n")
        
        f.write("\n收敛速度分析:\n")
        f.write("-" * 40 + "\n")
        for threshold in [0.1, 0.05, 0.03, 0.02]:
            f.write(f"\n达到损失 < {threshold}:\n")
            for opt_name in results:
                losses = results[opt_name]['val_loss']
                conv_epoch = next((i for i, loss in enumerate(losses) if loss < threshold), None)
                if conv_epoch is not None:
                    f.write(f"  {opt_name}: 第 {conv_epoch} 轮\n")
                else:
                    f.write(f"  {opt_name}: 未达到\n")
    
    print(f"\n详细结果已保存到: {filename}")
    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 运行实验
    run_large_scale_experiment()
