import torch

# 保存原本的 __repr__ 方法（便于恢复或在自定义中调用）
original_repr = torch.Tensor.__repr__

# 定义自定义的打印函数
def custom_repr(self):
    return f"{{Tensor: {tuple(self.shape)}, {self.device}, {self.dtype}}} {original_repr(self)}"

# 定义启用猴子补丁的函数
def enable_custom_repr():
    torch.Tensor.__repr__ = custom_repr  # 替换原方法

# （可选）定义恢复原方法的函数
def disable_custom_repr():
    torch.Tensor.__repr__ = original_repr
