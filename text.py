import torch
print(torch.cuda.is_available())  # 如果返回 True，则表示支持 CUDA
print(torch.cuda.device_count())  # 返回可用的 GPU 数量
print(torch.cuda.get_device_name(0))  # 返回第一个 GPU 的名称
