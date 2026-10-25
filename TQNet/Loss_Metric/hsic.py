import torch
"""
    计算二维张量中所有样本对之间的**欧氏距离平方矩阵**
    采用向量化实现，避免双重循环，时间复杂度O(m²d)，m为样本数，d为特征维度
    数学原理：(a-b)² = a² - 2ab + b²
    """
def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)# 调用上面的函数计算所有样本对的欧氏距离平方
    return torch.exp(-pairwise_distances_ /sigma) # 高斯核公式：K(i,j) = exp(-||x_i - x_j||² / σ)

# HSIC用于衡量两个随机变量之间的统计独立性：HSIC(X,Y) = (1/(m-1)²) * trace(L H K H)
# 其中K和L分别是X和Y的核矩阵，H是中心化矩阵，约束预测值与真实值之间的相关性
def HSIC(x, y, s_x=1, s_y=1):
    m,_ = x.shape #batch size# 获取样本数量m，忽略特征维度
    K = GaussianKernelMatrix(x,s_x)# 分别计算x和y的高斯核矩阵
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))# 构造中心化矩阵H，用于消除核矩阵的均值影响
    H = H.cuda()
    # H = H.double().cuda()
    # # 矩阵乘法顺序：L @ H @ K @ H，然后计算迹（对角线元素之和）
    # 除以(m-1)²是无偏估计的归一化因子（部分文献使用m²作为有偏估计）
    # torch.mm 是 PyTorch 中严格二维张量的矩阵乘法
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC

# outputs = torch.randn(32,7).cuda()
# batch_y = torch.normal(1,1,(32,7)).cuda()
#
# print(HSIC(outputs,batch_y))