import gymnasium as gym
from gymnasium.utils import seeding

# 初始化随机数生成器（设置种子确保可复现）
rng, seed = seeding.np_random(42)

# 从泊松分布中采样 100 个数字，λ=10 表示平均值为 10
samples = rng.poisson(lam=20, size=100)

# 打印结果
print(samples)
