import numpy as np

# 示例：指定一个具体的嵌入文件路径
npy_path = "/home/hyj/unknownDeviceIdentification/dataset/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings/test/uk/allure-speaker/idle/2019-04-25_idle/allure-speaker__2019-04-25_idle__00001_raw_embed.npy"

# 加载嵌入向量
embedding = np.load(npy_path)

# 查看维度和前几维
print("嵌入向量形状:", embedding.shape)
print("128维嵌入值:", embedding[:130])
