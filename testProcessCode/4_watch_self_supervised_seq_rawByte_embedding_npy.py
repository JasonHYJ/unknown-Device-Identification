import numpy as np

# 示例：指定一个具体的嵌入文件路径
# npy_path = "/home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_rawbyte_embeddings/train/uk/allure-speaker/activity/android_lan_audio_off/allure-speaker__android_lan_audio_off__00001_raw_embed.npy"
# npy_path = "/home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_rawbyte_embeddings/train/uk/allure-speaker/idle/2019-04-25_idle/allure-speaker__2019-04-25_idle__00007_raw_embed.npy"

# npy_path = "/home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_sequence_embeddings/train/uk/allure-speaker/activity/android_lan_audio_off/allure-speaker__android_lan_audio_off__00001_seq_embed.npy"
npy_path = "/home/hyj/unknownDeviceIdentification/dataset/9_learned_embeddings/9_learned_sequence_embeddings/train/uk/allure-speaker/idle/2019-04-25_idle/allure-speaker__2019-04-25_idle__00007_seq_embed.npy"

# 加载嵌入向量
embedding = np.load(npy_path)

# 查看维度和前几维
print("嵌入向量形状:", embedding.shape)
print("自监督学习后的128维嵌入向量:", embedding[:130])

