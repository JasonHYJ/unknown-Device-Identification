import numpy as np

# 读取 .npz 文件
# data = np.load("/home/hyj/unknownDeviceIdentification/dataset/6_extracted_features/6_csv_sequence_feature_matrix/uk/charger-camera/activity/android_lan_watch/charger-camera__android_lan_watch__00001_seq.npz")
data = np.load("/home/hyj/unknownDeviceIdentification/dataset/7_cleaned_features/7_cleaned_sequence_feature_matrix/uk/allure-speaker/idle/2019-04-25_idle/allure-speaker__2019-04-25_idle__00001_seq.npz")

# 查看所有字段名
print("字段名列表：", data.files)

# 查看具体某个字段
print(data["feature_matrix"])
print("feature_matrix 的 shape:", data["feature_matrix"].shape)
print("mask:", data["mask"])
print("设备类型标签:", data["type_label"])
print("厂商标签:", data["brand_label"])
print("设备型号标签:", data["device_label"])
print("原始样本文件:", data["sample_file"])

feature_matrix = data['feature_matrix']
print("是否含NaN:", np.isnan(feature_matrix).any())
print("最大最小值:", feature_matrix.max(), feature_matrix.min())
print("均值/方差:", feature_matrix.mean(), feature_matrix.std())
