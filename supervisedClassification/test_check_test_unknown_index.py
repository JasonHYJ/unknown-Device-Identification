"""
检查评估结果生成的test_result.csv和unknown_result.csv中的每一个index所对应的sample_test.csv和processed_unknown.csv情况。
正常来说，是每一行的样本一一对应的。
这个代码就是为了检查对应情况的，目前是只输出了前五行的一个实际情况，预测结果unknown_results，评估对象processed_unknown和未知样本uk_unknown，这三个文件中的对应样本情况。
"""

import pandas as pd

# 加载所有相关文件
test_results = pd.read_csv("/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/eval_results/test_results.csv")
sampled_test = pd.read_csv("/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/eval_results/sampled_test.csv")
uk_test = pd.read_csv("/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/uk_test.csv")
unknown_results = pd.read_csv("/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/eval_results/unknown_results.csv")
processed_unknown = pd.read_csv("/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/eval_results/processed_unknown.csv")
uk_unknown = pd.read_csv("/home/hyj/unknownDeviceIdentification/dataset/11_multitask_training/uk/uk_unknown.csv")

# 检查 unknown_results.csv
for index in range(5):  # 检查前 5 个样本
    result_row = unknown_results[unknown_results['index'] == index]
    processed_row = processed_unknown.iloc[index]
    sample_file = processed_row['sample_file']
    original_row = uk_unknown[uk_unknown['sample_file'] == sample_file]
    print(f"Index {index}:")
    print(f"  unknown_results: {result_row[['true_type', 'pred_type', 'is_behavior']].to_dict()}")
    print(f"  processed_unknown: {processed_row[['type_label', 'brand_label', 'device_label', 'is_behavior']].to_dict()}")
    print(f"  uk_unknown: {original_row[['type_label', 'brand_label', 'device_label', 'is_behavior']].to_dict()}")

# 检查 test_results.csv
for index in range(5):
    result_row = test_results[test_results['index'] == index]
    sampled_row = sampled_test.iloc[index]
    sample_file = sampled_row['sample_file']
    original_row = uk_test[uk_test['sample_file'] == sample_file]
    print(f"Index {index}:")
    print(f"  test_results: {result_row[['true_type', 'pred_type', 'is_behavior']].to_dict()}")
    print(f"  sampled_test: {sampled_row[['type_label', 'brand_label', 'device_label', 'is_behavior']].to_dict()}")
    print(f"  uk_test: {original_row[['type_label', 'brand_label', 'device_label', 'is_behavior']].to_dict()}")