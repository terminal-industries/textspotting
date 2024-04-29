import json
import random
def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def filter_data(json_data):
    data_list_true = []
    data_list_false = []

    # 遍历每个 img_data，检查其中的 instances
    for img_data in json_data['data_list']:
        has_true = any(instance['vertical_text'] for instance in img_data['instances'])
        has_false = all(not instance['vertical_text'] for instance in img_data['instances'])

        if has_true:
            data_list_true.append(img_data)
        if has_false:
            data_list_false.append(img_data)

    # 确保 data_list_true 和 data_list_false 的数量相等
    count_to_match = len(data_list_true)
    if count_to_match < len(data_list_false):
        data_list_false = data_list_false[:count_to_match]

    return data_list_true, data_list_false
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
def serialize_json(data):
    """
    Serialize a data structure containing numpy types to a JSON string.

    Parameters:
    - data: The data structure to serialize.

    Returns:
    - A JSON string representation of `data` with numpy types converted to native Python types.
    """
    try:
        json_str = json.dumps(data, cls=NumpyEncoder)
        return json_str
    except TypeError as e:
        return str(e)
# 使用示例

data = {"metainfo": {"dataset_type": "TextDetDataset",
                     "task_name": "textdet",
                     "category": [{"id": 0, "name": "text"}]}
        }

if __name__ == "__main__":
    filepath = 'textspotting_train.json'  # 将 'path_to_your_json_file.json' 替换成你的 JSON 文件路径
    json_data = load_json_data(filepath)
    data_list_true, data_list_false = filter_data(json_data)
    
    combined_list = data_list_true + data_list_false

    # Shuffle the combined list in place
    random.shuffle(combined_list)


    # 打印结果或进行其他处理
    
    print("Data with at least one vertical_text True:")
    print(len(data_list_true))
    print("Data with all vertical_text False:")
    print(len(data_list_false))


    data['data_list'] = combined_list
    
    json_str = serialize_json(data)
    with open('textspotting_train_sample.json', 'w') as file:
        file.write(json_str)

