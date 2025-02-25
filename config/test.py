import yaml

def validate_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        print("YAML文件格式正确")
    except yaml.YAMLError as e:
        print(f"YAML文件格式错误: {e}")

# 验证配置文件
validate_yaml('config/config.yaml')