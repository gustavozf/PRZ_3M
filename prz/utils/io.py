import json

def dump_json(data, out_path: str):
    with open(out_path, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)