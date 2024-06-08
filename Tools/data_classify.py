import json

data = {
    "bronze_vessel": [
        {
            "name": "鼎",
            "id": 0
        },
        {
            "name": "豆",
            "id": 1
        },
        {
            "name": "敦",
            "id": 2
        },
        {
            "name": "簠",
            "id": 3
        },
        {
            "name": "鬲",
            "id": 4
        },
        {
            "name": "觚",
            "id": 5
        },
        {
            "name": "簋",
            "id": 6
        },
        {
            "name": "盉",
            "id": 7
        },
        {
            "name": "壶",
            "id": 8
        },
        {
            "name": "斝",
            "id": 9
        },
        {
            "name": "爵",
            "id": 10
        },
        {
            "name": "罍",
            "id": 11
        },
        {
            "name": "盨",
            "id": 12
        },
        {
            "name": "甗",
            "id": 13
        },
        {
            "name": "卣",
            "id": 14
        },
        {
            "name": "觯",
            "id": 15
        },
        {
            "name": "尊",
            "id": 16
        }
    ]
}


def save_dict_to_json(file_path, dictionary):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dictionary, json_file, indent=4, ensure_ascii=False)

save_dict_to_json('../bronze_vessel.json', data)
