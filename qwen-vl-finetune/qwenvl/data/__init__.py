import re

# Define placeholders for dataset paths
ann_dir = "/mnt/ssd1/sh/projects/LGSC/LLaVAOV/instructions_lgsc_groma_style"
data_dir = "/mnt/ssd1/datasets"

COCO_MULTI = {
    "annotation_path": ann_dir + "coco/coco_train_instructions_multi_categories.json",
    "data_path": data_dir + "COCO/all"
}

COCO_SINGLE = {
    "annotation_path": ann_dir + "coco/coco_train_instructions_single_categories.json",
    "data_path": data_dir + "COCO/all"
}

REFCOCO = {
    "annotation_path": ann_dir + "refcoco/refcoco_train_instructions.json",
    "data_path": data_dir + "COCO/all"
}

REFCOCO_PLUS = {
    "annotation_path": ann_dir + "refcoco+/refcoco+_train_instructions.json",
    "data_path": data_dir + "COCO/all"
}

REFCOCOG = {
    "annotation_path": ann_dir + "refcocog/refcocog_train_instructions.json",
    "data_path": data_dir + "COCO/all"
}

data_dict = {
    # Detection
    "coco_multi": COCO_MULTI,
    "coco_single": COCO_SINGLE,
    # Referring Expression Comprehension
    "refcoco": REFCOCO,
    "refcoco_plus": REFCOCO_PLUS,
    "refcocog": REFCOCOG,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%([\d.]+)$", dataset_name)
    if match:
        return float(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%([\d.]+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
