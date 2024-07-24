import json

from configs._base_.datasets.bonai_instance import data_root, test_ann_file

with open(test_ann_file, 'r') as f:
    dataset = json.load(f)

for obj in dataset['annotations']:
    obj['segmentation'] = obj['roof_mask']

with open(data_root + '/bonai_shanghai_xian_test_modified.json', 'w') as f:
    json.dump(dataset, f, indent=2)
