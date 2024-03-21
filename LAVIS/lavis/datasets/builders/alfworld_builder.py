from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.alfworld_datasets import AlfworldDataset

from lavis.common.registry import registry

@registry.register_builder("alfworld_cap")
class AlfworldCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = AlfworldDataset
    eval_dataset_cls = AlfworldDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/alfworld/defaults_caption.yaml",
    }