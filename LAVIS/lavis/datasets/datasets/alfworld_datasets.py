import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": self._post_process_path(ann["image"]),
                "text_input": ann["text_input"],
                "text_output": ann["text_output"],
                "image": sample["image"],
            }
        )


class AlfworldDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, self._post_process_path(ann["image"]))
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        text_input = self.text_processor(ann["text_input"])
        text_output = self.text_processor(ann["text_output"], add_prompt=False)

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output
        }
        
    def _post_process_path(self, path):
        # temporally customized function 
        return path[2:]