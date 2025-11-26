"""VOC class name utilities shared between dataset and inference.

The 20 object classes follow the standard Pascal VOC ordering.
"""

VOC_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_NAME_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASS_NAMES)}
