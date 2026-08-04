class RandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        image = sample["image"]
        boxes = sample["boxes"]

        if random.random() >= self.probability:
            return sample

        width = image.shape[1]
        image = image[:, ::-1]

        boxes = boxes.copy()
        x1 = boxes[:, 0].copy()
        x2 = boxes[:, 2].copy()

        boxes[:, 0] = width - x2
        boxes[:, 2] = width - x1

        sample["image"] = image
        sample["boxes"] = boxes
        return sample