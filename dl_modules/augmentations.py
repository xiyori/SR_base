import albumentations as albu


def get_training_augmentation(crop_size: int):
    return albu.Compose([
        albu.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        albu.HorizontalFlip(p=0.5)
    ])


def get_validation_augmentation(crop_size: int):
    return albu.Compose([
        albu.PadIfNeeded(min_height=crop_size, min_width=crop_size, always_apply=True),
        albu.CenterCrop(height=crop_size, width=crop_size, always_apply=True)
    ])


def get_input_image_augmentation():
    return albu.Compose([
        albu.OneOf(
            [
                albu.Compose([
                    albu.Blur(blur_limit=2, p=1),
                    albu.IAASharpen(alpha=(0.5, 0.8), lightness=(0.9, 1.0), p=1)
                ], p=1),
                albu.Compose([
                    albu.GaussianBlur(blur_limit=(3, 3), p=1),
                    albu.IAASharpen(alpha=(0.5, 0.8), lightness=(0.9, 1.0), p=1)
                ], p=1),
                albu.Downscale(scale_min=0.5, scale_max=0.5, interpolation=cv2.INTER_AREA, p=1)
            ],
            p=1
        ),
        albu.OneOf(
            [
                albu.IAAAdditiveGaussianNoise(p=1),
                albu.GaussNoise(var_limit=(5.0, 25.0), p=1),
                albu.ImageCompression(quality_lower=98, p=1)
            ],
            p=1
        )
    ])
