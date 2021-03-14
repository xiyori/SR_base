import cv2
import albumentations as albu
# from albumentations.augmentations.transforms import ImageCompression


def get_training_transform(crop_size: int, scale: float):
    return albu.Compose([
        albu.RandomScale(scale_limit=(scale - 1.0, scale - 1.0), interpolation=cv2.INTER_AREA, always_apply=True),
        albu.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        albu.HorizontalFlip(p=0.5)
    ])


def get_validation_transform(scale: float):
    return albu.RandomScale(
        scale_limit=(scale - 1.0, scale - 1.0), interpolation=cv2.INTER_AREA, always_apply=True
    )


def get_input_image_augmentation():
    return albu.Downscale(scale_min=0.2, scale_max=0.3, interpolation=cv2.INTER_AREA, p=0.5)


def get_generate_noise_transform(width: int, height: int):
    return albu.Compose([
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True),
        albu.CenterCrop(height=height, width=width, always_apply=True)
    ])

    # albu.Compose([
    #     albu.OneOf(
    #         [
    #             albu.Compose([
    #                 albu.Blur(blur_limit=3, p=1),
    #                 albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.9, 1.0), p=1)
    #             ], p=1),
    #             albu.Compose([
    #                 albu.GaussianBlur(blur_limit=(3, 3), p=1),
    #                 albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.9, 1.0), p=1)
    #             ], p=1),
    #             albu.Downscale(scale_min=0.4, scale_max=0.6, interpolation=cv2.INTER_AREA, p=1)
    #         ],
    #         p=0.5
    #     ),
    #     albu.OneOf(
    #         [
    #             albu.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255), p=1),
    #             albu.GaussNoise(var_limit=(5.0, 30.0), p=1),
    #             albu.ImageCompression(quality_lower=95,
    #                                   compression_type=ImageCompression.ImageCompressionType.JPEG, p=1)
    #         ],
    #         p=1
    #     )
    # ])