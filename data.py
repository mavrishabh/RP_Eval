# %%
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout

# %%
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %%
def load_data(path):
    """ X = Images and Y = masks """
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))
    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))
    return (train_x, train_y), (test_x, test_y)

# %%
def rotate_image(image, angle):
    H, W = image.shape[:2]
    center = (W // 2, H // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_image

# %%
def augment_data(images, masks, save_path, augment=True, rotation_angles=None):
    H = 512
    W = 512

    if rotation_angles is None:
        rotation_angles = []

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting names """
        name = x.split("\\")[-1].split(".")[0]
        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        # Resize images and masks
        x = cv2.resize(x, (W, H))
        y = cv2.resize(y, (W, H))

        # Save original image and mask
        tmp_image_name = f"{name}_orig.jpg"
        tmp_mask_name = f"{name}_orig.jpg"
        image_path = os.path.join(save_path, "image", tmp_image_name)
        mask_path = os.path.join(save_path, "mask", tmp_mask_name)
        cv2.imwrite(image_path, x)
        cv2.imwrite(mask_path, y)

        if augment:
            for angle in rotation_angles:
                rotated_x = rotate_image(x, angle)
                rotated_y = rotate_image(y, angle)

                tmp_image_name = f"{name}_rot{angle}.jpg"
                tmp_mask_name = f"{name}_rot{angle}.jpg"

                image_path = os.path.join(save_path, "image", tmp_image_name)
                mask_path = os.path.join(save_path, "mask", tmp_mask_name)

                cv2.imwrite(image_path, rotated_x)
                cv2.imwrite(mask_path, rotated_y)

# %%
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "DRIVE/"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Creating directories """
    create_dir("new_data/train/image")
    create_dir("new_data/train/mask")
    create_dir("new_data/test/image")
    create_dir("new_data/test/mask")

    """ Augment training data with rotations """
    augment_data(train_x, train_y, "new_data/train/", augment=True, rotation_angles=[60, 120, 180, 240, 300, 360])

    """ Copy test data without augmentation """
    augment_data(test_x, test_y, "new_data/test/", augment=False)
