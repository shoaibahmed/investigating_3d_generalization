import glob
import numpy as np
import cv2

from PIL import Image


class RandomBackgroundTransform:
    def __init__(self, image_regex="./landscape_images/archive/*.jpg") -> None:
        # Images taken from: https://www.kaggle.com/datasets/arnaud58/landscape-pictures
        # Load the list of background images here
        self.background_img_paths = list(glob.glob(image_regex))
        print("Number of background images loaded:", len(self.background_img_paths))
        
        # Preload all the images
        self.img_size = (256, 256)  # Before cropping
        self.preload_images = True
        if self.preload_images:
            print("Preloading landscape images...")
            self.background_imgs = [cv2.resize(self.get_image(img_path), self.img_size) for img_path in self.background_img_paths]
            print(f"{len(self.background_imgs)} images preloaded in memory...")
    
    def __call__(self, img):
        img = np.array(img)
        
        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # threshold input image as mask
        # mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.threshold(gray, 25, 1, cv2.THRESH_BINARY)[1]

        # negate mask
        mask = 1. - mask
        mask = mask.clip(0., 1.)

        # Replace mask on three channels
        assert len(mask.shape) == 2
        mask = np.stack([mask, mask, mask], axis=2)
        assert len(mask.shape) == 3
        
        # Load the background img
        bg_img = self.get_new_bg()
        # bg_img = cv2.resize(bg_img, img.shape[:2])
        
        # Combine the bg image and fg image
        combined_img = bg_img * mask + img * (1 - mask)
        combined_img = np.clip(combined_img, 0, 255).astype(np.uint8)
        
        assert combined_img.shape == img.shape, f"{combined_img.shape} != {img.shape}"
        assert combined_img.dtype == img.dtype, f"{combined_img.dtype} != {img.dtype}"
        
        return Image.fromarray(combined_img)
    
    def get_new_bg(self):
        if self.preload_images:
            img_idx = np.random.choice(list(range(len(self.background_imgs))))
            return self.background_imgs[img_idx]
        
        img_path = np.random.choice(self.background_img_paths)
        img = self.get_image(img_path)
        return cv2.resize(img, self.img_size)
    
    def get_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return np.array(img)


if __name__ == "__main__":
    transform = RandomBackgroundTransform()
    model_idx = np.random.randint(0, 5000)
    image_paths = list(glob.glob(f"/mnt/sas/Datasets/Chairs_v4/model_{model_idx}/*/*.jpg"))
    input_img_path = np.random.choice(image_paths)
    
    print("Selected image:", input_img_path)
    img = cv2.imread(input_img_path)
    print("Image shape:", img.shape)
    
    transformed_img = transform(img)
    print("Transformed image shape:", transformed_img.shape)
    
    cv2.imwrite("transformed_img.jpg", transformed_img)
