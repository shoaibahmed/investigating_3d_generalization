import os
import json

import cv2
import numpy as np

import matplotlib.pyplot as plt


file_path = "/mnt/sas/Datasets/Paperclips/paperclip_0/coords.jsonl"
assert os.path.exists(file_path), file_path

with open(file_path, "r") as f:
    lines = f.readlines()

video_name = f'test_video_paperclip_0.avi'
frame = None
text_scale = 0.45
text_thickness = 1

for i, line in enumerate(lines):
    json_dict = json.loads(line)
    print(json_dict)
    joint_positions = np.array([json_dict[f"j{i}"]["image"] for i in range(8)])
    print("Joint positions:", joint_positions)
    
    capture_width = 1920
    capture_height = 1080
    aspect_ratio = float(capture_width) / capture_height
    print("Aspect ratio:", aspect_ratio)
    
    new_size = 256
    scaled_pos = joint_positions
    scaled_pos[:, 0] = joint_positions[:, 0] * new_size * aspect_ratio - 96
    scaled_pos[:, 1] = joint_positions[:, 1] * new_size
    scaled_pos = scaled_pos.astype(np.int32)
    
    img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    prev_coords = None
    for (x, y) in scaled_pos:
        # img[y, x, :] = 255
        color = (255, 255, 255)
        cv2.circle(img, (x, y), radius=2, color=color, thickness=1)
        if prev_coords is not None:  # Add a line between the two
            cv2.line(img, prev_coords, (x, y), color=color, thickness=1)
        prev_coords = (x, y)
    
    if i == 0:
        ref_img = cv2.imread("reference_img.jpg")
        print(img.shape, img.dtype)
        print(ref_img.shape, ref_img.dtype)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax[0].imshow(img)
        ax[1].imshow(ref_img)
        
        # for a in ax.ravel():
        #     a.set_axis_off()
        #     a.set_yticklabels([])
        #     a.set_xticklabels([])

        fig.tight_layout()
        output_file = "output.png"
        plt.savefig(output_file, bbox_inches=0.0, pad_inches=0, dpi=300)
        # plt.show()
        
        height, width, layers = img.shape

        fps = 30
        video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    video.write(img)

video.release()
