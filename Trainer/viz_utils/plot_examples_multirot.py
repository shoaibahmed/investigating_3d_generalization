import sys
import itertools

import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
import webdataset as wds

sys.path.append('..')
from dataset import CoordsDataset, getWebDatasetWrapper
from bg_transform import RandomBackgroundTransform


def plot(x, y=None, model=None, output_file=None, transforms=None, plot_rows=3, num_plots_per_row=3):
    if x is None:
        print("Plotting dummy data!")
        B, C, H, W = 100, 3, 224, 224
        num_cls = 10
        x = torch.empty(B, C, H, W).uniform_(0.0, 1.0)
        y = torch.LongTensor(B).random_(0, num_cls)
        print(y.unique())
    plot_size = 3
    fig, ax = plt.subplots(plot_rows, num_plots_per_row, figsize=(plot_size * num_plots_per_row, plot_size * plot_rows), sharex=True, sharey=True)

    pred = None
    if model is not None and y is not None:
        pred = model(x).argmax(dim=1).detach().cpu().numpy()

    input = x.cpu().numpy()
    is_grayscale = input.shape[1] == 1
    input = input[:, 0, :, :] if is_grayscale else np.transpose(input, (0, 2, 3, 1))
    if transforms is not None:
        dataset_mean, dataset_std = transforms
        input = np.clip((input * np.array(dataset_std)[None, :, None, None]) + np.array(dataset_mean)[None, :, None, None], 0.0, 1.0)

    assert input.min() >= 0.0 and input.max() <= 1.0, "Data should be normalized to [0.0, 1.0]. Please use the transforms argument to pass in the mean and std used for normalization."

    for idx in range(len(input)):
        ax[idx // num_plots_per_row, idx % num_plots_per_row].imshow(input[idx], cmap='gray' if is_grayscale else None)
        correct = (y[idx] == pred[idx]) if pred is not None else True
        if y is not None:
            ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(f"Label: {y[idx]}", color='b' if pred is None else 'g' if correct else 'r')

        if idx == plot_rows * num_plots_per_row - 1:
            break

    for a in ax.ravel():
        a.set_axis_off()

        # Turn off tick labels
        a.set_yticklabels([])
        a.set_xticklabels([])

    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, bbox_inches=0.0, pad_inches=0)
    plt.show()
    plt.close()


load_coords = False
load_models = False
use_rand_bg_transform = True

root_dir = "/netscratch/siddiqui/Datasets/Paperclips_v6_coords/"
if load_models:
    url = "/netscratch/siddiqui/Datasets/Chairs_train_v6/Chairs_v6_classes_10_eval.tar.xz"
else:
    url = "/netscratch/siddiqui/Datasets/Paperclips_train_v6/Paperclips_v6_classes_10_eval.tar.xz"

additional_transforms = [RandomBackgroundTransform(image_regex="../landscape_images/archive/*.jpg")] if use_rand_bg_transform else []
val_transform = transforms.Compose(
    additional_transforms +
    [
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
print("Transforms:", transforms)

if load_coords:
    print("Loading coords dataset:", root_dir)
    assert not load_models
    dataset = CoordsDataset(root_dir, "test", transform=val_transform, return_images=True,
                            multirot_stride=10, num_classes=10)
else:
    print("Loading WebDataset:", url)
    dataset = getWebDatasetWrapper(url, is_train=False, transform=val_transform,
                                   multirot_stride=10, load_models=load_models)

loader = wds.WebLoader(dataset, num_workers=4, batch_size=32)  # Convert to webloader
dataset_rotation_axis_list = dataset.get_rotation_axis_list()

num_examples = 1
num_views = 6
rotation_list = list(range(0, 360, 360 // num_views))
rotation_list = list(itertools.product(rotation_list, rotation_list))
print("Rotation list:", rotation_list)
assert num_examples == 1, "code is only designed to support one example"

rotation_axis = "xy"
idxmap = {"x": 0, "y": 1, "z": 2}
assert len(rotation_axis) == 2
selected_rotation_idx = [idxmap[x] for x in rotation_axis]
print("Selected rotation idx:", selected_rotation_idx)

example_list = list(range(num_examples))
image_dict = {ex: {k: None for k in rotation_list} for ex in example_list}

total_ex = 0
for idx, (data, (target, superclass_target, hardnegative_idx, rotation_axis_idx, rotation_angle)) in enumerate(loader):
    # print(type(data), data.shape)
    # print(target, superclass_target, hardnegative_idx, rotation_axis_idx, rotation_angle)
    rotation_angle = torch.stack(rotation_angle, dim=1)
    total_ex += len(data)
    for i in range(len(target)):
        ex_idx = int(target[i])
        if ex_idx < num_examples:
            current_rot_axis = dataset_rotation_axis_list[rotation_axis_idx[i]]
            if current_rot_axis != rotation_axis:  # Skip incorrect rotation axis
                continue
            rot_angle = [int(rotation_angle[i][j]) for j in range(len(rotation_angle[i]))]
            rot_angle = tuple([rot_angle[j] for j in selected_rotation_idx])
            if rot_angle in rotation_list:
                assert image_dict[ex_idx][rot_angle] is None
                image_dict[ex_idx][rot_angle] = data[i]
print("Total number of examples in eval set:", total_ex)

# Ensure that none of the slots are none
assert not any([image_dict[ex][rot] is None for rot in rotation_list for ex in example_list]), [rot for rot in rotation_list for ex in example_list if image_dict[ex][rot] is None]
image_list = []
for ex in example_list:
    for rot in rotation_list:
        image_list.append(image_dict[ex][rot])
print("Image list length:", len(image_list))

# Plot the images
x = torch.stack(image_list, dim=0)
print("Image shape:", x.shape)
output_file = f"{'chairs' if load_models else 'paperclips'}{'_coords' if load_coords else ''}_rot_val_examples_multirot_{rotation_axis}{'_rand_bg' if use_rand_bg_transform else ''}.png"
plot(x, output_file=output_file, plot_rows=num_views, num_plots_per_row=num_views)
print("Output written to file:", output_file)
