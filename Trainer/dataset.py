import os
import glob
import natsort
import pathlib
from tqdm import tqdm

import pickle
import numpy as np
import pandas as pd

import torch
from torchvision import transforms

import webdataset as wds

import cv2
import json
from PIL import Image


class CoordsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, transform, return_images, multirot_stride, num_classes=None,
                 train_stride=None, training_views=None):
        assert split in ["train", "test"]
        assert not (train_stride is not None and training_views is not None), "Training stride and training views are mutually exclusive"
        assert train_stride is None or (isinstance(train_stride, int) and train_stride in list(range(361)))  # Can include 360 i.e. only one view
        assert training_views is None or isinstance(training_views, list) or isinstance(training_views, tuple), training_views
        assert split != "train" or not (train_stride is None and training_views is None), "Either the training stride of the training views must be specified in train mode"
        
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.return_eval_tuple = self.split == "test"
        self.return_images = return_images
        self.hardnegative_class_names = [0]  # no hard negatives
        self.multirot_stride = multirot_stride
        self.num_classes = num_classes
        
        dataset_cache_dir = "./dataset_cache/"
        if not os.path.exists(dataset_cache_dir):
            os.mkdir(dataset_cache_dir)
        
        training_views_str = ""
        if split == "train":
            training_views_str = '_'.join([str(x) for x in training_views]) if training_views is not None else None
            training_views_str = f"_train_views_{training_views_str}" if training_views_str is not None else f"_train_stride_{train_stride}"
        cache_file = os.path.join(dataset_cache_dir, "cache_paperclip_coords_multirot" + training_views_str + ("_cls_" + str(num_classes) if num_classes is not None else "") + ".pkl")
        
        if os.path.exists(cache_file):
            print("Loading dataset from cache file:", cache_file)
            with open(cache_file, "rb") as stream:
                dataset_object = pickle.load(stream)
            self.joint_pos, self.labels, self.frame_idx, self.superclass_labels, self.hardnegative_idx, self.rotation_labels, self.class_names, self.rotation_class_names = dataset_object
        
        else:
            self.files = glob.glob(f"{self.root_dir}/*/coords.jsonl")
            self.files = natsort.natsorted(self.files)
            print("Total files:", len(self.files), self.files[:5])
            
            # Define the class labels
            self.paperclip_idx = [int(x.split(os.sep)[-2].replace("paperclip_", "")) for x in self.files]
            
            if num_classes is not None:
                print("Fixing the number of classes to be:", num_classes)
                selected_files = [x < num_classes for x in self.paperclip_idx]
                num_selected_files = int(np.sum(selected_files))
                print(f"# classes: {num_classes} / Initial files: {len(selected_files)} / Selected files: {num_selected_files}")
                self.files = [x for i, x in enumerate(self.files) if selected_files[i]]
                self.paperclip_idx = [x for i, x in enumerate(self.paperclip_idx) if selected_files[i]]
                assert len(self.paperclip_idx) == num_selected_files
            
            paperclip_instances = natsort.natsorted(np.unique(self.paperclip_idx))
            print("Number of paperclip instances:", len(paperclip_instances))
            if num_classes is not None:
                assert len(paperclip_instances) == num_classes
            self.class_names = paperclip_instances
            
            interim_cache_file = os.path.join(dataset_cache_dir, "cache_interim_paperclip_coords_multirot" + ("_cls_" + str(num_classes) if num_classes is not None else "") + ".pkl")
            if not os.path.exists(interim_cache_file):
                # Parse all the JSONL files
                self.labels = []
                self.frame_idx = []  # Refers to all three axes of rotation
                self.rotation_axis = []
                self.joint_pos = []
                
                for coords_file in tqdm(self.files):
                    paperclip_idx = int(coords_file.split(os.sep)[-2].replace("paperclip_", ""))
                    assert 0 <= paperclip_idx < 10000, paperclip_idx
                    
                    with open(coords_file, "r") as f:
                        lines = f.readlines()

                    label_map = {}
                    for line in lines:
                        json_dict = json.loads(line)
                        joint_positions = np.array([json_dict[f"j{i}"]["image"] for i in range(8)])
                        
                        if json_dict["rot_axis"] not in label_map:
                            label_map[json_dict["rot_axis"]] = {}
                        
                        x, y, z = json_dict["rot"]["x"], json_dict["rot"]["y"], json_dict["rot"]["z"]
                        assert (x, y, z) not in label_map[json_dict["rot_axis"]]
                        label_map[json_dict["rot_axis"]][(x, y, z)] = joint_positions
                        
                        # Add all the corresponding values to the list
                        self.labels.append(paperclip_idx)
                        self.frame_idx.append((x, y, z))
                        self.rotation_axis.append(json_dict["rot_axis"])
                        self.joint_pos.append(joint_positions)
            
                with open(interim_cache_file, "wb") as stream:
                    pickle.dump([self.labels, self.frame_idx, self.rotation_axis, self.joint_pos], stream, protocol=4)  # Since protocol 4 is compatible with Python3.7
            else:
                print("Loading dataset from interim cache file:", interim_cache_file)
                with open(interim_cache_file, "rb") as stream:
                    self.labels, self.frame_idx, self.rotation_axis, self.joint_pos = pickle.load(stream)
            
            obtained_rotation_class_names = np.unique(self.rotation_axis)
            self.rotation_class_names = ["x", "y", "z", "xy", "xz", "yz"]  # TODO: Maybe also add an argument to specify rotation axes
            assert all([k in self.rotation_class_names for k in obtained_rotation_class_names]), obtained_rotation_class_names
            self.rot2label = {k: v for v, k in enumerate(self.rotation_class_names)}
            assert all([x in self.rotation_class_names for x in self.rotation_axis])
            self.rotation_labels = [self.rot2label[k] for k in self.rotation_axis]
            self.superclass_labels = self.labels
            self.hardnegative_idx = [0 for _ in range(len(self.labels))]
            
            df = pd.DataFrame()  # Dataframe for caching results
            df["joint_pos"] = self.joint_pos
            df["label"] = self.labels
            df["superclass_label"] = self.superclass_labels
            df["frame_idx"] = self.frame_idx
            df["hard_negative_idx"] = self.hardnegative_idx
            df["rotation_axis"] = self.rotation_axis
            df["rotation_label"] = self.rotation_labels
            
            if split == "train":
                train_pose = "y"
                if train_stride is not None:
                    selected_files_idx = [i[1] % train_stride == 0 for i in self.frame_idx]  # only evaluate y-axis
                else:
                    assert training_views is not None
                    training_views = [x if x >= 0 else 360+x for x in training_views] # Convert negative numbers to positive ones
                    print("Selecting following views:", training_views)
                    selected_files_idx = [i[1] in training_views for i in self.frame_idx]
                selected_files_rot = [x == train_pose for x in self.rotation_axis]
                is_train_file = [x and y for x, y in zip(selected_files_idx, selected_files_rot)]
                df["is_train_file"] = is_train_file
                assert np.sum(is_train_file) > 0, f"Number of training examples ({np.sum(is_train_file)}) should be non-zero"
                
                self.joint_pos = [self.joint_pos[i] for i in range(len(self.joint_pos)) if is_train_file[i]]
                self.labels = [self.labels[i] for i in range(len(self.labels)) if is_train_file[i]]
                assert len(self.joint_pos) == len(self.labels)
                
                if train_stride is not None:
                    expected_num_instances_per_cls = int(360 / train_stride)
                    print(f"Training dataset size with stride of {train_stride}:", len(self.joint_pos), self.joint_pos[:5])
                    print(f"Number of instances expected per class with a stride of {train_stride}: {expected_num_instances_per_cls}")
                else:
                    assert training_views is not None
                    expected_num_instances_per_cls = len(training_views)
                    print(f"Training dataset size with training views {training_views}:", len(self.joint_pos), self.joint_pos[:5])
                    print(f"Number of instances expected per class with training views {training_views}: {expected_num_instances_per_cls}")
                
                for c in range(self.get_num_classes()):
                    assert np.sum([self.labels[i] == c for i in range(len(self.labels))]) == expected_num_instances_per_cls

                random_cls = np.random.randint(self.get_num_classes())
                relevant_files = [self.joint_pos[i] for i in range(len(self.labels)) if self.labels[i] == random_cls]
                print(f"Example from class {random_cls}: {len(relevant_files)} {relevant_files}")
            else:
                assert len(self.joint_pos) == len(self.labels) == len(self.superclass_labels) == len(self.rotation_labels)
            
            # metadata_output_file = os.path.join(metadata_output_dir, "metadata.csv")
            # df.to_csv(metadata_output_file, header=True, index=False)
            
            dataset_object = (self.joint_pos, self.labels, self.frame_idx, self.superclass_labels, self.hardnegative_idx, self.rotation_labels, self.class_names, self.rotation_class_names)
            with open(cache_file, "wb") as stream:
                pickle.dump(dataset_object, stream, protocol=4)  # Since protocol 4 is compatible with Python3.7
            print("Saved data to cache file:", cache_file)
        
        self.rot2label = {k: v for v, k in enumerate(self.rotation_class_names)}  # Regenerate label map when loading from cache
        rotations_per_axis = 360
        if split == "test":
            self.views_per_class_list = [rotations_per_axis if len(x) == 1 else ((rotations_per_axis // self.multirot_stride))**len(x) for x in self.rotation_class_names]
            self.views_per_class = sum(self.views_per_class_list)
        else:
            self.views_per_class_list = None
        print(f"# classes: {self.num_classes} / split: {self.split} / rotation classes: {self.rotation_class_names} / views per class: {self.views_per_class_list}")
    
    @staticmethod
    def get_image_from_coords(joint_positions, connected=False):
        capture_width = 1920
        capture_height = 1080
        aspect_ratio = float(capture_width) / capture_height
        
        new_size = 256
        scaled_pos = joint_positions
        scaled_pos[:, 0] = joint_positions[:, 0] * new_size * aspect_ratio - 96
        scaled_pos[:, 1] = joint_positions[:, 1] * new_size
        scaled_pos = scaled_pos.astype(np.int32)
        
        img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
        prev_coords = None
        if connected:
            for (x, y) in scaled_pos:
                color = (0, 0, 255)
                cv2.circle(img, (x, y), radius=2, color=color, thickness=1)
                if prev_coords is not None:  # Add a line between the two
                    cv2.line(img, prev_coords, (x, y), color=color, thickness=1)
                prev_coords = (x, y)
        else:
            for (x, y) in scaled_pos:
                color = (255, 255, 255)
                cv2.circle(img, (x, y), radius=3, color=color, thickness=-1)
            
            # Gaussian blur the image to make the coords smooth
            img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
            img = (img / img.max() * 255.).astype(np.uint8)  # Scale pixel intensities after smoothing
        
        return Image.fromarray(img)
    
    @staticmethod
    def get_array_from_coords(joint_positions):
        capture_width = 1920
        capture_height = 1080
        aspect_ratio = float(capture_width) / capture_height
        
        new_size = 256
        scaled_pos = joint_positions
        scaled_pos[:, 0] = joint_positions[:, 0] * new_size * aspect_ratio - 96
        scaled_pos[:, 1] = joint_positions[:, 1] * new_size
        scaled_pos = scaled_pos.astype(np.int32)
        scaled_pos = np.clip(scaled_pos, 0, new_size-1)  # important to ensure correct output range
        
        array = np.zeros((2, new_size), dtype=np.float32)
        weight = 1. / 8.  # 8 Coords at maximum
        for (x, y) in scaled_pos:
            array[0][x] += weight
            array[1][y] += weight
        return array

    def get_num_classes(self):
        return len(self.class_names)

    def get_num_hard_negatives(self):
        return len(self.hardnegative_class_names)

    def get_num_rotation_axis(self):
        return len(self.rotation_class_names)

    def get_rotation_axis_list(self):
        return self.rotation_class_names
    
    def get_num_views_per_rotation_axis(self, axis):
        if not isinstance(axis, int):
            axis = self.rot2label[axis]
        return self.views_per_class_list[axis]

    def get_num_views_per_rotation_axis_list(self):
        return self.views_per_class_list
    
    def get_rotation_cls_map(self):
        return self.rot2label
    
    def get_multirot_stride(self):
        return self.multirot_stride

    def __getitem__(self, idx):
        coords = self.joint_pos[idx]
        if self.return_images:
            img = CoordsDataset.get_image_from_coords(coords)
        else:  # Convert the image to 2 1D coords
            img = CoordsDataset.get_array_from_coords(coords)
        
        y = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.return_eval_tuple:
            return img, (y, self.superclass_labels[idx], self.hardnegative_idx[idx], self.rotation_labels[idx], self.frame_idx[idx])
        return img, y

    def __len__(self):
        return len(self.labels)


class WebDatasetWrapper(torch.utils.data.IterableDataset):
    def __init__(self, tar_file, is_train, transform, load_models, multirot_stride):
        print(">> Loading tar file:", tar_file)
        is_multirot = multirot_stride is not None
        self.multirot_stride = multirot_stride
        
        # Remove extension and path
        main_file_name = pathlib.Path(tar_file)
        main_file_name = str(main_file_name).rstrip(''.join(main_file_name.suffixes))
        main_file_name = os.path.split(main_file_name)[1]  # Strip the path
        print("Main file:", main_file_name)
        
        num_classes = None
        num_equidistant_views = None
        selected_views = None
        file_parts = main_file_name.split("_")
        is_eval = file_parts[-1].lower() == "eval"
        
        for idx, part in enumerate(file_parts):
            if part == "classes":
                num_classes = int(file_parts[idx+1])
            elif part == "views":
                if file_parts[idx-1] == "equidistant":
                    num_equidistant_views = int(file_parts[idx+1])
                else:
                    assert file_parts[idx-1] == "selected", file_parts
                    selected_views = file_parts[idx+1]
        
        assert is_eval or (num_equidistant_views is not None or selected_views is not None)
        assert num_classes in [10, 50, 100, 500, 1000, 2500, 5000, 10000], num_classes
        assert num_equidistant_views is None or num_equidistant_views in [1, 2, 3, 4, 6, 12], num_equidistant_views
        
        permitted_views = ["0", "-15,15", "-30,30", "-30,0,30", "-30,-10,10,30", "-30,-15,0,15,30", "-30,-20,-10,0,10,20,30",
                           "-30,-24,-18,-12,-6,0,6,12,18,24,30", "-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30",
                           "-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21,24,27,30", "-60,0,60",
                           "-60,-30,0,30,60", "-90,0,90", "-90,-60,-30,0,30,60,90"]
        assert selected_views is None or selected_views in permitted_views, selected_views
        
        if not is_eval:
            self.views_per_class = num_equidistant_views
            if num_equidistant_views is None:
                self.views_per_class = len(selected_views.split(","))
        
        self.num_equidistant_views = num_equidistant_views
        self.selected_views = selected_views
        self.is_eval = is_eval
        self.num_classes = num_classes
        self.num_hard_negatives = 1
        rotations_per_axis = 360
        
        if is_multirot:
            self.rotation_class_names = ["x", "y", "z", "xy", "xz", "yz"]
            assert rotations_per_axis % self.multirot_stride == 0, f"{rotations_per_axis} % {self.multirot_stride} != 0 ({rotations_per_axis%self.multirot_stride})"
        else:
            self.rotation_class_names = ["x", "y", "z"]
        self.rotation_cls_map = {k: i for i, k in enumerate(self.rotation_class_names)}
        
        if is_eval:
            # self.views_per_class = len(self.rotation_class_names) * rotations_per_axis
            self.views_per_class_list = [rotations_per_axis if len(x) == 1 else ((rotations_per_axis // self.multirot_stride))**len(x) for x in self.rotation_class_names]
            self.views_per_class = sum(self.views_per_class_list)
            self.num_expected_files = self.num_classes * self.views_per_class
        else:
            self.num_expected_files = self.num_classes * self.views_per_class
        print(f"# classes: {self.num_classes} / eval: {self.is_eval} / rotation classes: {self.rotation_class_names} / # views per class: {self.views_per_class} / # files expected: {self.num_expected_files} / multirot={is_multirot}")
        
        # Initialize webdataset instance
        assert os.path.exists(tar_file), f"{tar_file} not found!"  # Assumes that the given path is a single tar file, rather than an expression
        if is_train:
            def preprocess_train(sample):
                image, paperclip_idx = sample
                if transform is not None:
                    image = transform(image)
                return image, paperclip_idx
            
            dataset = wds.WebDataset(tar_file).shuffle(1000).decode("pil").to_tuple("image.jpg", "model_idx.cls" if load_models else "paperclip_idx.cls").map(preprocess_train)
        else:
            def preprocess_eval(sample):
                if load_models:
                    image, paperclip_idx, rotation_axis, rot_x, rot_y, rot_z = sample
                    hardneg_idx = 0
                else:
                    image, paperclip_idx, hardneg_idx, rotation_axis, rot_x, rot_y, rot_z = sample
                
                if transform is not None:
                    image = transform(image)
                assert hardneg_idx == 0
                superclass_idx = paperclip_idx  # No superclass
                rotation_cls = self.rotation_cls_map[rotation_axis]
                rotation_angles = [rot_x, rot_y, rot_z]
                # rotation_angle = rotation_angles[rotation_cls]
                return image, (paperclip_idx, superclass_idx, hardneg_idx, rotation_cls, rotation_angles)
            
            # Expected format: data, (target, superclass_target, hardnegative_idx, rotation_axis, rotation_angle)
            cls_fields = ["model_idx.cls"] if load_models else ["paperclip_idx.cls", "hardneg_idx.cls"]
            dataset = wds.WebDataset(tar_file).shuffle(1000).decode("pil").to_tuple("image.jpg", *cls_fields,
                                                                                    "rotation_axis.txt", "rotation_angle_x.cls", "rotation_angle_y.cls", 
                                                                                    "rotation_angle_z.cls").map(preprocess_eval)
        self.dataset = dataset
    
    def __iter__(self):
        return self.dataset.__iter__()

    def __len__(self):
        return self.num_expected_files
    
    def get_num_classes(self):
        return self.num_classes

    def get_num_hard_negatives(self):
        return self.num_hard_negatives

    def get_num_rotation_axis(self):
        return len(self.rotation_class_names)

    def get_rotation_axis_list(self):
        return self.rotation_class_names
    
    def get_num_views_per_class(self):
        return self.views_per_class

    def get_num_views_per_rotation_axis(self, axis):
        if not isinstance(axis, int):
            axis = self.rotation_cls_map[axis]
        return self.views_per_class_list[axis]
    
    def get_num_views_per_rotation_axis_list(self):
        return self.views_per_class_list
    
    def get_selected_views(self):
        return self.selected_views
    
    def get_rotation_cls_map(self):
        return self.rotation_cls_map
    
    def get_multirot_stride(self):
        return self.multirot_stride


def getWebDatasetWrapper(tar_file, is_train, transform, load_models, multirot_stride):
    return WebDatasetWrapper(tar_file, is_train, transform, load_models, multirot_stride)


def getWebDataset(tar_file, transform, is_train):
    def preprocess_train(sample):
        image, paperclip_idx = sample
        if transform is not None:
            image = transform(image)
        return image, paperclip_idx
    
    def preprocess_eval(sample):
        image, key, paperclip_idx, hardneg_idx, rotation_axis, frame_idx, rot_x, rot_y, rot_z = sample
        if transform is not None:
            image = transform(image)
        return image, key, paperclip_idx, hardneg_idx, rotation_axis, frame_idx, rot_x, rot_y, rot_z
    
    assert os.path.exists(tar_file), f"{tar_file} not found!"  # Assumes that the given path is a single tar file, rather than an expression
    if is_train:
        dataset = wds.WebDataset(tar_file).shuffle(1000).decode("pil").to_tuple("image.jpg", "paperclip_idx.cls").map(preprocess_train)
    else:
        dataset = wds.WebDataset(tar_file).shuffle(1000).decode("pil").to_tuple("image.jpg", "__key__", "paperclip_idx.cls", "hardneg_idx.cls", "rotation_axis.txt", 
                                                                                "frame_idx.cls", "rotation_angle_x.cls", "rotation_angle_y.cls", "rotation_angle_z.cls").map(preprocess_eval)
    return dataset
