import os
import webdataset as wds


def getPaperclipMultiRotWDS(root_dir, split, transform, train_stride=None, training_views=None, num_classes=None, training_axis="y", dataset_stride=36):
    """
    - Maybe also support training on multiple axes simultaneously
    """
    assert training_axis in ["x", "y", "z", "xy", "xz", "yz", "xyz"], training_axis
    assert split != "train" or not (training_views is None and train_stride is None), "Need to specify either train_stride or training_views (both were empty)"
    assert not (training_views is not None and train_stride is not None), "Both training views and training stride cannot be specified"
    
    if training_views is not None:
        assert isinstance(training_views, list) or isinstance(training_views, tuple)
        training_views = [x if x >= 0 else 360+x for x in training_views] # Convert negative numbers to positive ones
    
    def filter_seed_files(sample):
        return 'jpg' in sample
    
    def filter_specific_training_views(sample):
        key = sample['__key__']
        paths = key.split(os.sep)
        paperclip_idx = int(paths[-4].replace("paperclip_", ""))
        rotation_axis = paths[-2]
        filename = paths[-1].split("_")
        assert len(filename) == 4, filename
        assert filename[0] == "frame", filename
        assert filename[2] == "angle", filename
        rotation_angle = [int(x) for x in filename[3].split(",")]
        rot_x, rot_y, rot_z = rotation_angle
        
        if split == "test":  # Select everything for the test set
            return paperclip_idx < num_classes  # 0 indexed, so < condition
        
        # Add rotation axis constraint
        select_ex = rotation_axis == training_axis  # Restrict to the correct rotation angle
        
        # Add training views constraint
        if train_stride is not None:  # Train stride is specified
            assert training_views is None
            
            assert train_stride % dataset_stride == 0, f"Train stride ({train_stride}) is not compatible with dataset stride ({dataset_stride})"
            if "x" in training_axis:
                select_ex = select_ex and (rot_x % train_stride == 0)
            if "y" in training_axis:
                select_ex = select_ex and (rot_y % train_stride == 0)
            if "z" in training_axis:
                select_ex = select_ex and (rot_z % train_stride == 0)
        else:
            assert training_views is not None
            
            if "x" in training_axis:
                select_ex = select_ex and rot_x in training_views
            if "y" in training_axis:
                select_ex = select_ex and rot_y in training_views
            if "z" in training_axis:
                select_ex = select_ex and rot_z in training_views
        
        # Add constraint on the number of classes
        if num_classes is not None:
            select_ex = select_ex and (paperclip_idx < num_classes)  # 0 indexed, so < condition
        
        return select_ex

    def preprocess(sample):
        image, key = sample
        paths = key.split(os.sep)
        paperclip_idx = int(paths[-4].replace("paperclip_", ""))
        hardneg_idx = int(paths[-3].replace("hard_negative_", ""))
        rotation_axis = paths[-2]
        filename = paths[-1].split("_")
        assert len(filename) == 4, filename
        assert filename[0] == "frame", filename
        assert filename[2] == "angle", filename
        frame_idx = int(filename[1])
        rotation_angle = [int(x) for x in filename[3].split(",")]
        rot_x, rot_y, rot_z = rotation_angle
        
        if transform is not None:
            image = transform(image)  # Apply image transform
        
        return image, key, paperclip_idx, hardneg_idx, rotation_axis, frame_idx, rot_x, rot_y, rot_z

    assert os.path.exists(root_dir)
    dataset = wds.WebDataset(root_dir).select(filter_seed_files).select(filter_specific_training_views).shuffle(1000).decode("pil").to_tuple("jpg", "__key__").map(preprocess)

    return dataset


version = "v6"
url = f"/netscratch/siddiqui/Datasets/Paperclips_{version}.tar.xz"
dataset_stride = 1
output_dir = os.path.join(os.path.split(url)[0], f"Paperclips_train_{version}")
print("Output directory:", output_dir)
if not os.path.exists(output_dir):
    print("Creating output directory:", output_dir)
    os.mkdir(output_dir)

create_test_set_files = True
create_equidistant_dataset = True
create_limited_views_dataset = True

class_list = [10, 50, 100, 500, 1000, 2500, 5000, 10000]

if create_test_set_files:
    for num_classes in class_list:
        # Create the dataset instance
        dataset = getPaperclipMultiRotWDS(url, split="test", transform=None, num_classes=num_classes)
        
        # Create the output file
        output_file = os.path.join(output_dir, f"Paperclips_{version}_classes_{num_classes}_eval.tar.xz")
        if os.path.exists(output_file):
            print(f"Warning: file already exists ({output_file}). Skipping file generation...")
            continue
        sink = wds.TarWriter(output_file, compress=True)
        print("Creating output file:", output_file)

        total_ex = 0
        for idx, (image, key, paperclip_idx, hardneg_idx, rotation_axis, frame_idx, rot_x, rot_y, rot_z) in enumerate(dataset):
            print(f"# classes= {num_classes} (# ex={idx}) // {key} // {paperclip_idx} // {hardneg_idx} // {rotation_axis} // {frame_idx} // {rot_x}, {rot_y}, {rot_z}")
            sample = {
                "__key__": key,
                "image.jpg": image,
                "paperclip_idx.cls": paperclip_idx,
                "hardneg_idx.cls": hardneg_idx,
                "rotation_axis.txt": rotation_axis,
                "frame_idx.cls": frame_idx,
                "rotation_angle_x.cls": rot_x,
                "rotation_angle_y.cls": rot_y,
                "rotation_angle_z.cls": rot_z,
            }
            sink.write(sample)
            total_ex += 1
        
        sink.close()
        print(f"{total_ex} examples written to file: {output_file}")

if create_equidistant_dataset:
    # Write the equidistant views dataset
    for num_views in [1, 2, 3, 4, 6, 12]:
        assert 360 % num_views == 0, f"{360 % num_views} != 0"
        train_stide = 360 // num_views
        
        for num_classes in class_list:
            # Create the dataset instance
            dataset = getPaperclipMultiRotWDS(url, split="train", transform=None, train_stride=train_stide, dataset_stride=dataset_stride, num_classes=num_classes)
            
            # Create the output file
            output_file = os.path.join(output_dir, f"Paperclips_{version}_classes_{num_classes}_equidistant_views_{num_views}.tar.xz")
            if os.path.exists(output_file):
                print(f"Warning: file already exists ({output_file}). Skipping file generation...")
                continue
            sink = wds.TarWriter(output_file, compress=True)
            print("Creating output file:", output_file)

            total_ex = 0
            for idx, (image, key, paperclip_idx, hardneg_idx, rotation_axis, frame_idx, rot_x, rot_y, rot_z) in enumerate(dataset):
                print(f"# views={num_views} (# ex={idx}) // {key} // {paperclip_idx} // {hardneg_idx} // {rotation_axis} // {frame_idx} // {rot_x}, {rot_y}, {rot_z}")
                sample = {
                    "__key__": key,
                    "image.jpg": image,
                    "paperclip_idx.cls": paperclip_idx,
                    "hardneg_idx.cls": hardneg_idx,
                    "rotation_axis.txt": rotation_axis,
                    "frame_idx.cls": frame_idx,
                    "rotation_angle_x.cls": rot_x,
                    "rotation_angle_y.cls": rot_y,
                    "rotation_angle_z.cls": rot_z,
                }
                sink.write(sample)
                total_ex += 1
            
            sink.close()
            print(f"{total_ex} examples written to file: {output_file}")

if create_limited_views_dataset:
    # Write the limited views dataset
    for view_list in ["0", "-15,15", "-30,30", "-30,0,30", "-30,-10,10,30", "-30,-15,0,15,30", "-30,-20,-10,0,10,20,30", "-30,-24,-18,-12,-6,0,6,12,18,24,30",
                      "-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30", "-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21,24,27,30",
                      "-60,0,60", "-60,-30,0,30,60", "-90,0,90", "-90,-60,-30,0,30,60,90"]:
        for num_classes in class_list:
            # Create the dataset instance
            training_views = [int(x) for x in view_list.split(',')]
            dataset = getPaperclipMultiRotWDS(url, split="train", transform=None, training_views=training_views, dataset_stride=dataset_stride, num_classes=num_classes)

            # Create the output file
            output_file = os.path.join(output_dir, f"Paperclips_{version}_classes_{num_classes}_selected_views_{view_list}.tar.xz")
            if os.path.exists(output_file):
                print(f"Warning: file already exists ({output_file}). Skipping file generation...")
                continue
            sink = wds.TarWriter(output_file, compress=True)
            print("Creating output file:", output_file)

            total_ex = 0
            for idx, (image, key, paperclip_idx, hardneg_idx, rotation_axis, frame_idx, rot_x, rot_y, rot_z) in enumerate(dataset):
                print(f"View list={view_list} (# ex={idx}) // {key} // {paperclip_idx} // {hardneg_idx} // {rotation_axis} // {frame_idx} // {rot_x}, {rot_y}, {rot_z}")
                sample = {
                    "__key__": key,
                    "image.jpg": image,
                    "paperclip_idx.cls": paperclip_idx,
                    "hardneg_idx.cls": hardneg_idx,
                    "rotation_axis.txt": rotation_axis,
                    "frame_idx.cls": frame_idx,
                    "rotation_angle_x.cls": rot_x,
                    "rotation_angle_y.cls": rot_y,
                    "rotation_angle_z.cls": rot_z,
                }
                sink.write(sample)
                total_ex += 1
            
            sink.close()
            print(f"{total_ex} examples written to file: {output_file}")
