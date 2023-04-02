import os
import glob

import natsort
import simplejson

import numpy as np
import matplotlib.pyplot as plt


# Params for v6
multirot_stride = 10
num_views_per_axis = 360 // multirot_stride

dataset = "paperclips"
assert dataset in ["paperclips", "chairs"]
version = "v6"
examples = "normal"
assert examples in ["normal", "bg_aug", "rot_aug"]

log_dir = f"*{dataset}_{version}_wds*"
log_files = glob.glob(f"{log_dir}/*/*_eval.json")
if examples == "normal":
    log_files = [x for x in log_files if "bg_aug" not in x and "rot_aug" not in x]
elif examples == "bg_aug":
    log_files = [x for x in log_files if "bg_aug" in x]
else:
    assert examples == "rot_aug"
    log_files = [x for x in log_files if "rot_aug" in x]
print("Unfiltered file list:", len(log_files))
log_files = [x for x in log_files if "views" not in x]
print("JSON output files:", len(log_files), log_files[:3])

output_dir = f"{dataset}_{version}_{examples}_wds_plots/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dict = {}
for file in log_files:
    print("Loading file:", file)
    file_name = os.path.split(file)[1].replace("_eval.json", "").split("_")
    
    offset = 0
    stride = None
    views_list = None
    
    if dataset == "paperclips":
        assert file_name[0] == "paperclip"
    else:
        assert file_name[0] == "3d"
        assert file_name[1] == "models"
        offset = 1
    assert file_name[1+offset] == "wds"
    
    assert file_name[2+offset] == "stride"
    stride = int(file_name[3+offset])
    num_training_views = int(360 / stride)
    views = num_training_views
    
    assert file_name[4+offset] == "cls"
    num_classes = int(file_name[5+offset])
    if file_name[6+offset] == "no":
        assert file_name[7+offset] == "hard"
        assert file_name[8+offset] == "neg"
        if file_name[9+offset] == "rand":
            assert file_name[10+offset] == "bg"
            model = "_".join(file_name[11+offset:12+offset+(1 if file_name[12+offset] == "bn" else 0)])
        elif file_name[9+offset] == "rot":
            assert file_name[10+offset] == "aug"
            model = "_".join(file_name[11+offset:12+offset+(1 if file_name[12+offset] == "bn" else 0)])
        else:
            model = "_".join(file_name[9+offset:10+offset+(1 if file_name[10+offset] == "bn" else 0)])
    else:
        if file_name[6+offset] == "rand":
            assert file_name[7+offset] == "bg"
            model = "_".join(file_name[8+offset:9+offset+(1 if file_name[9+offset] == "bn" else 0)])
        elif file_name[6+offset] == "rot":
            assert file_name[7+offset] == "aug"
            model = "_".join(file_name[8+offset:9+offset+(1 if file_name[9+offset] == "bn" else 0)])
        else:
            model = "_".join(file_name[6+offset:7+offset+(1 if file_name[7+offset] == "bn" else 0)])
    
    print(f"Model: {model} / Frame stride: {stride} / Number of training views: {num_training_views} / # classes: {num_classes}")
    
    # Read the file contents
    with open(file, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l != ""]
    last_line = lines[-1]
    # print("Last line:", last_line)
    
    test_stats = simplejson.loads(last_line)
    # print("Test stats:", test_stats)
    
    if model not in output_dict:
        output_dict[model] = {}
    if views not in output_dict[model]:
        output_dict[model][views] = {}
    assert num_classes not in output_dict[model][views]
    output_dict[model][views][num_classes] = test_stats

model_list = natsort.natsorted(list(output_dict.keys()))
print("Model list:", model_list)

training_views_list = natsort.natsorted(list(output_dict[model_list[0]].keys()))
print("Training views list:", training_views_list)

num_classes_list = natsort.natsorted(list(output_dict[model_list[0]][training_views_list[0]].keys()))
print("Number classes list:", num_classes_list)

# Validate that the k settings are same for every run
# assert all([all([natsort.natsorted(list(output_dict[model_list[i]].keys())) == training_views_list]) for i in range(len(model_list))])
# assert all([all([natsort.natsorted(list(output_dict[model_list[i]][training_views_list[j]].keys())) == num_classes_list]) for j in range(len(training_views_list)) for i in range(len(model_list))])
# assert len(training_views_list) == 6

# Define the barchart
bar_width = 0.8 / len(num_classes_list) # 0.15

# Separate the models based on different number of training views
# start_point = -0.25 #- len(num_classes_list)*bar_width / 2
start_point = -(0.025 if len(num_classes_list) == 3 else 0.1 if len(num_classes_list) == 4 else 0.25) #- len(num_classes_list)*bar_width / 2
print(f"Bar width: {bar_width} / Starting point: {start_point}")
colors = ['green', 'red', 'orange', 'purple', 'magenta', 'blue']
fontsize = 14

r_list = []
for iterator, num_classes in enumerate(num_classes_list):  # Lower than training views since we pick the acc list from the 0 training views
    stride = start_point + iterator*bar_width
    r = [x + stride for x in np.arange(len(training_views_list))]
    r_list.append(r)
    
    # Plot the angle-wise results
    acc_list_one_view = None
    for i in range(len(model_list)):
        for training_views in training_views_list:
            if training_views not in output_dict[model_list[i]]:
                continue
            if num_classes not in output_dict[model_list[i]][training_views]:
                continue
            
            # Plot the data
            plt.figure(figsize=(8, 6))

            rotation_axes_all = natsort.natsorted(output_dict[model_list[i]][training_views][num_classes]["rotation_stats"])
            print("All rotation axes:", rotation_axes_all)
            
            rotation_axes = ["x", "y", "z"]
            assert [x in rotation_axes_all for x in rotation_axes], f"{rotation_axes} should all be in {rotation_axes_all}"
            
            for color_iter, plot_axis in enumerate(rotation_axes):
                # angle_list = [angle for angle in range(360)]
                angle_list = natsort.natsorted(list(output_dict[model_list[i]][training_views][num_classes]["rotation_stats"][plot_axis]["angles"].keys()))
                angle_list = [int(x) for x in angle_list]  # Important for axis values
                acc_list = [output_dict[model_list[i]][training_views][num_classes]["rotation_stats"][plot_axis]["angles"][str(angle)]["acc"] for angle in angle_list]
                # plt.plot(angle_list, acc_list, linewidth=3, c=colors[color_iter], label=f'{model_list[i]} trained with {training_views} training views')
                plt.plot(angle_list, acc_list, linewidth=3, c=colors[color_iter], label=f"{plot_axis}-axis")

            num_training_views = len(training_views) if isinstance(training_views, list) else training_views
            assert isinstance(num_training_views, int)
            
            assert 360 % num_training_views == 0
            stride = 360 // num_training_views
            training_views_angle = [x for x in range(360) if x % stride == 0]
            assert len(training_views_angle) == num_training_views
        
            # Add a symbol to mark a training view
            plt.plot(training_views_angle, [0.0 for _ in range(len(training_views_angle))], linewidth=0., marker='x', 
                    markersize=10.0, markeredgewidth=3.0, c='tab:blue', label=f'Training view location')
            plt.plot(training_views_angle, [100.0 for _ in range(len(training_views_angle))], linewidth=0., marker='o', 
                    markersize=5.0, markeredgewidth=3.0, c='tab:blue')
            plt.vlines(x=training_views_angle, ymin=0, ymax=100.0, colors='blue', linestyle='--', lw=2)

            plt.ylabel('Accuracy (%)', fontsize=fontsize)
            plt.xlabel(f'Model rotation angle', fontsize=fontsize)
            plt.ylim(-5, 105)
            # plt.legend(fontsize=fontsize)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.325),
                    ncol=1, fancybox=True, shadow=False, fontsize=fontsize)
            
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

            # plt.tight_layout()
            output_file = os.path.join(output_dir, f"results_{dataset}_{version}_num_views_{training_views}_{model_list[i]}_cls_{num_classes}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close('all')
            
            # Plot the individual images for different multirot axes
            rotation_axes_multirot = [x for x in rotation_axes_all if len(x) > 1]
            print("Multirot rotation axes:", rotation_axes_multirot)
            
            for plot_axis in rotation_axes_multirot:
                fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
                
                angle_list = natsort.natsorted(list(output_dict[model_list[i]][training_views][num_classes]["rotation_stats"][plot_axis]["angles"].keys()))
                angle_list = [int(x) for x in angle_list]  # Important for axis values
                acc_list = [output_dict[model_list[i]][training_views][num_classes]["rotation_stats"][plot_axis]["angles"][str(angle)]["acc"] for angle in angle_list]
                
                # Split them into the two different axes
                # angle_list_first = np.array(angle_list) // num_views_per_axis
                # angle_list_second = np.array(angle_list) % num_views_per_axis
                
                # Convert the grid to image
                angle_list = np.array(angle_list).reshape(-1, num_views_per_axis)
                acc_list = np.array(acc_list).reshape(-1, num_views_per_axis)
                
                im = plt.imshow(acc_list, cmap=plt.get_cmap('hot'), vmin=0, vmax=100)
                
                plt.ylabel(plot_axis.lower()[0], fontsize=fontsize)
                plt.xlabel(plot_axis.lower()[1], fontsize=fontsize)
                
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                
                # set axis ticks labels
                # ax.set_xticklabels([x for x in range(0, acc_list.shape[1] * multirot_stride, multirot_stride)])
                # ax.set_yticklabels([y for y in range(0, acc_list.shape[0] * multirot_stride, multirot_stride)])
                
                plt.colorbar(label="Accuracy", orientation="vertical")
                
                # Add the training views markers
                if plot_axis == "xy":
                    plt.scatter([x // 10 for x in training_views_angle], [0 for _ in range(len(training_views_angle))], 
                                linewidth=0., marker='o', c='tab:blue', s=100)
                elif plot_axis == "yz":
                    plt.scatter([0 for _ in range(len(training_views_angle))], [x // 10 for x in training_views_angle], 
                                linewidth=0., marker='o', c='tab:blue', s=100)
                
                # plt.tight_layout()
                output_file = os.path.join(output_dir, f"results_{dataset}_{version}_num_views_{training_views}_{model_list[i]}_cls_{num_classes}_{plot_axis}.png")
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.show()
                plt.close('all')

            for plot_axis in ["y"]:
                # angle_list = [angle for angle in range(360)]
                angle_list = natsort.natsorted(list(output_dict[model_list[i]][training_views][num_classes]["rotation_stats"][plot_axis]["angles"].keys()))
                angle_list = [int(x) for x in angle_list]  # Important for axis values
                acc_list = [output_dict[model_list[i]][training_views][num_classes]["rotation_stats"][plot_axis]["angles"][str(angle)]["acc"] for angle in angle_list]
                
                if training_views == 1:
                    acc_list_one_view = acc_list
                    continue
                assert acc_list_one_view is not None
                
                # Plot the overlaid figure
                plt.figure(figsize=(8, 6))
                plt.plot(angle_list, acc_list, linewidth=3, c='red', label=f'{model_list[i]} trained with {training_views} training views')
                
                # Plot the accuracy list from the one-view case and replicate it based on the expected number of peaks
                assert 360 % training_views == 0
                stride = 360 // training_views
                training_views_angle = [x for x in range(360) if x % stride == 0]
                assert len(training_views_angle) == training_views
                print(f"Training views for {training_views} peak replication: {training_views_angle}")
                
                cm = plt.get_cmap('hsv')
                color_list = [cm(1.*i/len(training_views_angle)) for i in range(len(training_views_angle))]
                
                max_acc_single_view = None
                for j, train_view_angle in enumerate(training_views_angle):
                    adjusted_acc_list = acc_list_one_view
                    adjusted_acc_list = adjusted_acc_list[-train_view_angle:] +  adjusted_acc_list[:-train_view_angle]
                    assert len(adjusted_acc_list) == len(acc_list_one_view)
                    # plt.plot(angle_list, adjusted_acc_list, linewidth=3, c=color_list[j], label=f'Single view peak at {train_view_angle}')
                    plt.plot(angle_list, adjusted_acc_list, linewidth=3, c='tab:gray', alpha=0.5, 
                                label=f'Replicated single view peak' if j == 0 else None)
                    if max_acc_single_view is None:
                        max_acc_single_view = adjusted_acc_list
                    else:
                        max_acc_single_view = np.maximum(max_acc_single_view, adjusted_acc_list)
                    
                    # Add a symbol to mark a training view
                    plt.plot([train_view_angle], [0.0], linewidth=0., marker='x', markersize=10.0, markeredgewidth=3.0,
                                c='tab:blue', label=f'Training view location' if j == 0 else None)
                    plt.plot([train_view_angle], [100.0], linewidth=0., marker='o', 
                            markersize=5.0, markeredgewidth=3.0, c='tab:blue')
                    plt.vlines(x=[train_view_angle], ymin=0, ymax=100.0, colors='blue', linestyle='--', lw=2)

                # Fill between the line between one view and multi-view setup
                plt.fill_between(angle_list, max_acc_single_view, acc_list, alpha=0.3, color='orange', label='Difference b/w 2D matching and deep networks')
                
                plt.ylabel('Accuracy (%)', fontsize=fontsize)
                plt.xlabel(f'Model rotation angle along {plot_axis}-axis', fontsize=fontsize)
                plt.ylim(-5, 105)
                # plt.legend(fontsize=fontsize)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.325),
                            ncol=1, fancybox=True, shadow=False, fontsize=fontsize)

                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

                # plt.tight_layout()
                output_file = os.path.join(output_dir, f"results_{dataset}_{version}_num_views_{training_views}_{model_list[i]}_cls_{num_classes}_axis_{plot_axis}_overlaid.png")
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.show()
                plt.close('all')

print("Index list:", r_list)

for model in model_list:
    for plot_type in ["acc", "superclass_acc", "rotation_x", "rotation_y", "rotation_z"]:
        plt.figure(figsize=(12, 8))

        # colors = ['green', 'red', 'orange', 'purple', 'magenta', 'blue']
        cm = plt.get_cmap('viridis')
        colors = [cm(1.*i/len(r_list)) for i in range(len(r_list))]
        
        for i, r in enumerate(r_list):
            if "rotation" in plot_type:
                plot_axis = plot_type.split("_")[1]
                acc_list = [output_dict[model][training_views][num_classes_list[i]]["rotation_stats"][plot_axis]["acc"]
                            if training_views in output_dict[model] and num_classes_list[i] in output_dict[model][training_views]
                            else 0.0
                            for training_views in training_views_list]
            else:
                acc_list = [output_dict[model][training_views][num_classes_list[i]][plot_type]
                            if training_views in output_dict[model] and num_classes_list[i] in output_dict[model][training_views]
                            else 0.0
                            for training_views in training_views_list]
            print(f"Accuracy list for model {model} with {num_classes_list[i]} classes: {acc_list}")
            plt.bar(r, acc_list, width=bar_width, color=colors[i], edgecolor='black', capsize=7, label=f"# classes: {num_classes_list[i]}")

        plt.xticks([r + bar_width for r in range(len(training_views_list))], training_views_list)
        plt.ylabel('Accuracy (%)', fontsize=fontsize)
        plt.xlabel('Number of training views', fontsize=fontsize)
        plt.ylim(0, 105)
        plt.legend(fontsize=fontsize)
        
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # plt.tight_layout()
        output_file = os.path.join(output_dir, f"results_{dataset}_{version}_{model}_{plot_type}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close('all')
