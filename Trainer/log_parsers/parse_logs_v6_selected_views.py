import os
import glob

import natsort
import simplejson

import numpy as np
import matplotlib.pyplot as plt


dataset = "paperclips"
assert dataset in ["paperclips", "chairs"]
plot_selected_views = True
version = "v6"
plot_num_views = False

log_dir = f"*{dataset}_{version}_wds*"
log_files = glob.glob(f"{log_dir}/*/*_eval.json")
print("Unfiltered file list:", len(log_files))
if plot_selected_views:
    log_files = [x for x in log_files if "views" in x]
    print("Files before view filtering:", len(log_files))
    if plot_num_views:
        view_list = ["-30,30", "-30,0,30", "-30,-10,10,30", "-30,-20,-10,0,10,20,30", "-30,-24,-18,-12,-6,0,6,12,18,24,30", "-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30", "-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21,24,27,30"]  # Specify the view range
    else:
        view_list = ["-15,15", "-30,30", "-30,0,30", "-60,0,60", "-60,-30,0,30,60", "-90,0,90", "-90,-60,-30,0,30,60,90"]  # Specify the view range
    log_files = [x for x in log_files if any([view in x for view in view_list])]
    print("Files after view filtering:", len(log_files))
else:
    log_files = [x for x in log_files if "views" not in x]
print("JSON output files:", len(log_files), log_files[:3])

output_dir = f"{dataset}_{version}_selected_views_wds_plots_compare/" if plot_selected_views else f"{dataset}_{version}_wds_plots/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dict = {}
for file in log_files:
    print("Loading file:", file)
    file_name = os.path.split(file)[1].replace("_eval.json", "").split("_")
    
    offset = 0
    stride = None
    views = None
    views_list = None
    
    if dataset == "paperclips":
        assert file_name[0] == "paperclip"
    else:
        assert file_name[0] == "3d"
        assert file_name[1] == "models"
        offset = 1
    assert file_name[1+offset] == "wds"
    if plot_selected_views:
        assert file_name[2+offset] == "views"
        views = file_name[3+offset]
        views_list = [int(x) for x in views.split(",")]
        num_training_views = len(views_list)
    else:
        assert file_name[2+offset] == "stride"
        stride = int(file_name[3+offset])
        num_training_views = int(360 / stride)
    assert file_name[4+offset] == "cls"
    num_classes = int(file_name[5+offset])
    if file_name[6+offset] == "no":
        assert file_name[7+offset] == "hard"
        assert file_name[8+offset] == "neg"
        model = "_".join(file_name[9+offset:10+offset+(1 if file_name[10+offset] == "bn" else 0)])
    else:
        model = "_".join(file_name[6+offset:7+offset+(1 if file_name[7+offset] == "bn" else 0)])
    
    if plot_selected_views:
        print(f"Model: {model} / Training views: {views_list} / Number of training views: {num_training_views} / # classes: {num_classes}")
    else:
        print(f"Model: {model} / Frame stride: {stride} / Number of training views: {num_training_views} / # classes: {num_classes}")
    
    # Read the file contents
    with open(file, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l != ""]
    last_line = lines[-1]
    # print("Last line:", last_line)
    
    test_stats = simplejson.loads(last_line)
    # print("Test stats:", test_stats)
    
    assert plot_selected_views
    
    if model not in output_dict:
        output_dict[model] = {}
    if views not in output_dict[model]:
        output_dict[model][views] = {}
    assert num_classes not in output_dict[model][views]
    output_dict[model][views][num_classes] = test_stats

model_list = natsort.natsorted(list(output_dict.keys()))
print("Model list:", model_list)

training_views_list = natsort.natsorted(list(output_dict[model_list[0]].keys()))
if not plot_num_views:
    assert all([x in view_list for x in training_views_list]), training_views_list
training_views_list = view_list  # Use the ordering specified in view list
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

            for color_iter, plot_axis in enumerate(["x", "y", "z"]):
                angle_list = [angle for angle in range(360)]
                acc_list = [output_dict[model_list[i]][training_views][num_classes]["rotation_stats"][plot_axis]["angles"][str(angle)]["acc"] for angle in angle_list]
                # plt.plot(angle_list, acc_list, linewidth=3, c=colors[color_iter], label=f'{model_list[i]} trained with {training_views} training views')
                plt.plot(angle_list, acc_list, linewidth=3, c=colors[color_iter], label=f"{plot_axis}-axis")

            if not plot_selected_views:
                num_training_views = len(training_views) if isinstance(training_views, list) else training_views
                assert isinstance(num_training_views, int)
                
                assert 360 % num_training_views == 0
                stride = 360 // num_training_views
                training_views_angle = [x for x in range(360) if x % stride == 0]
                assert len(training_views_angle) == num_training_views
            else:
                training_views_angle = [int(x) for x in training_views.split(",")]
                training_views_angle = [x+360 if x < 0 else x for x in training_views_angle]
            
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
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                       ncol=3, fancybox=True, shadow=False, fontsize=fontsize)
            
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

            # plt.tight_layout()
            output_file = os.path.join(output_dir, f"results_{dataset}_{version}_num_views_{training_views}_{model_list[i]}_cls_{num_classes}_compare.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close('all')
            
            if not plot_selected_views:
                for plot_axis in ["y"]:
                    angle_list = [angle for angle in range(360)]
                    acc_list = [output_dict[model_list[i]][training_views][num_classes]["rotation_stats"][plot_axis]["angles"][str(angle)]["acc"] for angle in angle_list]
                    
                    if len(training_views) == 1:
                        acc_list_one_view = acc_list
                        continue
                    assert acc_list_one_view is not None
                    
                    # Plot the overlaid figure
                    plt.figure(figsize=(8, 6))
                    plt.plot(angle_list, acc_list, linewidth=3, c='orange', label=f'{model_list[i]} trained with {training_views} training views')
                    
                    # Plot the accuracy list from the one-view case and replicate it based on the expected number of peaks
                    assert 360 % len(training_views) == 0
                    stride = 360 // len(training_views)
                    training_views_angle = [x for x in range(360) if x % stride == 0]
                    assert len(training_views_angle) == len(training_views)
                    print(f"Training views for {training_views} peak replication: {training_views_angle}")
                    
                    cm = plt.get_cmap('hsv')
                    color_list = [cm(1.*i/len(training_views_angle)) for i in range(len(training_views_angle))]
                    
                    for j, train_view_angle in enumerate(training_views_angle):
                        adjusted_acc_list = acc_list_one_view
                        adjusted_acc_list = adjusted_acc_list[-train_view_angle:] +  adjusted_acc_list[:-train_view_angle]
                        assert len(adjusted_acc_list) == len(acc_list_one_view)
                        # plt.plot(angle_list, adjusted_acc_list, linewidth=3, c=color_list[j], label=f'Single view peak at {train_view_angle}')
                        plt.plot(angle_list, adjusted_acc_list, linewidth=3, c='tab:gray', alpha=0.5, label=f'Replicated single view peak')
                    
                    plt.ylabel('Accuracy (%)', fontsize=fontsize)
                    plt.xlabel(f'Model rotation angle along {plot_axis}-axis', fontsize=fontsize)
                    plt.ylim(-5, 105)
                    # plt.legend(fontsize=fontsize)
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
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
    for plot_type in ["acc", "rotation_x", "rotation_y", "rotation_z"]:
        plt.figure(figsize=(16, 8))

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

        if plot_num_views:
            plt.xticks([r + bar_width for r in range(len(training_views_list))], [len(x.split(',')) for x in training_views_list])
        else:
            plt.xticks([r + bar_width for r in range(len(training_views_list))], [f"[{x}]" for x in training_views_list])
        plt.ylabel('Accuracy (%)', fontsize=fontsize)
        plt.xlabel('Number of training views' if plot_num_views else 'List of training views used', fontsize=fontsize)
        plt.ylim(0, 105)
        plt.legend(fontsize=fontsize)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
        #            ncol=2, fancybox=True, shadow=False, fontsize=fontsize)
        
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # plt.tight_layout()
        output_file = os.path.join(output_dir, f"results_{dataset}_{version}_{model}_{plot_type}_compare{'_num_views' if plot_num_views else ''}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close('all')
