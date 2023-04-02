# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import os
import time
import random
import warnings
import simplejson
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR

import data_utils
import dist_utils

import timm
from torchmore import layers, flex

args = None


class Input(nn.Module):
    def __init__(self, assume, reorder=None, range=None, sizes=None, dtype=torch.float32):
        """Declares the input for a network.
        :param order: order of axes (e.g., BDL, BHWD, etc.)
        :param dtype: dtype to convert to
        :param range: tuple giving low/high values
        :param assume: default input order (when tensor doesn't have order attribute; None=required)
        """
        super().__init__()
        self.assume = assume
        self.reorder = reorder if reorder is not None else assume
        self.dtype = dtype
        self.range = range
        self.sizes = sizes

    def forward(self, x):
        if self.range is not None:
            lo = x.min().item()
            hi = x.max().item()
            assert lo >= self.range[0] and hi <= self.range[1], (lo, hi, self.range)
        if self.reorder is not None:
            if hasattr(x, "order"):
                x = layers.reorder(x, x.order, self.reorder)
            else:
                if self.assume is True or self.assume==self.reorder:
                    pass
                elif self.assume is None:
                    raise ValueError("input is required to have a .order property")
                else:
                    x = layers.reorder(x, self.assume, self.reorder)
        if self.sizes is not None:
            for i, size in enumerate(self.sizes):
                if size is None:
                    continue
                elif isinstance(size, int):
                    assert x.size(i) == size, (i, x.size(i))
                elif isinstance(size, (list, tuple)):
                    lo, hi = size
                    assert x.size(i) >= lo and x.size(i) <= hi, (i, x.size(i), (lo, hi))
                else:
                    raise ValueError("bad size spec")
        x = x.type(self.dtype)
        return x

    def __repr__(self):
        return f"Input({self.assume}->{self.reorder} " + \
            f"{self.dtype} {self.range} {self.sizes})"


class LSTM_Reducer(nn.Module):
    """A 2D LSTM Reduction module.
    Input order as for 2D convolutions -- 2D tensor output to be fed to the final classifier
    """

    def __init__(self, ninput=None, noutput=None, nhidden=None,
                 num_layers=1, bidirectional=True, use_cell_state=True):
        super().__init__()
        nhidden = nhidden or noutput
        self.ndir = bidirectional+1
        self.use_cell_state = use_cell_state
        self.hlstm = nn.LSTM(ninput, nhidden, num_layers=num_layers,
                             bidirectional=bidirectional)
        self.vlstm = nn.LSTM(nhidden*self.ndir, noutput, num_layers=num_layers,
                             bidirectional=bidirectional)

    def forward(self, x):
        b, d, h, w = x.shape
        hin = layers.reorder(x, "BDHW", "WHBD").view(w, h*b, d)

        # Apply LSTM per row and obtain the final state for each row
        if self.use_cell_state:
            _, (_, c_n) = self.hlstm(hin)
            c_n = c_n[-self.ndir:, :, :]  # Use the hidden state from the final layer; h_n=(num_layers * num_directions, B, hidden_size)
            assert len(c_n.shape) == 3 and c_n.shape[0] in [1, 2]  # 1/2, H*B, D
            vin = c_n.permute(1, 2, 0).reshape(h, b, -1)  # H x B x D * (1/2 depending on whether using BDLSTM)
        else:
            hout, _ = self.hlstm(hin)
            hout = hout[-1, :, :]  # Get the final state for hout: (seq_len, batch, num_directions * hidden_size)
            vin = hout.view(h, b, -1)  # H x B x D * (1/2 depending on whether using BDLSTM)

        # Apply an LSTM vertically on the final state from each row and get the final state
        if self.use_cell_state:
            _, (_, c_n) = self.vlstm(vin)
            c_n = c_n[-self.ndir:, :, :]  # Use the hidden state from the final layer; h_n=(num_layers * num_directions, B, hidden_size)
            assert len(c_n.shape) == 3 and c_n.shape[0] in [1, 2]  # 1/2, B, D
            vout = c_n.permute(1, 2, 0).reshape(b, -1)  # B x D * (1/2 depending on whether using BDLSTM)
        else:
            vout, _ = self.vlstm(vin)
            vout = vout[-1, :, :]  # Get the final state for vout: (seq_len, batch, num_directions * hidden_size)

        assert len(vout.shape) == 2  # B x D * (1/2 depending on whether using BDLSTM)
        return vout


def make_lstm(input_dim, input_channels, num_cls, fp16=False, non_lin=nn.ReLU):
    layer_config = [32, 'BN', 'NL', 'MP', 64, 'BN', 'NL', 'MP', 128, 'BN', 'NL', 128, 'BN', 'NL', 'MP', 
                    256, 'BN', 'NL', 256, 'BN', 'NL', 256, 'BN', 'NL', 'MP', 
                    256, 'BN', 'NL', 256, 'BN', 'NL', 256, 'BN', 'NL', 'GAP']
    
    layer_list = []
    last_channels = input_channels
    for idx in range(len(layer_config)):
        is_last_layer = idx == len(layer_config) - 1
        assert isinstance(layer_config[idx], int) or layer_config[idx] in ["MP", "AP", "BN", "NL"] or (is_last_layer and layer_config[idx] in ["GAP", "GMP", "LSTM_Red"])
        layer = None
        if isinstance(layer_config[idx], int):
            layer = layers.BDHW_LSTM(ninput=last_channels, noutput=layer_config[idx],
                                     nhidden=None, num_layers=1, bidirectional=True)
            last_channels = layer_config[idx] * 2  # Factor of two due to BDLSTM
        elif layer_config[idx] == "GAP":
            layer = layers.Fun("lambda x: x.mean(dim=(2, 3))")  # GAP layer
        elif layer_config[idx] == "GMP":
            layer = layers.Fun("lambda x: x.max(dim=3)[0].max(2)[0]")  # GMP layer
        elif layer_config[idx] == "LSTM_Red":
            layer = LSTM_Reducer(ninput=last_channels, noutput=layer_config[idx+1],
                                 nhidden=None, num_layers=1, bidirectional=True)
        elif layer_config[idx] == "MP":
            layer = nn.MaxPool2d(2)
        elif layer_config[idx] == "AP":
            layer = nn.AvgPool2d(2)
        elif layer_config[idx] == "BN":
            layer = nn.BatchNorm2d(last_channels)
        elif layer_config[idx] == "NL":
            layer = non_lin()
        
        assert layer is not None
        layer_list += [layer]

    dtype = torch.half if fp16 else torch.float
    model = nn.Sequential(
        Input("BDHW", range=(0, 1), sizes=[None, input_channels, None, None]),
        *layer_list,
        torch.nn.Flatten(),
        flex.Linear(num_cls)
    )
    flex.shape_inference(model, (1, input_channels, input_dim, input_dim))
    list(model.named_modules())[1][1].dtype = dtype  # Replace the input layer dtype for FP16
    return model


def make_mlp(input_channels, sequence_length, num_cls, fp16=False):
    dtype = torch.half if fp16 else torch.float
    model = nn.Sequential(
        Input("BDHW", range=(0, 1), sizes=[None, input_channels, sequence_length]),
        torch.nn.Flatten(),
        flex.Linear(512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        flex.Linear(512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        flex.Linear(512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        flex.Linear(512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        flex.Linear(num_cls)
    )
    flex.shape_inference(model, (1, input_channels, sequence_length))
    list(model.named_modules())[1][1].dtype = dtype  # Replace the input layer dtype for FP16
    return model


def get_optimizer(model, args):
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    param_list = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        dist_utils.dist_print("Using SGD optimizer...")
        optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        assert args.optimizer == "adam"
        dist_utils.dist_print("Using Adam optimizer...")
        optimizer = optim.Adam(param_list, lr=args.lr, weight_decay=args.wd)
    return optimizer


def train(model, device, train_loader, optimizer, criterion, loss_scaler, batch_size_multiplier=1, log_interval=10, clip_grad=None):
    assert batch_size_multiplier == 1
    
    model.train()
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # loss.backward()
        loss_scaler.scale(loss).backward()
        
        optimizer_step = ((batch_idx + 1) % batch_size_multiplier) == 0
        if optimizer_step:
            if clip_grad is not None and clip_grad > 0.:
                # Unscales the gradients of optimizer's assigned params in-place
                loss_scaler.unscale_(optimizer)
                
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            # optimizer.step()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad()
        
        if batch_idx % log_interval == 0:
            pbar.set_description(f"Loss: {float(loss):.4f}")
        torch.cuda.synchronize()
    pbar.close()


def test(model, device, criterion, test_loader, use_tqdm=True, return_preds=False):
    model.eval()
    
    correct = torch.tensor([0]).to(device)
    correct_superclass = torch.tensor([0]).to(device)
    test_loss = torch.tensor([0.0]).to(device)
    total = torch.tensor([0]).to(device)
    
    num_examples = len(test_loader.dataset)
    num_classes = test_loader.dataset.get_num_classes()
    num_rotation_axis = test_loader.dataset.get_num_rotation_axis()
    rotation_axis_list = test_loader.dataset.get_rotation_axis_list()
    num_hard_negatives = test_loader.dataset.get_num_hard_negatives()
    num_views_list = test_loader.dataset.get_num_views_per_rotation_axis_list()
    rotation_cls_map = test_loader.dataset.get_rotation_cls_map()
    multirot_stride = test_loader.dataset.get_multirot_stride()
    per_axis_multirot_ex = 360 // multirot_stride
    dist_utils.dist_print(f"# examples: {num_examples} / # classes: {num_classes} / # hard negatives: {num_hard_negatives} / # rotation axis: {num_rotation_axis}")
    dist_utils.dist_print(f"Axis rotation list: {rotation_axis_list} / # views per rotation axis list: {num_views_list} / Rotation cls map: {rotation_cls_map} / Per axis # ex: {per_axis_multirot_ex}")
    assert len(rotation_axis_list) == num_rotation_axis
    
    rotation_stats = {}
    for i in range(num_rotation_axis):
        num_views = num_views_list[i]
        rotation_stats[i] = {"correct": torch.tensor([0]).to(device), "total": torch.tensor([0]).to(device)}
        rotation_stats[i]["angles"] = {k: {"correct": torch.tensor([0]).to(device), "total": torch.tensor([0]).to(device)} for k in range(num_views)}
    
    all_stats = {}
    for i in range(num_classes):
        all_stats[i] = {"correct": torch.tensor([0]).to(device), "superclass_correct": torch.tensor([0]).to(device), "total": torch.tensor([0]).to(device), "rotation": {}}
        for j in range(num_rotation_axis):
            num_views = num_views_list[j]
            all_stats[i]["rotation"][j] = {"correct": torch.tensor([0]).to(device), "superclass_correct": torch.tensor([0]).to(device), "total": torch.tensor([0]).to(device)}
            all_stats[i]["rotation"][j]["angles"] = {k: {"correct": torch.tensor([0]).to(device), "total": torch.tensor([0]).to(device)} for k in range(num_views)}
    
    preds, targets = [], []
    
    pbar = tqdm(test_loader) if use_tqdm else test_loader
    for data, (target, superclass_target, hardnegative_idx, rotation_axis, rotation_angles) in pbar:
        data, target, superclass_target = data.to(device), target.to(device), superclass_target.to(device)
        with torch.no_grad():
            output = model(data)
            if return_preds:
                preds.append(output.argmax(dim=1).detach())
                targets.append(target.detach())
            
            test_loss += float(criterion(output, target)) * len(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            
            # Convert the prediction the superclass
            superclass_pred = torch.div(pred, num_hard_negatives, rounding_mode='trunc')  # This is how classes are computed from superclasses
            correct_superclass += superclass_pred.eq(superclass_target.view_as(superclass_pred)).sum().item()
            
            # Count the correct prediction for rotations
            for i in range(num_rotation_axis):
                rot_mask = rotation_axis == i
                rot_correct = pred[rot_mask].eq(target[rot_mask].view_as(pred[rot_mask])).sum().item()
                rot_total = rot_mask.sum().item()
                rotation_stats[i]["correct"] += rot_correct
                rotation_stats[i]["total"] += rot_total
                axis_rep = rotation_axis_list[i]
                is_multiaxis = len(axis_rep) > 1
                
                # Convert angles to ints for matching
                rotation_angles = rotation_angles.to(torch.int32)
                
                # Compute angles stats
                num_views = num_views_list[i]
                for j in range(num_views):
                    if is_multiaxis:
                        if len(axis_rep) > 2:
                            raise NotImplementedError("Not implemented for rotation along 3 axis simultaneously")
                        axes_ind = [rotation_cls_map[x] for x in axis_rep]
                        first_axis_angle = (j // per_axis_multirot_ex)
                        second_axis_angle = (j % per_axis_multirot_ex)
                        rotation_angles_norm = rotation_angles // multirot_stride
                        rotation_angles_norm = torch.stack([rotation_angles_norm[:, i] for i in axes_ind], dim=1)
                        assert rotation_angles_norm.shape[1] == 2, f"{rotation_angles_norm.shape}"
                        angle_mask = torch.logical_and(rotation_angles_norm[:, 0] == first_axis_angle, rotation_angles_norm[:, 1] == second_axis_angle)
                        # print(f"{axis_rep} / view: {j} / 1st angle: {first_axis_angle} / 2nd angle: {second_axis_angle}")
                    else:
                        rotation_angle = rotation_angles[:, rotation_cls_map[axis_rep]]
                        angle_mask = rotation_angle == j
                    combined_mask = torch.logical_and(rot_mask, angle_mask)
                    
                    angle_correct = pred[combined_mask].eq(target[combined_mask].view_as(pred[combined_mask])).sum().item()
                    angle_total = combined_mask.sum().item()
                    
                    rotation_stats[i]["angles"][j]["correct"] += angle_correct
                    rotation_stats[i]["angles"][j]["total"] += angle_total
            
            # Include the per-example predictions
            for i in range(len(target)):
                cls_idx = int(target[i])
                ex_correct = int((pred[i] == target[i]).sum())
                superclass_correct = int((superclass_pred[i] == superclass_target[i]).sum())
                all_stats[cls_idx]["total"] += 1
                all_stats[cls_idx]["correct"] += ex_correct
                all_stats[cls_idx]["superclass_correct"] += superclass_correct
                
                # Set the angle stats
                all_stats[cls_idx]["rotation"][int(rotation_axis[i])]["total"] += 1
                all_stats[cls_idx]["rotation"][int(rotation_axis[i])]["correct"] += ex_correct
                
                axis_rep = rotation_axis_list[int(rotation_axis[i])]
                if len(axis_rep) > 1:
                    # Accommodate the multiaxis eval
                    current_rotation_axis_list = [rotation_cls_map[x] for x in axis_rep]
                    normalized_angles = [int(rotation_angles[i][x]) // multirot_stride for x in current_rotation_axis_list]
                    assert len(normalized_angles) == 2
                    
                    angle = int(normalized_angles[0] * per_axis_multirot_ex + normalized_angles[1])
                    all_stats[cls_idx]["rotation"][int(rotation_axis[i])]["angles"][angle]["total"] += 1
                    all_stats[cls_idx]["rotation"][int(rotation_axis[i])]["angles"][angle]["correct"] += ex_correct
                else:
                    # Ensure that the model can deal with the case that the rotation angles are now an array for all three axes
                    angle = int(rotation_angles[i, int(rotation_axis[i])])
                    all_stats[cls_idx]["rotation"][int(rotation_axis[i])]["angles"][angle]["total"] += 1
                    all_stats[cls_idx]["rotation"][int(rotation_axis[i])]["angles"][angle]["correct"] += ex_correct
    
    if use_tqdm:
        pbar.close()

    # Reduce all of the values in case of distributed processing
    torch.cuda.synchronize()
    correct = int(dist_utils.reduce_tensor(correct.data))
    correct_superclass = int(dist_utils.reduce_tensor(correct_superclass.data))
    test_loss = float(dist_utils.reduce_tensor(test_loss.data))
    total = int(dist_utils.reduce_tensor(total.data))
    
    assert total == len(test_loader.dataset)
    test_loss /= total
    test_acc = 100. * correct / total
    superclass_test_acc = 100. * correct_superclass / total
    output_dict = dict(loss=test_loss, acc=test_acc, superclass_acc=superclass_test_acc, correct=correct, correct_superclass=correct_superclass, total=total)
    output_dict["rotation_stats"] = {}
    
    for i in range(num_rotation_axis):
        rotation_stats[i]["correct"] = int(dist_utils.reduce_tensor(rotation_stats[i]["correct"].data))
        rotation_stats[i]["total"] = int(dist_utils.reduce_tensor(rotation_stats[i]["total"].data))
        rotation_stats[i]["acc"] = 100. * rotation_stats[i]["correct"] / rotation_stats[i]["total"]
        
        num_views = num_views_list[i]
        for j in range(num_views):
            rotation_stats[i]["angles"][j]["correct"] = int(dist_utils.reduce_tensor(rotation_stats[i]["angles"][j]["correct"].data))
            rotation_stats[i]["angles"][j]["total"] = int(dist_utils.reduce_tensor(rotation_stats[i]["angles"][j]["total"].data))
            rotation_stats[i]["angles"][j]["acc"] = 100. * rotation_stats[i]["angles"][j]["correct"] / rotation_stats[i]["angles"][j]["total"]
        output_dict["rotation_stats"][rotation_axis_list[i]] = rotation_stats[i]  # Add stats to the output dict
    
    # Aggregate all stats
    for i in range(num_classes):
        all_stats[i]["correct"] = int(dist_utils.reduce_tensor(all_stats[i]["correct"].data))
        all_stats[i]["superclass_correct"] = int(dist_utils.reduce_tensor(all_stats[i]["superclass_correct"].data))
        all_stats[i]["total"] = int(dist_utils.reduce_tensor(all_stats[i]["total"].data))
        all_stats[i]["acc"] = 100. * all_stats[i]["correct"] / all_stats[i]["total"]
        
        for rot_axis in range(num_rotation_axis):
            all_stats[i]["rotation"][rot_axis]["correct"] = int(dist_utils.reduce_tensor(all_stats[i]["rotation"][rot_axis]["correct"].data))
            all_stats[i]["rotation"][rot_axis]["superclass_correct"] = int(dist_utils.reduce_tensor(all_stats[i]["rotation"][rot_axis]["superclass_correct"].data))
            all_stats[i]["rotation"][rot_axis]["total"] = int(dist_utils.reduce_tensor(all_stats[i]["rotation"][rot_axis]["total"].data))
            
            num_views = num_views_list[rot_axis]
            for rot_angle in range(num_views):
                all_stats[i]["rotation"][rot_axis]["angles"][rot_angle]["correct"] = int(dist_utils.reduce_tensor(all_stats[i]["rotation"][rot_axis]["angles"][rot_angle]["correct"].data))
                all_stats[i]["rotation"][rot_axis]["angles"][rot_angle]["total"] = int(dist_utils.reduce_tensor(all_stats[i]["rotation"][rot_axis]["angles"][rot_angle]["total"].data))
                assert all_stats[i]["rotation"][rot_axis]["angles"][rot_angle]["total"] == 1, f"i: {i} / axis: {rot_axis} / angle: {rot_angle} / {all_stats[i]}"
                all_stats[i]["rotation"][rot_axis]["angles"][rot_angle]["acc"] = 100. * all_stats[i]["rotation"][rot_axis]["angles"][rot_angle]["correct"] / all_stats[i]["rotation"][rot_axis]["angles"][rot_angle]["total"]
    
    if return_preds:
        # Gather all the lists
        preds = torch.cat(dist_utils.gather_tensor(torch.cat(preds, dim=0)), dim=0).detach().cpu()
        targets = torch.cat(dist_utils.gather_tensor(torch.cat(targets, dim=0)), dim=0).detach().cpu()
        
        assert len(preds) == len(targets) == total
        assert (preds == targets).sum() == correct
        
        # Add the prediction lists
        output_dict = {}
        output_dict["preds"] = preds.numpy().tolist()
        output_dict["targets"] = targets.numpy().tolist()
    
    dist_utils.dist_print(f"Test set | Average loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}% ({correct}/{total}) | Superclass accuracy: {superclass_test_acc:.2f}% ({correct_superclass}/{total})")
    for i in range(num_rotation_axis):
        dist_utils.dist_print(f"Test set | Rotation axis: {rotation_axis_list[i]} | Accuracy: {rotation_stats[i]['acc']:.2f}% ({rotation_stats[i]['correct']}/{rotation_stats[i]['total']})")
    return output_dict, all_stats


def main():
    global args
    dataset_choices = ["paperclip_wds", "3d_models_wds", "paperclip_coords", "paperclip_coords_array"]
    lr_scheduler_choices = ["cosine", "step", "none"]
    
    # Training settings
    parser = argparse.ArgumentParser(description='Trainer to evaluate the generalization of different architectures on the paperclip dataset')
    parser.add_argument('--dataset', default=dataset_choices[0], choices=dataset_choices,
                        help=f'Dataset to be used for training the model (choices: {", ".join(dataset_choices)}; default: {dataset_choices[0]})')
    parser.add_argument('--train-stride', type=int, default=None, metavar='N',
                        help='defines how further apart are training examples in the dataset (defaults to 60 i.e. 6 images per paperclip)')
    parser.add_argument('--training-views', type=str, default=None,
                        help='views to be used for training the model (specify either the stride or the training views -- mutually exclusive)')
    parser.add_argument('--use-hard-negatives', action='store_true', default=False,
                        help='use hard negatives from the dataset')
    parser.add_argument('--model-name', type=str, default='vgg11_bn', 
                        help='name of the model to be used for training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--optimizer-batch-size', type=int, default=128, metavar='N',
                        help='final batch size to be used for training after combining all the processes (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr-scheduler', default=lr_scheduler_choices[0], choices=lr_scheduler_choices,
                        help=f'LR scheduler to be used (choices: {", ".join(lr_scheduler_choices)}; default: {lr_scheduler_choices[0]})')
    parser.add_argument('--lr-steps', type=str, default=None, help='epochs to be used for reducing the learning rate (separated by comma)')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='rate with which to decay to learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='M',
                        help='Gradient clipping threshold (disabled by default)')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        help='optimizer to be used for training the model (default: sgd)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=3, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--output-dir', type=str, default='.', help='Location to store models')
    parser.add_argument('--data', type=str, default='../data', help='Location to store data')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers allocated to the dataloader')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='number of classes to be used')
    parser.add_argument('--no-eval', action='store_true', default=False,
                        help='No evaluation while training the model')
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Only required when loading distributed models')
    
    parser.add_argument('--use-rotation-aug', action='store_true', default=False,
                        help='Train the model using rotation augmentation')
    parser.add_argument('--use-rand-bg-aug', action='store_true', default=False,
                        help='Train the model using random background augmentation')
    parser.add_argument('--multirot-stride', type=int, default=None,
                        help='stride for multirot')
    
    # WebDataset pararms
    parser.add_argument('--train_tar_file', type=str, default=None, help='Location of the training tar file (only used for WebDataset)')
    parser.add_argument('--val_tar_file', type=str, default=None, help='Location of the validation/test tar file (only used for WebDataset)')
    
    args = parser.parse_args()
    
    assert "resnet" in args.model_name or "vgg" in args.model_name or args.model_name in ["lstm", "vit_b_16", "mlp"]
    assert not args.model_name == "mlp" or args.dataset == "paperclip_coords_array"
    assert args.num_classes is None or args.dataset in ["paperclip_v2", "paperclip_multirot", "3d_models", "paperclip_coords", "paperclip_coords_array"]
    assert args.dataset != "3d_models" or not args.use_hard_negatives
    
    if args.lr_steps == "":
        args.lr_steps = None
    assert args.lr_steps is None or args.lr_scheduler == "step"
    
    if args.lr_steps != None:
        args.lr_steps = [int(x) for x in args.lr_steps.split(',')]
        dist_utils.dist_print("Epochs for decay:", args.lr_steps)
    else:
        args.lr_steps = []
    
    args.use_wds = "_wds" in args.dataset
    if args.use_wds:
        assert args.train_tar_file is not None and os.path.exists(args.train_tar_file), args.train_tar_file
        assert args.val_tar_file is not None and os.path.exists(args.val_tar_file), args.val_tar_file
    else:
        args.orig_training_views = args.training_views
        if args.training_views == "":
            args.training_views = None
        if args.training_views != None:
            args.training_views = [int(x) for x in args.training_views.split(',')]
            assert all([-360 <= x <= 360 for x in args.training_views])
            dist_utils.dist_print("Training views:", args.training_views)
        assert not (args.train_stride is not None and args.training_views is not None), "Training stride and training views are mutually exclusive"
        assert not (args.train_stride is None and args.training_views is None), "Either the training stride of the training views must be specified"
    
    # Initialize the distributed environment
    args.gpu = 0
    args.world_size = 1
    args.distributed = args.distributed or int(os.getenv('WORLD_SIZE', 1)) > 1
    args.rank = int(os.getenv('RANK', 0))
    args.local_rank = 0

    if "SLURM_NNODES" in os.environ:
        args.local_rank = args.rank % torch.cuda.device_count()
        dist_utils.dist_print(f"SLURM tasks/nodes: {os.getenv('SLURM_NTASKS', 1)}/{os.getenv('SLURM_NNODES', 1)}")
    elif "WORLD_SIZE" in os.environ:
        args.local_rank = int(os.getenv('LOCAL_RANK', 0))

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        assert int(os.getenv('WORLD_SIZE', 1)) == args.world_size
        dist_utils.dist_print(f"Initializing the environment with {args.world_size} processes | Current process rank: {args.local_rank}")
    
    if args.seed is not None:
        dist_utils.dist_print(f"Using seed: {args.seed}")
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    
    args.effective_batch_size = args.batch_size * args.world_size
    if args.optimizer_batch_size < 1:
        args.optimizer_batch_size = args.effective_batch_size
    assert args.optimizer_batch_size % args.effective_batch_size == 0, \
        f"Optimizer batch size should be divisible by the effective batch size ({args.optimizer_batch_size} % {args.effective_batch_size} != 0)"
    args.batch_size_multiplier = args.optimizer_batch_size // args.effective_batch_size
    main_proc = dist_utils.is_main_proc(args.local_rank, shared_fs=True)
    
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    # Ignore warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    warnings.filterwarnings("ignore", "The given NumPy array is not writeable", UserWarning)

    # Prepare the dataset and the dataloaders
    if args.num_classes is not None:
        assert args.dataset in ["paperclip_coords", "paperclip_coords_array"]
    num_classes = args.num_classes
    
    input_dim = 32 if args.dataset in ["cifar10", "cifar100"] else 224 if args.dataset in dataset_choices[2:] else None
    assert input_dim is not None
    train_loader, test_loader = data_utils.get_dataloaders(args)
    if args.use_wds:
        num_classes = train_loader.dataset.get_num_classes()
        args.num_classes = num_classes  # Required for checkpoint names
    if "paperclip" in args.dataset:
        assert train_loader.dataset.get_num_classes() == num_classes, f"{train_loader.dataset.get_num_classes()} != {num_classes}"
    
    # Supports ResNet, VGG, LSTM and ViT/B-16
    input_channels = 3
    dist_utils.dist_print(f"Model: {args.model_name.upper()} | # classes: {num_classes}")
    if args.model_name == "mlp":
        model = make_mlp(input_channels=2, sequence_length=256, num_cls=num_classes).to(device)
        dist_utils.dist_print("MLP architecture:", model)
    elif args.model_name == "lstm":
        model = make_lstm(input_dim, input_channels, num_classes).to(device)
        dist_utils.dist_print("LSTM architecture:", model)
    elif args.model_name == "vit_b_16":
        # HPs taken from: https://github.com/google-research/vision_transformer/issues/2
        model = timm.models.vit_base_patch16_224(img_size=input_dim, in_chans=input_channels, num_classes=num_classes, drop_rate=0.1, pretrained=True).to(device)
        dist_utils.dist_print("ViT architecture:", model)
    else:
        generator = getattr(models, args.model_name)
        model = generator(pretrained=False, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.use_wds:
        training_views = train_loader.dataset.get_selected_views()
        if training_views is not None:
            view_representation = f"{'_views_' + str(training_views)}"
        else:
            num_views = train_loader.dataset.get_num_views_per_class()
            assert 360 % num_views == 0
            stride = 360 // num_views
            view_representation = f"{'_stride_' + str(stride)}"
    else:
        view_representation = f"{'_stride_' + str(args.train_stride) if args.train_stride is not None else '_views_' + args.orig_training_views}"
    model_base = os.path.join(args.output_dir, f"{args.dataset}{view_representation}{'_cls_' + str(args.num_classes) if args.num_classes is not None else ''}{'_no_hard_neg' if not args.use_hard_negatives else ''}{'_rot_aug' if args.use_rotation_aug else ''}{'_rand_bg' if args.use_rand_bg_aug else ''}_{args.model_name}_{args.optimizer}_ep_{args.epochs}_{args.lr_scheduler}{('_' + ','.join([str(x) for x in args.lr_steps])) if len(args.lr_steps) != 0 else ''}{('_clip_' + str(args.clip_grad)) if args.clip_grad is not None and args.clip_grad > 0. else ''}")
    dist_utils.dist_print("Base model file name:", model_base)
    
    model_file = f"{model_base}{'_dist' if args.distributed else ''}.pt"
    dist_utils.dist_print("\n****************************************************************")
    
    # Convert the model to a distributed model
    model_dist = dist_utils.convert_to_distributed(model, args.local_rank, sync_bn=True, use_torch_ddp=True)
    
    # Get the optimizer
    optimizer = get_optimizer(model_dist, args)
    
    # Define the loss scaler
    loss_scaler = torch.cuda.amp.GradScaler()
    
    lr_scheduler = None
    if args.lr_scheduler == "step":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_decay)
        dist_utils.dist_print(f"Using step LR scheduler with a decay factor of {args.lr_decay} at epochs {args.lr_steps}...")
    elif args.lr_scheduler == "cosine":
        dist_utils.dist_print(f"Using cosine LR scheduler...")
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        assert args.lr_scheduler == "none"
    
    if not args.no_eval:
        dist_utils.dist_print("----------------------------------------------")
        dist_utils.dist_print("Evaluating model perfomance before training...")
        test(model_dist, device, criterion, test_loader)
        dist_utils.dist_print("----------------------------------------------")
    
    if not os.path.exists(model_file):
        dist_utils.dist_print(">> Model file not found. Training model from scratch...")
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print("Output directory created...")
        
        start_time = time.time()
        for epoch in tqdm(range(1, args.epochs + 1)):
            train(model_dist, device, train_loader, optimizer, criterion, loss_scaler, batch_size_multiplier=args.batch_size_multiplier, 
                  log_interval=args.log_interval, clip_grad=args.clip_grad)
            if not args.no_eval:
                test(model_dist, device, criterion, test_loader)
            if lr_scheduler is not None:
                lr_scheduler.step()
        if main_proc:
            torch.save(model_dist.state_dict(), model_file)
        dist_utils.wait_for_other_procs()
        elapsed_time_secs = time.time() - start_time
        dist_utils.dist_print(f"Training completed in {elapsed_time_secs/60.:.2f} mins.")
    else:
        dist_utils.dist_print(">> Model file already exists. Evaluating the pretrained model...")
    
    start_time = time.time()
    dist_utils.dist_print("Loading model file:", model_file)
    model_dist.load_state_dict(dist_utils.convert_state_dict(torch.load(model_file, map_location=device)), strict=True)
    results, all_stats = test(model_dist, device, criterion, test_loader)
    elapsed_time_secs = time.time() - start_time
    dist_utils.dist_print(f"Evaluation completed in {elapsed_time_secs/60.:.2f} mins.")
    
    if main_proc:
        output_file = f"{model_base}_eval.json"
        with open(output_file, "w") as f:
            f.write(simplejson.dumps(results))
        
        output_file = f"{model_base}_eval_all_stats.json"
        with open(output_file, "w") as f:
            f.write(simplejson.dumps(all_stats))


if __name__ == "__main__":
    main()
