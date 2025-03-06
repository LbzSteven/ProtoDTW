import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from utils.receptive_field import compute_rf_prototype
from utils.helpers import makedir, find_high_activation_crop, find_high_activation_subse
from scipy.interpolate import interp1d


# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel,  # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None,  # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved here
                    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
                    prototype_ts_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,  # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):
    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         ])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: ts start index
    2: ts end index
    3: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 4],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 4],
                                    fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 3],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 3],
                                    fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-' + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_ts_filename_prefix=prototype_ts_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir,
                             proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end - start))


# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist,  # this will be updated
                               global_min_fmap_patches,  # this will be updated
                               proto_rf_boxes,  # this will be updated
                               proto_bound_boxes,  # this will be updated
                               class_specific=True,
                               search_y=None,  # required if class_specific == True
                               num_classes=None,  # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes: str = None,
                               prototype_ts_filename_prefix: str = None,
                               prototype_self_act_filename_prefix: str = None,
                               prototype_activation_function_in_numpy=None):
    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda().float()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_ts_index_dict = {key: [] for key in range(num_classes)}
        # ts_y is the image's integer label
        for ts_index, ts_y in enumerate(search_y):
            ts_label = ts_y.item()
            class_to_ts_index_dict[ts_label].append(ts_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_l = prototype_shape[2]
    # proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2]

    for j in range(n_prototypes):
        # if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_ts_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_ts_index_dict[target_class]][:, j, :]

        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j, :]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmin_proto_dist_j[0] = class_to_ts_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            ts_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_ts_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_ts_end_index = fmap_ts_start_index + proto_l
            # fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            # fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[ts_index_in_batch,
                                     :,
                                     fmap_ts_start_index:fmap_ts_end_index,
                                     ]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info

            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)

            # get the whole image
            original_ts_j = search_batch_input[rf_prototype_j[0]]
            # originally rf_prototype_j[0] as index which is batch_argmin_proto_dist_j
            original_ts_j = original_ts_j.numpy()

            original_ts_j = np.transpose(original_ts_j, (1, 0))  # (May be need on less dim)
            original_ts_size = original_ts_j.shape[0]

            # crop out the receptive field
            # rf_ts_j = original_ts_j[rf_prototype_j[1]:rf_prototype_j[2],
            #           rf_prototype_j[3]:rf_prototype_j[4], :]

            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            # proto_rf_boxes[j, 3] = rf_prototype_j[3]
            # proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 4 and search_y is not None:
                proto_rf_boxes[j, 3] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_ts_j = proto_dist_[ts_index_in_batch, j, :]
            if prototype_network_parallel.module.prototype_activation_function == 'log':
                proto_act_ts_j = np.log(
                    (proto_dist_ts_j + 1) / (proto_dist_ts_j + prototype_network_parallel.module.epsilon))
            elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                proto_act_ts_j = max_dist - proto_dist_ts_j
            else:
                proto_act_ts_j = prototype_activation_function_in_numpy(proto_dist_ts_j)
            if original_ts_size == len(proto_act_ts_j):
                upsampled_act_ts_j = proto_act_ts_j
            else:
                x = np.linspace(0, 1, len(proto_act_ts_j))
                x_new = np.linspace(0, 1, original_ts_size)
                f = interp1d(x, proto_act_ts_j, kind='cubic')
                upsampled_act_ts_j = f(x_new)
            # upsampled_act_ts_j = cv2.resize(proto_act_ts_j, dsize=(original_ts_size, original_ts_size),
            #                                 interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_subse(upsampled_act_ts_j)
            # proto_bound_j = find_high_activation_crop(upsampled_act_ts_j)
            # crop out the image patch with high activation as prototype image
            proto_ts_j = original_ts_j[proto_bound_j[0]:proto_bound_j[1], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            # proto_bound_boxes[j, 3] = proto_bound_j[2]
            # proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 4 and search_y is not None:
                proto_bound_boxes[j, 3] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_ts_j)
                if prototype_ts_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    proto_path = os.path.join(dir_for_saving_prototypes,
                                              prototype_ts_filename_prefix + str(j) + '.tiff')
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

                    plt.plot(original_ts_j.flatten(), label='Original Data')

                    plt.plot(np.arange(proto_bound_j[0], proto_bound_j[1]), proto_ts_j,
                             label=f"class:{target_class}_proto" + str(j))
                    plt.legend(loc=1)
                    plt.savefig(proto_path, format='tiff', dpi=300)
                    plt.close()

                    # plt.close()
                    # overlay (upsampled) self activation on original image and save the result
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                    color_array = upsampled_act_ts_j.flatten()
                    cmap = plt.cm.coolwarm
                    norm = Normalize(vmin=color_array.min(), vmax=color_array.max())
                    x = np.arange(0, len(original_ts_j))
                    for i in range(len(x) - 1):
                        plt.plot(x[i:i + 2], original_ts_j.flatten()[i:i + 2], color=cmap(norm(color_array[i])),
                                 linewidth=3)

                    sm = ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, label='Class Activation Map')
                    cbar.set_label(label='Class activation map', size=24, weight='bold')
                    cbar.ax.tick_params(labelsize=20)
                    ax.grid(color="#2A3459")
                    plt.xticks(fontsize=20, weight='bold')
                    plt.yticks(fontsize=20, weight='bold')
                    plt.ylabel('Value', fontsize=24, weight='bold')
                    plt.xlabel("Time Step", fontweight="bold", fontsize=24)
                    plt.tight_layout()

                    self_activation_path = os.path.join(dir_for_saving_prototypes, prototype_ts_filename_prefix
                                                        + '-original_with_self_act' + str(j) + '.tiff')

                    plt.savefig(self_activation_path, format='tiff', dpi=300)
                    plt.close()
                    # rescaled_act_ts_j = upsampled_act_ts_j - np.amin(upsampled_act_ts_j)
                    # rescaled_act_ts_j = rescaled_act_ts_j / np.amax(rescaled_act_ts_j)
                    # heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_ts_j), cv2.COLORMAP_JET)
                    # heatmap = np.float32(heatmap) / 255
                    # heatmap = heatmap[..., ::-1]
                    # overlayed_original_ts_j = 0.5 * original_ts_j + 0.3 * heatmap
                    # plt.imsave(os.path.join(dir_for_saving_prototypes,
                    #                         prototype_ts_filename_prefix + '-original_with_self_act' + str(
                    #                             j) + '.png'),
                    #            overlayed_original_ts_j,
                    #            vmin=0.0,
                    #            vmax=1.0)

                    # if different from the original (whole) image, save the prototype receptive field as png
                    # if rf_ts_j.shape[0] != original_ts_size or rf_ts_j.shape[1] != original_ts_size:
                    #     plt.imsave(os.path.join(dir_for_saving_prototypes,
                    #                             prototype_ts_filename_prefix + '-receptive_field' + str(j) + '.png'),
                    #                rf_ts_j,
                    #                vmin=0.0,
                    #                vmax=1.0)
                    #     overlayed_rf_ts_j = overlayed_original_ts_j[rf_prototype_j[1]:rf_prototype_j[2],
                    #                         rf_prototype_j[3]:rf_prototype_j[4]]
                    #     plt.imsave(os.path.join(dir_for_saving_prototypes,
                    #                             prototype_ts_filename_prefix + '-receptive_field_with_self_act' + str(
                    #                                 j) + '.png'),
                    #                overlayed_rf_ts_j,
                    #                vmin=0.0,
                    #                vmax=1.0)

                    # save the prototype image (highly activated region of the whole image)
                    # plt.imsave(os.path.join(dir_for_saving_prototypes,
                    #                         prototype_img_filename_prefix + str(j) + '.png'),
                    #            proto_img_j,
                    #            vmin=0.0,
                    #            vmax=1.0)

    if class_specific:
        del class_to_ts_index_dict
