import os
import shutil

# import torch.utils.data.distributed

import argparse
import re

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import model
import push
import train_and_test as tnt
from utils import save
from utils.helpers import makedir, get_time, parser_else_settings
from utils.dataset_util import generate_loader, read_UCR_UEA
from utils.log import create_logger
from utils.optimizer_util import get_optimizer_scheduler

from utils.settings import base_architecture, experiment_run, prototype_shape, prototype_activation_function, add_on_layers_type
from utils.settings import num_train_epochs, num_warm_epochs, coefs, push_start, push_epochs, projection, push_per

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')  # python3 main.py -gpuid=0,1,2,3
parser.add_argument('--dataset', type=str, default='GunPoint')
parser.add_argument('--model', type=str, default='InceptionTime')
parser.add_argument('--pretrained_path', type=str, default='./pretrainedModels')
parser.add_argument('--projection', type=bool, default=True)
parser.add_argument('--base_architecture', type=str, default=None)
parser.add_argument('--experiment_run', type=str, default=None)
parser.add_argument('--num_prototypes', type=int, default=10, help='number of prototypes')
parser.add_argument('--proto_activ', type=str, default='log', help='activation function for prototypes')
parser.add_argument('--add_on', type=str, default='bottleneck', help='type of add-on layers')
parser.add_argument('--imagePushEach', type=bool, default=False, help='If generate image each time push, '
                                                                      'when false, only generate at the last push')
args = parser.parse_args()

dataset_name = args.dataset
model_dataset_path = os.path.join(args.pretrained_path, args.model, args.dataset)
weights_path = os.path.join(model_dataset_path, 'weight.pt')

# print(weights_path, os.path.exists(weights_path))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# projection
projection = parser_else_settings(args.projection, projection)

# base architecture
base_architecture = parser_else_settings(args.base_architecture,base_architecture)
base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
# print(base_architecture, base_architecture_type)

# experiment run
experiment_run = parser_else_settings(args.experiment_run, experiment_run)
model_dir = './saved_models/' + base_architecture + '/' + args.dataset + '/' + experiment_run + '_' + get_time() + '/'
makedir(model_dir)

# proto activate/ add on
prototype_activation_function = parser_else_settings(args.proto_activ, prototype_activation_function)
add_on_layers_type = parser_else_settings(args.add_on,add_on_layers_type)

# make log file image file setup prototype image prefix
log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'), display=False)
log_dir = os.path.join(model_dir, 'runs/')
ts_dir = os.path.join(model_dir, 'img')
makedir(ts_dir)
prototype_ts_filename_prefix = 'prototype-ts'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'
if log_dir is not None:
    writer = SummaryWriter(log_dir)

# load the data we set train loader and train push loader the same
train_batch_size = 64
train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset_name, UCR_UEA_dataloader=None)

in_channels = train_x.shape[-2]
input_size = train_x.shape[-1]
num_classes = len(np.unique(train_y))  # one-hot version train_y.shape[1]

prototype_dim = prototype_shape[1]
prototype_size = prototype_shape[2]
# set up prototype_shape
prototype_shape = (args.num_prototypes*num_classes, prototype_dim, prototype_size) \
                    if args.num_prototypes is not None else prototype_shape

train_loader, test_loader = generate_loader(train_x, test_x, train_y, test_y, batch_size_train=train_batch_size)
train_push_loader = train_loader
# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              weight_path=weights_path,
                              pretrained=True, ts_length=input_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = get_optimizer_scheduler(ppnet)

log('start training')



for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        result_train = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                                 class_specific=class_specific, coefs=coefs, log=log)

        result_test = tnt.test(model=ppnet_multi, dataloader=test_loader,
                               class_specific=class_specific, log=log)
        accu_test = result_test[0]
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'warm', accu=accu_test,
                                    target_accu=70, log=log)

    elif epoch >= push_start and epoch in push_epochs:
        if args.imagePushEach:
            root_dir_for_saving_prototypes = ts_dir
        else:
            root_dir_for_saving_prototypes = ts_dir if epoch == num_train_epochs - push_per else None
        if projection:
            push.push_prototypes(
                train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=None,  # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=root_dir_for_saving_prototypes,  # if not None, prototypes will be saved here  ts_dir
                epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
                prototype_ts_filename_prefix=prototype_ts_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            result_test = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                   class_specific=class_specific, log=log)
            accu_test = result_test[0]
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push',
                                        accu=accu_test,
                                        target_accu=0.70, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                result_train = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                         class_specific=class_specific, coefs=coefs, log=log)

                result_test = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                       class_specific=class_specific, log=log)
                accu_test = result_test[0]
                save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                            model_name=str(epoch) + '_' + str(i) + 'push', accu=accu_test,
                                            target_accu=70, log=log)

    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        result_train = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                                 class_specific=class_specific, coefs=coefs, log=log)

        result_test = tnt.test(model=ppnet_multi, dataloader=test_loader,
                               class_specific=class_specific, log=log)
        accu_test = result_test[0]
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'epoch', accu=accu_test,
                                    target_accu=70, log=log)

    accu, duration, cross_ent, cluster_cost, separation, avg_separation, total_loss, l1, p_dist_pair = result_train

    if log_dir is not None:
        writer.add_scalar('Time/Duration', duration, epoch)
        writer.add_scalar('Loss/CrossEntropy', cross_ent, epoch)
        writer.add_scalar('Loss/Cluster', cluster_cost, epoch)
        if class_specific:
            writer.add_scalar('Loss/Separation', separation, epoch)
            writer.add_scalar('Loss/AvgSeparation', avg_separation, epoch)
        writer.add_scalar('Loss/TotalLoss', total_loss, epoch)
        writer.add_scalar('Loss/L1Norm', l1, epoch)
        writer.add_scalar('Accuracy/Train', accu, epoch)
        writer.add_scalar('Accuracy/Test', accu_test, epoch)
        writer.add_scalar('Prototype Distance', p_dist_pair, epoch)
        writer.close()

log(f'Final Train ACC:{accu:.2f} Test ACC:{accu_test:.2f}')