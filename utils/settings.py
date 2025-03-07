base_architecture = 'inceptionTime' #'vgg19'
protosize = 3
prototype_shape = (20, 128, protosize)
# num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'bottleneck'
projection = True
experiment_run = 'push_try_padding' + str(protosize)

# data_path = './datasets/cub200_cropped/'
# train_dir = data_path + 'train_cropped_augmented/'
# test_dir = data_path + 'test_cropped/'
# train_push_dir = data_path + 'train_cropped/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 100
num_warm_epochs = 5

push_start = 10
push_per = 10
push_epochs = [i for i in range(num_train_epochs) if i % push_per == 0]
