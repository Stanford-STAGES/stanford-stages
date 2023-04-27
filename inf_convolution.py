import torch
import torch.nn.functional as F


def batch_norm(x, n_out, av_dims, is_training, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with torch.no_grad():
        beta = torch.zeros(n_out, requires_grad=True)
        gamma = torch.ones(n_out, requires_grad=True)
        running_mean = torch.zeros(n_out)
        running_var = torch.ones(n_out)
    if is_training:
        batch_mean, batch_var = torch.mean(x, dim=av_dims), torch.var(x, dim=av_dims)
        running_mean = 0.99 * running_mean + 0.01 * batch_mean.detach()
        running_var = 0.99 * running_var + 0.01 * batch_var.detach()
        normed = F.batch_norm(x, running_mean, running_var, beta, gamma, eps=1e-3, training=True)
    else:
        normed = F.batch_norm(x, running_mean, running_var, beta, gamma, eps=1e-3, training=False)
    return normed

    # phase_train = tf.Print(phase_train,[phase_train])
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    # meanV = tf.Print(mean,[mean])
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


"""
def _activation_summary(x):

  tensor_name = re.sub('%s summary', '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
"""


def _variable_on_cpu(name, shape, initializer):
    dtype = torch.float32
    var = torch.nn.Parameter(initializer(torch.empty(shape), dtype=dtype), requires_grad=True)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = torch.float32
    initializer = torch.nn.init.truncated_normal_
    var = _variable_on_cpu(name, shape, lambda x, dtype=dtype: initializer(x, mean=0, std=stddev, dtype=dtype))
    
    if wd is not None:
        weight_decay = wd * torch.sum(var ** 2)
        torch.nn.init.zeros_(weight_decay)
        torch.autograd.backward([weight_decay], retain_graph=True)
    
    return var


def conv_block(config, inputs, scope_name, fShape, stride):
    with torch.autograd.profiler.record_function(scope_name):
        kernel = _variable_with_weight_decay('weights', shape=fShape,
                                             stddev=1e-3, wd=0.000001)
        conv = torch.nn.functional.conv2d(inputs, kernel, stride=(1, stride), padding=(0, 1))
        biases = _variable_on_cpu('biases', fShape[3], torch.nn.init.constant_)
        bias = torch.nn.functional.bias_add(conv, biases)
        bnormed = batch_norm(bias, fShape[3], [0, 2, 3], config.is_training, scope=scope_name)
        conv = torch.nn.functional.relu(bnormed, inplace=True)
        # _activation_summary(conv)

        return conv


def conv2d_block(is_training, inputs, scope_name, fShape, stride):
    with torch.nn.ModuleList(scope_name) as scope:
        conv = torch.nn.Conv3d(inputs, fShape[4], kernel_size=fShape[:3], stride=stride, padding='SAME')
        bnormed = torch.nn.BatchNorm3d(fShape[4])(conv)
        conv = F.relu(bnormed, name=scope_name)

        return conv


def small_autocorr(inputs, is_training, segment_size, modality, batch_size):
    if modality == 'eeg':
        nIn = 2
        nOut = [64, 128, 256]
        inputs = inputs.view(batch_size, -1, segment_size, 2, 200).permute(0, 1, 4, 2, 3)

        strides = [[3, 2], [2, 1]]
    elif modality == 'eog':
        nIn = 3
        nOut = [64, 128, 256]
        inputs = inputs.view(batch_size, -1, segment_size, 3, 400).permute(0, 1, 4, 2, 3)

        strides = [[4, 2], [2, 1]]
    else:
        nIn = 1
        nOut = [16, 32, 64]
        inputs = inputs.view(batch_size, -1, segment_size, 1, 40).permute(0, 1, 4, 2, 3)

        strides = [[2, 2], [2, 1]]

    conv1 = torch.nn.Conv3d(nIn, nOut[0], kernel_size=(1, 7, 7), stride=(1, strides[0][0], strides[0][1]), padding=(0, 3, 3))(inputs)
    conv1 = F.relu(conv1)

    pool1 = F.max_pool3d(conv1, kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))

    conv3 = torch.nn.Conv3d(nOut[0], nOut[1], kernel_size=(1, 5, 5), stride=(1, strides[1][0], strides[1][1]), padding=(0, 2, 2))(pool1)
    conv3 = F.relu(conv3)

    conv4 = torch.nn.Conv3d(nOut[1], nOut[1], kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))(conv3)
    conv4 = F.relu(conv4)

    pool2 = F.max_pool3d(conv4, kernel_size=(1, 1, 2), stride=(1, 1, 2), padding=(0, 0, 0))

    conv5 = torch.nn.Conv3d(nOut[1], nOut[2], kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))(pool2)
    conv5 = F.relu(conv5)

    conv6 = torch.nn.Conv3d(nOut[2], nOut[2], kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))(conv5)
    conv6 = F.relu(conv6)

    meanPool = F.avg_pool3d(conv6, kernel_size=(1, conv6.shape[3], conv6.shape[4]), stride=(1, 1, 1))

    return meanPool.view(batch_size, -1)


def large_autocorr(inputs, is_training, segment_size, modality, batch_size):
    if modality == 'eeg':
        nIn = 2
        nOut = [64, 128, 256, 512]
        inputs = inputs.view(batch_size, -1, 2, 200, segment_size)
        inputs = inputs.permute(0, 1, 3, 4, 2)
        strides = [[3, 2], [2, 1]]
    elif modality == 'eog':
        nIn = 3
        nOut = [64, 128, 256, 512]
        inputs = inputs.view(batch_size, -1, 3, 400, segment_size)
        inputs = inputs.permute(0, 1, 3, 4, 2)
        strides = [[4, 2], [2, 1]]
    else:
        nIn = 1
        nOut = [16, 32, 64, 512]
        inputs = inputs.view(batch_size, -1, 1, 40, segment_size)
        inputs = inputs.permute(0, 1, 3, 4, 2)
        strides = [[2, 2], [2, 1]]

    conv1 = conv2d_block(inputs, nIn, nOut[0], 'conv1' + modality, strides[0], is_training)

    pool1 = F.max_pool3d(conv1, kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

    conv3 = conv2d_block(pool1, nOut[0], nOut[1], 'conv3' + modality, strides[1], is_training)
    conv4 = conv2d_block(conv3, nOut[1], nOut[1], 'conv4' + modality, [1, 1], is_training)

    pool2 = F.max_pool3d(conv4, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 0, 0])

    conv5 = conv2d_block(pool2, nOut[1], nOut[2], 'conv5' + modality, [1, 1], is_training)
    conv6 = conv2d_block(conv5, nOut[2], nOut[2], 'conv6' + modality, [1, 1], is_training)

    pool3 = F.max_pool3d(conv6, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 0, 0])

    conv7 = conv2d_block(pool3, nOut[2], nOut[3], 'conv7' + modality, [1, 1], is_training)
    conv8 = conv2d_block(is_training, conv7, 'conv8' + modality, [nOut[3], nOut[3], 3, 3, 1], [1, 1])

    meanPool = F.avg_pool2d(conv8, kernel_size=[conv8.size(2), conv8.size(3)])
    
    return meanPool.squeeze()


def random_autocorr(inputs, is_training, segment_size, modality, batch_size):
    torch.manual_seed(0)
    if modality == 'eeg':
        nIn = 2
        nOut = [torch.randint(32, 96, (1,)).item(),
                torch.randint(64, 192, (1,)).item(),
                torch.randint(128, 384, (1,)).item()]
        inputs = inputs.view(batch_size, -1, 2, 200, segment_size)
        inputs = inputs.permute(0, 1, 3, 4, 2)

        strides = [[3, 2], [2, 1]]
    elif modality == 'eog':
        nIn = 3
        nOut = [torch.randint(32, 96, (1,)).item(),
                torch.randint(64, 192, (1,)).item(),
                torch.randint(128, 384, (1,)).item()]
        inputs = inputs.view(batch_size, -1, 3, 400, segment_size)
        inputs = inputs.permute(0, 1, 3, 4, 2)

        strides = [[4, 2], [2, 1]]
    else:
        nIn = 1
        nOut = [torch.randint(8, 24, (1,)).item(),
                torch.randint(16, 48, (1,)).item(),
                torch.randint(32, 96, (1,)).item()]
        inputs = inputs.view(batch_size, -1, 1, 40, segment_size)
        inputs = inputs.permute(0, 1, 3, 4, 2)

        strides = [[2, 2], [2, 1]]

    conv1 = conv2d_block(is_training, inputs, nIn, nOut[0], 'conv1' + modality, [1, 7, 7], strides[0])

    pool1 = torch.nn.functional.max_pool3d(conv1, kernel_size=[1, 1, 3], stride=[1, 1, 2],
                             padding=[0, 0, 1])

    conv2 = conv2d_block(is_training, pool1, nOut[0], nOut[1], 'conv2' + modality, [1, 5, 5], strides[1])
    conv3 = conv2d_block(is_training, conv2, nOut[1], nOut[1], 'conv3' + modality, [1, 3, 3], [1, 1])

    pool2 = torch.nn.functional.max_pool3d(conv3, kernel_size=[1, 1, 2], stride=[1, 1, 2],
                             padding=[0, 0, 0])

    conv4 = conv2d_block(is_training, pool2, nOut[1], nOut[2], 'conv4' + modality, [1, 3, 3], [1, 1])
    conv5 = conv2d_block(is_training, conv4, nOut[2], nOut[2], 'conv5' + modality, [1, 3, 3], [1, 1])

    meanPool = torch.mean(conv5, dim=2, keepdim=True, name='mean_pool1' + modality)

    meanPool = torch.mean(meanPool, dim=2, keepdim=False, name='mean_pool2' + modality)

    return meanPool


def main(inputs, model_name, is_training, segment_size, modality, batch_size):
    if model_name.startswith('ac'):
        if model_name[3:5] == 'lh':
            hidden = large_autocorr(inputs, is_training, segment_size, modality, batch_size)
        elif model_name[3:5] == 'rh':
            hidden = random_autocorr(inputs, is_training, segment_size, modality, batch_size)
        else:
            hidden = small_autocorr(inputs, is_training, segment_size, modality, batch_size)

    return hidden
