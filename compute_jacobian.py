# python 3.7
"""Computes the Jacobian matrix with the help of random syntheses."""

import os
import sys
import time
import argparse
import signal
import pickle
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.parallel_for.gradients import jacobian

import dnnlib
import dnnlib.tflib as tflib
from utils.visualizer import save_image

import warnings  # pylint: disable=wrong-import-order
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    """Parses arguments."""
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()

    parser.add_argument('restore_path', type=str, default='',
                        help='The pre-trained encoder pkl file path')
    parser.add_argument("--image_size", type=int,
                        default=256, help="size of the images")
    parser.add_argument("--batch_size", type=int,
                        default=1, help="size of the input batch")
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified,'
                             '`./outputs/jacobian_seed_{}` will be used '
                             'by default.')
    parser.add_argument('--total_num', type=int, default=5,
                        help='Number of latent codes to sample')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')
    parser.add_argument('--compute_z', action='store_true',
                        help='Whether to compute jacobian on z '
                             '(default: False)')
    parser.add_argument('--save_png', action='store_false',
                        help='Whether or not to save the images '
                             '(default: True)')
    parser.add_argument('--fused_channel', action='store_false',
                        help='Whether or not to mean RGB channel '
                             '(default: True)')
    parser.add_argument('--seed', type=int, default=4,
                        help='Random seed')
    parser.add_argument('--d_name', type=str, default='ffhq',
                        help='Name of the dataset.')
    return parser.parse_args()


def resize_image(images, size=256):
    """Resizes the image with data format NCHW."""
    images = tf.transpose(images, perm=[0, 2, 3, 1])
    images = tf.image.resize_images(images, size=[size, size], method=1)
    images = tf.transpose(images, perm=[0, 3, 1, 2])
    return images


def main():
    """Main function."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tf_config = {'rnd.np_random_seed': 1000}
    tflib.init_tf(tf_config)
    assert os.path.exists(args.restore_path)
    with open(args.restore_path, 'rb') as f:
        _, _, Gs = pickle.load(f)
    num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]
    # Building graph
    latent_z_ph = tf.placeholder(tf.float32, [1, latent_dim], name='latent_z')
    latent_w_ph = tf.placeholder(tf.float32, [1, latent_dim], name='latent_w')
    sess = tf.get_default_session()
    latent_w_temp = tf.expand_dims(latent_w_ph, axis=1)
    latent_wp = tf.tile(latent_w_temp, [1, num_layers, 1])

    # Sampling w from z
    latent_w = Gs.components.mapping.get_output_for(latent_z_ph, None)
    # Build graph for w
    images_w = Gs.components.synthesis.get_output_for(
        latent_wp, randomize_noise=False)
    output_size = images_w.shape[-1]
    if output_size != args.image_size:
        images_w = resize_image(images=images_w, size=args.image_size)
    images_w_255 = tflib.convert_images_to_uint8(images_w, nchw_to_nhwc=True)
    if args.fused_channel:
        images_w = tf.reduce_mean(images_w, axis=1)
    # Build graph for z
    images_z = Gs.get_output_for(
        latent_z_ph, None, is_validation=True, randomize_noise=False)
    if output_size != args.image_size:
        images_z = resize_image(images=images_z, size=args.image_size)
    if args.fused_channel:
        images_z = tf.reduce_mean(images_z, axis=1)

    jaco_w = jacobian(images_w, latent_w_ph, use_pfor=False)
    print(f'Jacobian w shape: {jaco_w.shape}')
    if args.compute_z:
        jaco_z = jacobian(images_z, latent_z_ph, use_pfor=False)
        print(f'Jacobian z shape: {jaco_z.shape}')

    save_dir = args.output_dir or f'./outputs/jacobians_seed_{args.seed}'
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(args.seed)
    print(f'Starting calculating jacobian...')
    sys.stdout.flush()
    jaco_ws = []
    jaco_zs = []
    start_time = time.time()
    latent_codes = np.random.randn(args.total_num, latent_dim)
    for num in tqdm(range(latent_codes.shape[0])):
        latent_code = latent_codes[num:num+1]
        feed_dict = {latent_z_ph: latent_code}
        latent_w_i = sess.run(latent_w, feed_dict)
        feed_dict = {latent_w_ph: latent_w_i[:, 0]}
        jaco_w_i = sess.run(jaco_w, feed_dict)
        if args.compute_z:
            jaco_z_i = sess.run(jaco_z, {latent_z_ph: latent_code})
            jaco_zs.append(jaco_z_i)
        jaco_ws.append(jaco_w_i)

        if args.save_png:
            img_i = sess.run(images_w_255, {latent_w_ph: latent_w_i[:, 0]})
            save_path0 = (f'{save_dir}/syn_dataset_{args.d_name}_'
                          f'number_{num:04d}.png')
            save_image(save_path0, img_i[0])

    jaco_w = np.concatenate(jaco_ws, axis=0)
    print(f'jaco_w shape {jaco_w.shape}')
    np.save(f'{save_dir}/w_dataset_{args.d_name}.npy', jaco_w)

    if args.compute_z:
        jaco_z = np.concatenate(jaco_zs, axis=0)
        print(f'jaco_z shape {jaco_z.shape}')
        np.save(f'{save_dir}/z_dataset_{args.d_name}.npy', jaco_z)

    print(f'Finished! and Time: '
          f'{dnnlib.util.format_time(time.time() - start_time):12s}')


if __name__ == "__main__":
    main()
