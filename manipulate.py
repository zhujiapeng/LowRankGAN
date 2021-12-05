# python 3.7
"""Manipulates the latent codes using existing model and directions."""

import os
import sys
import argparse
import signal
import pickle
from tqdm import tqdm
import tensorflow as tf
import numpy as np

import dnnlib.tflib as tflib
from utils.visualizer import save_image, adjust_pixel_range
from utils.visualizer import HtmlPageVisualizer
from utils.editor import parse_indices
from utils.editor import manipulate_codes

import warnings  # pylint: disable=wrong-import-order
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    """Parses arguments."""
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()

    parser.add_argument('restore_path', type=str, default='',
                        help='The pre-trained encoder pkl file path')
    parser.add_argument('matrix_path', type=str,
                        help='Path to the low rank matrix.')
    parser.add_argument("--batch_size", type=int,
                        default=1, help="size of the input batch")
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified,'
                             '`./outputs/manipulation` will be used '
                             'by default.')
    parser.add_argument('--total_num', type=int, default=10,
                        help='number of loops for sampling')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index of the manipulation direction')
    parser.add_argument('--end', type=int, default=1,
                        help='Number of direction will be used in VH')
    parser.add_argument('--mani_layers', type=str, default='4,5,6,7',
                        help='The layer will be manipulated')
    parser.add_argument('--start_distance', type=float, default=-4.0,
                        help='Start distance for manipulation. (default: -4.0)')
    parser.add_argument('--end_distance', type=float, default=4.0,
                        help='End distance for manipulation. (default: 4.0)')
    parser.add_argument('--step', type=int, default=7,
                        help='Number of steps for manipulation. (default: 7)')
    parser.add_argument('--save_raw', action='store_true',
                        help='Whether to save raw images (default: False)')
    parser.add_argument('--seed', type=int, default=4,
                        help='random seed')
    parser.add_argument('--name', type=str, default='lowrankgan',
                        help='The name to help save the file')
    parser.add_argument('--latent_path', type=str, default=None,
                        help='The path to the existing latent codes.')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tf_config = {'rnd.np_random_seed': 1000}
    tflib.init_tf(tf_config)
    np.random.seed(args.seed)
    assert os.path.exists(args.restore_path)
    assert os.path.exists(args.matrix_path)
    with open(args.restore_path, 'rb') as f:
        _, _, Gs = pickle.load(f)
    directions = np.load(args.matrix_path)
    num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]
    # Building graph
    latent_z_ph = tf.placeholder(
        tf.float32, [None, latent_dim], name='latent_z')
    latent_w_ph = tf.placeholder(
        tf.float32, [None, num_layers, latent_dim], name='latent_w')
    sess = tf.get_default_session()
    latent_w = Gs.components.mapping.get_output_for(latent_z_ph, None)
    images = Gs.components.synthesis.get_output_for(
        latent_w_ph, randomize_noise=False)
    print(f'Load or Generate latent codes.')
    if os.path.exists(args.latent_path):
        latent_codes = np.load(args.latent_path)
        latent_codes = latent_codes[:args.total_num]
    else:
        latent_codes = np.random.randn(args.total_num, latent_dim)
    feed_dict = {latent_z_ph: latent_codes}
    latent_ws = sess.run(latent_w, feed_dict)
    num_images = latent_ws.shape[0]
    image_list = []
    for i in range(num_images):
        image_list.append(f'{i:06d}')
    save_dir = args.output_dir or f'./outputs/manipulations'
    os.makedirs(save_dir, exist_ok=True)
    delta_num = args.end - args.start
    visualizer = HtmlPageVisualizer(num_rows=num_images * delta_num,
                                    num_cols=args.step + 2,
                                    viz_size=256)
    layer_index = parse_indices(args.mani_layers)
    print(f'Manipulate on layers {layer_index}')
    for row in tqdm(range(num_images)):
        latent_code = latent_ws[row:row+1]
        images_ori = sess.run(images, {latent_w_ph: latent_code})
        images_ori = adjust_pixel_range(images_ori)
        if args.save_raw:
            save_image(f'{save_dir}/ori_{row:06d}.png', images_ori[0])
        for num_direc in range(args.start, args.end):
            html_row = num_direc - args.start
            direction = directions[num_direc:num_direc + 1][:, np.newaxis]
            direction = np.tile(direction, [1, num_layers, 1])
            visualizer.set_cell(row * delta_num + html_row, 0,
                                text=f'{image_list[row]}_{num_direc:03d}')
            visualizer.set_cell(row * delta_num + html_row, 1,
                                image=images_ori[0])
            mani_codes = manipulate_codes(latent_code=latent_code,
                                          boundary=direction,
                                          layer_index=layer_index,
                                          start_distance=args.start_distance,
                                          end_distance=args.end_distance,
                                          steps=args.step)
            mani_images = sess.run(images, {latent_w_ph: mani_codes})
            mani_images = adjust_pixel_range(mani_images)
            for i in range(mani_images.shape[0]):
                visualizer.set_cell(row * delta_num + html_row, i + 2,
                                    image=mani_images[i])
                if args.save_raw:
                    save_name = (
                        f'mani_{row:06d}_ind_{num_direc:06d}_{i:06d}.png')
                    save_image(f'{save_dir}/{save_name}', mani_images[i])
    visualizer.save(f'{save_dir}/manipulate_results_{args.name}.html')


if __name__ == "__main__":
    main()
