# python 3.7
"""Computes the editing directions regarding the region of interest."""

import os
import sys
import argparse
import signal
import numpy as np
from tqdm import tqdm

from coordinate import COORDINATE_face
from coordinate import get_mask
from RobustPCA import RobustPCA


def parse_args():
    """Parses arguments."""
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument('jaco_path', type=str,
                        help='Path to jacobian matrix.')
    parser.add_argument('--region', type=str, default='eyes',
                        help='The region to be used to compute jacobian.')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified,'
                             '`./outputs/directions` will be used by default.')
    parser.add_argument('--lamb', type=int, default=60,
                        help='The coefficient to control the sparsity')
    parser.add_argument('--max_iter', type=int, default=10000,
                        help='The max iteration for low-rank factorization')
    parser.add_argument('--num_relax', type=int, default=0,
                        help='Factor of relaxation for the non-zeros singular'
                             ' values')
    parser.add_argument('--name', type=str, default='lowrankgan',
                        help='Name of help save the results.')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    assert os.path.exists(args.jaco_path)
    Jacobians = np.load(args.jaco_path)
    image_size = Jacobians.shape[2]
    w_dim = Jacobians[-1]
    assert args.region in COORDINATE_face, \
        f"{args.region} coordinate is not defined in " \
        f"COORDINATE_face. Please define this region first!"
    coords = COORDINATE_face[args.region]
    mask = get_mask(image_size, coordinate=coords)
    foreground_ind = np.where(mask == 1)
    background_ind = np.where((1 - mask) == 1)
    save_dir = args.output_dir or f'./outputs/directions'
    os.makedirs(save_dir, exist_ok=True)
    for ind in tqdm(range(Jacobians.shape[0])):
        Jacobian = Jacobians[ind]
        if len(Jacobian.shape) == 4:  # [H, W, 1, latent_dim]
            Jaco_fore = Jacobian[foreground_ind[0], foreground_ind[1], 0]
            Jaco_back = Jacobian[background_ind[0], background_ind[1], 0]
        elif len(Jacobian.shape) == 5:  # [channel, H, W, 1, latent_dim]
            Jaco_fore = Jacobian[:, foreground_ind[0], foreground_ind[1], 0]
            Jaco_back = Jacobian[:, background_ind[0], background_ind[1], 0]
        else:
            raise ValueError(f'Shape of Jacobian is not correct!')
        Jaco_fore = np.reshape(Jaco_fore, [-1, w_dim])
        Jaco_back = np.reshape(Jaco_back, [-1, w_dim])
        coef_f = 1 / Jaco_fore.shape[0]
        coef_b = 1 / Jaco_back.shape[0]
        M_fore = coef_f * Jaco_fore.T.dot(Jaco_fore)
        B_back = coef_b * Jaco_back.T.dot(Jaco_back)
        # low-rank factorization on foreground
        RPCA = RobustPCA(M_fore, lamb=1/args.lamb)
        L_f, _ = RPCA.fit(max_iter=args.max_iter)
        rank_f = np.linalg.matrix_rank(L_f)
        # low-rank factorization on background
        RPCA = RobustPCA(B_back, lamb=1/args.lamb)
        L_b, _ = RPCA.fit(max_iter=args.max_iter)
        rank_b = np.linalg.matrix_rank(L_b)
        # SVD on the low-rank matrix
        _, _, VHf = np.linalg.svd(L_f)
        _, _, VHb = np.linalg.svd(L_b)
        F_principal = VHf[:rank_f]  # Principal space of foreground
        relax_subspace = min(max(1, rank_b - args.num_relax), w_dim-1)
        B_null = VHb[relax_subspace:].T  # Null space of background

        F_principal_proj = B_null.dot(B_null.T).dot(F_principal.T)  # Projection
        F_principal_proj = F_principal_proj.T
        F_principal_proj /= np.linalg.norm(
            F_principal_proj, axis=1, keepdims=True)
        save_name = (f'{save_dir}/directions_img_{ind:02d}_'
                     f'region_{args.region}_{args.name}')
        np.save(f'{save_name}.npy', F_principal)
        np.save(f'{save_name}_projected.npy', F_principal_proj)


if __name__ == "__main__":
    main()
