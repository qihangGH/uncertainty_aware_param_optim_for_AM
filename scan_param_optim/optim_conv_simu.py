import argparse
import datetime
import json
import os
import pdb

import sys

sys.path.append('..')
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from tensorflow import keras as tfk

from utils.dataload_utils import save_args, load_args
from utils.test_utils import test_with_non_bayesian, test_with_bayesian_gaussian, test_with_bayesian_stu
from MPModels.mp_models import MPSigRegressor, transform_to_StudentT

tfd = tfp.distributions
tfpl = tfp.layers

parser = argparse.ArgumentParser("Optimize scan parameters based on the convolutional model")

# ------------------------------------load model------------------------------------
parser.add_argument("-m", "--model_filepath", type=str, default=
"../trained_models/20240504_simu_rd20_128_b256",
# "../trained_models/20240506_simu_stu_rd20_128_b256",
                    help="Path to the trained model file")
parser.add_argument("--weight_suffix", type=str, default='_relative_error',
                    choices=['_relative_error', '_loss', '_mse'])
# --------------------------load layer to be optimized------------------------------
parser.add_argument("--scan_info_filepath", type=str, default='../scan_params/warp.npy')
parser.add_argument("--scan_cache_dir", type=str, default='../cache/warp')
parser.add_argument("--dt", type=float, default=0.00002)  # in second
parser.add_argument("--p_max", type=float, default=250)  # in Watt
# --------------------------optimization parameters------------------------------
parser.add_argument('--save_dir', type=str, default=
'opt_warp_det',
# 'opt_warp_prob',
                    help="The folder to save optimization results"
                    )
parser.add_argument('--warm_start_filepath', type=str, default=
None
# 'opt_warp_det/80.npy'
                    )
parser.add_argument('--test_others_dir', type=str, default=
None
                    )
parser.add_argument('--gt_file', type=str, default=
None
                    )
parser.add_argument('--save_every', type=int, default=
1,
# 10
                    help="The frequency to save optimization results"
                    )
parser.add_argument('--test_layer_every', type=int, default=100)
parser.add_argument('--epochs', type=int, default=5000,
                    help='The epochs of the optimization. You can set it as a large number and let the `break_tol`'
                         'assess whether the optimization converges.')
parser.add_argument('--train_infer_times', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=
3200,
# 500,
                    help="The batch size in the batch-based gradient descent. Tune it smaller if out of memory error occurs"
                    )
parser.add_argument('--test_layer_infer_times', type=int, default=1)
parser.add_argument('--test_layer_batch_size', type=int, default=
8000
# 3000
                    )
parser.add_argument('--gt', type=float, default=20)
parser.add_argument('--du_max', type=float, default=5)
parser.add_argument('--reduce_var_weight', type=float, default=0.6)
parser.add_argument('--variation_weight', type=float, default=100.)
parser.add_argument('--constant_powers', action='store_true', default=False)
parser.add_argument('--downsample_rate', type=str, default=
'128,64,32,16,8,4,2'
# '16,8,4,2'
                    )
parser.add_argument('--lr', type=float, default=
0.2,
# 0.0625,
                    help='learning rate or step size'
                    )
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--window_size', type=int, default=
3,
# 4,
                    )
parser.add_argument('--tol', type=float, default=
0.128
# 0.2
                    )
parser.add_argument('--break_tol', type=float, default=
0.01
# 0.02
                    )
parser.add_argument('--smooth_factor', type=float, default=0.01)
parser.add_argument('--threshold', type=float, default=10)
parser.add_argument('--radius', type=int, default=15)
opts = parser.parse_args()
model_filename = os.path.split(opts.model_filepath)[-1]
args = load_args(os.path.join(opts.model_filepath, model_filename + '.json'), is_print_args=False)

patch_sizes = [int(size) for size in args.patch_sizes.split(',')]
if isinstance(args.units_after_concat, int):
    units_after_concat = args.units_after_concat
else:
    units_after_concat = [int(i) for i in args.units_after_concat.split(',')]
model_params_dict = dict(
    blocks_feature_extractor=args.blocks_feature_extractor,
    filters_feature_extractor=args.filters_feature_extractor,
    units_before_concat=args.units_before_concat,
    units_after_concat=units_after_concat,
    activation=args.activation,
    output_dist=args.model_name,
    scales=len(patch_sizes),
    pv_units=args.pv_units,
    fusion_opt=args.fusion_opt,
    dropout_rate=args.dropout_rate,
    preprocess_pvt_mode=args.preprocess_pvt_mode,
    weight_decay=None
)

# p_min and v_min must be zeros here,
# if p_min is not zero, e.g., p_min = 50, then the grid that has p = 50 will be normalized to 0,
# which cannot be distinguished from empty grids
p_min, p_max = 0., opts.p_max


def smooth_custom_loss(y_true, y_pred, threshold=10, smooth_factor=10):
    abs_diff = tf.abs(y_true - y_pred)
    sigmoid_weight = tf.sigmoid(smooth_factor * (abs_diff - threshold))
    loss = tf.reduce_sum(sigmoid_weight * tf.square(abs_diff)) / (tf.reduce_sum(sigmoid_weight) + 1e-9)
    return loss


def bayesian_smooth_custom_loss(y_true, y_pred, nll, threshold=10, smooth_factor=10):
    abs_diff = tf.abs(y_true - y_pred)
    sigmoid_weight = tf.sigmoid(smooth_factor * (abs_diff - threshold))
    loss = tf.reduce_sum(sigmoid_weight * nll) / (tf.reduce_sum(sigmoid_weight) + 1e-9)
    return loss


class ConvParamOptim:
    def __init__(self, scan_info, model, radius, cache_dir, downsample_rate=None, constant_powers=False):
        self.scan_info = scan_info
        assert np.min(scan_info[:, 4]) > 0
        self.reg_ind = self.seg_lines(visualize=False)
        self.radius = radius
        self.downsample_rate = downsample_rate
        self.downsample = 0
        if downsample_rate is not None:
            self.downsample = downsample_rate[0]
        self.model = model
        if args.model_name != 'non_bayesian':
            if args.model_name == 'gaussian':
                self.transform_fn = lambda t: tfd.Normal(loc=t[..., 0], scale=t[..., 1])
            elif args.model_name == 'student':
                self.transform_fn = transform_to_StudentT
            else:
                raise ValueError(f'Unknown model name {args.model_name}')
        self.model.trainable = False
        print("The model is trainable or not:", self.model.trainable)
        print('Loading data ...')
        time1 = time.time()
        self.p_seqs = np.load(os.path.join(cache_dir, f'p_seqs_{radius}.npy'), allow_pickle=True)
        self.center_p_ind = np.load(os.path.join(cache_dir, f'center_p_ind_{radius}.npy'))
        self.t_maps = np.load(os.path.join(cache_dir, f't_maps_{radius}_0.001.npy'))
        print(f'Finish loading in {time.time() - time1} seconds.')
        self.constant_powers = constant_powers
        if constant_powers:
            self.power = tf.Variable([0.], trainable=True, dtype=tf.float32)
        else:
            # self.power = tf.Variable(scan_info[:, 2], trainable=True, dtype=tf.float32)
            self.power = tf.Variable(tf.zeros_like(scan_info[:, 2]), trainable=True, dtype=tf.float32)
            if self.downsample > 1:
                # set visible indices
                self.visible_ind = self.get_visible_ind(downsample=self.downsample)
                # for each invisible index, store its two interpolation indices and interpolation coefficient
                self.invisible_ind, self.interp_start, self.interp_end, self.interp_t = self.get_interp()
                assert len(set(self.visible_ind).intersection(set(self.invisible_ind))) == 0 and \
                       len(self.visible_ind) + len(self.invisible_ind) == len(self.scan_info)

    def optim_params(self, gt_s, epochs, batch_size, save_dir, optimizer, lr_decay,
                     du_max, variation_weight, threshold, smooth_factor, window_size,
                     tol, converge_key='mse', save_every=20, test_layer_every=50):
        assert window_size > 2, "`window_size` must be greater than 2"
        os.makedirs(save_dir, exist_ok=True)
        ospj = lambda x: os.path.join(save_dir, x)
        print(f"Downsample rate: {self.downsample}")
        print(f"Learning rate: {optimizer.lr.numpy():.4f}")
        mean_loss = tfk.metrics.Mean(name='loss')
        mean_mse = tfk.metrics.Mean(name='mse')
        mean_re = tfk.metrics.Mean(name='mre')
        mean_nll = tfk.metrics.Mean(name='nll')
        # mean_out_loss = tfk.metrics.Mean(name='out_of_bound')
        mean_var_loss = tfk.metrics.Mean(name='var')
        mean_var_reduce_loss = tfk.metrics.Mean(name='var_red')
        if args.model_name == 'non_bayesian':
            metric_list = [mean_loss, mean_mse, mean_re, mean_var_loss]
        else:
            metric_list = [mean_loss, mean_nll, mean_mse, mean_re, mean_var_loss, mean_var_reduce_loss]
        epoch_loss = {m.name: [] for m in metric_list}
        epoch_time = [0.]

        steps = self.num_points // batch_size
        if self.num_points % batch_size != 0:
            steps += 1
        epoch_after_update = 0
        self.model.trainable = False
        self.model.training = True
        upsample_count = 0
        gt_s = gt_s[self.center_p_ind[:, 1]]
        gt_tensor = tf.constant(gt_s, dtype=tf.float32)
        all_ind = np.arange(len(self.scan_info))
        start_time = time.time()
        # test at the beginning
        self.downsample = 1
        pred_results = self.pred_layer(gt_s, batch_size=opts.test_layer_batch_size,
                                       infer_times=opts.test_layer_infer_times)
        self.visualize_pred(gt_s, pred_results, figsize=[10, 10], marker_size=20)
        # pdb.set_trace()
        self.downsample = self.downsample_rate[0]
        record_layer = {k: [v] for k, v in pred_results.items()}
        for i in range(1, epochs + 1):
            print("Epoch: {}/{}".format(i, epochs))
            ind = np.arange(self.num_points)
            np.random.shuffle(ind)
            progress_bar = tqdm(range(1, steps + 1), total=steps)
            for j in progress_bar:
                batch_ind = ind[(j - 1) * batch_size:j * batch_size]
                with tf.GradientTape() as tape:
                    # interpolate then activate
                    if self.constant_powers:
                        interp_act_power = self.get_power(tf.tile(self.power, [len(self.scan_info[:, 2])]), all_ind)
                    else:
                        interp_act_power = self.get_power(self.get_interp_power(all_ind), all_ind)
                    pv_data, n_data = self.sample_neighborhoods_from_cache_training(batch_ind, interp_act_power)
                    batch_gt = tf.gather(gt_tensor, batch_ind)
                    if args.model_name == 'non_bayesian':
                        pred_s = self.model([pv_data, n_data])
                        if smooth_factor > 0.:
                            main_loss = smooth_custom_loss(batch_gt, pred_s, threshold, smooth_factor)
                        else:
                            main_loss = tf.reduce_mean((batch_gt - pred_s) ** 2)
                        main_rel_error = tf.stop_gradient(tf.reduce_mean(tf.abs(pred_s - batch_gt) / batch_gt) * 100)
                    else:
                        if opts.train_infer_times > 1:

                            def single_infer(_):
                                params_output = self.model.layers[-2]([pv_data, n_data], training=True)
                                dist_output = tfpl.DistributionLambda(self.transform_fn)(params_output)
                                nnl_output = -dist_output.log_prob(batch_gt)
                                return params_output, nnl_output

                            all_params_pred, all_nll = tf.map_fn(
                                single_infer,
                                tf.range(opts.train_infer_times),
                                dtype=(tf.float32, tf.float32)
                            )
                            mean_nnl = tf.reduce_mean(all_nll, axis=0)  # [B, ]
                            mean = tf.reduce_mean(all_params_pred[..., 0], axis=0)  # [B, ]
                            if smooth_factor > 0.:
                                main_loss = bayesian_smooth_custom_loss(
                                    batch_gt, mean, mean_nnl, threshold, smooth_factor
                                )
                            else:
                                main_loss = tf.reduce_mean(mean_nnl)
                            if opts.reduce_var_weight > 0:
                                if args.model_name == 'student':
                                    variance = tf.reduce_mean(
                                        all_params_pred[..., 3] * (1 + all_params_pred[..., 1]) / (
                                                all_params_pred[..., 2] - 1)
                                        / all_params_pred[..., 1] + all_params_pred[..., 0] ** 2, axis=0
                                    ) - mean ** 2  # [B, ]
                                elif args.model_name == 'gaussian':
                                    variance = tf.reduce_mean(
                                        all_params_pred[..., 1] ** 2 + all_params_pred[..., 0] ** 2, axis=0
                                    ) - mean ** 2  # [B, ]
                                else:
                                    raise ValueError(f'Unknown model name {args.model_name}')
                                # var_reduce_loss = tf.reduce_mean(variance)
                                var_reduce_loss = tf.reduce_mean(tf.sqrt(variance + 1e-9))
                                main_loss += opts.reduce_var_weight * var_reduce_loss
                            mse_loss = tf.stop_gradient(tf.reduce_mean((mean - batch_gt) ** 2))
                            main_rel_error = tf.stop_gradient(tf.reduce_mean(tf.abs(mean - batch_gt) / batch_gt) * 100)
                        else:
                            params_pred = self.model.layers[-2]([pv_data, n_data], training=True)
                            pred_s_dist = tfpl.DistributionLambda(self.transform_fn)(params_pred)
                            if smooth_factor > 0.:
                                main_loss = bayesian_smooth_custom_loss(
                                    batch_gt, params_pred[..., 0], -pred_s_dist.log_prob(batch_gt), threshold,
                                    smooth_factor
                                )
                            else:
                                main_loss = tf.reduce_mean(-pred_s_dist.log_prob(batch_gt))
                            if opts.reduce_var_weight > 0:
                                # Only uncertainty of data because inference only once
                                if args.model_name == 'student':
                                    # [m, beta, a, b] b(1+beta) / (a-1) / beta
                                    variance = params_pred[..., 3] * (1 + params_pred[..., 1]) / (
                                            params_pred[..., 2] - 1) / params_pred[..., 1]
                                elif args.model_name == 'gaussian':
                                    variance = params_pred[..., 1] ** 2
                                else:
                                    raise ValueError(f'Unknown model name {args.model_name}')
                                # var_reduce_loss = tf.reduce_mean(variance)
                                var_reduce_loss = tf.reduce_mean(tf.sqrt(variance + 1e-9))
                                main_loss += opts.reduce_var_weight * var_reduce_loss
                            mse_loss = tf.stop_gradient(tf.reduce_mean((params_pred[..., 0] - batch_gt) ** 2))
                            main_rel_error = tf.stop_gradient(
                                tf.reduce_mean(tf.abs(params_pred[..., 0] - batch_gt) / batch_gt) * 100)
                    # regularization
                    var_loss_per = tf.square(tf.gather(interp_act_power, self.reg_ind + 1) - tf.gather(interp_act_power,
                                                                                                       self.reg_ind)) - du_max ** 2
                    masked_var_loss = tf.boolean_mask(var_loss_per, var_loss_per > 0)
                    var_loss = variation_weight * tf.cond(
                        tf.size(masked_var_loss) > 0,
                        lambda: tf.reduce_sum(masked_var_loss),
                        lambda: tf.constant(0.0, dtype=tf.float32)
                    )
                    train_loss = tf.add_n([main_loss, var_loss])
                gradients = tape.gradient(train_loss, [self.power])
                optimizer.apply_gradients(zip(gradients, [self.power]))
                mean_loss.update_state([train_loss] * len(batch_ind))
                mean_re.update_state([main_rel_error] * len(batch_ind))
                if args.model_name == 'non_bayesian':
                    mean_mse.update_state([main_loss] * len(batch_ind))
                else:
                    mean_nll.update_state([main_loss] * len(batch_ind))
                    mean_mse.update_state([mse_loss] * len(batch_ind))
                    if opts.reduce_var_weight > 0:
                        mean_var_reduce_loss.update_state([var_reduce_loss] * len(batch_ind))
                mean_var_loss.update_state([var_loss] * len(batch_ind))
                progress_bar.set_description(" ".join(["{}: {:.2f}".format(m.name, m.result()) for m in metric_list]))

            epoch_time.append(time.time() - start_time)
            for m in metric_list:
                epoch_loss[m.name].append(m.result())
                m.reset_states()

            if i % save_every == 0:
                self.save_params(ospj(f'{i}.npy'))
                np.save(ospj(f'epoch_time'), np.array(epoch_time))
                for k, v in epoch_loss.items():
                    np.save(ospj(f'epoch_{k}.npy'), np.array(v))

            update_cond = False
            if len(epoch_loss[converge_key]) >= window_size:
                window = np.array(epoch_loss[converge_key][-window_size:])
                moving_avg = np.mean(window)
                avg_diff = np.abs(moving_avg - epoch_loss[converge_key][-1])
                print(f'Moving avg: {moving_avg} / {avg_diff}')
                update_cond = avg_diff < tol
                if self.downsample == self.downsample_rate[-1] and update_cond and epoch_after_update:
                    break
            if upsample_count < len(self.downsample_rate) - 1 and epoch_after_update >= window_size and update_cond:
                upsample_count += 1
                tol /= 2
                if tol < opts.break_tol:
                    tol = opts.break_tol
                if self.constant_powers:
                    print(f'\033[31mtol from {tol * 2} to {tol}')
                    self.downsample = self.downsample_rate[upsample_count]
                else:
                    print(f'\033[31mUpsampling: downsample rate from {self.downsample_rate[upsample_count - 1]} '
                          f'to {self.downsample_rate[upsample_count]}, tol from {tol * 2} to {tol}')
                    # interpolate powers
                    self.power.assign(self.get_interp_power(all_ind))
                    self.downsample = self.downsample_rate[upsample_count]
                    if self.downsample > 1:
                        self.visible_ind = self.get_visible_ind(downsample=self.downsample)
                        self.invisible_ind, self.interp_start, self.interp_end, self.interp_t = self.get_interp()
                        assert len(set(self.visible_ind).intersection(set(self.invisible_ind))) == 0 and \
                               len(self.visible_ind) + len(self.invisible_ind) == len(self.scan_info)
                # set new learning rate
                print(f'The learning rate is downsampled from {optimizer.lr.numpy():.4f} '
                      f'to {optimizer.lr.numpy() * lr_decay:.4f}\033[m')
                optimizer.lr.assign(optimizer.lr * lr_decay)
                epoch_after_update = 0
            else:
                epoch_after_update += 1

            # calculate var loss
            if not self.constant_powers:
                ori_power = self.original_power
                ori_var_loss = np.abs(ori_power[self.reg_ind + 1] - ori_power[self.reg_ind]) - du_max * p_max
                ori_var_loss = ori_var_loss[ori_var_loss > 0]
                if len(ori_var_loss) > 0:
                    print(f'Out of bound: {len(ori_var_loss)}/{len(ori_power)} = {len(ori_var_loss) / len(ori_power)} '
                          f'Max: {np.max(ori_var_loss)}, Min: {np.min(ori_var_loss)}, Mean: {np.mean(ori_var_loss)}')

            if i % test_layer_every == 0:
                marker_size = 20
                figsize = [10, 10]
                pred_results = self.pred_layer(gt_s, batch_size=opts.test_layer_batch_size,
                                               infer_times=opts.test_layer_infer_times)
                self.visualize_pred(gt_s, pred_results, figsize, marker_size, visual_tuple=())
                for k in record_layer:
                    record_layer[k].append(pred_results[k])
                    np.save(ospj(f'layer_{k}.npy'), np.vstack(record_layer[k]))

    def visualize_pred(self, gt, pred_results, figsize, marker_size,
                       visual_tuple=('nnl', 's_star', 'aleatoric', 'epistemic')):
        print(f"MSE: {np.mean((pred_results['m_star'] - gt) ** 2):.4f}, "
              f"MRE: {np.mean(np.abs(pred_results['m_star'] - gt) / gt) * 100:.4f}%")
        if args.model_name != 'non_bayesian':
            print(f"NNL: {np.mean(pred_results['nnl']):.4f}, Uncertainty: {np.mean(pred_results['s_star']):.4f}, "
                  f"Aleatoric: {np.mean(pred_results['aleatoric']):.4f}, Epistemic: {np.mean(pred_results['epistemic']):.4f}")
            print(f"Mean max 100 uncertainty: {np.mean(np.sort(pred_results['s_star'])[-100:]):.4f}")
        # self.visual_a_field(self.original_power[self.center_p_ind[:, 1]], figsize, marker_size,
        #                     cmap='plasma', title='optimized power')
        # # abs_diff = tf.abs(tf.constant(gt) - tf.constant(pred_results['m_star']))
        # # sigmoid_weight = tf.sigmoid(smooth_factor * (abs_diff - threshold))
        # # weighted_mse = sigmoid_weight * tf.square(abs_diff)
        # self.visual_a_field(pred_results['m_star'], figsize, marker_size, title='optimized mp size')
        # if args.model_name != 'non_bayesian':
        #     for k in visual_tuple:
        #         self.visual_a_field(pred_results[k], figsize, marker_size, title=k)
        # # self.visual_a_field(sigmoid_weight.numpy(), figsize, marker_size, title='sigmoid weight')
        # # self.visual_a_field(abs_diff.numpy(), figsize, marker_size, title='abs diff')
        # # self.visual_a_field(weighted_mse.numpy(), figsize, marker_size, title='weighted mse')

    def sample_neighborhoods_from_cache(self, sel_ind):
        neighbor_p_seq = self.p_seqs[sel_ind]
        patch_ind = tf.concat(
            [i * tf.ones(len(neighbor), dtype=tf.int32) for i, neighbor in enumerate(neighbor_p_seq)], axis=0)
        neighbor_p_seq = np.concatenate(neighbor_p_seq)
        interp_power = self.get_interp_power(np.hstack([self.center_p_ind[sel_ind, 1], neighbor_p_seq[:, 3]]))
        pv_data = self.get_power(interp_power, self.center_p_ind[sel_ind, 1])[:, None]
        row_ind, col_ind = \
            tf.constant(neighbor_p_seq[:, 0], dtype=tf.int32), tf.constant(neighbor_p_seq[:, 1], dtype=tf.int32)
        p_fill_ind = tf.stack([patch_ind, row_ind, col_ind], axis=-1)
        p_maps = tf.zeros((len(sel_ind), 2 * self.radius + 2, 2 * self.radius + 2), dtype=tf.float32)
        p_maps = tf.tensor_scatter_nd_update(p_maps, p_fill_ind, self.get_power(interp_power, neighbor_p_seq[:, 3]))
        n_data = tf.stack([
            p_maps,
            tf.constant(self.t_maps[sel_ind], dtype=tf.float32)
        ], axis=-1)
        return pv_data, n_data

    def sample_neighborhoods_from_cache_training(self, sel_ind, interp_act_power):
        neighbor_p_seq = self.p_seqs[sel_ind]
        patch_ind = tf.concat(
            [i * tf.ones(len(neighbor), dtype=tf.int32) for i, neighbor in enumerate(neighbor_p_seq)], axis=0)
        neighbor_p_seq = np.concatenate(neighbor_p_seq)
        pv_data = tf.gather(interp_act_power, self.center_p_ind[sel_ind, 1])[:, None]
        row_ind, col_ind = \
            tf.constant(neighbor_p_seq[:, 0], dtype=tf.int32), tf.constant(neighbor_p_seq[:, 1], dtype=tf.int32)
        p_fill_ind = tf.stack([patch_ind, row_ind, col_ind], axis=-1)
        p_maps = tf.zeros((len(sel_ind), 2 * self.radius + 2, 2 * self.radius + 2), dtype=tf.float32)
        p_maps = tf.tensor_scatter_nd_update(p_maps, p_fill_ind, tf.gather(interp_act_power, neighbor_p_seq[:, 3]))
        n_data = tf.stack([
            p_maps,
            tf.constant(self.t_maps[sel_ind], dtype=tf.float32)
        ], axis=-1)
        return pv_data, n_data

    @staticmethod
    def get_power(power, sel_ind):
        return tf.nn.sigmoid(tf.gather(power, sel_ind))

    def set_power(self, original_p):
        assert original_p.ndim == 1, 'Please input 1-D ndarray, e.g., np.array([0.])'
        original_p = original_p / p_max
        self.power = tf.Variable(inverse_sigmoid(original_p), trainable=True, dtype=tf.float32)

    def get_visible_ind(self, downsample):
        # get segment starts and ends
        diff_ind = self.scan_info[1:, -1] - self.scan_info[:-1, -1]
        end_ind = np.where(diff_ind > 1)[0]
        start_ind = end_ind + 1
        end_ind = np.append(end_ind, len(self.scan_info) - 1)
        start_ind = np.insert(start_ind, 0, 0)
        visible_ind = [start_ind, end_ind]
        for i in range(len(start_ind)):
            seg_len = end_ind[i] - start_ind[i] + 1
            if seg_len > downsample:
                visible_ind.append(np.arange(start_ind[i], end_ind[i] + 1, downsample))
        visible_ind = np.asarray(list(set(np.hstack(visible_ind))))
        # Note that an order can be permuted in a set
        visible_ind = np.sort(visible_ind)
        return visible_ind

    def get_interp(self):
        invisible_ind = []
        interp_start = []
        interp_end = []
        # for i in tqdm(range(1, len(self.visible_ind))):
        for i in range(1, len(self.visible_ind)):
            len_ind = self.visible_ind[i] - self.visible_ind[i - 1]
            if len_ind > 1:
                invisible_ind.append(np.arange(self.visible_ind[i - 1] + 1, self.visible_ind[i]))
                interp_start.append(self.visible_ind[i - 1] * np.ones(len_ind - 1, dtype=int))
                interp_end.append(self.visible_ind[i] * np.ones(len_ind - 1, dtype=int))
        invisible_ind, interp_start, interp_end = \
            np.hstack(invisible_ind), np.hstack(interp_start), np.hstack(interp_end)
        invisible_loc = self.scan_info[invisible_ind, :2]
        dis2start = np.linalg.norm(invisible_loc - self.scan_info[interp_start, :2], axis=-1)
        dis2end = np.linalg.norm(invisible_loc - self.scan_info[interp_end, :2], axis=-1)
        interp_t = dis2start / (dis2start + dis2end + 1e-9)
        return invisible_ind, interp_start, interp_end, interp_t

    def get_interp_power(self, sel_ind):
        if self.constant_powers:
            return tf.tile(self.power, [len(self.scan_info[:, 2])])
        else:
            if self.downsample > 1:
                # remove redundant indices
                sel_ind_set = set(sel_ind)
                sel_ind_array = np.sort(np.asarray(list(sel_ind_set), dtype=int))
                is_invisible_ind = np.isin(self.invisible_ind, sel_ind_array)
                invisible_ind = self.invisible_ind[is_invisible_ind]
                invisible_start, invisible_end, invisible_t = \
                    self.interp_start[is_invisible_ind], self.interp_end[is_invisible_ind], self.interp_t[
                        is_invisible_ind]
                invisible_t = tf.constant(invisible_t, dtype=tf.float32)
                invisible_power = (1 - invisible_t) * tf.gather(self.power, invisible_start) + \
                                  invisible_t * tf.gather(self.power, invisible_end)
                interp_power = tf.tensor_scatter_nd_update(self.power, invisible_ind[:, None], invisible_power)
                return interp_power
            else:
                return self.power

    def seg_lines(self, visualize=False):
        ind_diff = self.scan_info[1:, 4] - self.scan_info[:-1, 4]
        jump_pts = np.where(ind_diff > 1)[0]
        start_ind = np.insert(jump_pts + 1, 0, 0)
        end_ind = np.append(jump_pts, len(self.scan_info) - 1)
        seg = [np.arange(s_ind, e_ind + 1) for s_ind, e_ind in zip(start_ind, end_ind)]
        if visualize:
            for i in range(len(seg)):
                plt.scatter(self.scan_info[seg[i], 0], self.scan_info[seg[i], 1], s=0.5)
            plt.axis('equal')
            plt.show()
        reg_ind = np.hstack([s[:-1] for s in seg])
        return reg_ind

    def pred_layer(self, gt, batch_size, infer_times=10, verbose=True):
        steps = self.num_points // batch_size
        if self.num_points % batch_size != 0:
            steps += 1
        ind = np.arange(self.num_points)
        record_keys = ['m_star']
        if args.model_name != 'non_bayesian':
            record_keys.extend(['s_star', 'aleatoric', 'epistemic', 'nnl'])
        record_results = []
        progess_bar = tqdm(range(1, steps + 1)) if verbose else range(1, steps + 1)
        for j in progess_bar:
            pv_data, n_data = self.sample_neighborhoods_from_cache(ind[(j - 1) * batch_size:j * batch_size])
            test_data = {'X': pv_data, 'n': n_data, 'y': gt[(j - 1) * batch_size:j * batch_size]}
            if args.model_name == 'non_bayesian':
                record_results.append(test_with_non_bayesian(self.model, test_data, batch_size, verbose=0))
            elif args.model_name == 'gaussian':
                record_results.append(
                    test_with_bayesian_gaussian(self.model, test_data, batch_size, infer_times, verbose=False,
                                                visualize=False))
            elif args.model_name == 'student':
                record_results.append(
                    test_with_bayesian_stu(self.model, test_data, batch_size, infer_times, verbose=False,
                                           visualize=False))
        record_results = {k: np.concatenate([record_results[i][k] for i in range(steps)]) for k in record_keys}
        return record_results

    @property
    def num_points(self):
        return len(self.center_p_ind)

    @property
    def original_power(self):
        all_ind = np.arange(len(self.scan_info))
        interp_power = self.get_interp_power(all_ind)
        return self.get_power(interp_power, all_ind).numpy() * p_max

    def save_params(self, filepath):
        np.save(filepath, self.original_power)

    def visual_a_field(self, field, figsize, marker_size, cmap='viridis', vmin=None, vmax=None, title=None):
        plt.figure(figsize=figsize)
        plt.scatter(self.scan_info[self.center_p_ind[:, 1], 0],
                    self.scan_info[self.center_p_ind[:, 1], 1],
                    c=field, s=marker_size, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(), plt.axis('equal'), plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.show()


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def find_line_seg(scan_info):
    disp = np.linalg.norm(scan_info[1:, :2] - scan_info[:-1, :2], axis=-1)
    within_seg_disp = np.mean(disp[:5])
    jump_pts = np.where(disp > 4 * within_seg_disp)[0]
    start_ind = np.insert(jump_pts + 1, 0, 0)
    end_ind = np.append(jump_pts, len(scan_info) - 1)
    seg = [np.arange(s_ind, e_ind + 1) for s_ind, e_ind in zip(start_ind, end_ind)]
    scan_info[:, 4] = np.hstack([seg[i] + 100 * (i + 1) for i in range(len(seg))])


def predict_layers(optim: ConvParamOptim, scan_params_dir, gt_s):
    assert args.model_name != 'non_bayesian'
    assert optim.downsample == 1
    record_layer = {'nnl': [], 'm_star': [], 's_star': [], 'aleatoric': [], 'epistemic': []}
    scan_param_files = [file for file in os.listdir(scan_params_dir) if file[0].isdigit()]
    scan_param_files = sorted(scan_param_files, key=lambda x: int(x[:-4]))
    gt_s = gt_s[optim.center_p_ind[:, 1]]
    for i in range(len(scan_param_files) - 20, len(scan_param_files)):  # n, n_points
        optim.set_power(np.load(os.path.join(scan_params_dir, scan_param_files[i])))
        pred_results = optim.pred_layer(gt_s, batch_size=opts.test_layer_batch_size,
                                        infer_times=opts.test_layer_infer_times, verbose=False)
        for k in record_layer:
            record_layer[k].append(pred_results[k])
        mse = (pred_results['m_star'] - gt_s) ** 2
        print(
            f"{scan_param_files[i]}: "
            f"Max mse: {np.max(mse):.3f}, Mean mse: {np.mean(mse):.3f}, "
            f"Max u: {np.max(pred_results['s_star']):.3f}, Mean u: {np.mean(pred_results['s_star']):.3f}, "
            f"Max alea: {np.max(pred_results['aleatoric']):.3f}, Mean alea: {np.mean(pred_results['aleatoric']):.3f}, "
            f"Max epis: {np.max(pred_results['epistemic']):.3f}, Mean epis: {np.mean(pred_results['epistemic']):.3f}"
        )
    for k in record_layer:
        np.save(os.path.join(scan_params_dir, f'{args.model_name}_layer_{k}_last20.npy'), np.vstack(record_layer[k]))


if __name__ == '__main__':
    save_args(opts, save_dir=opts.save_dir)
    # load original scan data [x, y, p, v, idx];
    # because there are some points that are not fused, so the `idx` is not continuous
    scan_data = np.load(opts.scan_info_filepath)
    dt = opts.dt
    temp_speed = np.hstack([
        np.zeros(1), np.linalg.norm(scan_data[2:, :2] - scan_data[:-2, :2], axis=1) / (2 * dt), np.zeros(1)])
    # Note that time indices starts from 1
    scan_data = np.hstack([scan_data, temp_speed.reshape((-1, 1)), np.arange(1, len(scan_data) + 1).reshape(-1, 1)])
    scan_data[:, 2] = np.clip(scan_data[:, 2], a_min=p_min, a_max=p_max)
    scan_data[:, 2] = scan_data[:, 2] / p_max
    find_line_seg(scan_data)
    scan_data = scan_data.astype(np.float32)

    if args.model_name == 'non_bayesian':
        test_model = MPSigRegressor(is_bayesian=False, **model_params_dict)
        test_model.compile(loss='mse')
    else:
        center_p_ind = np.load(os.path.join(opts.scan_cache_dir, f'center_p_ind_{opts.radius}.npy'))
        # `kl_weight` does not matter
        kl_weight = 1 / len(center_p_ind) if args.dropout_rate is None else None
        test_model = MPSigRegressor(is_bayesian=True, kl_weight=kl_weight, **model_params_dict)
    weight_filepath = os.path.join(opts.model_filepath, model_filename + opts.weight_suffix + '.tf')
    print('Use the model with weight {} ...'.format(weight_filepath))
    test_model.load_weights(weight_filepath).expect_partial()
    test_model.trainable = False

    optimizer = tf.optimizers.Adam(learning_rate=opts.lr)

    downsample_rate = [int(i) for i in opts.downsample_rate.split(',')]
    # Note that a larger batch size is better
    print(f"Downsample rate: {downsample_rate}")
    gt_s = opts.gt * np.ones(len(scan_data), dtype=np.float32)
    if opts.gt_file is not None:
        gt_s = np.load(opts.gt_file)

    if opts.test_others_dir is not None:
        print(f'Test the model:', opts.test_others_dir)
        param_optim = ConvParamOptim(
            scan_info=scan_data,
            model=test_model,
            radius=opts.radius,
            cache_dir=opts.scan_cache_dir,
            downsample_rate=[1]
        )
        predict_layers(param_optim, opts.test_others_dir, gt_s)
    else:
        param_optim = ConvParamOptim(
            scan_info=scan_data,
            model=test_model,
            radius=opts.radius,
            cache_dir=opts.scan_cache_dir,
            downsample_rate=downsample_rate,
            constant_powers=opts.constant_powers
        )
        if opts.warm_start_filepath is not None:
            print(f'Warm start with powers from {opts.warm_start_filepath}')
            param_optim.set_power(np.load(opts.warm_start_filepath))

        param_optim.optim_params(
            gt_s=gt_s, epochs=opts.epochs, batch_size=opts.batch_size,
            save_dir=opts.save_dir,
            optimizer=optimizer,
            lr_decay=opts.lr_decay,
            du_max=opts.du_max / p_max,
            variation_weight=opts.variation_weight,
            threshold=opts.threshold,
            smooth_factor=opts.smooth_factor,
            window_size=opts.window_size,
            tol=opts.tol,
            save_every=opts.save_every,
            test_layer_every=opts.test_layer_every
        )
