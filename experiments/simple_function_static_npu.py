import os
import ast
import math
import torch
import stable_nalu
import argparse
import stable_nalu.functional.regualizer as Regualizer
from decimal import Decimal
import misc.utils as utils
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='Runs the simple function static task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.SimpleFunctionStaticNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. Tanh, ReLU, NAC, NALU')
parser.add_argument('--operation',
                    action='store',
                    default='add',
                    choices=[
                        'add', 'sub', 'mul', 'div', 'squared', 'root'
                    ],
                    type=str,
                    help='Specify the operation to use, e.g. add, mul, squared')
parser.add_argument('--num-subsets',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the number of subsets to use')
parser.add_argument('--regualizer',
                    action='store',
                    default=10,
                    type=float,
                    help='Specify the regualization lambda to be used')
parser.add_argument('--regualizer-z',
                    action='store',
                    default=0,
                    type=float,
                    help='Specify the z-regualization lambda to be used')
parser.add_argument('--regualizer-oob',
                    action='store',
                    default=1,
                    type=float,
                    help='Specify the oob-regualization lambda to be used')
parser.add_argument('--first-layer',
                    action='store',
                    default=None,
                    help='Set the first layer to be a different type')

parser.add_argument('--max-iterations',
                    action='store',
                    default=100000,
                    type=int,
                    help='Specify the max number of iterations to use')
parser.add_argument('--batch-size',
                    action='store',
                    default=128,
                    type=int,
                    help='Specify the batch-size to be used for training')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')

parser.add_argument('--interpolation-range',
                    action='store',
                    default=[1,2],
                    type=ast.literal_eval,
                    help='Specify the interpolation range that is sampled uniformly from')
parser.add_argument('--extrapolation-range',
                    action='store',
                    default=[2,6],
                    type=ast.literal_eval,
                    help='Specify the extrapolation range that is sampled uniformly from')
parser.add_argument('--input-size',
                    action='store',
                    default=100,
                    type=int,
                    help='Specify the input size')
parser.add_argument('--subset-ratio',
                    action='store',
                    default=0.25,
                    type=float,
                    help='Specify the subset-size as a fraction of the input-size')
parser.add_argument('--overlap-ratio',
                    action='store',
                    default=0.5,
                    type=float,
                    help='Specify the overlap-size as a fraction of the input-size')
parser.add_argument('--simple',
                    action='store_true',
                    default=False,
                    help='Use a very simple dataset with t = sum(v[0:2]) + sum(v[4:6])')

parser.add_argument('--hidden-size',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the vector size of the hidden layer.')
parser.add_argument('--nac-mul',
                    action='store',
                    default='none',
                    choices=['none', 'normal', 'safe', 'max-safe', 'mnac', 'npu', 'real-npu'],
                    type=str,
                    help='Make the second NAC a multiplicative NAC, used in case of a just NAC network.')
parser.add_argument('--oob-mode',
                    action='store',
                    default='clip',
                    choices=['regualized', 'clip'],
                    type=str,
                    help='Choose of out-of-bound should be handled by clipping or regualization.')
parser.add_argument('--regualizer-scaling',
                    action='store',
                    default='linear',
                    choices=['exp', 'linear'],
                    type=str,
                    help='Use an expoentational scaling from 0 to 1, or a linear scaling.')
parser.add_argument('--regualizer-scaling-start',
                    action='store',
                    default=1000000,
                    type=int,
                    help='Start linear scaling at this global step.')
parser.add_argument('--regualizer-scaling-end',
                    action='store',
                    default=2000000,
                    type=int,
                    help='Stop linear scaling at this global step.')
parser.add_argument('--regualizer-shape',
                    action='store',
                    default='linear',
                    choices=['squared', 'linear', 'none'],
                    type=str,
                    help='Use either a squared or linear shape for the bias and oob regualizer. Use none so W reg in tensorboard is logged at 0')
parser.add_argument('--mnac-epsilon',
                    action='store',
                    default=0,
                    type=float,
                    help='Set the idendity epsilon for MNAC.')
parser.add_argument('--nalu-bias',
                    action='store_true',
                    default=False,
                    help='Enables bias in the NALU gate')
parser.add_argument('--nalu-two-nac',
                    action='store_true',
                    default=False,
                    help='Uses two independent NACs in the NALU Layer')
parser.add_argument('--nalu-two-gate',
                    action='store_true',
                    default=False,
                    help='Uses two independent gates in the NALU Layer')
parser.add_argument('--nalu-mul',
                    action='store',
                    default='normal',
                    choices=['normal', 'safe', 'trig', 'max-safe', 'mnac'],
                    help='Multplication unit, can be normal, safe, trig')
parser.add_argument('--nalu-gate',
                    action='store',
                    default='normal',
                    choices=['normal', 'regualized', 'obs-gumbel', 'gumbel'],
                    type=str,
                    help='Can be normal, regualized, obs-gumbel, or gumbel')

parser.add_argument('--optimizer',
                    action='store',
                    default='adam',
                    choices=['adam', 'sgd'],
                    type=str,
                    help='The optimization algorithm to use, Adam or SGD')
parser.add_argument('--learning-rate',
                    action='store',
                    default=1e-3,
                    type=float,
                    help='Specify the learning-rate')
parser.add_argument('--momentum',
                    action='store',
                    default=0.0,
                    type=float,
                    help='Specify the nestrov momentum, only used with SGD')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='simple_function_static',
                    type=str,
                    help='Where the data should be stored')
parser.add_argument('--remove-existing-data',
                    action='store_true',
                    default=False,
                    help='Should old results be removed')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Should network measures (e.g. gates) and gradients be shown')

parser.add_argument('--reg-scale-type',
                    action='store',
                    default='heim',
                    choices=['heim', 'madsen'],
                    type=str,
                    help='Type of npu regularisation scaling to use. Matches respective author\'s papers')
parser.add_argument('--regualizer-beta-start',
                    action='store',
                    default=1e-5,
                    type=float,
                    help='Starting value of the beta scale factor.')
parser.add_argument('--regualizer-beta-end',
                    action='store',
                    default=1e-4,
                    type=float,
                    help='Final value of the beta scale factor.')
parser.add_argument('--regualizer-beta-step',
                    action='store',
                    default=10000,
                    type=int,
                    help='Update the regualizer-beta-start value every x steps.')
parser.add_argument('--regualizer-beta-growth',
                    action='store',
                    default=10,
                    type=int,
                    help='Scale factor to grow the regualizer-beta-start value by.')
parser.add_argument('--regualizer-l1',
                    action='store_true',
                    default=False,
                    help='Add L1 regularization loss term. Be sure the regualizer-scaling is set')
parser.add_argument('--regualizer-npu-w',
                    action='store',
                    default=0,
                    type=int,
                    help='Use sparisty reg on npu weights. Int represents the amount to scale reg by. 0 means off')
parser.add_argument('--regualizer-gate',
                    type=int,
                    default=0,
                    help='Use sparisty reg on npu gate. Int represents the amount to scale reg by. 0 means off')
parser.add_argument('--npu-clip',
                    action='store',
                    default='none',
                    choices=['none', 'w', 'g', 'wg', 'wig'],
                    help='Type of parameters (if any) to clip in a NPU/RealNPU module')
parser.add_argument('--npu-Wr-init',
                    action='store',
                    default='xavier-uniform',
                    choices=['xavier-uniform', 'xavier-uniform-constrained'],
                    help='Init method to use for the W_real of the NPU. xavier-uniform= NPU paper init method,'
                         'xavier-uniform-constrained= NAU init method')

parser.add_argument('--pytorch-precision',
                    type=int,
                    default=32,
                    help='Precision for pytorch to work in')

parser.add_argument('--nmu-noise',
                    action='store_true',
                    default=False,
                    help='Applies/ unapplies multiplicative noise from a ~U[1,5] during training. Aids with failure ranges on a vinilla NMU.')
parser.add_argument('--nau-noise',
                    action='store_true',
                    default=False,
                    help='Applies/ unapplies additive noise from a ~U[1,5] during training.')

parser.add_argument('--W-set-count',
                    action='store',
                    default=5,
                    type=int,
                    help='Number of independant weight matricies to use for NAU in relaxed param sharing.')
parser.add_argument('--weight-pull-start',
                    action='store',
                    default=10000,
                    type=int,
                    help='Start linear scaling of pull factor for relaxed weight sharing at this step (epoch).')
parser.add_argument('--weight-pull-end',
                    action='store',
                    default=150000,
                    type=int,
                    help='Stop linear scaling of pull factor for relaxed weight sharing at this step (epoch).')

parser.add_argument('--pcc2mse-iteration',
                    action='store',
                    default=750000,
                    type=int,
                    help='Epoch to switch from pcc to mse loss for training')
parser.add_argument('--train-criterion',
                    action='store',
                    default='mse',
                    choices=['mse', 'pcc', 'pcc-mse', 'mape'],
                    type=str,
                    help='Train criterion to optimise on. pcc is the stable version (with epsilon). '
                         'pcc-mse will use pcc until iteration args.pcc2mse-iteration and then switch to mse.')
parser.add_argument('--trash-cell',
                    action='store_true',
                    default=False,
                    help='For realnpu only atm. Allows for an additional trainable scalar to act as a scratchpad')

parser.add_argument('--beta-nau',
                    action='store_true',
                    default=False,
                    help='Have nau weights initialised using a beta distribution B(7,7)')

parser.add_argument('--save-best-model',
                    action='store_true',
                    default=False,
                    help='Saves the model and its params for the epochs with the lowest interpolation error seen in training.')
parser.add_argument('--no-save',
                    action='store_true',
                    default=False,
                    help='Do not save model at the end of training')
parser.add_argument('--load-checkpoint',
                    action='store_true',
                    default=False,
                    help='Loads a saved checkpoint and resumes training')

parser.add_argument('--log-interval',
                    action='store',
                    default=1000,
                    type=int,
                    help='Log to tensorboard every X epochs.')

args = parser.parse_args()

if args.pytorch_precision == 32:
  torch.set_default_dtype(torch.float32)
elif args.pytorch_precision == 64:
  torch.set_default_dtype(torch.float64)
else:
  raise ValueError(f'Unsupported pytorch_precision option ({args.pytorch_precision})')

setattr(args, 'cuda', torch.cuda.is_available() and not args.no_cuda)

# Print configuration
print(f'running')
print(f'  - layer_type: {args.layer_type}')
print(f'  - first_layer: {args.first_layer}')
print(f'  - operation: {args.operation}')
print(f'  - num_subsets: {args.num_subsets}')
print(f'  - regualizer: {args.regualizer}')
print(f'  - regualizer_z: {args.regualizer_z}')
print(f'  - regualizer_oob: {args.regualizer_oob}')
print(f'  -')
print(f'  - max_iterations: {args.max_iterations}')
print(f'  - batch_size: {args.batch_size}')
print(f'  - seed: {args.seed}')
print(f'  -')
print(f'  - interpolation_range: {args.interpolation_range}')
print(f'  - extrapolation_range: {args.extrapolation_range}')
print(f'  - input_size: {args.input_size}')
print(f'  - subset_ratio: {args.subset_ratio}')
print(f'  - overlap_ratio: {args.overlap_ratio}')
print(f'  - simple: {args.simple}')
print(f'  -')
print(f'  - hidden_size: {args.hidden_size}')
print(f'  - nac_mul: {args.nac_mul}')
print(f'  - oob_mode: {args.oob_mode}')
print(f'  - regualizer_scaling: {args.regualizer_scaling}')
print(f'  - regualizer_scaling_start: {args.regualizer_scaling_start}')
print(f'  - regualizer_scaling_end: {args.regualizer_scaling_end}')
print(f'  - regualizer_shape: {args.regualizer_shape}')
print(f'  - mnac_epsilon: {args.mnac_epsilon}')
print(f'  - nalu_bias: {args.nalu_bias}')
print(f'  - nalu_two_nac: {args.nalu_two_nac}')
print(f'  - nalu_two_gate: {args.nalu_two_gate}')
print(f'  - nalu_mul: {args.nalu_mul}')
print(f'  - nalu_gate: {args.nalu_gate}')
print(f'  -')
print(f'  - optimizer: {args.optimizer}')
print(f'  - learning_rate: {args.learning_rate}')
print(f'  - momentum: {args.momentum}')
print(f'  -')
print(f'  - cuda: {args.cuda}')
print(f'  - name_prefix: {args.name_prefix}')
print(f'  - remove_existing_data: {args.remove_existing_data}')
print(f'  - verbose: {args.verbose}')
print(f'  -')
print(f'  - reg_scale_type: {args.reg_scale_type}')
print(f'  - regualizer_beta_start: {args.regualizer_beta_start}')
print(f'  - regualizer_beta_end: {args.regualizer_beta_end}')
print(f'  - regualizer_beta_step: {args.regualizer_beta_step}')
print(f'  - regualizer_beta_growth: {args.regualizer_beta_growth}')
print(f'  - regualizer_l1: {args.regualizer_l1}')
print(f'  - regualizer-npu-w: {args.regualizer_npu_w}')
print(f'  - regualizer-gate: {args.regualizer_gate}')
print(f'  - npu-clip: {args.npu_clip}')
print(f'  - npu-Wr-init: {args.npu_Wr_init}')
print(f'  -')
print(f'  - pytorch-precision: {torch.get_default_dtype()}')
print(f'  -')
print(f'  - nmu-noise: {args.nmu_noise}')
print(f'  - nau-noise: {args.nau_noise}')
print(f'  -')
# print(f'  - W-set-count: {args.W_set_count}')
# print(f'  - weight-pull-start: {args.weight_pull_start}')
# print(f'  - weight-pull-end: {args.weight_pull_end}')
# print(f'  -')
# print(f'  -')
print(f'  - save-best-model: {args.save_best_model}')
print(f'  - no-save: {args.no_save}')
print(f'  - load-checkpoint: {args.load_checkpoint}')
print(f'  - log-interval: {args.log_interval}')
print(f'  -')
print(f'  - train-criterion: {args.train_criterion}')
print(f'  - pcc2mse-iteration: {args.pcc2mse_iteration}')
print(f'  - trash-cell: {args.trash_cell}')
print(f'  -')
print(f'  - beta-nau: {args.beta_nau}')
print(f'  -')


def get_npu_Wr_init_writer_value():
    if args.npu_Wr_init == 'xavier-uniform':
        return 'xu'
    elif args.npu_Wr_init == 'xavier-uniform-constrained':
        return 'xuc'
    else:
        raise ValueError(f'Invalid arg ({args.npu_Wr_init}) given for npu_Wr_init')


def get_train_criterion_writer_value():
    val = args.train_criterion
    if args.train_criterion == 'pcc-mse':
        val += f'-{str(args.pcc2mse_iteration)}'
    if args.trash_cell:
        if args.layer_type != 'RealNPU' and args.layer_type != 'NRU':
            raise ValueError(f'trash-cell is only implemented for the RealNPU and NRU layer')
        val += f'-TC'
    return val

# Prepear logging
# summary_writer = stable_nalu.writer.DummySummaryWriter()
summary_writer = stable_nalu.writer.SummaryWriter(
    f'{args.name_prefix}/{args.layer_type.lower()}'
    f'{"-nac-" if args.nac_mul != "none" else ""}'
    f'{"-mwnac" if args.first_layer == "MultiWeightNAU" else ""}'
    f'{"n" if args.nac_mul == "normal" else ""}'
    f'{"s" if args.nac_mul == "safe" else ""}'
    f'{"s" if args.nac_mul == "max-safe" else ""}'
    f'{"t" if args.nac_mul == "trig" else ""}'
    f'{"m" if args.nac_mul == "mnac" else ""}'
    f'{"npu" if args.nac_mul == "npu" else ""}'
    f'{"npur" if args.nac_mul == "real-npu" else ""}'
    f'{"-nalu-" if (args.nalu_bias or args.nalu_two_nac or args.nalu_two_gate or args.nalu_mul != "normal" or args.nalu_gate != "normal") else ""}'
    f'{"b" if args.nalu_bias else ""}'
    f'{"2n" if args.nalu_two_nac else ""}'
    f'{"2g" if args.nalu_two_gate else ""}'
    f'{"s" if args.nalu_mul == "safe" else ""}'
    f'{"s" if args.nalu_mul == "max-safe" else ""}'
    f'{"t" if args.nalu_mul == "trig" else ""}'
    f'{"m" if args.nalu_mul == "mnac" else ""}'
    f'{"r" if args.nalu_gate == "regualized" else ""}'
    f'{"u" if args.nalu_gate == "gumbel" else ""}'
    f'{"uu" if args.nalu_gate == "obs-gumbel" else ""}'
    f'_op-{args.operation.lower()}'
    f'_oob-{"c" if args.oob_mode == "clip" else "r"}'
    f'_rs-{args.regualizer_scaling}-{args.regualizer_shape}'
    f'_eps-{args.mnac_epsilon}'
    f'_rl-{args.regualizer_scaling_start}-{args.regualizer_scaling_end}'
    f'_r-{args.regualizer}-{args.regualizer_z}-{args.regualizer_oob}'
    f'_i-{args.interpolation_range[0]}-{args.interpolation_range[1]}'
    f'_e-{args.extrapolation_range[0]}-{args.extrapolation_range[1]}'
    f'_z-{"simple" if args.simple else f"{args.input_size}-{args.subset_ratio}-{args.overlap_ratio}"}'
    f'_b{args.batch_size}'
    f'_s{args.seed}'
    f'_h{args.hidden_size}'
    f'_z{args.num_subsets}'
    f'_lr-{args.optimizer}-{"%.5f" % args.learning_rate}-{args.momentum}'
    f'_L1{"T" if args.regualizer_l1 else f"F"}'
    f'_rb-{args.regualizer_beta_start}-{args.regualizer_beta_end}-{args.regualizer_beta_step}-{args.regualizer_beta_growth}'
    f'_rWnpu-{args.regualizer_npu_w}'
    f'_rg-{args.regualizer_gate}'
    f'_r{"H" if args.reg_scale_type == "heim" else f"M"}'
    f'_clip-{args.npu_clip if args.npu_clip != "none" else args.npu_clip[0]}'
    f'_WrI-{get_npu_Wr_init_writer_value()}'
    f'_p-{args.pytorch_precision}'
    f'_nM{"T" if args.nmu_noise else f"F"}'
    f'_nA{"T" if args.nau_noise else f"F"}'
    f'_Bnau{"T" if args.beta_nau else f"F"}'
    f'_{get_train_criterion_writer_value()}',
    remove_existing_data=args.remove_existing_data
)

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    torch.set_num_threads(int(os.environ['LSB_DJOB_NUMPROC']))

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
eps = torch.finfo().eps

# Setup datasets
dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
    operation=args.operation,
    input_size=args.input_size,
    subset_ratio=args.subset_ratio,
    overlap_ratio=args.overlap_ratio,
    num_subsets=args.num_subsets,
    simple=args.simple,
    use_cuda=args.cuda,
    seed=args.seed,
)
print(f'  -')
print(f'  - dataset: {dataset.print_operation()}')
# Interpolation and extrapolation seeds are from random.org
dataset_train = iter(dataset.fork(sample_range=args.interpolation_range).dataloader(batch_size=args.batch_size))
dataset_valid_interpolation_data = next(iter(dataset.fork(sample_range=args.interpolation_range, seed=43953907).dataloader(batch_size=10000)))
dataset_test_extrapolation_data = next(iter(dataset.fork(sample_range=args.extrapolation_range, seed=8689336).dataloader(batch_size=10000)))

# setup model
model = stable_nalu.network.SimpleFunctionStaticNetwork(
    args.layer_type,
    input_size=dataset.get_input_size(),
    writer=summary_writer.every(args.log_interval).verbose(args.verbose),
    first_layer=args.first_layer,
    hidden_size=args.hidden_size,
    nac_oob=args.oob_mode,
    regualizer_shape=args.regualizer_shape,
    regualizer_z=args.regualizer_z,
    mnac_epsilon=args.mnac_epsilon,
    nac_mul=args.nac_mul,
    nalu_bias=args.nalu_bias,
    nalu_two_nac=args.nalu_two_nac,
    nalu_two_gate=args.nalu_two_gate,
    nalu_mul=args.nalu_mul,
    nalu_gate=args.nalu_gate,
    fixed_gate=False,      # TODO - Remove option from NPU and NPUReal
    regualizer_gate=args.regualizer_gate,
    regualizer_npu_w=args.regualizer_npu_w,
    nmu_noise=args.nmu_noise,
    nau_noise=args.nau_noise,
    W_set_count=args.W_set_count,
    beta_nau=args.beta_nau,
    npu_clip=args.npu_clip,
    npu_Wr_init=args.npu_Wr_init,
    trash_cell=args.trash_cell
)
model.reset_parameters()


def print_tab_delimited_tensor_elems(name, t1, t2=None):
    print(name)
    print(*list(map(lambda x: x.item(), list(t1[0]) + list(t1[1]) + list(t2[0]))), sep='\t')
# print_tab_delimited_tensor_elems('L1 and L2 W', model.layer_1.layer.W, model.layer_2.layer.W)

if args.cuda:
    model.cuda()
criterion = torch.nn.MSELoss()

# use when you want separate param_groups for each sub_module for doing optimsiation
# would require passing in module_params to the optimiser rather than model.parameters()
#from stable_nalu.abstract import ExtendedTorchModule    # uncomment if using weight sharing where opt uses separate param groups
# module_params = []
# for c in model.children():
#  if isinstance(c, ExtendedTorchModule):
#    module_params.append({'params': c.parameters()})
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
else:
    raise ValueError(f'{args.optimizer} is not a valid optimizer algorithm')

def test_model(data):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        model.eval()
        x, t = data
        err = criterion(model(x), t)
        model.train()
        return err

def calc_W_pull_factor(epoch):
    # linear growth pull, which grows in given epoch range. Afterwhich pull remains at 1
    return max(0, min(1, ((epoch - args.weight_pull_start) /
            (args.weight_pull_end - args.weight_pull_start)
        )))


print(model)
print('')
print(summary_writer.name)
print('')
utils.print_model_params(model)
print()

use_npu_scaling = args.regualizer_l1 or (args.regualizer_npu_w and args.reg_scale_type == 'heim') \
                  or (args.regualizer_gate and args.reg_scale_type == 'heim')
if use_npu_scaling:
    # Decimal type required to avoid accumulation of fp precision errors when multiplying by growth factor
    args.regualizer_beta_start = Decimal(str(args.regualizer_beta_start))
    # Decimal and fp arithmetic don't mix so beta end must also be a decimal
    args.regualizer_beta_end = Decimal(str(args.regualizer_beta_end))
r_l1_scale = args.regualizer_beta_start

# # flag to check if the W set has been averaged (when doing relaxed param sharing)
# is_Ws_averaged = False
# # weight sharing layer
# wsNAU_layer = True if args.first_layer == 'MultiWeightNAU' or args.layer_type == 'MultiWeightNAU' else False
# # must average either a bit before the bias reg for W starts or a fixed amount after the pull values reaches 1
# pull_reg_max_epoch = max(1000, min(args.regualizer_scaling_start - 7000, args.weight_pull_end + 50000))

# Set starting interpolation error check to highest possible values (i.e. inf)
if args.save_best_model:
    lowest_interp_error = math.inf
    print(f"lowest_interp_error starting at: {lowest_interp_error}")

'''Resuming previous training'''
resume_epoch = 0
if args.load_checkpoint:
    resume_epoch = stable_nalu.writer.load_model(summary_writer.name, model, optimizer) # add one for next epoch
    if resume_epoch > args.max_iterations:
        raise ValueError(
            f'{args.max_iterations} must be larger than or equal to the loaded models resume epoch {resume_epoch}')
    if resume_epoch != 0:
        for i, j in zip(range(resume_epoch), dataset_train):
            (x_train, t_train) = j
    print("Checkpoint loaded")
    print('train %d: %.5f, inter: %.5f, extra: %.5f' % (resume_epoch, test_model((x_train, t_train)), test_model(dataset_valid_interpolation_data), test_model(dataset_test_extrapolation_data)))
'''------------------'''
# Train model
for epoch_i, (x_train, t_train) in zip(range(resume_epoch, args.max_iterations + 1), dataset_train):
    summary_writer.set_iteration(epoch_i)

    # Prepear model
    model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
    optimizer.zero_grad()

    # Log validation
    if epoch_i % args.log_interval == 0:
        interpolation_error = test_model(dataset_valid_interpolation_data)
        extrapolation_error = test_model(dataset_test_extrapolation_data)

        summary_writer.add_scalar('metric/valid/interpolation', interpolation_error)
        summary_writer.add_scalar('metric/test/extrapolation', extrapolation_error)

    # forward
    y_train = model(x_train)
    regualizers = model.regualizer()    # logs 3 reg metrics to tensorbord if verbose

    if (args.regualizer_scaling == 'linear'):
        r_w_scale = max(0, min(1, (
                (epoch_i - args.regualizer_scaling_start) /
                (args.regualizer_scaling_end - args.regualizer_scaling_start)
        )))
    elif (args.regualizer_scaling == 'exp'):
        r_w_scale = 1 - math.exp(-1e-5 * epoch_i)

    # FOR NAU
    # nau_reg_buffer = 5000
    # if (args.regualizer_scaling == 'linear'):
    #     r_w_scale_NAU = max(0, min(1, (
    #         (epoch_i - args.regualizer_scaling_start - nau_reg_buffer) /
    #         (args.regualizer_scaling_end - args.regualizer_scaling_start)
    #     )))

    l1_loss = 0
    if args.regualizer_l1:
        l1_loss = Regualizer.l1(model.parameters())
        if args.verbose:
            summary_writer.add_scalar('L1/train/L1-loss', l1_loss)

    if use_npu_scaling:
        # the beta_start value will be updated accordingly to be the correct beta value for the epoch.
        # It is done this way to avoid having initalise another variable outside the epoch loop
        if args.regualizer_beta_start <= args.regualizer_beta_end:
            if epoch_i % args.regualizer_beta_step == 0 and epoch_i != 0:
                if args.regualizer_beta_start < args.regualizer_beta_end:
                    args.regualizer_beta_start *= args.regualizer_beta_growth
        else:
            if epoch_i % args.regualizer_beta_step == 0 and epoch_i != 0:
                if args.regualizer_beta_start > args.regualizer_beta_end:
                    args.regualizer_beta_start /= args.regualizer_beta_growth

        r_l1_scale = float(args.regualizer_beta_start)  # Decimal doesn't work for tensorboard or mixed fp arithmetic
        if args.verbose:
            summary_writer.add_scalar('L1/train/beta', r_l1_scale)

    # mse loss
    mse_loss = criterion(y_train, t_train)

    # pcc loss
    # Acknowledgement: https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739
    vx = y_train - torch.mean(y_train)
    vy = t_train - torch.mean(t_train)
    # r = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + eps)  # + eps to avoid denominator = 0

    """
    Acknowledgement: https://www.johndcook.com/blog/2008/11/05/how-to-calculate-pearson-correlation-accurately/
    Use population stats (i.e., no Bassels' correction) and +eps to avoid div by 0.
    """
    n = t_train.shape[0]
    sx = torch.sqrt((torch.sum(vx ** 2) / n).clamp(eps))
    sy = torch.sqrt((torch.sum(vy ** 2) / n).clamp(eps))
    r = torch.sum((vx / (sx + eps)) * (vy / (sy + eps))) / n
    # assert -1 <= r <= 1, f"r (={r}) value is not in valid range of [-1,1]"
    pcc_loss = 1 - r

    # mean abs percentage error
    mape_loss = torch.mean(torch.abs((t_train - y_train) / t_train))

    loss_train_criterion = stable_nalu.functional.get_train_criterion(args.train_criterion, mse_loss, pcc_loss, mape_loss,
                                                                      epoch_i, args.pcc2mse_iteration)

    loss_train_regualizer = args.regualizer * r_w_scale * regualizers['W'] + \
                            regualizers['g'] + \
                            args.regualizer_z * regualizers['z'] + \
                            args.regualizer_oob * regualizers['W-OOB'] + \
                            args.regualizer_l1 * r_l1_scale * l1_loss + \
                            args.regualizer_npu_w * (r_l1_scale if args.reg_scale_type == 'heim' else r_w_scale) * regualizers['W-NPU'] + \
                            args.regualizer_gate * (r_l1_scale if args.reg_scale_type == 'heim' else r_w_scale) * regualizers['g-NPU'] #+ \
                            # (torch.norm(model.layer_2.layer.trash_cell, 1) if args.trash_cell else 0) #+ \ #TODO: Leave out for speed
                            # args.regualizer * r_w_scale_NAU * regualizers['W-NAU'] #+ \
                            # calc_W_pull_factor(epoch_i) * (0 if is_Ws_averaged else 1) * regualizers['W-set'] #+ \

    loss_train = loss_train_criterion + loss_train_regualizer

    # Averages set of indep. weights if thr req. is met.
    # Indep weights will be frozen and no longer used. Adds the new param for the optimizer to track.
    # if wsNAU_layer and not is_Ws_averaged:
    #     # + 1000 to avoid averaging as soon as you start tightening param values.
    #     if (regualizers['W-set'] < 0.05 and epoch_i > args.weight_pull_start + 1000) or epoch_i == pull_reg_max_epoch:
    #         print('Ws have been averaged')
    #         avg_W = model.layer_1.layer.average_Ws()
    #         optimizer.add_param_group({'params': avg_W})
    #         # Uncomment if you want to remove unused W_i from the opt. (Would require changing init parameters given to optimiser)
    #         # del optimizer.param_groups[0]
    #         is_Ws_averaged = True

    # Log loss
    if args.verbose or epoch_i % args.log_interval == 0:
        summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
        summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
        summary_writer.add_scalar('loss/train/total', loss_train)
    if epoch_i % args.log_interval == 0:
        print('train %d: %.5f, inter: %.5f, extra: %.5f' % (epoch_i, loss_train_criterion, interpolation_error, extrapolation_error))

        # save the best model (.pth) and it's params (.txt)
        if args.save_best_model and (interpolation_error < lowest_interp_error):
            lowest_interp_error = interpolation_error
            stable_nalu.writer.save_model("lowest_interp_model/" + summary_writer.name, model.state_dict())
            stable_nalu.writer.save_model_params("lowest_interp_params/" + summary_writer.name, model)

    # Optimize model
    if loss_train.requires_grad:
        loss_train.backward()
        optimizer.step()
    model.optimize(loss_train_criterion)

    # Log gradients if in verbose mode
    if args.verbose and epoch_i % args.log_interval == 0:
        model.log_gradients()

# Compute validation loss
loss_valid_inter = test_model(dataset_valid_interpolation_data)
loss_valid_extra = test_model(dataset_test_extrapolation_data)

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - loss_valid_inter: {loss_valid_inter}')
print(f'  - loss_valid_extra: {loss_valid_extra}')

utils.print_model_params(model)

is_solved_last_epoch = False
if loss_valid_extra <= 1e-7:
    is_solved_last_epoch = True
print(f"Success (last epoch): \t{is_solved_last_epoch}")


if not args.no_save:
    model.writer._root.close()  # fix - close summary writer before saving model to avoid thread locking issues
    # Use saved weights to visualize the intermediate values.
    stable_nalu.writer.save_model_checkpoint(summary_writer.name, epoch_i + 1, model, optimizer,
                                             {'torch': torch.get_rng_state(),
                                              'numpy': np.random.get_state()}
                                             )

