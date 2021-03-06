#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific langcduage governing permissions and
#  limitations under the License.

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def define_flags():
    parser = argparse.ArgumentParser(description="Parser for FLAGS to the model.")

    parser.add_argument('--RUN_ID', type=str, help="Unique identifier for the experiment.")

    # ____________________________________________________ #
    # ====================== MODEL ======================= #

    parser.add_argument('--experiment', type=str, help="Experiment to run.")
    parser.add_argument('--experiment_type', type=str, default='scribble', help="Experiment to run - scribbles.")

    # ____________________________________________________ #
    # ========== ARCHITECTURE HYPER-PARAMETERS ========== #

    parser.add_argument('--lr', type=float, nargs='?', default=1e-4, help="Learning rate generator.")
    parser.add_argument('--batch_size', type=int, nargs='?', default=12, help="Batch size.")
    parser.add_argument('--n_epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--warm_up_period', type=int, nargs='?', default=0,
                        help="Time before evaluating the model the first time.")
    parser.add_argument('--w_regularisation', type=float, nargs='?', default=0.2, help="Weigh for regularisation.")

    # ____________________________________________________ #
    # =============== TRAINING STRATEGY ================== #

    parser.add_argument('--augment', type=str2bool, nargs='?', const=True, default=True, help="Perform data augmentation.")
    parser.add_argument('--standardize', type=str2bool, nargs='?', const=True, default=False,
                        help="Perform data standardization (z-score).")

    # parser.add_argument('--load_pre_initialized', type=str, help="Load pre-initialized model (to have a fixed seed). "
    #                                                              "This will make the initialization really fixed. Also "
    #                                                              "consider fixing tensorflow, numpy and random seeds.")

    # ____________________________________________________________ #
    # =============== LOGS AND REPORTS SETTINGS ================== #

    # global
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help="Verbosity, for print reports.")
    parser.add_argument('--notify', type=str2bool, nargs='?', const=True, default=False,
                        help="If True: add telegram notifier to notify the end of training.")

    # tensorboard
    parser.add_argument('--tensorboard_on', type=str2bool, nargs='?', const=True, default=True,
                        help="If True: save tensorboard logs.")
    parser.add_argument('--tensorboard_verbose', type=str2bool, nargs='?', const=True, default=True,
                        help="If True: save also layers weights every N epochs.")
    parser.add_argument('--skip_step', type=int, nargs='?', default=4000, help="Frequency of printing batch report.")
    parser.add_argument('--train_summaries_skip', type=int, nargs='?', default=100,
                        help="Number of skips before writing summaries for training steps "
                             "(used to reduce its verbosity; put 1 to avoid this).")

    # test report
    parser.add_argument('--results_dir', type=str, nargs='?', default='.', help="results directory")

    # ____________________________________________________ #
    # ==================== HARDWARE ====================== #

    # internal variables:
    parser.add_argument('--num_threads', type=int, nargs='?', default=20, help="Number of threads for loading data.")
    parser.add_argument('--CUDA_VISIBLE_DEVICE', type=int, nargs='?', default=0, help="Visible GPU.")

    # ____________________________________________________ #
    # ===================== DATA SET ====================== #

    # path for the data set:
    parser.add_argument('--dataset_name', type=str, help="Dataset name.")
    parser.add_argument('--data_path', type=str, help="Path of data files.")

    # ids for the data
    parser.add_argument('--n_sup_vols', type=str, help="Number of labelled data to use as training volumes (e.g. '2vols').")
    parser.add_argument('--split_number', type=str, help="Split number for cross-validation (e.g. 'split0').")

    # downsample factor: if X > 1, then downsample of a factor X
    parser.add_argument('--downsample_factor', type=int, nargs='?', default=1,
                        help="downsample factor: if X > 1, then downsample of a factor X.")

    # ------------------------
    args = parser.parse_args()

    return args
