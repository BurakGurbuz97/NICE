import os
from launch_utils import get_argument_parser, get_experience_streams
from launch_utils import set_seeds, create_log_dirs
from Source import architecture, learner


if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    args = get_argument_parser()
    log_dirpath = create_log_dirs(args)
    set_seeds(args.seed)
    scenario, input_size, output_size, task2classes = get_experience_streams(args)
    backbone = architecture.get_backbone(args, input_size, output_size)
    nice = learner.Learner(args, backbone, scenario, input_size, task2classes, log_dirpath)
    nice.learn_all_episodes()
