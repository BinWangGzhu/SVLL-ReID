import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from utils.logger import setup_logger
from datasets.make_dataloader_savllreid import make_dataloader
from model.make_model_savllreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_savllreid_stage1 import do_train_stage1
from processor.processor_savllreid_stage2 import do_train_stage2
import random
import torch
import numpy as np
import argparse
from config import cfg
torch.set_num_threads(6)

from torch.nn.parallel import DistributedDataParallel as DDP

# 分别是：cuda是否可用，可见cuda数量，当前cuda设备号，GPU名字
print('cuda是否可用:',torch.cuda.is_available())
print('可见cuda数量：',torch.cuda.device_count())
print('当前cuda设备号:',torch.cuda.current_device())
print('GPU名字：',torch.cuda.get_device_name())

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_savllreid.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument("--local_rank", default=0, type=int)


    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        print("DDP！")
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)


    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg.SOLVER.STAGE1.LR_MIN, \
                        warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)

    #stage1
    do_train_stage1(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )
    logger.info('stage1 is over!stage1 is over!stage1 is over!stage1 is over!stage1 is over!stage1 is over!stage1 is over!stage1 is over!')
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA, cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                  cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

    # stage2
    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query,
        args.local_rank,
    )