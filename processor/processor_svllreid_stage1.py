import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

from loss.SIMCLR_loss import SIMCLRLoss

def do_train_stage1(cfg,
                    model,
                    train_loader_stage1,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = 'cuda:{}'.format(local_rank)
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        print('当前local_rank对于device:',device)
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    aug_features = []
    labels = []

    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())

        labels_list = torch.stack(labels, dim=0).cuda()  # N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features,aug_features

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image)
        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]

            target = labels_list[b_list.cuda()]

            image_features = image_features_list[b_list.cuda()]

            with amp.autocast(enabled=True):
                # SATPL loss,which modified from simclr
                #Firstly getting the unique_prompts from the batch for language self-supervision because the batch contains multiple identical text prompts.
                unique_target = torch.unique(target)
                prompt_aug_parameter=[]
                prompt_aug_parameter.append(cfg.SOLVER.STAGE1.PROMPT_AUG_SCALE)
                prompt_aug_parameter.append(cfg.SOLVER.STAGE1.PROMPT_AUG_METHOD)
                text_features, aug_outputs = model(label=target,unique_label=unique_target, get_text=True,USE_SSL_stage1=True, prompt_aug_parameter= prompt_aug_parameter)
                simclr = SIMCLRLoss()
                simclrloss = simclr(aug_outputs)
                # CLIP-ReID loss: i2t +t2i
                loss_i2t = xent(image_features, text_features, target, target)
                loss_t2i = xent(text_features, image_features, target, target)
                #Overall loss of stage1
                loss = loss_i2t + loss_t2i + cfg.SOLVER.STAGE1.SSL_LOSS_WEIGHT * simclrloss

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), img.shape[0])
            torch.cuda.synchronize()

            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (i + 1) % log_period == 0:
                        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (i + 1), len(train_loader_stage1),
                                            loss_meter.avg, scheduler._get_lr(epoch)[0]))
            else:
                if (i + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (i + 1), len(train_loader_stage1),
                                        loss_meter.avg, scheduler._get_lr(epoch)[0]))


        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:

                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)



    logger.info("Stage1 running time: {}".format(total_time))
