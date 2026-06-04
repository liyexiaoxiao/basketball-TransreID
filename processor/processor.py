import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.ema import ModelEMA
from torch.cuda import amp
import torch.distributed as dist

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        rerank_k1=cfg.TEST.RERANK_K1,
        rerank_k2=cfg.TEST.RERANK_K2,
        rerank_lambda=cfg.TEST.RERANK_LAMBDA,
        qe=cfg.TEST.QE,
        qe_k=cfg.TEST.QE_K,
        qe_alpha=cfg.TEST.QE_ALPHA,
        qe_iter=cfg.TEST.QE_ITER,
    )
    scaler = amp.GradScaler()
    # Initialize EMA model
    ema_model = ModelEMA(model, decay=0.9998)
    logger.info('Using EMA with decay=0.9998')

    # TensorBoard
    tb_writer = None
    if getattr(cfg.SOLVER, 'TB_LOG', False) and HAS_TB:
        tb_dir = os.path.join(cfg.OUTPUT_DIR, 'tensorboard')
        tb_writer = SummaryWriter(log_dir=tb_dir)
        logger.info('TensorBoard logging enabled: {}'.format(tb_dir))
    elif getattr(cfg.SOLVER, 'TB_LOG', False) and not HAS_TB:
        logger.warning('TensorBoard not available (install tensorboard). Skipping TB logging.')

    # Gradient clipping
    grad_clip = getattr(cfg.SOLVER, 'GRAD_CLIP', 0.0)
    if grad_clip > 0:
        logger.info('Using gradient clipping with max_norm={}'.format(grad_clip))

    # Best model tracking
    eval_best = getattr(cfg.SOLVER, 'EVAL_BEST', False)
    best_mAP = 0.0
    best_epoch = 0

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            # Gradient clipping (AMP-aware)
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()
            ema_model.update(model)

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                current_lr = scheduler.get_epoch_values(epoch)
                if current_lr is not None and len(current_lr) > 0:
                    lr_str = "{:.2e}".format(current_lr[0])
                else:
                    lr_str = "N/A"
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, lr_str))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # TensorBoard: log training metrics
        if tb_writer is not None:
            current_lr = scheduler.get_epoch_values(epoch)
            if current_lr is not None and len(current_lr) > 0:
                tb_writer.add_scalar('Train/LR', current_lr[0], epoch)
            tb_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
            tb_writer.add_scalar('Train/Acc', acc_meter.avg, epoch)

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                    torch.save(ema_model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ema_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                torch.save(ema_model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ema_{}.pth'.format(epoch)))
                logger.info('Saved checkpoint: {}'.format(cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                # Evaluate with EMA model for better results
                ema_model.ema.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = ema_model.ema(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("EMA Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

                # TensorBoard: log validation metrics
                if tb_writer is not None:
                    tb_writer.add_scalar('Val/mAP', mAP, epoch)
                    tb_writer.add_scalar('Val/Rank-1', cmc[0], epoch)
                    tb_writer.add_scalar('Val/Rank-5', cmc[4], epoch)
                    tb_writer.add_scalar('Val/Rank-10', cmc[9], epoch)

                # Save best model
                if eval_best and mAP > best_mAP:
                    best_mAP = mAP
                    best_epoch = epoch
                    best_model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth')
                    best_ema_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ema_best.pth')
                    torch.save(model.state_dict(), best_model_path)
                    torch.save(ema_model.state_dict(), best_ema_path)
                    logger.info('>>> New best mAP: {:.1%} at epoch {} — saved as {}'.format(
                        best_mAP, best_epoch, cfg.MODEL.NAME + '_best.pth'))

    # End of training summary
    if eval_best:
        logger.info('=== Training complete. Best mAP: {:.1%} at epoch {} ==='.format(best_mAP, best_epoch))
    else:
        logger.info('=== Training complete ===')

    if tb_writer is not None:
        tb_writer.close()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        rerank_k1=cfg.TEST.RERANK_K1,
        rerank_k2=cfg.TEST.RERANK_K2,
        rerank_lambda=cfg.TEST.RERANK_LAMBDA,
        qe=cfg.TEST.QE,
        qe_k=cfg.TEST.QE_K,
        qe_alpha=cfg.TEST.QE_ALPHA,
        qe_iter=cfg.TEST.QE_ITER,
    )

    evaluator.reset()

    if device:
        if isinstance(model, (list, tuple)):
            for m in model:
                m.to(device)
        else:
            if torch.cuda.device_count() > 1:
                print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(device)

    if isinstance(model, (list, tuple)):
        for m in model:
            m.eval()
    else:
        model.eval()
    img_path_list = []

    logger.info("Using Test-Time Augmentation (horizontal flip)")
    if getattr(cfg.TEST, 'QE', False):
        logger.info("Using Query Expansion (k={}, alpha={}, iter={})".format(cfg.TEST.QE_K, cfg.TEST.QE_ALPHA, cfg.TEST.QE_ITER))
    if getattr(cfg.TEST, 'RE_RANKING', False):
        logger.info("Using Re-Ranking (k1={}, k2={}, lambda={})".format(cfg.TEST.RERANK_K1, cfg.TEST.RERANK_K2, cfg.TEST.RERANK_LAMBDA))
    if getattr(cfg.TEST, 'MULTI_SCALE', False):
        logger.info("Using Multi-Scale Models: {}".format(list(getattr(cfg.TEST, 'SCALES', []))))

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            if isinstance(model, (list, tuple)):
                weights = list(getattr(cfg.TEST, 'SCALE_WEIGHTS', []))
                if weights and len(weights) == len(model):
                    ws = torch.tensor(weights, dtype=torch.float32, device=device)
                    ws = ws / (ws.sum() + 1e-12)
                else:
                    ws = torch.full((len(model),), 1.0 / len(model), dtype=torch.float32, device=device)

                fused = None
                for i, m in enumerate(model):
                    if hasattr(m, 'base') and hasattr(m.base, 'patch_embed') and hasattr(m.base.patch_embed, 'img_size'):
                        size = m.base.patch_embed.img_size
                        img_s = torch.nn.functional.interpolate(img, size=size, mode='bilinear', align_corners=False)
                    else:
                        img_s = img

                    feat_s = m(img_s, cam_label=camids, view_label=target_view)
                    feat_s = torch.nn.functional.normalize(feat_s, dim=1, p=2)
                    feat_flip_s = m(torch.flip(img_s, dims=[3]), cam_label=camids, view_label=target_view)
                    feat_flip_s = torch.nn.functional.normalize(feat_flip_s, dim=1, p=2)
                    feat_s = (feat_s + feat_flip_s) / 2.0
                    feat_s = torch.nn.functional.normalize(feat_s, dim=1, p=2)
                    fused = feat_s * ws[i] if fused is None else fused + feat_s * ws[i]
                feat = torch.nn.functional.normalize(fused, dim=1, p=2)
            else:
                feat = model(img, cam_label=camids, view_label=target_view)
                feat = torch.nn.functional.normalize(feat, dim=1, p=2)
                feat_flip = model(torch.flip(img, dims=[3]), cam_label=camids, view_label=target_view)
                feat_flip = torch.nn.functional.normalize(feat_flip, dim=1, p=2)
                feat = (feat + feat_flip) / 2.0
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
