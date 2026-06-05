"""
Ensemble test with feature-level fusion: load multiple model weights,
extract features with TTA, average features across models, and evaluate.

Feature-level fusion is more effective than score-level (distance matrix)
fusion because it preserves the geometric relationship between features.

Usage:
    python ensemble_test.py --config_file configs/BallShow/vit_transreid_stride.yml \
        --weights logs/BallShow_vit_transreid_stride/transformer_ema_80.pth,logs/BallShow_vit_transreid_stride/transformer_ema_100.pth,logs/BallShow_vit_transreid_stride/transformer_ema_120.pth
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval, euclidean_distance, eval_func


def extract_features_with_tta(model, val_loader, device):
    """Extract features using TTA (horizontal flip)."""
    model.eval()
    feats_list = []
    pids_list = []
    camids_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids_dev = camids.to(device)
            target_view = target_view.to(device)
            # Original
            feat = model(img, cam_label=camids_dev, view_label=target_view)
            # Horizontal flip TTA
            feat_flip = model(torch.flip(img, dims=[3]), cam_label=camids_dev, view_label=target_view)
            feat = (feat + feat_flip) / 2.0

            feats_list.append(feat.cpu())
            pids_list.extend(np.asarray(pid))
            camids_list.extend(np.asarray(camid))

    feats = torch.cat(feats_list, dim=0)
    return feats, pids_list, camids_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Ensemble Test with Feature-Level Fusion")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--weights", required=True, type=str,
                        help="Comma-separated list of model weight paths, e.g. path1.pth,path2.pth")
    parser.add_argument("--fusion", default="feature", choices=["feature", "score"],
                        help="Fusion method: 'feature' averages feature vectors, 'score' averages distance matrices")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    weight_list = [w.strip() for w in args.weights.split(',')]

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info("Ensemble test with {} models, fusion method: {}".format(len(weight_list), args.fusion))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    if args.fusion == "feature":
        # Feature-level fusion: average features from all models, then compute distance
        all_feats = []
        for i, weight_path in enumerate(weight_list):
            logger.info("Loading model {}: {}".format(i + 1, weight_path))
            model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
            model.load_param(weight_path)
            model.to(device)

            feats, pids, camids = extract_features_with_tta(model, val_loader, device)
            logger.info("Extracted features shape: {}".format(feats.shape))

            del model
            torch.cuda.empty_cache()

            # L2-normalize independently before averaging
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            all_feats.append(feats)

            if i == 0:
                q_pids = np.asarray(pids[:num_query])
                q_camids = np.asarray(camids[:num_query])
                g_pids = np.asarray(pids[num_query:])
                g_camids = np.asarray(camids[num_query:])

        # Average normalized features across models (feature-level fusion)
        logger.info("Averaging features from {} models".format(len(all_feats)))
        avg_feats = sum(all_feats) / len(all_feats)
        # Re-normalize after averaging
        avg_feats = torch.nn.functional.normalize(avg_feats, dim=1, p=2)

        qf = avg_feats[:num_query]
        gf = avg_feats[num_query:]
        distmat = euclidean_distance(qf, gf)

    else:
        # Score-level fusion: average distance matrices (original behavior)
        distmats = []
        for i, weight_path in enumerate(weight_list):
            logger.info("Loading model {}: {}".format(i + 1, weight_path))
            model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
            model.load_param(weight_path)
            model.to(device)

            feats, pids, camids = extract_features_with_tta(model, val_loader, device)
            logger.info("Extracted features shape: {}".format(feats.shape))

            del model
            torch.cuda.empty_cache()

            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

            qf = feats[:num_query]
            gf = feats[num_query:]

            if i == 0:
                q_pids = np.asarray(pids[:num_query])
                q_camids = np.asarray(camids[:num_query])
                g_pids = np.asarray(pids[num_query:])
                g_camids = np.asarray(camids[num_query:])

            distmat = euclidean_distance(qf, gf)
            distmats.append(distmat)

        logger.info("Averaging distance matrices from {} models".format(len(distmats)))
        distmat = sum(distmats) / len(distmats)

    # Evaluate
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info("=== Ensemble Results ({}-level fusion) ===".format(args.fusion))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
