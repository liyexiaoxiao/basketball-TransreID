import os
import time
import re
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

from config import cfg
from datasets.ballshow import BallShow
from datasets.bases import read_image
from model import make_model
from utils.metrics import R1_mAP_eval, euclidean_distance, cosine_similarity

class ModelWorker:
    """
    Singleton-like wrapper for model and dataset states
    """
    def __init__(self):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = None
        self.model = None
        self.num_classes = 0
        self.camera_num = 0
        self.view_num = 0
        self.val_transforms = None
        
        # Gallery cache
        self.gallery_features = None
        self.gallery_paths = []
        self.gallery_pids = []
        self.gallery_cams = []

    def set_device(self, dev_str):
        self.device = dev_str
        if self.model is not None:
            self.model.to(self.device)

    def load_config_and_model(self, config_path, weight_path, progress_callback=None):
        # 1. Defrost and load config with UTF-8 encoding (fixes Windows default encoding crash)
        if progress_callback: progress_callback("Loading config file...")
        self.cfg.defrost()
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_cfg = self.cfg.load_cfg(f)
        self.cfg.merge_from_other_cfg(loaded_cfg)
        if weight_path:
            self.cfg.TEST.WEIGHT = weight_path
        self.cfg.freeze()

        # 2. Setup transforms
        self.val_transforms = T.Compose([
            T.Resize(self.cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        ])

        # 3. Load dataset info to get classes/cams/views metadata
        if progress_callback: progress_callback("Parsing dataset (BallShow)...")
        self.dataset = BallShow(root=self.cfg.DATASETS.ROOT_DIR, verbose=False)
        self.num_classes = self.dataset.num_train_pids
        self.camera_num = self.dataset.num_train_cams
        self.view_num = self.dataset.num_train_vids

        # 4. Make model
        if progress_callback: progress_callback(f"Building model: {self.cfg.MODEL.NAME}...")
        self.model = make_model(self.cfg, num_class=self.num_classes, camera_num=self.camera_num, view_num=self.view_num)

        # 5. Load model weights if provided
        # Skip if weights match pre-trained ImageNet backbone weights file (which has no classification heads and throws KeyError)
        if self.cfg.TEST.WEIGHT:
            is_backbone_weights = False
            if self.cfg.MODEL.PRETRAIN_PATH:
                pretrain_name = os.path.basename(self.cfg.MODEL.PRETRAIN_PATH).lower()
                weight_name = os.path.basename(self.cfg.TEST.WEIGHT).lower()
                if pretrain_name == weight_name or "vit_base_p16" in weight_name:
                    is_backbone_weights = True
            
            if is_backbone_weights:
                if progress_callback: progress_callback("Using model initialization weights (skipped loading backbone weights as final model weights).")
            else:
                if progress_callback: progress_callback(f"Loading weights from {self.cfg.TEST.WEIGHT}...")
                self.model.load_param(self.cfg.TEST.WEIGHT)
        
        # Move model to device and eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Clear gallery cache on reload
        self.gallery_features = None
        self.gallery_paths = []
        self.gallery_pids = []
        self.gallery_cams = []
        
        if progress_callback: progress_callback("Model loaded successfully!")

# Global model state
state = ModelWorker()

class ModelLoaderThread(QThread):
    finished_signal = pyqtSignal(bool, str, dict)
    log_signal = pyqtSignal(str)

    def __init__(self, config_path, weight_path, device_str):
        super().__init__()
        self.config_path = config_path
        self.weight_path = weight_path
        self.device_str = device_str

    def run(self):
        try:
            state.set_device(self.device_str)
            state.load_config_and_model(
                self.config_path, 
                self.weight_path, 
                progress_callback=self.log_signal.emit
            )
            details = {
                "backbone": state.cfg.MODEL.NAME,
                "transformer_type": state.cfg.MODEL.TRANSFORMER_TYPE,
                "num_classes": state.num_classes,
                "camera_num": state.camera_num,
                "view_num": state.view_num,
                "input_size": str(state.cfg.INPUT.SIZE_TEST),
                "device": state.device
            }
            self.finished_signal.emit(True, "Model loaded successfully!", details)
        except Exception as e:
            self.finished_signal.emit(False, f"Error loading model: {str(e)}", {})

class GalleryFeatureExtractorThread(QThread):
    finished_signal = pyqtSignal(bool, str, int)
    progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)

    def run(self):
        try:
            if state.model is None:
                self.finished_signal.emit(False, "Model not loaded. Load model first.", 0)
                return

            self.log_signal.emit("Starting gallery feature extraction...")
            gallery_data = state.dataset.gallery
            total_images = len(gallery_data)
            
            if total_images == 0:
                self.finished_signal.emit(False, "Gallery dataset is empty.", 0)
                return

            features_list = []
            paths_list = []
            pids_list = []
            cams_list = []

            # Batch processing for efficiency
            batch_size = state.cfg.TEST.IMS_PER_BATCH
            
            for i in range(0, total_images, batch_size):
                batch_data = gallery_data[i : i + batch_size]
                imgs = []
                batch_pids = []
                batch_cams = []
                batch_paths = []
                
                for img_path, pid, camid, _ in batch_data:
                    img = read_image(img_path)
                    img = state.val_transforms(img)
                    imgs.append(img)
                    batch_pids.append(pid)
                    batch_cams.append(camid)
                    batch_paths.append(img_path)

                imgs_tensor = torch.stack(imgs, dim=0).to(state.device)
                
                # SIE camera and view labels (dummy or real)
                # Gallery images camera ids are from datasets. Note: camid in gallery_data is already 0-indexed (subtracted 1)
                cam_labels = torch.tensor(batch_cams, dtype=torch.int64).to(state.device)
                view_labels = torch.zeros(len(batch_data), dtype=torch.int64).to(state.device) # Default views to 0

                with torch.no_grad():
                    # Compute feature
                    feat = state.model(imgs_tensor, cam_label=cam_labels, view_label=view_labels)
                    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
                    
                    # TTA (Test-Time Augmentation - horizontal flip)
                    imgs_tensor_flip = torch.flip(imgs_tensor, dims=[3])
                    feat_flip = state.model(imgs_tensor_flip, cam_label=cam_labels, view_label=view_labels)
                    feat_flip = torch.nn.functional.normalize(feat_flip, dim=1, p=2)
                    
                    feat = (feat + feat_flip) / 2.0
                    feat = torch.nn.functional.normalize(feat, dim=1, p=2)

                features_list.append(feat.cpu())
                paths_list.extend(batch_paths)
                pids_list.extend(batch_pids)
                cams_list.extend(batch_cams)

                self.progress_signal.emit(min(i + batch_size, total_images), total_images)

            state.gallery_features = torch.cat(features_list, dim=0)
            state.gallery_paths = paths_list
            state.gallery_pids = pids_list
            state.gallery_cams = cams_list

            self.log_signal.emit(f"Gallery extraction complete. Extracted {len(paths_list)} images.")
            self.finished_signal.emit(True, "Gallery features extracted successfully!", len(paths_list))
        except Exception as e:
            self.finished_signal.emit(False, f"Error extracting gallery features: {str(e)}", 0)

class QuerySearchThread(QThread):
    finished_signal = pyqtSignal(bool, str, list)
    log_signal = pyqtSignal(str)

    def __init__(self, query_path, top_k=10):
        super().__init__()
        self.query_path = query_path
        self.top_k = top_k

    def run(self):
        try:
            if state.model is None:
                self.finished_signal.emit(False, "Model not loaded.", [])
                return
            if state.gallery_features is None:
                self.finished_signal.emit(False, "Gallery features not extracted yet.", [])
                return

            self.log_signal.emit(f"Running search for query: {os.path.basename(self.query_path)}")
            
            # Extract query details
            # Parse pid and camid if follow standard name structure
            pid, camid = -1, -1
            filename = os.path.basename(self.query_path)
            pattern = re.compile(r'([-\d]+)_c(\d)')
            match = pattern.search(filename)
            if match:
                pid, camid = map(int, match.groups())
                camid -= 1  # 0-indexed alignment

            # Extract feature of query image
            img = read_image(self.query_path)
            img_tensor = state.val_transforms(img).unsqueeze(0).to(state.device)
            
            cam_labels = torch.tensor([max(0, camid)], dtype=torch.int64).to(state.device)
            view_labels = torch.zeros(1, dtype=torch.int64).to(state.device)

            with torch.no_grad():
                feat = state.model(img_tensor, cam_label=cam_labels, view_label=view_labels)
                feat = torch.nn.functional.normalize(feat, dim=1, p=2)
                
                # TTA
                img_tensor_flip = torch.flip(img_tensor, dims=[3])
                feat_flip = state.model(img_tensor_flip, cam_label=cam_labels, view_label=view_labels)
                feat_flip = torch.nn.functional.normalize(feat_flip, dim=1, p=2)
                
                feat = (feat + feat_flip) / 2.0
                feat = torch.nn.functional.normalize(feat, dim=1, p=2)
                
            qf = feat.cpu()
            gf = state.gallery_features

            # Calculate distances (Euclidean)
            distmat = euclidean_distance(qf, gf)[0]
            
            # Sort distances (ascending order)
            indices = np.argsort(distmat)
            
            # Build top_k results
            results = []
            count = 0
            for idx in indices:
                g_path = state.gallery_paths[idx]
                g_pid = state.gallery_pids[idx]
                g_cam = state.gallery_cams[idx]
                dist = distmat[idx]
                
                # Check if it is a match (same pid, but exclude same camera of same person in standard ReID if requested.
                # However, for user display, we should just show the absolute closest.
                # Let's show everything but flag if same pid)
                is_same_id = (g_pid == pid) if (pid != -1 and g_pid != -1) else None
                
                results.append({
                    "path": g_path,
                    "filename": os.path.basename(g_path),
                    "pid": g_pid,
                    "camid": g_cam + 1, # Display 1-indexed camera
                    "distance": float(dist),
                    "similarity": float(1.0 / (1.0 + dist)), # Normalizing distance to [0, 1] range for display
                    "is_same_id": is_same_id
                })
                count += 1
                if count >= self.top_k:
                    break

            self.finished_signal.emit(True, self.query_path, results)
        except Exception as e:
            self.finished_signal.emit(False, f"Search failed: {str(e)}", [])

class BatchEvaluationThread(QThread):
    finished_signal = pyqtSignal(bool, float, float, float, float, str)
    progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)

    def run(self):
        try:
            if state.model is None:
                self.finished_signal.emit(False, 0.0, 0.0, 0.0, 0.0, "Model not loaded.")
                return

            self.log_signal.emit("Starting full dataset evaluation...")
            query_data = state.dataset.query
            gallery_data = state.dataset.gallery
            
            num_query = len(query_data)
            num_gallery = len(gallery_data)
            total_images = num_query + num_gallery
            
            self.log_signal.emit(f"Query size: {num_query}, Gallery size: {num_gallery}")

            # Instantiate standard evaluator
            evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=state.cfg.TEST.FEAT_NORM == 'yes')
            evaluator.reset()

            # Process all images (query + gallery combined)
            combined_data = query_data + gallery_data
            batch_size = state.cfg.TEST.IMS_PER_BATCH
            
            processed = 0
            for i in range(0, len(combined_data), batch_size):
                batch = combined_data[i : i + batch_size]
                imgs = []
                pids = []
                cams = []
                
                for img_path, pid, camid, _ in batch:
                    img = read_image(img_path)
                    img = state.val_transforms(img)
                    imgs.append(img)
                    pids.append(pid)
                    cams.append(camid)

                imgs_tensor = torch.stack(imgs, dim=0).to(state.device)
                cam_labels = torch.tensor(cams, dtype=torch.int64).to(state.device)
                view_labels = torch.zeros(len(batch), dtype=torch.int64).to(state.device)

                with torch.no_grad():
                    # Feature extraction
                    feat = state.model(imgs_tensor, cam_label=cam_labels, view_label=view_labels)
                    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
                    
                    # TTA
                    imgs_tensor_flip = torch.flip(imgs_tensor, dims=[3])
                    feat_flip = state.model(imgs_tensor_flip, cam_label=cam_labels, view_label=view_labels)
                    feat_flip = torch.nn.functional.normalize(feat_flip, dim=1, p=2)
                    
                    feat = (feat + feat_flip) / 2.0
                    
                evaluator.update((feat.cpu(), pids, cams))
                
                processed += len(batch)
                self.progress_signal.emit(processed, total_images)

            self.log_signal.emit("Computing metrics...")
            cmc, mAP, _, _, _, _, _ = evaluator.compute()

            log_result = f"Evaluation Results:\n"
            log_result += f"mAP: {mAP:.2%}\n"
            log_result += f"Rank-1: {cmc[0]:.2%}\n"
            log_result += f"Rank-5: {cmc[4]:.2%}\n"
            log_result += f"Rank-10: {cmc[9]:.2%}\n"
            
            self.log_signal.emit("Evaluation complete!")
            self.finished_signal.emit(True, float(mAP), float(cmc[0]), float(cmc[4]), float(cmc[9]), log_result)
        except Exception as e:
            self.finished_signal.emit(False, 0.0, 0.0, 0.0, 0.0, f"Evaluation error: {str(e)}")
