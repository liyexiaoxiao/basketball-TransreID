import os
import time
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QPushButton, QFileDialog, QComboBox, QSpinBox,
    QTabWidget, QScrollArea, QProgressBar, QGroupBox, QMessageBox,
    QSplitter, QFrame
)
from PyQt5.QtGui import QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, pyqtSlot

from gui.model_thread import (
    state, ModelLoaderThread, GalleryFeatureExtractorThread,
    QuerySearchThread, BatchEvaluationThread
)
from gui.widgets import ImageCard, LogViewer, DatasetListWidget

class MainWindow(QMainWindow):
    """
    Main dashboard for basketball ReID frontend.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basketball Player Identity Re-Identification (TransReID)")
        self.resize(1200, 800)
        
        # Thread references
        self.load_thread = None
        self.extract_thread = None
        self.search_thread = None
        self.eval_thread = None
        
        # Selected Paths
        self.config_path = ""
        self.weight_path = ""
        self.query_path = ""
        
        self.init_ui()
        self.apply_dark_theme()
        
        # Try to find default files to pre-fill
        self.prefill_defaults()

    def init_ui(self):
        # Central widget and horizontal layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Splitter to allow resizing of left sidebar and right content panel
        splitter = QSplitter(Qt.Horizontal, self)
        main_layout.addWidget(splitter)
        
        # ==========================================
        # LEFT PANEL: SIDEBAR (Controls & Config)
        # ==========================================
        sidebar = QWidget(self)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(15)
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(420)
        
        # Group 1: Model & Configuration Loader
        model_group = QGroupBox("Model Configuration", sidebar)
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(10)
        
        # Config Path
        cfg_label = QLabel("Config YML Path:", self)
        model_layout.addWidget(cfg_label)
        
        cfg_file_layout = QHBoxLayout()
        self.cfg_path_label = QLabel("No config selected", self)
        self.cfg_path_label.setStyleSheet("color: #888888; background-color: #121212; padding: 6px; border: 1px dashed #333; border-radius: 4px;")
        self.cfg_path_label.setWordWrap(True)
        self.btn_browse_cfg = QPushButton("Browse...", self)
        self.btn_browse_cfg.clicked.connect(self.browse_config)
        cfg_file_layout.addWidget(self.cfg_path_label, 1)
        cfg_file_layout.addWidget(self.btn_browse_cfg)
        model_layout.addLayout(cfg_file_layout)
        
        # Weight Path
        weight_label = QLabel("Model Weights (.pth):", self)
        model_layout.addWidget(weight_label)
        
        weight_file_layout = QHBoxLayout()
        self.weight_path_label = QLabel("No weights selected", self)
        self.weight_path_label.setStyleSheet("color: #888888; background-color: #121212; padding: 6px; border: 1px dashed #333; border-radius: 4px;")
        self.weight_path_label.setWordWrap(True)
        self.btn_browse_weight = QPushButton("Browse...", self)
        self.btn_browse_weight.clicked.connect(self.browse_weights)
        weight_file_layout.addWidget(self.weight_path_label, 1)
        weight_file_layout.addWidget(self.btn_browse_weight)
        model_layout.addLayout(weight_file_layout)
        
        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Inference Device:", self))
        self.combo_device = QComboBox(self)
        self.combo_device.addItem("CPU", "cpu")
        if state.device == "cuda":
            self.combo_device.addItem("CUDA/GPU", "cuda")
            self.combo_device.setCurrentIndex(1)  # Default to CUDA
        device_layout.addWidget(self.combo_device)
        model_layout.addLayout(device_layout)
        
        # Model Load Button
        self.btn_load_model = QPushButton("Load Config & Model", self)
        self.btn_load_model.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 8px;")
        self.btn_load_model.clicked.connect(self.load_model_action)
        model_layout.addWidget(self.btn_load_model)
        
        # Model Metadata details
        self.lbl_model_details = QLabel("Model Status: Not Loaded", self)
        self.lbl_model_details.setStyleSheet("color: #aaa; font-style: italic;")
        self.lbl_model_details.setWordWrap(True)
        model_layout.addWidget(self.lbl_model_details)
        
        sidebar_layout.addWidget(model_group)
        
        # Group 2: ReID Query Selector & Search Options
        query_group = QGroupBox("Query Panel", sidebar)
        query_layout = QVBoxLayout(query_group)
        query_layout.setSpacing(10)
        
        # Query Preview Image
        self.lbl_query_preview = QLabel("No Image Selected", self)
        self.lbl_query_preview.setAlignment(Qt.AlignCenter)
        self.lbl_query_preview.setMinimumHeight(180)
        self.lbl_query_preview.setStyleSheet("border: 2px dashed #444; border-radius: 6px; background-color: #121212; color: #666; font-size: 14px;")
        query_layout.addWidget(self.lbl_query_preview)
        
        # Choose Image Button
        query_file_layout = QHBoxLayout()
        self.btn_browse_query = QPushButton("Choose Query Image...", self)
        self.btn_browse_query.clicked.connect(self.browse_query_image)
        query_file_layout.addWidget(self.btn_browse_query)
        query_layout.addLayout(query_file_layout)
        
        # Top-K Matches Options
        topk_layout = QHBoxLayout()
        topk_layout.addWidget(QLabel("Matches to Display (Top-K):", self))
        self.spin_topk = QSpinBox(self)
        self.spin_topk.setRange(1, 50)
        self.spin_topk.setValue(10)
        topk_layout.addWidget(self.spin_topk)
        query_layout.addLayout(topk_layout)
        
        # Extract gallery cache button
        self.btn_extract_gallery = QPushButton("Pre-Extract Gallery Features", self)
        self.btn_extract_gallery.setEnabled(False)
        self.btn_extract_gallery.clicked.connect(self.extract_gallery_action)
        query_layout.addWidget(self.btn_extract_gallery)
        
        # Retrieval Action Button
        self.btn_search = QPushButton("Search & Match", self)
        self.btn_search.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.btn_search.setEnabled(False)
        self.btn_search.clicked.connect(self.search_action)
        query_layout.addWidget(self.btn_search)
        
        sidebar_layout.addWidget(query_group)
        sidebar_layout.addStretch(1)
        
        splitter.addWidget(sidebar)
        
        # ==========================================
        # RIGHT PANEL: MAIN TABS & RESULTS
        # ==========================================
        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        self.tab_widget = QTabWidget(self)
        right_layout.addWidget(self.tab_widget)
        
        # Tab 1: Single Search Results
        self.tab_search = QWidget(self)
        tab_search_layout = QVBoxLayout(self.tab_search)
        tab_search_layout.setContentsMargins(10, 10, 10, 10)
        
        self.lbl_search_summary = QLabel("Results will appear here after clicking Search.", self)
        self.lbl_search_summary.setStyleSheet("font-size: 13px; color: #888888; margin-bottom: 5px;")
        tab_search_layout.addWidget(self.lbl_search_summary)
        
        # Scroll Area for result cards grid
        self.results_scroll = QScrollArea(self)
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setStyleSheet("background-color: #161616; border: 1px solid #2b2b2b; border-radius: 6px;")
        
        self.results_container = QWidget(self)
        self.results_container.setStyleSheet("background-color: transparent;")
        self.grid_results = QGridLayout(self.results_container)
        self.grid_results.setSpacing(10)
        self.grid_results.setContentsMargins(10, 10, 10, 10)
        
        self.results_scroll.setWidget(self.results_container)
        tab_search_layout.addWidget(self.results_scroll)
        
        self.tab_widget.addTab(self.tab_search, "Single Image Search")
        
        # Tab 2: Dataset Explorer (Lazy-loaded upon model load)
        self.tab_explorer = QWidget(self)
        self.explorer_layout = QVBoxLayout(self.tab_explorer)
        self.explorer_layout.setContentsMargins(10, 10, 10, 10)
        self.explorer_placeholder = QLabel("Please load a model first to initialize dataset exploration.", self)
        self.explorer_placeholder.setAlignment(Qt.AlignCenter)
        self.explorer_placeholder.setStyleSheet("color: #666666; font-size: 14px;")
        self.explorer_layout.addWidget(self.explorer_placeholder)
        
        self.tab_widget.addTab(self.tab_explorer, "Dataset Explorer")
        
        # Tab 3: Performance Batch Evaluation
        self.tab_eval = QWidget(self)
        tab_eval_layout = QVBoxLayout(self.tab_eval)
        tab_eval_layout.setContentsMargins(15, 15, 15, 15)
        tab_eval_layout.setSpacing(15)
        
        eval_intro = QLabel("Run Batch Evaluation on all query and gallery images to calculate metrics (mAP, Rank-1, Rank-5, Rank-10).", self)
        eval_intro.setFont(QFont("Segoe UI", 10))
        eval_intro.setStyleSheet("color: #bbbbbb;")
        tab_eval_layout.addWidget(eval_intro)
        
        self.btn_run_eval = QPushButton("Start Batch Evaluation", self)
        self.btn_run_eval.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold; padding: 10px; font-size: 13px;")
        self.btn_run_eval.setEnabled(False)
        self.btn_run_eval.clicked.connect(self.run_batch_eval_action)
        tab_eval_layout.addWidget(self.btn_run_eval)
        
        # Progress section
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                border-radius: 4px;
                text-align: center;
                background-color: #121212;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #9b59b6;
            }
        """)
        self.progress_bar.setValue(0)
        self.lbl_progress_status = QLabel("Idle", self)
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.lbl_progress_status)
        tab_eval_layout.addLayout(progress_layout)
        
        # Score Summary Cards Panel
        scores_panel = QHBoxLayout()
        scores_panel.setSpacing(15)
        
        self.card_map = self.create_metric_card("mAP", "0.0%")
        self.card_rank1 = self.create_metric_card("Rank-1", "0.0%")
        self.card_rank5 = self.create_metric_card("Rank-5", "0.0%")
        self.card_rank10 = self.create_metric_card("Rank-10", "0.0%")
        
        scores_panel.addWidget(self.card_map)
        scores_panel.addWidget(self.card_rank1)
        scores_panel.addWidget(self.card_rank5)
        scores_panel.addWidget(self.card_rank10)
        tab_eval_layout.addLayout(scores_panel)
        
        # Log Viewer Panel
        tab_eval_layout.addWidget(QLabel("Terminal Log console:", self))
        self.log_viewer = LogViewer(self)
        tab_eval_layout.addWidget(self.log_viewer, 1)
        
        self.tab_widget.addTab(self.tab_eval, "Batch Evaluation")
        
        splitter.addWidget(right_panel)
        
        # Set splitter sizes (35% left, 65% right)
        splitter.setSizes([350, 850])

    def create_metric_card(self, title, val):
        card = QFrame(self)
        card.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignCenter)
        
        title_lbl = QLabel(title, card)
        title_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title_lbl.setStyleSheet("color: #888888; border: none;")
        title_lbl.setAlignment(Qt.AlignCenter)
        
        val_lbl = QLabel(val, card)
        val_lbl.setFont(QFont("Segoe UI", 18, QFont.Bold))
        val_lbl.setStyleSheet("color: #9b59b6; border: none;")
        val_lbl.setAlignment(Qt.AlignCenter)
        
        # Save reference
        card.val_lbl = val_lbl
        
        layout.addWidget(title_lbl)
        layout.addWidget(val_lbl)
        return card

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QGroupBox {
                font-family: 'Segoe UI';
                font-size: 13px;
                font-weight: bold;
                color: #3498db;
                border: 1px solid #2b2b2b;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                font-family: 'Segoe UI';
                color: #e0e0e0;
                font-size: 12px;
            }
            QPushButton {
                font-family: 'Segoe UI';
                font-size: 12px;
                color: #ffffff;
                background-color: #2c2c2c;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border: 1px solid #555555;
            }
            QPushButton:pressed {
                background-color: #1e1e1e;
            }
            QPushButton:disabled {
                background-color: #1c1c1c;
                color: #555555;
                border: 1px solid #252525;
            }
            QComboBox {
                font-family: 'Segoe UI';
                background-color: #222222;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QSpinBox {
                font-family: 'Segoe UI';
                background-color: #222222;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #2b2b2b;
                background-color: #161616;
                border-radius: 6px;
                top: -1px;
            }
            QTabBar::tab {
                font-family: 'Segoe UI';
                font-weight: bold;
                background-color: #222222;
                color: #888888;
                border: 1px solid #2b2b2b;
                border-bottom-color: transparent;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #161616;
                color: #3498db;
                border-bottom-color: #161616;
            }
            QTabBar::tab:hover {
                background: #2a2a2a;
            }
            QScrollBar:vertical {
                border: none;
                background: #121212;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #333333;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #444444;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)

    def prefill_defaults(self):
        # Scan configs/BallShow
        default_cfg_dir = os.path.join(os.getcwd(), "configs", "BallShow")
        if os.path.exists(default_cfg_dir):
            configs = [f for f in os.listdir(default_cfg_dir) if f.endswith(".yml")]
            if "vit_transreid_stride.yml" in configs:
                self.config_path = os.path.join(default_cfg_dir, "vit_transreid_stride.yml")
                self.cfg_path_label.setText("vit_transreid_stride.yml")
                self.cfg_path_label.setStyleSheet("color: #2ecc71; background-color: #121212; padding: 6px; border: 1px solid #27ae60; border-radius: 4px;")
            elif configs:
                self.config_path = os.path.join(default_cfg_dir, configs[0])
                self.cfg_path_label.setText(configs[0])
                self.cfg_path_label.setStyleSheet("color: #2ecc71; background-color: #121212; padding: 6px; border: 1px solid #27ae60; border-radius: 4px;")
                
        # Scan pre-trained weights in root
        if os.path.exists("jx_vit_base_p16_224-80ecf9dd.pth"):
            self.weight_path = os.path.abspath("jx_vit_base_p16_224-80ecf9dd.pth")
            self.weight_path_label.setText("jx_vit_base_p16_224-80ecf9dd.pth")
            self.weight_path_label.setStyleSheet("color: #2ecc71; background-color: #121212; padding: 6px; border: 1px solid #27ae60; border-radius: 4px;")

    # ==========================================
    # FILE BROWSER CONNECTORS
    # ==========================================
    def browse_config(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Config File", os.getcwd(), "YAML Config Files (*.yml *.yaml)"
        )
        if filename:
            self.config_path = filename
            self.cfg_path_label.setText(os.path.basename(filename))
            self.cfg_path_label.setStyleSheet("color: #2ecc71; background-color: #121212; padding: 6px; border: 1px solid #27ae60; border-radius: 4px;")
            self.log_viewer.append_log(f"Selected config: {filename}")

    def browse_weights(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Model Weights File", os.getcwd(), "PyTorch Model Weights (*.pth *.pt *.bin)"
        )
        if filename:
            self.weight_path = filename
            self.weight_path_label.setText(os.path.basename(filename))
            self.weight_path_label.setStyleSheet("color: #2ecc71; background-color: #121212; padding: 6px; border: 1px solid #27ae60; border-radius: 4px;")
            self.log_viewer.append_log(f"Selected weights: {filename}")

    def browse_query_image(self):
        # Start looking in query folder if exists
        start_dir = os.path.join(os.getcwd(), "data", "BallShow", "query")
        if not os.path.exists(start_dir):
            start_dir = os.getcwd()
            
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Query Image", start_dir, "Images (*.jpg *.png *.jpeg)"
        )
        if filename:
            self.set_query_image(filename)

    def set_query_image(self, path):
        self.query_path = path
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(
            self.lbl_query_preview.width() - 10,
            self.lbl_query_preview.height() - 10,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.lbl_query_preview.setPixmap(scaled_pixmap)
        self.lbl_query_preview.setStyleSheet("border: 2px solid #3498db; border-radius: 6px; background-color: #121212;")
        self.log_viewer.append_log(f"Selected query image: {os.path.basename(path)}")
        
        # Check if model is loaded and gallery feature extracted
        if state.model is not None and state.gallery_features is not None:
            self.btn_search.setEnabled(True)

    # ==========================================
    # ACTION: LOAD MODEL
    # ==========================================
    def load_model_action(self):
        if not self.config_path:
            QMessageBox.warning(self, "Warning", "Please select a config file first!")
            return
            
        # Disable buttons during load
        self.btn_load_model.setEnabled(False)
        self.btn_load_model.setText("Loading Model...")
        self.lbl_model_details.setText("Model Status: Loading...")
        
        dev = self.combo_device.currentData()
        
        # Run loader thread
        self.load_thread = ModelLoaderThread(self.config_path, self.weight_path, dev)
        self.load_thread.log_signal.connect(self.log_viewer.append_log)
        self.load_thread.finished_signal.connect(self.on_model_loaded)
        self.load_thread.start()

    @pyqtSlot(bool, str, dict)
    def on_model_loaded(self, success, message, details):
        self.btn_load_model.setEnabled(True)
        self.btn_load_model.setText("Load Config & Model")
        
        if success:
            self.log_viewer.append_log(message)
            self.lbl_model_details.setText(
                f"Backbone: {details['backbone']}\n"
                f"Type: {details['transformer_type']}\n"
                f"Classes: {details['num_classes']}\n"
                f"Cams: {details['camera_num']}, Views: {details['view_num']}\n"
                f"Size: {details['input_size']}\n"
                f"Device: {details['device'].upper()}"
            )
            self.lbl_model_details.setStyleSheet("color: #2ecc71; font-weight: bold;")
            
            # Enable next actions
            self.btn_extract_gallery.setEnabled(True)
            self.btn_run_eval.setEnabled(True)
            
            # Auto initialize explorer tab
            self.initialize_explorer_tab()
        else:
            self.log_viewer.append_log(f"Failed to load: {message}")
            self.lbl_model_details.setText(f"Load failed: {message}")
            self.lbl_model_details.setStyleSheet("color: #e74c3c; font-weight: bold;")
            QMessageBox.critical(self, "Load Error", message)

    def initialize_explorer_tab(self):
        # Clear tab_explorer layout
        while self.explorer_layout.count():
            child = self.explorer_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        # Load query directory from dataset config
        query_dir = os.path.join(state.cfg.DATASETS.ROOT_DIR, "query")
        if os.path.exists(query_dir):
            explorer = DatasetListWidget(query_dir, self)
            explorer.image_selected.connect(self.on_explorer_image_selected)
            self.explorer_layout.addWidget(explorer)
            self.log_viewer.append_log("Dataset Explorer initialized successfully.")
        else:
            lbl_err = QLabel(f"Query directory not found at: {query_dir}", self)
            lbl_err.setStyleSheet("color: #e74c3c;")
            self.explorer_layout.addWidget(lbl_err)

    def on_explorer_image_selected(self, img_path):
        self.set_query_image(img_path)
        # Switch to single search tab
        self.tab_widget.setCurrentIndex(0)
        # Automatically run search if gallery features are already extracted
        if state.gallery_features is not None:
            self.search_action()
        else:
            self.log_viewer.append_log("Please pre-extract gallery features first to search immediately.")

    # ==========================================
    # ACTION: EXTRACT GALLERY FEATURES
    # ==========================================
    def extract_gallery_action(self):
        self.btn_extract_gallery.setEnabled(False)
        self.btn_extract_gallery.setText("Extracting...")
        self.progress_bar.setValue(0)
        self.lbl_progress_status.setText("Extracting Gallery...")
        
        self.extract_thread = GalleryFeatureExtractorThread()
        self.extract_thread.log_signal.connect(self.log_viewer.append_log)
        self.extract_thread.progress_signal.connect(self.on_extraction_progress)
        self.extract_thread.finished_signal.connect(self.on_extraction_finished)
        self.extract_thread.start()

    @pyqtSlot(int, int)
    def on_extraction_progress(self, current, total):
        pct = int(current / total * 100)
        self.progress_bar.setValue(pct)
        self.lbl_progress_status.setText(f"Extracting: {current}/{total}")

    @pyqtSlot(bool, str, int)
    def on_extraction_finished(self, success, message, count):
        self.btn_extract_gallery.setEnabled(True)
        self.btn_extract_gallery.setText("Pre-Extract Gallery Features")
        
        if success:
            self.log_viewer.append_log(message)
            self.lbl_progress_status.setText("Gallery Extracted")
            # Enable Search
            if self.query_path:
                self.btn_search.setEnabled(True)
            QMessageBox.information(self, "Success", f"Features extracted successfully for {count} gallery images!")
        else:
            self.log_viewer.append_log(f"Extraction failed: {message}")
            self.lbl_progress_status.setText("Extraction Failed")
            QMessageBox.critical(self, "Extraction Error", message)

    # ==========================================
    # ACTION: RUN REID RETRIEVAL SEARCH
    # ==========================================
    def search_action(self):
        if not self.query_path:
            QMessageBox.warning(self, "Warning", "Please select a query image first!")
            return
            
        self.btn_search.setEnabled(False)
        self.btn_search.setText("Searching...")
        self.lbl_search_summary.setText("Searching gallery database...")
        
        # Clear previous results grid
        while self.grid_results.count():
            child = self.grid_results.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        top_k = self.spin_topk.value()
        
        start_time = time.time()
        self.search_thread = QuerySearchThread(self.query_path, top_k)
        self.search_thread.log_signal.connect(self.log_viewer.append_log)
        
        # We define search finish handler taking execution time
        def on_search_done(success, q_path, results):
            elapsed = time.time() - start_time
            self.btn_search.setEnabled(True)
            self.btn_search.setText("Search & Match")
            
            if success:
                self.lbl_search_summary.setText(f"Retrieved top {len(results)} matches in {elapsed:.3f} seconds.")
                
                # Display matches in a nice grid (5 columns)
                cols = 5
                for idx, res in enumerate(results):
                    row = idx // cols
                    col = idx % cols
                    
                    card = ImageCard(
                        image_path=res["path"],
                        pid=res["pid"],
                        camid=res["camid"],
                        score=res["similarity"],
                        is_same_id=res["is_same_id"],
                        parent=self.results_container
                    )
                    self.grid_results.addWidget(card, row, col)
            else:
                self.lbl_search_summary.setText("Search failed. Check logs.")
                QMessageBox.critical(self, "Search Error", q_path) # q_path acts as error message in failure
                
        self.search_thread.finished_signal.connect(on_search_done)
        self.search_thread.start()

    # ==========================================
    # ACTION: RUN BATCH EVALUATION
    # ==========================================
    def run_batch_eval_action(self):
        self.btn_run_eval.setEnabled(False)
        self.btn_run_eval.setText("Evaluating...")
        self.progress_bar.setValue(0)
        self.lbl_progress_status.setText("Initializing evaluation...")
        
        # Reset card texts
        self.card_map.val_lbl.setText("0.0%")
        self.card_rank1.val_lbl.setText("0.0%")
        self.card_rank5.val_lbl.setText("0.0%")
        self.card_rank10.val_lbl.setText("0.0%")
        
        self.eval_thread = BatchEvaluationThread()
        self.eval_thread.log_signal.connect(self.log_viewer.append_log)
        self.eval_thread.progress_signal.connect(self.on_eval_progress)
        self.eval_thread.finished_signal.connect(self.on_eval_finished)
        self.eval_thread.start()

    @pyqtSlot(int, int)
    def on_eval_progress(self, current, total):
        pct = int(current / total * 100)
        self.progress_bar.setValue(pct)
        self.lbl_progress_status.setText(f"Processed: {current}/{total}")

    @pyqtSlot(bool, float, float, float, float, str)
    def on_eval_finished(self, success, mAP, rank1, rank5, rank10, log_summary):
        self.btn_run_eval.setEnabled(True)
        self.btn_run_eval.setText("Start Batch Evaluation")
        
        if success:
            self.log_viewer.append_log(log_summary)
            self.lbl_progress_status.setText("Evaluation Complete")
            
            # Update metrics cards
            self.card_map.val_lbl.setText(f"{mAP * 100:.1f}%")
            self.card_rank1.val_lbl.setText(f"{rank1 * 100:.1f}%")
            self.card_rank5.val_lbl.setText(f"{rank5 * 100:.1f}%")
            self.card_rank10.val_lbl.setText(f"{rank10 * 100:.1f}%")
            
            QMessageBox.information(
                self, "Evaluation Successful", 
                f"Metrics Calculated:\nmAP: {mAP * 100:.2f}%\nRank-1: {rank1 * 100:.2f}%"
            )
        else:
            self.log_viewer.append_log(f"Evaluation failed: {log_summary}")
            self.lbl_progress_status.setText("Evaluation Failed")
            QMessageBox.critical(self, "Evaluation Error", log_summary)
            
    def closeEvent(self, event):
        # Cleanup running threads on window exit
        for thread in [self.load_thread, self.extract_thread, self.search_thread, self.eval_thread]:
            if thread is not None and thread.isRunning():
                thread.terminate()
                thread.wait()
        event.accept()
