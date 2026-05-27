import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from gui.main_window import MainWindow

def run_app():
    # Enable High DPI scaling
    QApplication.setAttribute(15, True) # Qt.AA_EnableHighDpiScaling = 15
    QApplication.setAttribute(17, True) # Qt.AA_UseHighDpiPixmaps = 17
    
    app = QApplication(sys.argv)
    
    # Set default premium font
    default_font = QFont("Segoe UI", 10)
    app.setFont(default_font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()
