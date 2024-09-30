import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMessageBox)
from MainWindow import MainWindow
from PointCloudManager import PointCloudManager
import argparse

if __name__ == "__main__":
    np.bool = np.bool_
    # Set up the argument parser to get the root directory and JSON filename from the command line
    parser = argparse.ArgumentParser(description="PyVista Point Cloud Visualizer")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory containing subdirectories with JSON files")
    parser.add_argument('--json_file', type=str, default='dsm.json', help="Name of the JSON file to load in each subdirectory")

    args = parser.parse_args()

    app = QApplication(sys.argv)

    try:
        point_cloud_manager = PointCloudManager(args.root_dir, args.json_file)
        
        # Pass the root directory and JSON file from the command line arguments
        window = MainWindow(point_cloud_manager)
        window.show()

    except Exception as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(str(e))
        msg.setWindowTitle("Error")
        msg.exec_()
        sys.exit()

    sys.exit(app.exec_())
