import os
import json
import pyvista as pv
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QMessageBox)
from PyQt5.QtCore import Qt
from pyvistaqt import QtInteractor

class MainWindow(QWidget):
    def __init__(self, point_cloud_manager):
        super().__init__()

        self.point_cloud_manager = point_cloud_manager
        self.current_index = 0

        # Initialize the two PyVista plotters
        self.plotter_mesh = QtInteractor(self)
        self.plotter_surface = QtInteractor(self)

        # Initialize the UI elements
        self.init_ui()

        # Load the first point cloud
        self.load_point_cloud()

        # Sync the cameras between the two plotters
        self.sync_cameras()

    def show_error_message(self, message):
        """Display an error message dialog."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(message)
        msg.setWindowTitle("Error")
        msg.exec_()

    def init_ui(self):
        layout = QVBoxLayout()

        # Horizontal layout for directory selection and opacity slider
        controls_layout = QHBoxLayout()

        # Dropdown menu for directory selection
        dropdown_layout = QHBoxLayout()
        dropdown_label = QLabel("Select Directory:")
        dropdown_label.setFixedWidth(100)
        self.directory_dropdown = QComboBox()
        self.directory_dropdown.addItems([os.path.basename(d) for d in self.point_cloud_manager.directories])
        self.directory_dropdown.currentIndexChanged.connect(self.on_directory_changed)

        dropdown_layout.addWidget(dropdown_label)
        dropdown_layout.addWidget(self.directory_dropdown)
        controls_layout.addLayout(dropdown_layout)

        # Slider for opacity
        slider_layout = QHBoxLayout()
        self.opacity_label = QLabel("Opacity: 50%")  # Initial label with default opacity value

        self.opacity_slider = QSlider()
        self.opacity_slider.setOrientation(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)  # Opacity from 0% to 100%
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)

        slider_layout.addWidget(self.opacity_label)
        slider_layout.addWidget(self.opacity_slider)
        controls_layout.addLayout(slider_layout)

        # Add controls to the main layout
        layout.addLayout(controls_layout)

        # Horizontal layout to hold both plotters side by side
        plotter_layout = QHBoxLayout()
        plotter_layout.addWidget(self.plotter_mesh.interactor)
        plotter_layout.addWidget(self.plotter_surface.interactor)

        layout.addLayout(plotter_layout)
        self.setLayout(layout)

    def load_point_cloud(self):
        """Load the point cloud and initialize both the mesh and the surface plotters."""
        try:
            points = self.point_cloud_manager.load_point_cloud(self.current_index)
        except (FileNotFoundError, IndexError, json.JSONDecodeError) as e:
            self.show_error_message(str(e))
            return

        self.mesh = pv.PolyData(points)
        self.surface = self.mesh.delaunay_2d()

        # Show both the mesh and surface simultaneously
        self.show_mesh()
        self.show_surface()

    def show_mesh(self):
        """Show the point cloud (mesh) in the first plotter."""
        self.plotter_mesh.clear()
        self.mesh_actor = self.plotter_mesh.add_mesh(self.mesh, color="blue", point_size=5, opacity=0.5)
        self.plotter_mesh.add_text(f"Point Cloud Mesh", position="upper_left", font_size=12)
        self.plotter_mesh.reset_camera()

    def show_surface(self):
        """Show the triangulated surface in the second plotter."""
        self.plotter_surface.clear()
        self.surface_actor = self.plotter_surface.add_mesh(self.surface, color="lightblue", show_edges=True, opacity=0.5)
        self.plotter_surface.add_text(f"Surface (Delaunay)", position="upper_left", font_size=12)
        self.plotter_surface.reset_camera()

    def sync_cameras(self):
        """Synchronize the cameras between the two plotters."""
        self.plotter_mesh.enable_trackball_style()
        self.plotter_surface.enable_trackball_style()

        # Trigger an update whenever there are camera changes
        self.plotter_mesh.interactor.AddObserver("EndInteractionEvent", self.update_surface_camera)

        self.plotter_surface.interactor.AddObserver("EndInteractionEvent", self.update_mesh_camera)

    def update_surface_camera(self, caller, event):
        """Update the surface plotter's camera to match the mesh plotter."""
        self.plotter_surface.camera_position = self.plotter_mesh.camera_position
        self.plotter_surface.camera.focal_point = self.plotter_mesh.camera.focal_point
        self.plotter_surface.camera.position = self.plotter_mesh.camera.position
        self.plotter_surface.camera.up = self.plotter_mesh.camera.up
        self.plotter_surface.update()

    def update_mesh_camera(self, caller, event):
        """Update the mesh plotter's camera to match the surface plotter."""
        self.plotter_mesh.camera_position = self.plotter_surface.camera_position
        self.plotter_mesh.camera.focal_point = self.plotter_surface.camera.focal_point
        self.plotter_mesh.camera.position = self.plotter_surface.camera.position
        self.plotter_mesh.camera.up = self.plotter_surface.camera.up
        self.plotter_mesh.update()

    def on_directory_changed(self, index):
        """Handle the event when the user changes the selected directory."""
        self.current_index = index
        self.load_point_cloud()

    def on_opacity_changed(self, value):
        """Change the opacity of the visualizations and update the opacity label."""
        opacity = value / 100.0  # Convert slider value to opacity (0.0 to 1.0)
        self.opacity_label.setText(f"Opacity: {value}%")
        self.mesh_actor.GetProperty().SetOpacity(opacity)
        self.surface_actor.GetProperty().SetOpacity(opacity)

        # Re-render both plots to apply opacity change
        self.plotter_mesh.render()  
        self.plotter_surface.render()
