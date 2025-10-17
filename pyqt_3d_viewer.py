#!/usr/bin/env python3
"""
PyQt 3D Viewer and Converter

- Loads DICOM series, NRRD volumes, or existing mesh files (STL/OBJ/PLY/GLB/GLTF)
- Segments bone from volume using HU thresholds and generates a surface via marching cubes
- Repairs meshes for better viewer compatibility (normals, holes, inversions, connected component)
- Displays meshes in an embedded PyVista scene
- Exports to STL/OBJ/PLY/GLB/GLTF
"""
import os
import sys
import traceback
from typing import List, Optional, Tuple

import numpy as np

# Volume IO and processing
import pydicom
from scipy import ndimage
from skimage import measure, morphology

# NRRD support
try:
    import nrrd  # provided by package 'pynrrd'
except Exception:  # pragma: no cover
    nrrd = None

# Mesh processing
import trimesh

# PyVista + Qt viewer
import pyvista as pv
from pyvistaqt import QtInteractor

# PyQt
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QAction,
)


class ViewerState:
    """Holds current working data for the viewer."""

    def __init__(self):
        self.volume_data: Optional[np.ndarray] = None
        self.volume_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (z, y, x)
        self.segment_mask: Optional[np.ndarray] = None
        self.current_mesh: Optional[trimesh.Trimesh] = None

        # Default HU thresholds for bone
        self.hu_min: int = 150
        self.hu_max: int = 3000


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Viewer & Converter (PyQt + PyVista)")
        self.resize(1280, 820)

        self.state = ViewerState()

        # Central 3D viewport
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter)
        self._setup_plotter()

        # Actions and menus
        self._create_actions()
        self._create_menus()
        self._create_toolbar()

        # Status bar for feedback
        self.statusBar().showMessage("Ready")

    # UI scaffolding
    def _setup_plotter(self) -> None:
        self.plotter.enable_anti_aliasing()
        self.plotter.add_axes(interactive=True)
        self.plotter.set_background("#1e1e1e")

    def _create_actions(self) -> None:
        # File
        self.action_open_mesh = QAction("Open Mesh…", self)
        self.action_open_mesh.triggered.connect(self.open_mesh_file)

        self.action_open_dicom = QAction("Open DICOM Folder…", self)
        self.action_open_dicom.triggered.connect(self.open_dicom_folder)

        self.action_open_nrrd = QAction("Open NRRD…", self)
        self.action_open_nrrd.triggered.connect(self.open_nrrd_file)

        self.action_export = QAction("Export As…", self)
        self.action_export.triggered.connect(self.export_mesh)

        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close)

        # Process
        self.action_segment = QAction("Segment Volume (HU)…", self)
        self.action_segment.triggered.connect(self.segment_volume_dialog)

        self.action_surface = QAction("Generate Surface (Marching Cubes)", self)
        self.action_surface.triggered.connect(self.generate_surface)

        self.action_repair = QAction("Repair Mesh for Viewer", self)
        self.action_repair.triggered.connect(self.repair_current_mesh)

        # View
        self.action_reset_cam = QAction("Reset Camera", self)
        self.action_reset_cam.triggered.connect(lambda: self.plotter.reset_camera())

    def _create_menus(self) -> None:
        menu_file = self.menuBar().addMenu("File")
        menu_file.addAction(self.action_open_mesh)
        menu_file.addAction(self.action_open_dicom)
        menu_file.addAction(self.action_open_nrrd)
        menu_file.addSeparator()
        menu_file.addAction(self.action_export)
        menu_file.addSeparator()
        menu_file.addAction(self.action_exit)

        menu_process = self.menuBar().addMenu("Process")
        menu_process.addAction(self.action_segment)
        menu_process.addAction(self.action_surface)
        menu_process.addAction(self.action_repair)

        menu_view = self.menuBar().addMenu("View")
        menu_view.addAction(self.action_reset_cam)

    def _create_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.addAction(self.action_open_mesh)
        tb.addAction(self.action_open_dicom)
        tb.addAction(self.action_open_nrrd)
        tb.addSeparator()
        tb.addAction(self.action_segment)
        tb.addAction(self.action_surface)
        tb.addAction(self.action_repair)
        tb.addSeparator()
        tb.addAction(self.action_export)

    # Helpers
    def _show_error(self, title: str, err: Exception) -> None:
        traceback.print_exc()
        QMessageBox.critical(self, title, f"{title}:\n{err}")
        self.statusBar().showMessage(f"{title}: {err}")

    def _show_info(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)
        self.statusBar().showMessage(text)

    def _faces_to_pv(self, faces: np.ndarray) -> np.ndarray:
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Faces must be (N, 3) triangle indices")
        counts = np.full((faces.shape[0], 1), 3, dtype=np.int64)
        faces_with_counts = np.hstack((counts, faces.astype(np.int64)))
        return faces_with_counts.ravel()

    def _update_view_with_trimesh(self, tri: trimesh.Trimesh, reset_camera: bool = True) -> None:
        """Convert trimesh to PyVista and display."""
        self.state.current_mesh = tri
        try:
            pv_faces = self._faces_to_pv(tri.faces)
            pv_mesh = pv.PolyData(tri.vertices, pv_faces)
        except Exception:
            # Force rebuild as triangles if needed
            tri = trimesh.Trimesh(vertices=tri.vertices, faces=tri.faces, process=True)
            pv_faces = self._faces_to_pv(tri.faces)
            pv_mesh = pv.PolyData(tri.vertices, pv_faces)

        self.plotter.clear()
        self.plotter.add_mesh(
            pv_mesh,
            color="#cccccc",
            smooth_shading=True,
            show_edges=False,
            specular=0.1,
            ambient=0.4,
        )
        self.plotter.add_axes(interactive=True)
        if reset_camera:
            self.plotter.reset_camera()
        self.statusBar().showMessage("Mesh displayed")

    # File open operations
    def open_mesh_file(self) -> None:
        filters = (
            "Mesh Files (*.stl *.obj *.ply *.gltf *.glb);;"
            "All Files (*)"
        )
        path, _ = QFileDialog.getOpenFileName(self, "Open Mesh", "", filters)
        if not path:
            return
        try:
            # Force single mesh when possible (e.g., GLTF/GLB scenes)
            mesh_or_scene = trimesh.load(path, force="mesh")
            if isinstance(mesh_or_scene, trimesh.Scene):
                tri = mesh_or_scene.dump(concatenate=True)
            else:
                tri = mesh_or_scene
            if not isinstance(tri, trimesh.Trimesh):
                # Last resort: try concatenating geometry
                scene = trimesh.load(path)
                if isinstance(scene, trimesh.Scene):
                    tri = scene.dump(concatenate=True)
            # Ensure triangles and processed topology
            tri = trimesh.Trimesh(vertices=tri.vertices, faces=tri.faces, process=True)
            self._update_view_with_trimesh(tri)
        except Exception as e:
            self._show_error("Failed to open mesh", e)

    def open_dicom_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder", "")
        if not folder:
            return
        try:
            files = self._find_dicom_files(folder)
            if not files:
                raise RuntimeError("No DICOM files found in folder")
            volume, spacing = self._load_dicom_series(files)
            self.state.volume_data = volume
            self.state.volume_spacing = spacing
            self.state.segment_mask = None
            self._show_info("DICOM Loaded", f"Loaded {len(files)} files. Spacing: {spacing}")
        except Exception as e:
            self._show_error("Failed to load DICOM", e)

    def open_nrrd_file(self) -> None:
        if nrrd is None:
            QMessageBox.warning(self, "NRRD Unavailable", "pynrrd is not installed.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open NRRD", "", "NRRD Files (*.nrrd);;All Files (*)")
        if not path:
            return
        try:
            data, header = nrrd.read(path)
            spacing = self._spacing_from_nrrd_header(header)
            self.state.volume_data = data.astype(np.float32)
            self.state.volume_spacing = spacing
            self.state.segment_mask = None
            self._show_info("NRRD Loaded", f"Shape: {data.shape} Spacing: {spacing}")
        except Exception as e:
            self._show_error("Failed to load NRRD", e)

    # Volume helpers
    def _find_dicom_files(self, folder: str) -> List[str]:
        dicom_files: List[str] = []
        for root, _dirs, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith((".dcm", ".dicom")):
                    dicom_files.append(os.path.join(root, fname))
        if not dicom_files:
            return []
        # Try to sort by InstanceNumber if available
        try:
            def read_inst(fp: str) -> int:
                try:
                    ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                    return int(getattr(ds, "InstanceNumber", 0))
                except Exception:
                    return 0
            dicom_files.sort(key=read_inst)
        except Exception:
            dicom_files.sort()
        return dicom_files

    def _load_dicom_series(self, files: List[str]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        ds0 = pydicom.dcmread(files[0], force=True)
        rows, cols = int(ds0.Rows), int(ds0.Columns)
        num_slices = len(files)
        volume = np.zeros((num_slices, rows, cols), dtype=np.float32)

        pixel_spacing = getattr(ds0, "PixelSpacing", [1.0, 1.0])
        slice_thickness = float(getattr(ds0, "SliceThickness", 1.0))
        spacing = (slice_thickness, float(pixel_spacing[1]), float(pixel_spacing[0]))  # (z, y, x)

        for i, fp in enumerate(files):
            dsi = pydicom.dcmread(fp, force=True)
            arr = dsi.pixel_array.astype(np.float32)
            slope = float(getattr(dsi, "RescaleSlope", 1.0))
            intercept = float(getattr(dsi, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
            volume[i] = arr
        return volume, spacing

    def _spacing_from_nrrd_header(self, header: dict) -> Tuple[float, float, float]:
        # Try 'space directions' vectors
        try:
            sdirs = header.get("space directions")
            if sdirs is not None:
                vecs = [np.array(v, dtype=float) for v in sdirs]
                return (
                    float(np.linalg.norm(vecs[0])),
                    float(np.linalg.norm(vecs[1])),
                    float(np.linalg.norm(vecs[2])),
                )
        except Exception:
            pass
        # Try 'spacings'
        try:
            sp = header.get("spacings")
            if sp is not None and len(sp) >= 3:
                return float(sp[0]), float(sp[1]), float(sp[2])
        except Exception:
            pass
        return 1.0, 1.0, 1.0

    # Segmentation and surface generation
    def segment_volume_dialog(self) -> None:
        if self.state.volume_data is None:
            QMessageBox.warning(self, "No Volume", "Load a DICOM or NRRD volume first.")
            return
        hu_min, ok1 = QtWidgets.QInputDialog.getInt(self, "Min HU", "Min HU:", value=self.state.hu_min)
        if not ok1:
            return
        hu_max, ok2 = QtWidgets.QInputDialog.getInt(self, "Max HU", "Max HU:", value=self.state.hu_max)
        if not ok2:
            return
        self.state.hu_min, self.state.hu_max = int(hu_min), int(hu_max)
        self.segment_volume()

    def segment_volume(self) -> None:
        try:
            vol = self.state.volume_data
            if vol is None:
                raise RuntimeError("No volume available")
            mask = (vol >= self.state.hu_min) & (vol <= self.state.hu_max)
            mask = morphology.remove_small_objects(mask, min_size=1000)
            mask = ndimage.binary_fill_holes(mask)
            mask = morphology.binary_closing(mask, morphology.ball(2))
            self.state.segment_mask = mask.astype(np.uint8)
            self._show_info("Segmentation", "Bone segmentation completed.")
        except Exception as e:
            self._show_error("Segmentation failed", e)

    def generate_surface(self) -> None:
        try:
            if self.state.segment_mask is None:
                # If no segmentation yet, attempt with current HU range
                if self.state.volume_data is None:
                    raise RuntimeError("No data to surface. Load a volume or mesh first.")
                self.segment_volume()
                if self.state.segment_mask is None:
                    return
            spacing = self.state.volume_spacing
            verts, faces, _normals, _vals = measure.marching_cubes(
                self.state.segment_mask, level=0.5, spacing=spacing
            )
            tri = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            self._update_view_with_trimesh(tri)
        except Exception as e:
            self._show_error("Surface generation failed", e)

    # Mesh repair and export
    def repair_current_mesh(self) -> None:
        try:
            tri = self.state.current_mesh
            if tri is None:
                raise RuntimeError("There is no mesh to repair.")
            # Attempt to keep the largest connected component
            try:
                components = tri.split(only_watertight=False)
                if len(components) > 1:
                    tri = max(components, key=lambda c: len(c.faces))
            except Exception:
                pass
            # Run a stronger repair pipeline
            try:
                trimesh.repair.fix_normals(tri)
            except Exception:
                pass
            try:
                trimesh.repair.fill_holes(tri)
            except Exception:
                pass
            try:
                trimesh.repair.fix_inversion(tri)
            except Exception:
                pass
            try:
                trimesh.repair.broken_faces(tri)
            except Exception:
                pass
            # Ensure triangulated, processed topology
            tri = trimesh.Trimesh(vertices=tri.vertices, faces=tri.faces, process=True)
            self._update_view_with_trimesh(tri, reset_camera=False)
            self._show_info(
                "Repair complete",
                f"Watertight: {bool(tri.is_watertight)} | Faces: {len(tri.faces)} | Vertices: {len(tri.vertices)}",
            )
        except Exception as e:
            self._show_error("Mesh repair failed", e)

    def export_mesh(self) -> None:
        try:
            tri = self.state.current_mesh
            if tri is None:
                raise RuntimeError("No mesh to export. Open or generate a mesh first.")
            filters = (
                "3D Files (*.stl *.obj *.ply *.gltf *.glb);;"
                "All Files (*)"
            )
            path, _ = QFileDialog.getSaveFileName(self, "Export Mesh", "mesh.glb", filters)
            if not path:
                return
            ext = os.path.splitext(path)[1].lower().strip(".")
            # Let trimesh handle by extension
            tri.export(path, file_type=ext if ext else None)
            self._show_info("Export", f"Saved to: {path}")
        except Exception as e:
            self._show_error("Export failed", e)


def main() -> int:
    # On some systems, PyVista prefers off-screen until a widget is attached
    pv.set_error_output_file(False)
    pv.global_theme.smooth_shading = True

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
