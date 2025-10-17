import os
import sys
import time
from typing import List, Optional, Dict, Any

import numpy as np
import pydicom
import nrrd  # provided by pynrrd
from scipy import ndimage
from skimage import measure, morphology

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import pyvista as pv
import trimesh

from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QProgressBar,
    QFileDialog,
    QMessageBox,
)


def count_non_manifold_edges(tri_mesh: trimesh.Trimesh) -> int:
    try:
        edges_face_count = tri_mesh.edges_face_count  # type: ignore[attr-defined]
        return int(np.sum(edges_face_count != 2))
    except Exception:
        try:
            # Fallback heuristic in case of API changes
            return int(np.sum(tri_mesh.edges_unique_length > 2))
        except Exception:
            return -1


def find_dicom_files(folder_path: str) -> List[str]:
    dicom_files: List[str] = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith((".dcm", ".dicom")):
                dicom_files.append(os.path.join(root, file_name))
    return sorted(dicom_files)


def load_dicom_volume(dicom_files: List[str]) -> np.ndarray:
    first_ds = pydicom.dcmread(dicom_files[0])
    rows = int(first_ds.Rows)
    cols = int(first_ds.Columns)
    num_slices = len(dicom_files)
    volume = np.zeros((num_slices, rows, cols), dtype=np.float32)

    for i, file_path in enumerate(dicom_files):
        ds = pydicom.dcmread(file_path)
        if hasattr(ds, "pixel_array"):
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                pixel_array = ds.pixel_array.astype(np.float32)
                pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            else:
                pixel_array = ds.pixel_array.astype(np.float32)
            volume[i] = pixel_array
    return volume


def create_bone_segmentation(volume: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
    bone_mask = (volume >= hu_min) & (volume <= hu_max)
    bone_mask = morphology.remove_small_objects(bone_mask, min_size=1000)
    bone_mask = ndimage.binary_fill_holes(bone_mask)
    bone_mask = morphology.binary_closing(bone_mask, morphology.ball(2))
    return bone_mask.astype(np.uint8)


class LoadWorker(QObject):
    finished = pyqtSignal(object, object, str)  # volume, header, volume_type
    error = pyqtSignal(str)
    message = pyqtSignal(str)

    def __init__(self, dicom_files: List[str], nrrd_file: Optional[str]):
        super().__init__()
        self.dicom_files = dicom_files
        self.nrrd_file = nrrd_file

    def run(self) -> None:
        try:
            if self.dicom_files:
                self.message.emit("Loading DICOM series...")
                volume = load_dicom_volume(self.dicom_files)
                self.finished.emit(volume, None, "dicom")
            elif self.nrrd_file:
                self.message.emit("Loading NRRD volume...")
                data, header = nrrd.read(self.nrrd_file)
                self.finished.emit(data, header, "nrrd")
            else:
                self.error.emit("No DICOM or NRRD selected")
        except Exception as exc:
            self.error.emit(f"Failed to load volume: {exc}")


class SegmentWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    message = pyqtSignal(str)

    def __init__(self, volume: np.ndarray, hu_min: int, hu_max: int):
        super().__init__()
        self.volume = volume
        self.hu_min = hu_min
        self.hu_max = hu_max

    def run(self) -> None:
        try:
            self.message.emit("Segmenting bones...")
            segmented = create_bone_segmentation(self.volume, self.hu_min, self.hu_max)
            self.finished.emit(segmented)
        except Exception as exc:
            self.error.emit(f"Segmentation failed: {exc}")


class ExportWorker(QObject):
    finished = pyqtSignal(dict)  # summary info
    error = pyqtSignal(str)
    message = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        segmented_data: np.ndarray,
        volume_type: str,
        dicom_files: List[str],
        nrrd_header: Optional[Dict[str, Any]],
        output_path: str,
    ) -> None:
        super().__init__()
        self.segmented_data = segmented_data
        self.volume_type = volume_type
        self.dicom_files = dicom_files
        self.nrrd_header = nrrd_header
        self.output_path = output_path

    def run(self) -> None:
        try:
            steps = 4
            self.progress.emit(0)
            progress = 0
            inc = int(100 / steps)

            # Step 1: marching cubes
            self.message.emit("STEP 1: Extracting surface with Marching Cubes...")
            if self.volume_type == "nrrd" and self.nrrd_header is not None:
                space_dirs = self.nrrd_header.get(
                    "space directions", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                )
                spacing = [float(np.linalg.norm(np.array(space_dirs[i]))) for i in range(3)]
            elif self.volume_type == "dicom" and self.dicom_files:
                ds0 = pydicom.dcmread(self.dicom_files[0])
                ps = getattr(ds0, "PixelSpacing", [1.0, 1.0])
                st = getattr(ds0, "SliceThickness", 1.0)
                spacing = [float(st), float(ps[1]), float(ps[0])]
            else:
                spacing = [1.0, 1.0, 1.0]

            verts, faces, _, _ = measure.marching_cubes(
                self.segmented_data, level=0.5, spacing=spacing
            )
            progress += inc
            self.progress.emit(progress)
            self.message.emit("STEP 1 complete")
            time.sleep(0.02)

            # Step 2: create PyVista mesh & clean/smooth
            self.message.emit("STEP 2: Creating PyVista mesh and cleaning...")
            faces_with_counts = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
            pv_mesh = pv.PolyData(verts, faces_with_counts.ravel())
            try:
                pv_mesh = pv_mesh.extract_surface().triangulate().clean(tolerance=1e-3)
            except Exception:
                pass
            try:
                pv_mesh = pv_mesh.smooth(n_iter=15, relaxation_factor=0.05)
            except Exception:
                pass
            progress += inc
            self.progress.emit(progress)
            self.message.emit("STEP 2 complete")
            time.sleep(0.02)

            # Step 3: convert to trimesh and repair
            self.message.emit("STEP 3: Smoothing (Taubin), component filter, and repair...")
            try:
                pv_mesh = pv_mesh.fill_holes(1000.0)
            except Exception:
                pass

            try:
                voxel_surface = pv_mesh.smooth_taubin(n_iter=30, pass_band=0.1)
            except Exception:
                voxel_surface = pv_mesh

            try:
                tri_tmp = trimesh.Trimesh(
                    vertices=voxel_surface.points,
                    faces=voxel_surface.faces.reshape(-1, 4)[:, 1:],
                    process=False,
                )
                components = tri_tmp.split(only_watertight=False)
                if len(components) > 1:
                    tri_clean = max(components, key=lambda c: len(c.faces))
                else:
                    tri_clean = tri_tmp
            except Exception as exc:
                # Fallback to constructing directly from pv_mesh
                try:
                    pv_faces = voxel_surface.faces.reshape(-1, 4)[:, 1:]
                    tri_clean = trimesh.Trimesh(
                        vertices=voxel_surface.points, faces=pv_faces, process=False
                    )
                except Exception:
                    raise RuntimeError(f"Trimesh conversion failed: {exc}")

            tri = tri_clean
            before_nm = count_non_manifold_edges(tri)
            before_watertight = bool(tri.is_watertight)

            try:
                trimesh.repair.fix_normals(tri)
                trimesh.repair.fill_holes(tri)
                trimesh.repair.fix_inversion(tri)
                trimesh.repair.broken_faces(tri)
            except Exception:
                # continue even if repair fails
                pass

            after_nm = count_non_manifold_edges(tri)
            after_watertight = bool(tri.is_watertight)

            progress += inc
            self.progress.emit(progress)
            self.message.emit("STEP 3 complete")
            time.sleep(0.02)

            # Step 4: save STL
            self.message.emit("STEP 4: Saving STL...")
            tri.export(self.output_path, file_type="stl")

            progress = 100
            self.progress.emit(progress)
            self.message.emit("Export complete")
            self.finished.emit(
                {
                    "output": self.output_path,
                    "before_nm": before_nm,
                    "after_nm": after_nm,
                    "watertight": after_watertight,
                }
            )
        except Exception as exc:
            self.error.emit(f"STL export failed: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CT Bone Segmentation Tool (Qt)")
        self.resize(1280, 860)

        # State
        self.dicom_files: List[str] = []
        self.nrrd_file: Optional[str] = None
        self.nrrd_header: Optional[Dict[str, Any]] = None
        self.volume_type: Optional[str] = None
        self.volume_data: Optional[np.ndarray] = None
        self.segmented_data: Optional[np.ndarray] = None

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(self._build_file_group())
        layout.addWidget(self._build_hu_group())
        layout.addWidget(self._build_actions_group())
        layout.addWidget(self._build_viz_group(), stretch=1)

        self.status_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate by default
        hl = QHBoxLayout()
        hl.addWidget(self.status_label, 1)
        hl.addWidget(self.progress, 0)
        layout.addLayout(hl)

    # --- UI builders ---
    def _build_file_group(self) -> QGroupBox:
        box = QGroupBox("DICOM/NRRD File Selection")
        grid = QGridLayout(box)
        self.btn_dicom = QPushButton("Select DICOM Folder")
        self.btn_dicom.clicked.connect(self.select_dicom_folder)
        self.btn_nrrd = QPushButton("Select NRRD File")
        self.btn_nrrd.clicked.connect(self.select_nrrd_file)
        self.file_label = QLabel("No DICOM/NRRD files selected")
        grid.addWidget(self.btn_dicom, 0, 0)
        grid.addWidget(self.file_label, 0, 1)
        grid.addWidget(self.btn_nrrd, 0, 2)
        return box

    def _build_hu_group(self) -> QGroupBox:
        box = QGroupBox("Hounsfield Unit Range")
        grid = QGridLayout(box)
        grid.addWidget(QLabel("Min HU:"), 0, 0)
        self.hu_min = QSpinBox()
        self.hu_min.setRange(-2000, 4000)
        self.hu_min.setValue(150)
        grid.addWidget(self.hu_min, 0, 1)
        grid.addWidget(QLabel("Max HU:"), 0, 2)
        self.hu_max = QSpinBox()
        self.hu_max.setRange(-2000, 4000)
        self.hu_max.setValue(3000)
        grid.addWidget(self.hu_max, 0, 3)
        self.btn_preview = QPushButton("Preview Segmentation")
        self.btn_preview.clicked.connect(self.preview_segmentation)
        grid.addWidget(self.btn_preview, 0, 4)
        return box

    def _build_actions_group(self) -> QGroupBox:
        box = QGroupBox("Processing")
        hl = QHBoxLayout(box)
        self.btn_load = QPushButton("Load DICOM/NRRD Series")
        self.btn_load.clicked.connect(self.load_volume)
        self.btn_segment = QPushButton("Segment Bones")
        self.btn_segment.clicked.connect(self.segment_bones)
        self.btn_export = QPushButton("Export STL")
        self.btn_export.clicked.connect(self.export_stl)
        hl.addWidget(self.btn_load)
        hl.addWidget(self.btn_segment)
        hl.addWidget(self.btn_export)
        return box

    def _build_viz_group(self) -> QGroupBox:
        box = QGroupBox("Visualization")
        vbl = QVBoxLayout(box)
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        vbl.addWidget(self.canvas)
        return box

    # --- Helpers ---
    def set_busy(self, busy: bool, message: Optional[str] = None) -> None:
        for btn in (self.btn_dicom, self.btn_nrrd, self.btn_load, self.btn_segment, self.btn_export, self.btn_preview):
            btn.setEnabled(not busy)
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        if message is not None:
            self.status_label.setText(message)

    def info(self, text: str) -> None:
        self.status_label.setText(text)

    def warn_dialog(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def error_dialog(self, title: str, text: str) -> None:
        QMessageBox.critical(self, title, text)

    # --- Actions ---
    def select_dicom_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.dicom_files = find_dicom_files(folder)
            self.nrrd_file = None
            self.nrrd_header = None
            if self.dicom_files:
                self.file_label.setText(f"Found {len(self.dicom_files)} DICOM files")
                self.info("DICOM folder selected")
            else:
                self.file_label.setText("No DICOM files found")
                self.warn_dialog("No Files", "No DICOM files found in selected folder")

    def select_nrrd_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select NRRD File", "", "NRRD files (*.nrrd);;All files (*.*)"
        )
        if file_path:
            self.nrrd_file = file_path
            self.nrrd_header = None
            self.dicom_files = []
            base = os.path.basename(file_path)
            self.file_label.setText(f"Selected NRRD file: {base}")
            self.info("NRRD file selected")

    def load_volume(self) -> None:
        if not self.dicom_files and not self.nrrd_file:
            self.error_dialog("Error", "Please select a valid DICOM folder or NRRD file first")
            return
        self.set_busy(True, "Loading volume...")

        self.load_thread = QThread()
        self.load_worker = LoadWorker(self.dicom_files, self.nrrd_file)
        self.load_worker.moveToThread(self.load_thread)
        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.message.connect(self.info)
        self.load_worker.error.connect(self._on_load_error)
        self.load_worker.finished.connect(self._on_load_finished)
        self.load_worker.finished.connect(self.load_thread.quit)
        self.load_worker.finished.connect(self.load_worker.deleteLater)
        self.load_thread.finished.connect(self.load_thread.deleteLater)
        self.load_thread.start()

    def _on_load_error(self, text: str) -> None:
        self.set_busy(False)
        self.error_dialog("Error", text)

    def _on_load_finished(self, volume: object, header: object, volume_type: str) -> None:
        self.volume_data = volume  # type: ignore[assignment]
        self.nrrd_header = header if isinstance(header, dict) else None
        self.volume_type = volume_type
        self.set_busy(False, "Volume loaded successfully")
        QMessageBox.information(self, "Success", "Volume loaded successfully")

    def preview_segmentation(self) -> None:
        if self.volume_data is None:
            self.error_dialog("Error", "Please load DICOM/NRRD series first")
            return
        hu_min = int(self.hu_min.value())
        hu_max = int(self.hu_max.value())
        middle = int(self.volume_data.shape[0] // 2)  # type: ignore[index]
        slice_data = self.volume_data[middle]  # type: ignore[index]
        bone_mask = (slice_data >= hu_min) & (slice_data <= hu_max)
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 3, 1)
        ax1.imshow(slice_data, cmap="gray")
        ax1.set_title("Original CT Slice")
        ax1.axis("off")
        ax2 = self.fig.add_subplot(1, 3, 2)
        ax2.imshow(bone_mask, cmap="gray")
        ax2.set_title("Bone Segmentation")
        ax2.axis("off")
        ax3 = self.fig.add_subplot(1, 3, 3)
        ax3.imshow(slice_data, cmap="gray", alpha=0.7)
        ax3.imshow(bone_mask, cmap="Reds", alpha=0.3)
        ax3.set_title("Overlay")
        ax3.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()

    def segment_bones(self) -> None:
        if self.volume_data is None:
            self.error_dialog("Error", "Please load DICOM/NRRD series first")
            return
        hu_min = int(self.hu_min.value())
        hu_max = int(self.hu_max.value())
        self.set_busy(True, "Segmenting bones...")
        self.seg_thread = QThread()
        self.seg_worker = SegmentWorker(self.volume_data, hu_min, hu_max)  # type: ignore[arg-type]
        self.seg_worker.moveToThread(self.seg_thread)
        self.seg_thread.started.connect(self.seg_worker.run)
        self.seg_worker.message.connect(self.info)
        self.seg_worker.error.connect(self._on_seg_error)
        self.seg_worker.finished.connect(self._on_seg_finished)
        self.seg_worker.finished.connect(self.seg_thread.quit)
        self.seg_worker.finished.connect(self.seg_worker.deleteLater)
        self.seg_thread.finished.connect(self.seg_thread.deleteLater)
        self.seg_thread.start()

    def _on_seg_error(self, text: str) -> None:
        self.set_busy(False)
        self.error_dialog("Error", text)

    def _on_seg_finished(self, segmented: object) -> None:
        self.segmented_data = segmented  # type: ignore[assignment]
        self.set_busy(False, "Bone segmentation completed")
        QMessageBox.information(self, "Success", "Bone segmentation completed")

    def export_stl(self) -> None:
        if self.segmented_data is None:
            self.error_dialog("Error", "Please segment bones first")
            return
        if self.volume_type is None:
            self.error_dialog("Error", "Volume type unknown; load data first")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save STL as...", "", "STL files (*.stl);;All files (*.*)"
        )
        if not out_path:
            return
        self.set_busy(True, "Exporting STL...")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.exp_thread = QThread()
        self.exp_worker = ExportWorker(
            segmented_data=self.segmented_data,  # type: ignore[arg-type]
            volume_type=self.volume_type,
            dicom_files=self.dicom_files,
            nrrd_header=self.nrrd_header,
            output_path=out_path,
        )
        self.exp_worker.moveToThread(self.exp_thread)
        self.exp_thread.started.connect(self.exp_worker.run)
        self.exp_worker.message.connect(self.info)
        self.exp_worker.progress.connect(self.progress.setValue)
        self.exp_worker.error.connect(self._on_export_error)
        self.exp_worker.finished.connect(self._on_export_finished)
        self.exp_worker.finished.connect(self.exp_thread.quit)
        self.exp_worker.finished.connect(self.exp_worker.deleteLater)
        self.exp_thread.finished.connect(self.exp_thread.deleteLater)
        self.exp_thread.start()

    def _on_export_error(self, text: str) -> None:
        self.set_busy(False)
        self.error_dialog("Error", text)

    def _on_export_finished(self, info: dict) -> None:
        self.set_busy(False, "Export complete")
        QMessageBox.information(
            self,
            "STL Export Complete",
            (
                f"STL saved to:\n{info.get('output')}\n\n"
                f"Before repair: {info.get('before_nm')} non-manifold edges\n"
                f"After repair:  {info.get('after_nm')} non-manifold edges\n"
                f"Watertight: {info.get('watertight')}"
            ),
        )


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
