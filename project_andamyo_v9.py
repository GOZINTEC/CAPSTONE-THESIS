"""
project_andamyo_v9.py
=====================
Two-Tab Clinical CT Bone Segmentation, 3D Implicit Osteotomy Cutting,
Directional Sub-Voxel TPMS & Strut-Lattice Workstation (v9)

Architecture Overview:
    - Tab 1 ("Segmentation"):
        * 2×2 Mimics-style MPR Quad-Viewport (Coronal, Axial, Sagittal + 3D)
        * ±90° 2D viewport planar reorientation with rotating crosshairs & labels
        * Real-time HU Thresholding with "Keep Largest Island Only" toggle and
          automatic background volume-preserving Taubin 3D surface preview
        * Anatomical cavity & medullary canal preservation (no 3D hole-filling)
        * Segmented host bone 3MF / STL export
    - Tab 2 ("Scaffold Design"):
        * Dedicated full-height interactive 3D viewport (EDL + depth peeling)
        * Interactive 3D Volumetric Implicit Cutting Tools (two-way synced):
            1. No Cut (Entire Segmented Bone)
            2. Cut by Rectangle Primitive (Oriented Box ROI via VTK BoxWidget)
            3. Cut by Dual Planes (Osteotomy Wedge / Resection via VTK LineWidget
               + individual non-parallel Plane 1 / Plane 2 tilt angles)
          with automatic Post-Cut Largest Island enforcement, automatic gizmo
          hiding upon scaffold synthesis, and automatic pattern reset on new cuts
        * Set Pattern Direction Tool:
            - Interactive 3D orientation vector arrow at the ROI centroid
            - 3×3 SO(3) coordinate rotation X' = R^T (X - X_c) aligning TPMS or
              Strut unit cells along arbitrary surgical / load-bearing axes
        * Anti-aliased local sub-voxel isotropic resampling (Δ_iso ∈ [40, 120] µm)
          with resampled defect envelope caching, multi-threaded VTK Flying Edges
          surface extraction (`vtkFlyingEdges3D` + `vtkStaticCleanPolyData`),
          pristine 20-iteration Taubin smoothing, dual 3MF / STL export with
          explicit millimeter units, and optional separate-file Host Bone Overlay
          export (automatically exporting every disconnected host bone piece as
          its own companion file) for surgical fit validation & 3D printing
"""

import os
import sys
import math
import time
import uuid
import zipfile
import hashlib
import multiprocessing
from xml.sax.saxutils import escape as xml_escape
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
import pyvista as pv
import vtk
import trimesh

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QSplitter, QTabWidget, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox, QFrame,
    QScrollArea, QSizePolicy, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QImage, QPainter, QColor, QPen, QFont, QIcon

from pyvistaqt import QtInteractor


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WL = 400
DEFAULT_WW = 1800
BONE_COLOR = [0.95, 0.93, 0.90]          # RGB – anatomical bone-white / ivory
SCAFFOLD_COLOR = [0.0, 0.82, 1.0]        # RGB – titanium / surgical implant cyan
OVERLAY_BONE_COLOR = [0.95, 0.93, 0.90]  # Host bone outside cut in overlay mode
ARROW_COLOR = [1.0, 0.78, 0.10]          # Gold vector arrow for pattern direction
PLANE1_COLOR = [1.0, 0.55, 0.15]         # Proximal / Plane 1 translucent tint
PLANE2_COLOR = [0.25, 0.85, 1.0]         # Distal / Plane 2 translucent tint
BOX_GIZMO_COLOR = [0.15, 0.90, 0.65]     # Oriented box translucent preview tint
DEFAULT_BONE_ALPHA = 0.35

# Pore size configuration (micrometers)
MIN_PORE_UM = 50
MAX_PORE_UM = 1500
DEFAULT_PORE_UM = 500
STEP_PORE_UM = 25

# Strut diameter configuration (micrometers)
MIN_STRUT_UM = 50
MAX_STRUT_UM = 800
DEFAULT_STRUT_UM = 200
STEP_STRUT_UM = 25

# Sub-voxel isotropic resampling bounds (millimeters)
MIN_DELTA_ISO_MM = 0.04
MAX_DELTA_ISO_MM = 0.12
MAX_HIGHRES_VOXELS = 45_000_000
MAX_PREVIEW_VOXELS = 35_000_000

# Cutting tool modes
CUT_MODE_NONE = "No Cut (Entire Segmented Bone)"
CUT_MODE_BOX = "Cut by Rectangle Primitive (Box ROI)"
CUT_MODE_PLANES = "Cut by Dual Planes (Osteotomy Wedge / Resection)"
ALL_CUT_MODES = [CUT_MODE_NONE, CUT_MODE_BOX, CUT_MODE_PLANES]

# Scaffold target envelope modes
TARGET_MODE_ANATOMICAL = (
    "Anatomical Cortical Bone (Preserve Medullary Canal & Cavities)"
)
TARGET_MODE_SOLID_GRAFT = (
    "Solid Resection Graft (Fill Defect Cavity)"
)
SCAFFOLD_TARGET_MODES = [
    TARGET_MODE_ANATOMICAL,
    TARGET_MODE_SOLID_GRAFT,
]

# HU preset tuples: (hu_min, hu_max, wl, ww)
HU_PRESETS = {
    "Bone":           (150,  3000, 400,  1800),
    "Cortical Bone":  (500,  3000, 1000, 2500),
    "Cancellous":     (150,  500,  325,  350),
    "Soft Tissue":    (-200, 400,  40,   400),
}

TPMS_PATTERNS = [
    "Gyroid", "Primitive", "Diamond",
    "Lidinoid", "Split-P", "Schwarz-P",
]

STRUT_PATTERNS = [
    "Cubic", "BCC", "BCCZ", "FCC",
]

ALL_SCAFFOLD_PATTERNS = ["None"] + TPMS_PATTERNS + STRUT_PATTERNS

TPMS_ALPHA_FACTORS = {
    "Gyroid":    1.35,
    "Primitive": 1.25,
    "Schwarz-P": 1.25,
    "Diamond":   1.45,
    "Lidinoid":  1.60,
    "Split-P":   1.50,
}

TPMS_ANALYTICAL_BOUNDS = {
    "Gyroid":    1.5,
    "Primitive": 3.0,
    "Schwarz-P": 3.0,
    "Diamond":   2.0,
    "Lidinoid":  2.5,
    "Split-P":   2.5,
}

COLOR_CORONAL  = QColor(0, 200, 0)
COLOR_AXIAL    = QColor(220, 40, 40)
COLOR_SAGITTAL = QColor(60, 120, 255)
COLOR_3D       = QColor(0, 200, 200)
MASK_RGBA      = (255, 80, 0)

# 3D additive manufacturing export formats
EXPORT_FMT_3MF_LABEL = "3MF (.3mf) [Recommended - Compressed & Watertight]"
EXPORT_FMT_STL_LABEL = "STL (.stl) [Standard Mesh]"
ALL_EXPORT_FORMATS = [EXPORT_FMT_3MF_LABEL, EXPORT_FMT_STL_LABEL]
FILE_FILTER_3MF = "3MF files (*.3mf)"
FILE_FILTER_STL = "STL files (*.stl)"
EXPORT_FILE_FILTERS = f"{FILE_FILTER_3MF};;{FILE_FILTER_STL}"


# ══════════════════════════════════════════════════════════════════════════════
#  3D ROTATION, IMPLICIT CUTTING & ORIENTATION MATHEMATICS
# ══════════════════════════════════════════════════════════════════════════════

def euler_zyx_to_matrix(yaw_deg, pitch_deg, roll_deg):
    """
    Construct a 3×3 right-handed rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    from intrinsic Z-Y-X Euler angles (in degrees).
    """
    rz = math.radians(float(yaw_deg))
    ry = math.radians(float(pitch_deg))
    rx = math.radians(float(roll_deg))

    cz, sz = math.cos(rz), math.sin(rz)
    cy, sy = math.cos(ry), math.sin(ry)
    cx, sx = math.cos(rx), math.sin(rx)

    Rz = np.array([[cz, -sz, 0.0],
                   [sz,  cz, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[ cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cx, -sx],
                   [0.0,  sx,  cx]], dtype=np.float64)
    return Rz @ Ry @ Rx


def matrix_to_euler_zyx(R):
    """
    Extract (yaw_deg, pitch_deg, roll_deg) in degrees from a 3×3 rotation matrix
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll), with SVD orthogonalization for stability.
    """
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    R_orth = U @ Vt
    if np.linalg.det(R_orth) < 0:
        U[:, -1] *= -1.0
        R_orth = U @ Vt

    sin_p = float(np.clip(-R_orth[2, 0], -1.0, 1.0))
    pitch = math.degrees(math.asin(sin_p))
    if abs(sin_p) < 0.99999:
        yaw = math.degrees(math.atan2(R_orth[1, 0], R_orth[0, 0]))
        roll = math.degrees(math.atan2(R_orth[2, 1], R_orth[2, 2]))
    else:
        yaw = math.degrees(math.atan2(-R_orth[0, 1], R_orth[1, 1]))
        roll = 0.0
    return float(yaw), float(pitch), float(roll)


def spherical_angles_to_unit_vec(yaw_deg, pitch_deg):
    """
    Convert azimuth/yaw ∈ [-180°, 180°] and elevation/pitch ∈ [-90°, 90°]
    to a 3D unit vector (nx, ny, nz).
    Note: (yaw=0°, pitch=90°) -> (0, 0, 1) (+Z anatomical longitudinal axis).
    """
    az = math.radians(float(yaw_deg))
    el = math.radians(float(pitch_deg))
    ce = math.cos(el)
    vec = np.array([
        ce * math.cos(az),
        ce * math.sin(az),
        math.sin(el),
    ], dtype=np.float64)
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-12)


def unit_vec_to_spherical_angles(vec):
    """
    Recover (yaw_deg, pitch_deg) ∈ [-180°, 180°] × [-90°, 90°] from a 3D vector.
    """
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return 0.0, 90.0
    v = v / n
    pitch_deg = math.degrees(math.asin(float(np.clip(v[2], -1.0, 1.0))))
    if abs(v[2]) > 0.99999:
        yaw_deg = 0.0
    else:
        yaw_deg = math.degrees(math.atan2(float(v[1]), float(v[0])))
    return float(yaw_deg), float(pitch_deg)


def build_local_tangent_basis(n_hat):
    """
    Build a smooth right-handed orthonormal basis (u_hat, v_hat, n_hat)
    given unit vector n_hat.
    """
    n = np.asarray(n_hat, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return u, v, n


def rodrigues_rotate(vec, axis, angle_deg):
    """Rotate 3D vector `vec` about unit `axis` by `angle_deg` degrees."""
    rad = math.radians(float(angle_deg))
    if abs(rad) < 1e-12:
        return np.asarray(vec, dtype=np.float64).copy()
    k = np.asarray(axis, dtype=np.float64)
    k = k / max(float(np.linalg.norm(k)), 1e-12)
    v = np.asarray(vec, dtype=np.float64)
    c, s = math.cos(rad), math.sin(rad)
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def pattern_direction_to_matrix(yaw_deg=0.0, pitch_deg=90.0, roll_deg=0.0):
    """
    Construct a 3×3 SO(3) rotation matrix R(yaw, pitch, roll) aligning the
    scaffold's primary Z' axis with the pattern direction vector (yaw, pitch)
    and spinning around that vector by `roll_deg`:
        R = Rz(yaw) @ Ry(90° - pitch) @ Rz(roll)
    At default (yaw=0°, pitch=90°, roll=0°), R = I_3 (identity).
    """
    rz1 = math.radians(float(yaw_deg))
    ry  = math.radians(90.0 - float(pitch_deg))
    rz2 = math.radians(float(roll_deg))

    c1, s1 = math.cos(rz1), math.sin(rz1)
    cy, sy = math.cos(ry),  math.sin(ry)
    c2, s2 = math.cos(rz2), math.sin(rz2)

    Rz1 = np.array([[c1, -s1, 0.0],
                    [s1,  c1, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[ cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=np.float64)
    Rz2 = np.array([[c2, -s2, 0.0],
                    [s2,  c2, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz1 @ Ry @ Rz2


def compute_dual_planes_geometry(cut_params):
    """
    Given dual-planes cut parameters dict, compute:
        midpoint  : (3,) float64
        n_line    : (3,) unit normal along connecting line (p1 -> p2)
        p1, p2    : (3,) anchor centers of Plane 1 and Plane 2
        n1, n2    : (3,) tilted plane normals such that the interior slab/wedge
                    satisfies: (x - p1)·n1 >= 0  AND  (x - p2)·n2 <= 0
    """
    mx = float(cut_params.get("mid_x", 0.0))
    my = float(cut_params.get("mid_y", 0.0))
    mz = float(cut_params.get("mid_z", 0.0))
    gap = max(0.2, float(cut_params.get("gap_l", 15.0)))
    line_yaw = float(cut_params.get("line_yaw", 0.0))
    line_pitch = float(cut_params.get("line_pitch", 90.0))

    p1_tilt_a = float(cut_params.get("p1_tilt", 0.0))
    p1_tilt_b = float(cut_params.get("p1_cross_tilt", 0.0))
    p2_tilt_a = float(cut_params.get("p2_tilt", 0.0))
    p2_tilt_b = float(cut_params.get("p2_cross_tilt", 0.0))

    mid = np.array([mx, my, mz], dtype=np.float64)
    u_hat, v_hat, n_line = build_local_tangent_basis(
        spherical_angles_to_unit_vec(line_yaw, line_pitch)
    )

    p1 = mid - 0.5 * gap * n_line
    p2 = mid + 0.5 * gap * n_line

    # Tilt Plane 1 and Plane 2 normals relative to connecting line n_line
    n1 = rodrigues_rotate(n_line, v_hat, p1_tilt_a)
    n1 = rodrigues_rotate(n1, u_hat, p1_tilt_b)
    n1 = n1 / max(float(np.linalg.norm(n1)), 1e-12)

    n2 = rodrigues_rotate(n_line, v_hat, p2_tilt_a)
    n2 = rodrigues_rotate(n2, u_hat, p2_tilt_b)
    n2 = n2 / max(float(np.linalg.norm(n2)), 1e-12)

    return mid, n_line, p1, p2, n1, n2


def evaluate_cut_implicit_phi(X_mm, Y_mm, Z_mm, cut_mode, cut_params):
    """
    Evaluate the analytical 3D implicit cutting signed distance field Φ_cut(X, Y, Z)
    in physical millimeter coordinates:
        Φ_cut >= 0  inside the selected cutting ROI
        Φ_cut <  0  outside the cutting ROI

    Supports:
        1. CUT_MODE_BOX:    -SDF_box(X, Y, Z) (exact Oriented Bounding Box SDF)
        2. CUT_MODE_PLANES: min((x - p1)·n1, -(x - p2)·n2) (Dual-Plane Wedge/Slab)
    """
    if cut_mode == CUT_MODE_BOX and cut_params is not None:
        cx = float(cut_params.get("cx", 0.0))
        cy = float(cut_params.get("cy", 0.0))
        cz = float(cut_params.get("cz", 0.0))
        lx = max(0.2, float(cut_params.get("lx", 20.0)))
        ly = max(0.2, float(cut_params.get("ly", 20.0)))
        lz = max(0.2, float(cut_params.get("lz", 20.0)))
        yaw = float(cut_params.get("yaw", 0.0))
        pitch = float(cut_params.get("pitch", 0.0))
        roll = float(cut_params.get("roll", 0.0))

        R = euler_zyx_to_matrix(yaw, pitch, roll)
        Rt = R.T.astype(np.float32)

        dX = X_mm - np.float32(cx)
        dY = Y_mm - np.float32(cy)
        dZ = Z_mm - np.float32(cz)

        qx = np.abs(Rt[0, 0] * dX + Rt[0, 1] * dY + Rt[0, 2] * dZ) - np.float32(0.5 * lx)
        qy = np.abs(Rt[1, 0] * dX + Rt[1, 1] * dY + Rt[1, 2] * dZ) - np.float32(0.5 * ly)
        qz = np.abs(Rt[2, 0] * dX + Rt[2, 1] * dY + Rt[2, 2] * dZ) - np.float32(0.5 * lz)

        out_x = np.maximum(qx, np.float32(0.0))
        out_y = np.maximum(qy, np.float32(0.0))
        out_z = np.maximum(qz, np.float32(0.0))
        outside_dist = np.sqrt(out_x * out_x + out_y * out_y + out_z * out_z)
        inside_dist = np.minimum(np.maximum(qx, np.maximum(qy, qz)), np.float32(0.0))
        sdf_box = outside_dist + inside_dist
        return (-sdf_box).astype(np.float32)

    elif cut_mode == CUT_MODE_PLANES and cut_params is not None:
        _, _, p1, p2, n1, n2 = compute_dual_planes_geometry(cut_params)
        d1 = (
            (X_mm - np.float32(p1[0])) * np.float32(n1[0])
            + (Y_mm - np.float32(p1[1])) * np.float32(n1[1])
            + (Z_mm - np.float32(p1[2])) * np.float32(n1[2])
        )
        d2 = -(
            (X_mm - np.float32(p2[0])) * np.float32(n2[0])
            + (Y_mm - np.float32(p2[1])) * np.float32(n2[1])
            + (Z_mm - np.float32(p2[2])) * np.float32(n2[2])
        )
        return np.minimum(d1, d2).astype(np.float32)

    return np.ones_like(X_mm, dtype=np.float32)


def enforce_largest_island_3d(mask_bool):
    """Keep strictly the single largest 3D connected component of a binary mask."""
    b = mask_bool > 0
    if not np.any(b):
        return np.zeros(mask_bool.shape, dtype=np.uint8)
    labeled, ncomp = ndimage.label(b)
    if ncomp > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        b = (labeled == np.argmax(counts))
    return b.astype(np.uint8)


def compute_mask_centroid_xyz(mask, spacing_zyx):
    """Compute physical geometric centroid (Xc, Yc, Zc) in mm of foreground voxels."""
    nz = np.nonzero(mask > 0)
    if len(nz[0]) == 0:
        return (0.0, 0.0, 0.0)
    sz, sy, sx = (float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2]))
    zc = float(nz[0].mean()) * sz
    yc = float(nz[1].mean()) * sy
    xc = float(nz[2].mean()) * sx
    return (xc, yc, zc)


def compute_cut_roi_and_host_masks(seg_bone_mask, spacing_zyx, cut_mode, cut_params):
    """
    Evaluate the 3D analytical implicit cutting tool (No Cut, Oriented Box,
    or Dual-Plane Wedge) over `seg_bone_mask` in memory-safe Z-chunks and
    enforce Post-Cut Largest Island isolation on the active ROI.

    Returns
    -------
    (roi_mask_uint8, host_bone_mask_uint8, roi_centroid_xyz)
    """
    bone_bool = seg_bone_mask > 0
    if not np.any(bone_bool):
        empty = np.zeros_like(seg_bone_mask, dtype=np.uint8)
        return empty, empty.copy(), (0.0, 0.0, 0.0)

    if cut_mode in (None, CUT_MODE_NONE) or cut_params is None:
        roi_u8 = bone_bool.astype(np.uint8)
        host_u8 = np.zeros_like(roi_u8)
        centroid = compute_mask_centroid_xyz(roi_u8, spacing_zyx)
        return roi_u8, host_u8, centroid

    sz, sy, sx = (float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2]))
    zdim, ydim, xdim = bone_bool.shape
    zz_mm = np.arange(zdim, dtype=np.float32) * np.float32(sz)
    yy_mm = np.arange(ydim, dtype=np.float32) * np.float32(sy)
    xx_mm = np.arange(xdim, dtype=np.float32) * np.float32(sx)

    inside_cut_bone = np.zeros(bone_bool.shape, dtype=bool)
    chunk_z = max(1, min(48, zdim))
    for z0 in range(0, zdim, chunk_z):
        z1 = min(zdim, z0 + chunk_z)
        sub_b = bone_bool[z0:z1]
        if not np.any(sub_b):
            continue
        Z_c, Y_c, X_c = np.meshgrid(zz_mm[z0:z1], yy_mm, xx_mm, indexing="ij")
        phi_c = evaluate_cut_implicit_phi(X_c, Y_c, Z_c, cut_mode, cut_params)
        inside_cut_bone[z0:z1] = sub_b & (phi_c >= 0.0)

    # Post-Cut Largest Island Enforcement (discards peripheral sliced fragments)
    roi_u8 = enforce_largest_island_3d(inside_cut_bone)
    host_u8 = (bone_bool & ~(roi_u8 > 0)).astype(np.uint8)
    centroid = compute_mask_centroid_xyz(roi_u8, spacing_zyx)
    return roi_u8, host_u8, centroid


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY, MORPHOLOGY & LATTICE GENERATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_windowing(slice_data, wl, ww):
    """Apply Window/Level to a 2D HU array, returning uint8 grayscale."""
    vmin = wl - ww / 2.0
    vmax = wl + ww / 2.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    out = ((slice_data.astype(np.float32) - vmin) / (vmax - vmin) * 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def numpy_to_qimage_rgba(arr_rgba):
    """Convert an (H, W, 4) uint8 RGBA numpy array to a QImage."""
    h, w = arr_rgba.shape[:2]
    arr_rgba = np.ascontiguousarray(arr_rgba)
    return QImage(arr_rgba.data, w, h, w * 4,
                  QImage.Format_RGBA8888).copy()


def _clean_anatomical_bone_region(mask_bool, min_size=500):
    """
    Clean a binary bone mask by filtering small disconnected speckle noise and
    bridging 1-voxel micro-cracks via zero-padded radius-1 spherical closing,
    WITHOUT ever calling binary_fill_holes so medullary canals and biological
    cavities remain strictly open voids (0).
    """
    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=bool)

    cleaned = mask_bool
    if min_size > 0 and int(mask_bool.sum()) > min_size:
        filtered = morphology.remove_small_objects(mask_bool, min_size=min_size)
        if np.any(filtered):
            cleaned = filtered

    padded = np.pad(cleaned, 2, mode="constant", constant_values=False)
    closed = morphology.binary_closing(padded, morphology.ball(1))[2:-2, 2:-2, 2:-2]
    return closed


def create_bone_mask(vol, hu_min, hu_max, keep_largest=True):
    """
    Create anatomical bone mask from HU volume preserving medullary canals
    and biological cavities. If `keep_largest=True`, isolates only the single
    largest connected bone island; otherwise preserves all bone islands.
    """
    m = (vol >= hu_min) & (vol <= hu_max)
    if not np.any(m):
        return np.zeros_like(vol, dtype=np.uint8)

    min_sz = 500 if keep_largest else 64
    m = _clean_anatomical_bone_region(m, min_size=min_sz)

    if keep_largest:
        return enforce_largest_island_3d(m)
    return m.astype(np.uint8)


def create_bone_mask_keep_largest(vol, hu_min, hu_max):
    """Backwards-compatible wrapper isolating largest anatomical bone structure."""
    return create_bone_mask(vol, hu_min, hu_max, keep_largest=True)


def fill_osteotomy_defect_envelope(mask_crop):
    """
    Convert a hollowed/trabecular cropped bone mask into a solid resection graft
    implant envelope by sealing cortical breaches, filling internal 3D pockets,
    and filling 2D axial medullary canals along the resection axis.
    """
    mask_bool = mask_crop > 0
    if not np.any(mask_bool):
        return mask_crop.astype(np.uint8)

    filled = ndimage.binary_fill_holes(mask_bool)
    for z in range(filled.shape[0]):
        if np.any(filled[z]):
            filled[z] = ndimage.binary_fill_holes(filled[z])

    radius = 3
    pad = radius + 1
    padded = np.pad(filled, pad, mode="constant", constant_values=False)
    closed_pad = morphology.binary_closing(padded, morphology.ball(radius))
    closed = closed_pad[pad:-pad, pad:-pad, pad:-pad]

    for z in range(closed.shape[0]):
        if np.any(closed[z]):
            closed[z] = ndimage.binary_fill_holes(closed[z])
    solid = ndimage.binary_fill_holes(closed)

    return solid.astype(np.uint8)


def is_strut_pattern(pattern_name):
    return pattern_name in STRUT_PATTERNS


def compute_target_delta_iso_mm(pore_size_um, strut_diam_um=None, pattern_name="None"):
    """
    Compute nominal target isotropic voxel spacing Δ_iso (in mm):
        Δ_iso = clip(D_p / 6.0, 0.04 mm, 0.12 mm)
    """
    dp_mm = max(0.01, float(pore_size_um) / 1000.0)
    base_delta = dp_mm / 6.0
    if is_strut_pattern(pattern_name) and strut_diam_um is not None:
        d_strut_mm = max(0.01, float(strut_diam_um) / 1000.0)
        base_delta = min(base_delta, d_strut_mm / 3.0)
    return float(np.clip(base_delta, MIN_DELTA_ISO_MM, MAX_DELTA_ISO_MM))


def dist_segment(px, py, pz, ax, ay, az, bx, by, bz):
    """Analytical Euclidean distance from 3D points (px,py,pz) to segment A-B."""
    abx, aby, abz = bx - ax, by - ay, bz - az
    ab_len_sq = abx * abx + aby * aby + abz * abz
    if ab_len_sq < 1e-12:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)

    apx, apy, apz = px - ax, py - ay, pz - az
    t = np.clip((apx * abx + apy * aby + apz * abz) / ab_len_sq, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    cz = az + t * abz
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)


def compute_unit_cell_period_mm(pattern_name, tpms_type, pore_size_um,
                                level_set=0.0, thickness=0.10,
                                strut_diam_um=DEFAULT_STRUT_UM):
    """Compute physical unit-cell period L (in mm) for target pore size."""
    dp_mm = max(0.01, float(pore_size_um) / 1000.0)
    d_strut_mm = max(0.01, float(strut_diam_um) / 1000.0)

    if is_strut_pattern(pattern_name):
        if pattern_name == "Cubic":
            L_mm = dp_mm + d_strut_mm
        elif pattern_name in ("BCC", "BCCZ"):
            L_mm = 1.25 * dp_mm + d_strut_mm * np.sqrt(2.0)
        elif pattern_name == "FCC":
            L_mm = np.sqrt(2.0) * dp_mm + d_strut_mm
        else:
            L_mm = dp_mm + d_strut_mm
        return max(0.05, float(L_mm))

    alpha = TPMS_ALPHA_FACTORS.get(pattern_name, 1.35)
    if tpms_type == "Shell":
        void_fraction = max(0.25, 1.0 - 1.5 * float(thickness))
        L_mm = (alpha * dp_mm) / void_fraction
    else:
        void_mod = np.clip(1.0 - 0.15 * float(level_set), 0.55, 1.65)
        L_mm = alpha * dp_mm * void_mod

    return max(0.05, float(L_mm))


def evaluate_strut_lattice_sdf(X_mm, Y_mm, Z_mm, pattern_name, L_mm):
    """Evaluate periodic strut lattice SDF for Cubic, BCC, BCCZ, or FCC."""
    h = 0.5 * L_mm
    u = np.abs(np.mod(X_mm, L_mm) - h)
    v = np.abs(np.mod(Y_mm, L_mm) - h)
    w = np.abs(np.mod(Z_mm, L_mm) - h)

    if pattern_name == "Cubic":
        d_edge_x = dist_segment(u, v, w, 0.0, h, h, h, h, h)
        d_edge_y = dist_segment(u, v, w, h, 0.0, h, h, h, h)
        d_edge_z = dist_segment(u, v, w, h, h, 0.0, h, h, h)
        d_axis_x = dist_segment(u, v, w, 0.0, 0.0, 0.0, h, 0.0, 0.0)
        d_axis_y = dist_segment(u, v, w, 0.0, 0.0, 0.0, 0.0, h, 0.0)
        d_axis_z = dist_segment(u, v, w, 0.0, 0.0, 0.0, 0.0, 0.0, h)
        return np.minimum.reduce([
            d_edge_x, d_edge_y, d_edge_z,
            d_axis_x, d_axis_y, d_axis_z
        ])

    elif pattern_name == "BCC":
        return dist_segment(u, v, w, 0.0, 0.0, 0.0, h, h, h)

    elif pattern_name == "BCCZ":
        d_bcc = dist_segment(u, v, w, 0.0, 0.0, 0.0, h, h, h)
        d_z_edge = dist_segment(u, v, w, h, h, 0.0, h, h, h)
        d_z_center = dist_segment(u, v, w, 0.0, 0.0, 0.0, 0.0, 0.0, h)
        return np.minimum(d_bcc, np.minimum(d_z_edge, d_z_center))

    elif pattern_name == "FCC":
        d_fx1 = dist_segment(u, v, w, h, 0.0, 0.0, h, h, h)
        d_fx2 = dist_segment(u, v, w, h, h, 0.0, h, 0.0, h)
        d_fy1 = dist_segment(u, v, w, 0.0, h, 0.0, h, h, h)
        d_fy2 = dist_segment(u, v, w, 0.0, h, h, 0.0, 0.0, h)
        d_fz1 = dist_segment(u, v, w, 0.0, 0.0, h, h, h, h)
        d_fz2 = dist_segment(u, v, w, h, 0.0, h, 0.0, h, h)
        return np.minimum.reduce([d_fx1, d_fx2, d_fy1, d_fy2, d_fz1, d_fz2])

    return np.zeros_like(u)


def evaluate_tpms_field(Xs, Ys, Zs, pattern_name):
    """Evaluate raw analytical TPMS implicit function F(X, Y, Z)."""
    if pattern_name == "Gyroid":
        return (np.cos(Xs) * np.sin(Ys)
                + np.cos(Ys) * np.sin(Zs)
                + np.cos(Zs) * np.sin(Xs))

    elif pattern_name in ("Primitive", "Schwarz-P"):
        return np.cos(Xs) + np.cos(Ys) + np.cos(Zs)

    elif pattern_name == "Diamond":
        return (np.sin(Xs) * np.sin(Ys) * np.sin(Zs)
                + np.sin(Xs) * np.cos(Ys) * np.cos(Zs)
                + np.cos(Xs) * np.sin(Ys) * np.cos(Zs)
                + np.cos(Xs) * np.cos(Ys) * np.sin(Zs))

    elif pattern_name == "Lidinoid":
        return (np.sin(2*Xs)*np.cos(Ys)*np.sin(Zs)
                + np.sin(Xs)*np.sin(2*Ys)*np.cos(Zs)
                + np.cos(Xs)*np.sin(Ys)*np.sin(2*Zs)
                - np.cos(2*Xs)*np.cos(2*Ys)
                - np.cos(2*Ys)*np.cos(2*Zs)
                - np.cos(2*Zs)*np.cos(2*Xs) + 0.3)

    elif pattern_name == "Split-P":
        return (1.1 * (np.sin(2*Xs)*np.cos(Ys)*np.sin(Zs)
                       + np.sin(Xs)*np.sin(2*Ys)*np.cos(Zs)
                       + np.cos(Xs)*np.sin(Ys)*np.sin(2*Zs))
                - 0.2 * (np.cos(2*Xs)*np.cos(2*Ys)
                         + np.cos(2*Ys)*np.cos(2*Zs)
                         + np.cos(2*Zs)*np.cos(2*Xs))
                - 0.4 * (np.cos(2*Xs) + np.cos(2*Ys) + np.cos(2*Zs)))

    return np.zeros_like(Xs)


def _transform_coords_chunk(X_c, Y_c, Z_c, rot_matrix=None, centroid_xyz=(0.0, 0.0, 0.0)):
    """
    Transform physical chunk coordinates (X_c, Y_c, Z_c) into the rotated
    pattern frame centered at `centroid_xyz`:
        [X', Y', Z']^T = R^T @ [X - X_c, Y - Y_c, Z - Z_c]^T
    """
    xc, yc, zc = (
        np.float32(centroid_xyz[0]),
        np.float32(centroid_xyz[1]),
        np.float32(centroid_xyz[2]),
    )
    dX = X_c - xc
    dY = Y_c - yc
    dZ = Z_c - zc
    if rot_matrix is None:
        return dX, dY, dZ

    Rt = np.asarray(rot_matrix, dtype=np.float32).T
    if np.allclose(Rt, np.eye(3, dtype=np.float32), atol=1e-6):
        return dX, dY, dZ

    Xp = Rt[0, 0] * dX + Rt[0, 1] * dY + Rt[0, 2] * dZ
    Yp = Rt[1, 0] * dX + Rt[1, 1] * dY + Rt[1, 2] * dZ
    Zp = Rt[2, 0] * dX + Rt[2, 1] * dY + Rt[2, 2] * dZ
    return Xp, Yp, Zp


def apply_scaffold_on_physical_grid(mask_bool, zz_mm, yy_mm, xx_mm,
                                    pattern_name, tpms_type, pore_size_um,
                                    level_set, thickness,
                                    strut_diam_um=DEFAULT_STRUT_UM,
                                    invert=False,
                                    rot_matrix=None,
                                    centroid_xyz=(0.0, 0.0, 0.0)):
    """
    Evaluate oriented TPMS or Strut Lattice SDF in memory-safe Z-chunks over
    physical coordinate axes (zz_mm, yy_mm, xx_mm) and intersect with mask_bool.
    """
    if pattern_name in (None, "None") or not np.any(mask_bool):
        return mask_bool.astype(np.uint8)

    zdim = mask_bool.shape[0]
    L_mm = compute_unit_cell_period_mm(
        pattern_name, tpms_type, pore_size_um,
        level_set=level_set, thickness=thickness,
        strut_diam_um=strut_diam_um,
    )

    out_mask = np.zeros(mask_bool.shape, dtype=np.uint8)
    chunk_z = max(1, min(32, zdim))

    if is_strut_pattern(pattern_name):
        r_strut_mm = max(0.025, float(strut_diam_um) / 2000.0)
        for z0 in range(0, zdim, chunk_z):
            z1 = min(zdim, z0 + chunk_z)
            sub_bone = mask_bool[z0:z1]
            if not np.any(sub_bone):
                continue
            Z_c, Y_c, X_c = np.meshgrid(
                zz_mm[z0:z1], yy_mm, xx_mm, indexing="ij"
            )
            Xp, Yp, Zp = _transform_coords_chunk(
                X_c, Y_c, Z_c, rot_matrix=rot_matrix, centroid_xyz=centroid_xyz
            )
            d_min = evaluate_strut_lattice_sdf(
                Xp, Yp, Zp, pattern_name, L_mm
            )
            pat_c = (d_min <= r_strut_mm)
            if invert:
                pat_c = ~pat_c
            out_mask[z0:z1] = np.logical_and(sub_bone, pat_c).astype(np.uint8)
        return out_mask

    omega = np.float32((2.0 * np.pi) / L_mm)
    bound = TPMS_ANALYTICAL_BOUNDS.get(pattern_name, 1.5)

    for z0 in range(0, zdim, chunk_z):
        z1 = min(zdim, z0 + chunk_z)
        sub_bone = mask_bool[z0:z1]
        if not np.any(sub_bone):
            continue
        Z_c, Y_c, X_c = np.meshgrid(
            zz_mm[z0:z1], yy_mm, xx_mm, indexing="ij"
        )
        Xp, Yp, Zp = _transform_coords_chunk(
            X_c, Y_c, Z_c, rot_matrix=rot_matrix, centroid_xyz=centroid_xyz
        )
        field_c = evaluate_tpms_field(Xp * omega, Yp * omega, Zp * omega, pattern_name)

        if tpms_type == "Skeletal":
            pat_c = (field_c > level_set)
        elif tpms_type == "Shell":
            field_norm = field_c / bound
            pat_c = (np.abs(field_norm - level_set) <= thickness)
        else:
            pat_c = (field_c > 0.0)

        if invert:
            pat_c = ~pat_c

        out_mask[z0:z1] = np.logical_and(sub_bone, pat_c).astype(np.uint8)

    return out_mask


def evaluate_continuous_lattice_phi_inplace(bone_phi, zz_mm, yy_mm, xx_mm,
                                            pattern_name, tpms_type,
                                            pore_size_um, level_set, thickness,
                                            strut_diam_um=DEFAULT_STRUT_UM,
                                            invert=False,
                                            rot_matrix=None,
                                            centroid_xyz=(0.0, 0.0, 0.0)):
    """
    Evaluate oriented continuous signed scalar field Φ_lattice(X', Y', Z')
    where [X', Y', Z']^T = R^T (X - X_c) and intersect in-place with bone_phi:
        Φ_CSG = min(Φ_ROI, Φ_lattice)
    """
    zdim = bone_phi.shape[0]
    L_mm = compute_unit_cell_period_mm(
        pattern_name, tpms_type, pore_size_um,
        level_set=level_set, thickness=thickness,
        strut_diam_um=strut_diam_um,
    )
    chunk_z = max(1, min(32, zdim))

    if is_strut_pattern(pattern_name):
        r_strut_mm = max(0.025, float(strut_diam_um) / 2000.0)
        for z0 in range(0, zdim, chunk_z):
            z1 = min(zdim, z0 + chunk_z)
            if np.all(bone_phi[z0:z1] <= 0.0):
                continue
            Z_c, Y_c, X_c = np.meshgrid(
                zz_mm[z0:z1], yy_mm, xx_mm, indexing="ij"
            )
            Xp, Yp, Zp = _transform_coords_chunk(
                X_c, Y_c, Z_c, rot_matrix=rot_matrix, centroid_xyz=centroid_xyz
            )
            d_min = evaluate_strut_lattice_sdf(
                Xp, Yp, Zp, pattern_name, L_mm
            )
            pat_phi = (r_strut_mm - d_min).astype(np.float32)
            if invert:
                pat_phi = -pat_phi
            np.minimum(bone_phi[z0:z1], pat_phi, out=bone_phi[z0:z1])
        return bone_phi

    omega = np.float32((2.0 * np.pi) / L_mm)
    metric_scale = float(L_mm / (2.0 * np.pi))
    bound = TPMS_ANALYTICAL_BOUNDS.get(pattern_name, 1.5)

    for z0 in range(0, zdim, chunk_z):
        z1 = min(zdim, z0 + chunk_z)
        if np.all(bone_phi[z0:z1] <= 0.0):
            continue
        Z_c, Y_c, X_c = np.meshgrid(
            zz_mm[z0:z1], yy_mm, xx_mm, indexing="ij"
        )
        Xp, Yp, Zp = _transform_coords_chunk(
            X_c, Y_c, Z_c, rot_matrix=rot_matrix, centroid_xyz=centroid_xyz
        )
        field_c = evaluate_tpms_field(Xp * omega, Yp * omega, Zp * omega, pattern_name)

        if tpms_type == "Skeletal":
            pat_phi = ((field_c - level_set) / bound * metric_scale).astype(np.float32)
        elif tpms_type == "Shell":
            field_norm = field_c / bound
            pat_phi = (
                (thickness - np.abs(field_norm - level_set)) * metric_scale
            ).astype(np.float32)
        else:
            pat_phi = (field_c / bound * metric_scale).astype(np.float32)

        if invert:
            pat_phi = -pat_phi

        np.minimum(bone_phi[z0:z1], pat_phi, out=bone_phi[z0:z1])

    return bone_phi


def _count_positive_phi_chunks(phi_3d, chunk_z=32):
    """Count voxels with Φ > 0 in memory-safe Z-chunks without full-array allocation."""
    total = 0
    zdim = phi_3d.shape[0]
    step = max(1, min(int(chunk_z), zdim))
    for cz0 in range(0, zdim, step):
        cz1 = min(zdim, cz0 + step)
        total += int(np.count_nonzero(phi_3d[cz0:cz1] > 0.0))
    return total


def estimate_unit_cell_porosity_pct(pattern_name, tpms_type="Skeletal",
                                    pore_size_um=DEFAULT_PORE_UM,
                                    level_set=0.0, thickness=0.25,
                                    strut_diam_um=DEFAULT_STRUT_UM,
                                    invert=False, n_grid=40):
    """
    Fast midpoint quadrature over a single periodic unit cell [0, L)^3 to
    estimate nominal lattice porosity (%) prior to full 3D scaffold generation.
    """
    if pattern_name in (None, "None"):
        return 0.0

    n = max(16, int(n_grid))
    L_mm = compute_unit_cell_period_mm(
        pattern_name, tpms_type, pore_size_um,
        level_set=level_set, thickness=thickness,
        strut_diam_um=strut_diam_um,
    )

    if is_strut_pattern(pattern_name):
        ax = ((np.arange(n, dtype=np.float32) + 0.5) / float(n)) * np.float32(L_mm)
        Z, Y, X = np.meshgrid(ax, ax, ax, indexing="ij")
        r_strut_mm = max(0.025, float(strut_diam_um) / 2000.0)
        d_min = evaluate_strut_lattice_sdf(X, Y, Z, pattern_name, L_mm)
        solid = (d_min <= r_strut_mm)
    else:
        ax = ((np.arange(n, dtype=np.float32) + 0.5) / float(n)) * np.float32(2.0 * np.pi)
        Z, Y, X = np.meshgrid(ax, ax, ax, indexing="ij")
        field = evaluate_tpms_field(X, Y, Z, pattern_name)
        bound = TPMS_ANALYTICAL_BOUNDS.get(pattern_name, 1.5)
        if tpms_type == "Skeletal":
            solid = (field > level_set)
        elif tpms_type == "Shell":
            solid = (np.abs(field / bound - level_set) <= thickness)
        else:
            solid = (field > 0.0)

    if invert:
        solid = ~solid

    solid_frac = float(np.count_nonzero(solid)) / float(solid.size)
    return float(np.clip((1.0 - solid_frac) * 100.0, 0.0, 100.0))


def _fast_clean_polydata(poly):
    """Multi-threaded VTK static exact-duplicate point cleaner with safe PyVista fallback."""
    if poly is None or poly.n_points == 0:
        return pv.PolyData()
    try:
        sc = vtk.vtkStaticCleanPolyData()
        sc.SetInputData(poly)
        sc.ToleranceIsAbsoluteOn()
        sc.SetAbsoluteTolerance(0.0)
        sc.RemoveUnusedPointsOn()
        sc.Update()
        out = pv.wrap(sc.GetOutput())
        if out is not None and out.n_points > 0:
            return out
    except Exception:
        pass
    return poly.extract_surface().clean(tolerance=0.0, absolute=True)


def _marching_cubes_fallback_polydata(arr, spacing_zyx, origin_xyz, lvl):
    """Safe single-threaded Lewiner Marching Cubes fallback returning pv.PolyData."""
    verts_zyx, faces, _, _ = measure.marching_cubes(
        arr, level=lvl, spacing=spacing_zyx
    )
    ox, oy, oz = (float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2]))
    verts_xyz = np.column_stack([
        verts_zyx[:, 2] + ox,
        verts_zyx[:, 1] + oy,
        verts_zyx[:, 0] + oz,
    ]).astype(np.float64)

    faces_with_n = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int64),
        faces.astype(np.int64),
    ]).ravel().astype(np.int64)
    return _fast_clean_polydata(pv.PolyData(verts_xyz, faces_with_n))


def _extract_isosurface_polydata(mask, spacing_zyx, origin_xyz=(0.0, 0.0, 0.0), level=0.5):
    """
    Multi-threaded VTK Flying Edges (`vtkFlyingEdges3D`) isosurface extractor on a
    (Z, Y, X) scalar/binary field with spacing (sz, sy, sx) and origin (ox, oy, oz).
    Applies symbolic tie-breaking (`|val - level| < 1e-5`) so planar cuts coinciding
    with grid coordinates never produce degenerate zero-length edges, cleans exact
    duplicate points via multi-threaded `vtkStaticCleanPolyData`, and falls back to
    `skimage.measure.marching_cubes` on any VTK contouring exception.
    """
    arr = np.asarray(mask, dtype=np.float32)
    lvl = float(level)
    near_iso = np.abs(arr - lvl) < 1e-5
    if np.any(near_iso):
        arr = arr.copy()
        arr[near_iso] = np.float32(lvl - 1e-5)

    ox, oy, oz = (float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2]))
    sz, sy, sx = (float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2]))

    try:
        grid = pv.ImageData()
        grid.dimensions = np.array(arr.shape[::-1], dtype=np.int32)  # (nx, ny, nz)
        grid.origin = (ox, oy, oz)
        grid.spacing = (sx, sy, sz)                                  # (dx, dy, dz)
        # Transpose (Z, Y, X) -> (X, Y, Z) prior to Fortran ravel so VTK point order matches Cartesian axes:
        grid.point_data["values"] = np.transpose(arr, (2, 1, 0)).ravel(order="F")
        contour = grid.contour(
            [lvl],
            scalars="values",
            method="flying_edges",
            compute_normals=False,
            compute_gradients=False,
            compute_scalars=False,
        )
        surf = contour.extract_surface()
        if not surf.is_all_triangles:
            surf = surf.triangulate()
        mesh = _fast_clean_polydata(surf)
        if mesh.n_points > 0 and mesh.n_cells > 0:
            return mesh
    except Exception:
        pass

    return _marching_cubes_fallback_polydata(arr, (sz, sy, sx), (ox, oy, oz), lvl)


def prepare_mesh_from_mask(mask, spacing_zyx, origin_xyz=(0.0, 0.0, 0.0), level=0.5):
    """
    Extract multi-threaded VTK Flying Edges (with Marching Cubes fallback) surface
    from a (Z, Y, X) scalar/binary field and return `(verts_xyz, faces_flat)`.
    """
    mesh = _extract_isosurface_polydata(
        mask, spacing_zyx=spacing_zyx, origin_xyz=origin_xyz, level=level
    )
    verts_xyz = np.asarray(mesh.points, dtype=np.float64)
    faces_flat = np.asarray(mesh.faces, dtype=np.int64)
    return verts_xyz, faces_flat


def _hash_binary_mask(mask_arr):
    """Fast compact BLAKE2b digest of a 3D binary mask's shape and bit-packed contents."""
    if mask_arr is None:
        return "none"
    b = np.asarray(mask_arr > 0, dtype=np.uint8)
    packed = np.packbits(b.ravel(order="C"))
    h = hashlib.blake2b(digest_size=12)
    h.update(repr(b.shape).encode("ascii"))
    h.update(packed.tobytes())
    return h.hexdigest()


def _canonicalize_cut_params(cut_params):
    """Convert `cut_params` dict into a deterministic, hashable tuple of rounded floats."""
    if not cut_params:
        return ()
    items = []
    for k in sorted(cut_params.keys()):
        v = cut_params[k]
        if isinstance(v, (int, float, np.floating, np.integer)):
            items.append((str(k), round(float(v), 5)))
        else:
            items.append((str(k), str(v)))
    return tuple(items)


def make_envelope_cache_key(roi_mask_hash, cut_mode, cut_params, fill_defect, spacing_zyx):
    """Build deterministic cache key `(roi_mask_hash, cut_mode, cut_params_tuple, fill_defect, spacing_zyx)`."""
    sp_tuple = tuple(round(float(s), 6) for s in spacing_zyx)
    return (
        str(roi_mask_hash),
        str(cut_mode),
        _canonicalize_cut_params(cut_params),
        bool(fill_defect),
        sp_tuple,
    )


def build_highres_scaffold_and_preview(mask_crop, lo_slice, spacing_zyx,
                                       pattern_name, tpms_type, pore_size_um,
                                       level_set, thickness, strut_diam_um,
                                       invert=False,
                                       rot_matrix=None,
                                       centroid_xyz=None,
                                       cut_mode=CUT_MODE_NONE,
                                       cut_params=None,
                                       full_bone_crop=None,
                                       progress_cb=None,
                                       return_metrics=False,
                                       cached_envelope=None,
                                       out_envelope_cache=None):
    """
    Local Sub-Voxel Isotropic Resampling + Continuous Implicit Cutting +
    Oriented Lattice CSG Engine:
      1. Extracts tight 3D bounding box of mask_crop (> 0) + zero border
      2. If `cached_envelope` matches the exact target high-resolution grid, reuses
         `csg_phi_envelope` and coordinate axes directly (skipping `ndimage.zoom`
         and `evaluate_cut_implicit_phi`); otherwise resamples at full v6 sub-voxel
         resolution and optionally populates `out_envelope_cache`
      3. Evaluates oriented TPMS / Strut SDF in rotated frame R^T (X - X_c)
      4. Extracts sub-voxel manifold 3D mesh via multi-threaded VTK Flying Edges
         (level=0.0) and computes exact sub-voxel volumetric porosity metrics
    """
    sz, sy, sx = (float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2]))
    mask_bool = mask_crop > 0
    if not np.any(mask_bool):
        raise RuntimeError("Selected scaffold ROI is empty.")

    nz_z, nz_y, nz_x = np.nonzero(mask_bool)
    z0, z1 = int(nz_z.min()), int(nz_z.max())
    y0, y1 = int(nz_y.min()), int(nz_y.max())
    x0, x1 = int(nz_x.min()), int(nz_x.max())

    if centroid_xyz is None:
        centroid_xyz = (
            float(nz_x.mean()) * sx,
            float(nz_y.mean()) * sy,
            float(lo_slice + nz_z.mean()) * sz,
        )

    has_analytic_cut = (
        cut_mode in (CUT_MODE_BOX, CUT_MODE_PLANES) and cut_params is not None
    )

    # Plain solid bone (no porous pattern) without analytical sub-voxel cut
    if pattern_name in (None, "None") and not has_analytic_cut:
        sub_padded = np.pad(
            mask_bool[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].astype(np.uint8),
            1, mode="constant", constant_values=0
        )
        origin_xyz = (
            (x0 - 1) * sx,
            (y0 - 1) * sy,
            (lo_slice + z0 - 1) * sz,
        )
        mesh = _extract_isosurface_polydata(
            sub_padded, (sz, sy, sx), origin_xyz=origin_xyz, level=0.5
        )
        if return_metrics:
            n_env = int(len(nz_z))
            env_vol = float(n_env) * sz * sy * sx
            metrics = {
                "porosity_pct": 0.0,
                "solid_fraction_pct": 100.0,
                "scaffold_vol_mm3": env_vol,
                "envelope_vol_mm3": env_vol,
                "void_vol_mm3": 0.0,
                "cache_hit": False,
            }
            return mesh, mask_bool.astype(np.uint8), min(sz, sy, sx), metrics
        return mesh, mask_bool.astype(np.uint8), min(sz, sy, sx)

    # Expand sub-box by up to 2 native voxels when analytical cut is active so
    # bone collar across the cut plane interpolates cleanly without rounding
    margin = 2 if (has_analytic_cut and full_bone_crop is not None) else 0
    bz0 = max(0, z0 - margin)
    bz1 = min(mask_bool.shape[0] - 1, z1 + margin)
    by0 = max(0, y0 - margin)
    by1 = min(mask_bool.shape[1] - 1, y1 + margin)
    bx0 = max(0, x0 - margin)
    bx1 = min(mask_bool.shape[2] - 1, x1 + margin)

    npad_z = (bz1 - bz0 + 1) + 2
    npad_y = (by1 - by0 + 1) + 2
    npad_x = (bx1 - bx0 + 1) + 2

    span_z = (npad_z - 1) * sz
    span_y = (npad_y - 1) * sy
    span_x = (npad_x - 1) * sx

    # Exact v6 fine grid spacing logic (zero degradation of pore/strut resolution)
    if pattern_name in (None, "None"):
        delta_iso = float(np.clip(min(sz, sy, sx) * 0.5, 0.08, 0.25))
    else:
        delta_iso = compute_target_delta_iso_mm(
            pore_size_um, strut_diam_um=strut_diam_um, pattern_name=pattern_name
        )

    est_voxels = (span_z / delta_iso + 1) * (span_y / delta_iso + 1) * (span_x / delta_iso + 1)
    if est_voxels > MAX_HIGHRES_VOXELS:
        phys_vol = max(1e-3, span_z * span_y * span_x)
        delta_iso = float((phys_vol / float(MAX_HIGHRES_VOXELS)) ** (1.0 / 3.0))

    out_nz = max(4, int(round(span_z / delta_iso)) + 1)
    out_ny = max(4, int(round(span_y / delta_iso)) + 1)
    out_nx = max(4, int(round(span_x / delta_iso)) + 1)
    target_shape = (out_nz, out_ny, out_nx)
    bbox_sig = (int(lo_slice), bz0, bz1, by0, by1, bx0, bx1)

    cache_hit = (
        isinstance(cached_envelope, dict)
        and cached_envelope.get("csg_phi_envelope") is not None
        and cached_envelope["csg_phi_envelope"].shape == target_shape
        and cached_envelope.get("bbox_sig") == bbox_sig
    )

    if cache_hit:
        if progress_cb is not None:
            progress_cb(
                32,
                f"Reusing cached sub-voxel defect envelope "
                f"(Δ={delta_iso * 1000:.0f} µm, grid {out_nx}×{out_ny}×{out_nz})…"
            )
        csg_phi = cached_envelope["csg_phi_envelope"].copy()
        zz_hi_mm = cached_envelope["zz_hi_mm"]
        yy_hi_mm = cached_envelope["yy_hi_mm"]
        xx_hi_mm = cached_envelope["xx_hi_mm"]
        dz_hi, dy_hi, dx_hi = cached_envelope["spacing_hi_zyx"]
        origin_x, origin_y, origin_z = cached_envelope["origin_xyz"]
        n_env_voxels = (
            int(cached_envelope["n_env_voxels"])
            if (return_metrics and "n_env_voxels" in cached_envelope)
            else (_count_positive_phi_chunks(csg_phi) if return_metrics else 0)
        )
    else:
        if has_analytic_cut and full_bone_crop is not None:
            roi_sub = mask_bool[bz0:bz1 + 1, by0:by1 + 1, bx0:bx1 + 1]
            bone_sub = (full_bone_crop[bz0:bz1 + 1, by0:by1 + 1, bx0:bx1 + 1] > 0)
            collar_mask = ndimage.binary_dilation(roi_sub, iterations=2) & bone_sub
            base_sub_float = collar_mask.astype(np.float32)
        else:
            base_sub_float = mask_bool[bz0:bz1 + 1, by0:by1 + 1, bx0:bx1 + 1].astype(np.float32)

        sub_padded = np.pad(base_sub_float, 1, mode="constant", constant_values=0.0)
        zf_z = out_nz / float(npad_z)
        zf_y = out_ny / float(npad_y)
        zf_x = out_nx / float(npad_x)

        if progress_cb is not None:
            progress_cb(
                28,
                f"Sub-voxel isotropic resampling (Δ={delta_iso * 1000:.0f} µm, "
                f"grid {out_nx}×{out_ny}×{out_nz})…"
            )

        bone_zoom = ndimage.zoom(
            sub_padded, zoom=(zf_z, zf_y, zf_x), order=1, mode="constant", cval=0.0
        )
        env_scale = float(min(sz, sy, sx))
        csg_phi = ((bone_zoom - 0.5) * env_scale).astype(np.float32)
        del bone_zoom

        hz, hy, hx = csg_phi.shape
        dz_hi = span_z / float(hz - 1)
        dy_hi = span_y / float(hy - 1)
        dx_hi = span_x / float(hx - 1)

        origin_x = (bx0 - 1) * sx
        origin_y = (by0 - 1) * sy
        origin_z = (lo_slice + bz0 - 1) * sz

        zz_hi_mm = origin_z + np.arange(hz, dtype=np.float32) * dz_hi
        yy_hi_mm = origin_y + np.arange(hy, dtype=np.float32) * dy_hi
        xx_hi_mm = origin_x + np.arange(hx, dtype=np.float32) * dx_hi

        # Apply exact analytical 3D implicit cutting halfspaces on high-res grid
        if has_analytic_cut:
            chunk_z = max(1, min(32, hz))
            for cz0 in range(0, hz, chunk_z):
                cz1 = min(hz, cz0 + chunk_z)
                if np.all(csg_phi[cz0:cz1] <= 0.0):
                    continue
                Z_c, Y_c, X_c = np.meshgrid(
                    zz_hi_mm[cz0:cz1], yy_hi_mm, xx_hi_mm, indexing="ij"
                )
                cut_phi = evaluate_cut_implicit_phi(X_c, Y_c, Z_c, cut_mode, cut_params)
                np.minimum(csg_phi[cz0:cz1], cut_phi, out=csg_phi[cz0:cz1])

        # Strictly negative outer 1-voxel frame guarantees closed watertight surface caps
        csg_phi[0, :, :] = -1.0
        csg_phi[-1, :, :] = -1.0
        csg_phi[:, 0, :] = -1.0
        csg_phi[:, -1, :] = -1.0
        csg_phi[:, :, 0] = -1.0
        csg_phi[:, :, -1] = -1.0

        # Measure sub-voxel target envelope volume before carving porous lattice
        n_env_voxels = _count_positive_phi_chunks(csg_phi) if return_metrics else 0

        if isinstance(out_envelope_cache, dict) and csg_phi.size <= MAX_HIGHRES_VOXELS:
            out_envelope_cache.clear()
            out_envelope_cache.update({
                "csg_phi_envelope": csg_phi.copy(),
                "zz_hi_mm": zz_hi_mm,
                "yy_hi_mm": yy_hi_mm,
                "xx_hi_mm": xx_hi_mm,
                "spacing_hi_zyx": (float(dz_hi), float(dy_hi), float(dx_hi)),
                "origin_xyz": (float(origin_x), float(origin_y), float(origin_z)),
                "n_env_voxels": int(n_env_voxels),
                "bbox_sig": bbox_sig,
            })

    if pattern_name not in (None, "None"):
        if progress_cb is not None:
            arch_label = "Strut Lattice" if is_strut_pattern(pattern_name) else "TPMS"
            progress_cb(
                42,
                f"Evaluating oriented {arch_label} ({pattern_name}, Pore: {pore_size_um} µm)…"
            )

        evaluate_continuous_lattice_phi_inplace(
            csg_phi, zz_hi_mm, yy_hi_mm, xx_hi_mm,
            pattern_name=pattern_name, tpms_type=tpms_type,
            pore_size_um=pore_size_um, level_set=level_set,
            thickness=thickness, strut_diam_um=strut_diam_um,
            invert=invert, rot_matrix=rot_matrix, centroid_xyz=centroid_xyz,
        )
        n_scaf_voxels = _count_positive_phi_chunks(csg_phi)
    else:
        n_scaf_voxels = (
            n_env_voxels if return_metrics else _count_positive_phi_chunks(csg_phi)
        )

    if n_scaf_voxels == 0:
        raise RuntimeError(
            "Resulting high-res scaffold is empty — check cut region or pore/strut settings."
        )

    if progress_cb is not None:
        progress_cb(58, "Extracting sub-voxel surface (multi-threaded Flying Edges)…")

    scaffold_pv_mesh = _extract_isosurface_polydata(
        csg_phi,
        spacing_zyx=(dz_hi, dy_hi, dx_hi),
        origin_xyz=(origin_x, origin_y, origin_z),
        level=0.0,
    )
    del csg_phi

    # Keep only largest connected structural scaffold island if analytical cut grazed edge
    if has_analytic_cut and pattern_name in (None, "None"):
        try:
            scaffold_pv_mesh = scaffold_pv_mesh.connectivity(extraction_mode="largest")
            scaffold_pv_mesh = _fast_clean_polydata(scaffold_pv_mesh.extract_surface())
        except Exception:
            pass

    # Build native-resolution 2D mask preview
    zz_nat_mm = (lo_slice + np.arange(mask_crop.shape[0], dtype=np.float32)) * sz
    yy_nat_mm = np.arange(mask_crop.shape[1], dtype=np.float32) * sy
    xx_nat_mm = np.arange(mask_crop.shape[2], dtype=np.float32) * sx
    native_crop_mask = apply_scaffold_on_physical_grid(
        mask_bool, zz_nat_mm, yy_nat_mm, xx_nat_mm,
        pattern_name=pattern_name, tpms_type=tpms_type,
        pore_size_um=pore_size_um, level_set=level_set,
        thickness=thickness, strut_diam_um=strut_diam_um,
        invert=invert, rot_matrix=rot_matrix, centroid_xyz=centroid_xyz,
    )

    if return_metrics:
        dv_mm3 = float(dz_hi) * float(dy_hi) * float(dx_hi)
        env_vol_mm3 = float(n_env_voxels) * dv_mm3
        scaf_vol_mm3 = float(n_scaf_voxels) * dv_mm3
        void_vol_mm3 = max(0.0, env_vol_mm3 - scaf_vol_mm3)
        if n_env_voxels > 0:
            porosity_pct = float(
                np.clip((1.0 - float(n_scaf_voxels) / float(n_env_voxels)) * 100.0, 0.0, 100.0)
            )
        else:
            porosity_pct = 0.0
        metrics = {
            "porosity_pct": porosity_pct,
            "solid_fraction_pct": 100.0 - porosity_pct,
            "scaffold_vol_mm3": scaf_vol_mm3,
            "envelope_vol_mm3": env_vol_mm3,
            "void_vol_mm3": void_vol_mm3,
            "cache_hit": bool(cache_hit),
        }
        return scaffold_pv_mesh, native_crop_mask, float(dz_hi), metrics

    return scaffold_pv_mesh, native_crop_mask, float(dz_hi)


def _extract_smoothed_bone_mesh_from_mask(mask_3d, spacing_zyx, n_taubin=28):
    """
    Fast helper to extract a Taubin-smoothed pv.PolyData surface from a 3D
    binary bone mask in canonical (X, Y, Z) coordinates (preserving cavities).
    """
    mask_bool = mask_3d > 0
    if not np.any(mask_bool):
        return None

    nz_z, nz_y, nz_x = np.nonzero(mask_bool)
    z0, z1 = int(nz_z.min()), int(nz_z.max())
    y0, y1 = int(nz_y.min()), int(nz_y.max())
    x0, x1 = int(nz_x.min()), int(nz_x.max())

    sub_bool = mask_bool[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    sz, sy, sx = (
        float(spacing_zyx[0]),
        float(spacing_zyx[1]),
        float(spacing_zyx[2]),
    )

    step_z, step_y, step_x = 1, 1, 1
    if sub_bool.size > MAX_PREVIEW_VOXELS:
        if sub_bool.shape[0] > 512:
            step_z = 2
        if sub_bool.shape[1] > 384:
            step_y = 2
        if sub_bool.shape[2] > 384:
            step_x = 2
        sub_bool = sub_bool[::step_z, ::step_y, ::step_x]
        if not np.any(sub_bool):
            return None

    eff_sz, eff_sy, eff_sx = sz * step_z, sy * step_y, sx * step_x
    sub_padded = np.pad(
        sub_bool.astype(np.uint8), 1, mode="constant", constant_values=0
    )
    origin_xyz = (
        x0 * sx - eff_sx,
        y0 * sy - eff_sy,
        z0 * sz - eff_sz,
    )
    mesh = _extract_isosurface_polydata(
        sub_padded, (eff_sz, eff_sy, eff_sx), origin_xyz=origin_xyz, level=0.5
    )
    if n_taubin > 0 and mesh.n_points > 0:
        try:
            mesh = mesh.smooth_taubin(
                n_iter=n_taubin, pass_band=0.1, normalize_coordinates=True
            )
        except TypeError:
            mesh = mesh.smooth_taubin(n_iter=n_taubin, pass_band=0.1)
        except Exception:
            pass
        mesh = _fast_clean_polydata(mesh)
    return mesh


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREADS  (non-blocking background computation)
# ══════════════════════════════════════════════════════════════════════════════

class LoadVolumeWorker(QThread):
    """Background thread for loading DICOM series or NRRD volumes."""
    finished = pyqtSignal(object, tuple)   # (np.ndarray, spacing_zyx)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, dicom_folder=None, nrrd_path=None, parent=None):
        super().__init__(parent)
        self.dicom_folder = dicom_folder
        self.nrrd_path = nrrd_path

    def run(self):
        try:
            self.progress.emit(10, "Reading image data…")
            if self.nrrd_path:
                img = sitk.ReadImage(self.nrrd_path)
            elif self.dicom_folder:
                reader = sitk.ImageSeriesReader()
                paths = reader.GetGDCMSeriesFileNames(self.dicom_folder)
                if not paths:
                    raise RuntimeError("No DICOM series found in folder")
                reader.SetFileNames(paths)
                img = reader.Execute()
            else:
                raise RuntimeError("No input file specified")

            self.progress.emit(60, "Converting to NumPy array…")
            arr = sitk.GetArrayFromImage(img).astype(np.int16)  # (Z, Y, X)
            sp = img.GetSpacing()                                # (X, Y, Z)
            spacing_zyx = (sp[2], sp[1], sp[0])

            self.progress.emit(100, "Volume loaded")
            self.finished.emit(arr, spacing_zyx)
        except Exception as e:
            self.error.emit(str(e))


class Preview3DWorker(QThread):
    """
    Asynchronous real-time 3D bone segmentation preview worker for Tab 1.
    Supports toggling `keep_largest` ("Keep Largest Island Only") in real time
    and applies automatic volume-preserving Taubin smoothing.
    """
    finished = pyqtSignal(object, object)  # (seg_mask_uint8, pv.PolyData_or_None)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, volume, spacing, hu_min, hu_max,
                 keep_largest=True, auto_smooth=True, parent=None):
        super().__init__(parent)
        self.volume       = volume
        self.spacing      = spacing
        self.hu_min       = hu_min
        self.hu_max       = hu_max
        self.keep_largest = keep_largest
        self.auto_smooth  = auto_smooth
        self._cancelled   = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self.volume is None or self.volume.size == 0:
                return

            self.progress.emit(20, "Segmenting bone & filtering islands…")
            seg_mask = create_bone_mask(
                self.volume,
                self.hu_min,
                self.hu_max,
                keep_largest=self.keep_largest,
            )

            if self._cancelled:
                return

            if not np.any(seg_mask):
                self.progress.emit(100, "No bone voxels in HU range")
                self.finished.emit(seg_mask, None)
                return

            self.progress.emit(55, "Extracting & Taubin-smoothing 3D bone surface…")
            n_iter = 30 if self.auto_smooth else 0
            mesh = _extract_smoothed_bone_mesh_from_mask(
                seg_mask, self.spacing, n_taubin=n_iter
            )

            if self._cancelled:
                return

            self.progress.emit(100, "Segmentation preview ready")
            self.finished.emit(seg_mask, mesh)

        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))


class CutPreviewWorker(QThread):
    """
    Asynchronous interactive cutting preview worker for Tab 2 ("Scaffold Design").
    Evaluates the 3D implicit Box or Dual-Plane Wedge cut, enforces Post-Cut
    Largest Island selection, and produces both the cyan candidate ROI preview
    mesh and the translucent host-bone mesh outside the cut.
    """
    # (roi_mask, host_mask, roi_mesh, host_mesh, roi_centroid_xyz)
    finished = pyqtSignal(object, object, object, object, tuple)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, seg_bone_mask, spacing, cut_mode, cut_params, parent=None):
        super().__init__(parent)
        self.seg_bone_mask = seg_bone_mask
        self.spacing       = spacing
        self.cut_mode      = cut_mode
        self.cut_params    = dict(cut_params) if cut_params is not None else None
        self._cancelled    = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self.seg_bone_mask is None or not np.any(self.seg_bone_mask):
                empty = np.zeros((1, 1, 1), dtype=np.uint8)
                self.finished.emit(empty, empty, None, None, (0.0, 0.0, 0.0))
                return

            self.progress.emit(20, "Evaluating 3D implicit cut & largest island…")
            roi_mask, host_mask, centroid = compute_cut_roi_and_host_masks(
                self.seg_bone_mask, self.spacing, self.cut_mode, self.cut_params
            )

            if self._cancelled:
                return

            roi_mesh = None
            if np.any(roi_mask):
                self.progress.emit(55, "Building cut ROI preview surface…")
                if self.cut_mode in (CUT_MODE_BOX, CUT_MODE_PLANES) and self.cut_params is not None:
                    # Use analytical cut CSG at coarse preview spacing for crisp cut planes
                    roi_mesh, _, _ = build_highres_scaffold_and_preview(
                        mask_crop=roi_mask,
                        lo_slice=0,
                        spacing_zyx=self.spacing,
                        pattern_name="None",
                        tpms_type="Skeletal",
                        pore_size_um=DEFAULT_PORE_UM,
                        level_set=0.0,
                        thickness=0.25,
                        strut_diam_um=DEFAULT_STRUT_UM,
                        cut_mode=self.cut_mode,
                        cut_params=self.cut_params,
                        full_bone_crop=self.seg_bone_mask,
                    )
                    try:
                        roi_mesh = _fast_clean_polydata(
                            roi_mesh.smooth_taubin(
                                n_iter=20, pass_band=0.1, normalize_coordinates=True
                            )
                        )
                    except Exception:
                        roi_mesh = _fast_clean_polydata(roi_mesh)
                else:
                    roi_mesh = _extract_smoothed_bone_mesh_from_mask(
                        roi_mask, self.spacing, n_taubin=25
                    )

            if self._cancelled:
                return

            host_mesh = None
            if np.any(host_mask):
                self.progress.emit(80, "Building host bone overlay surface…")
                host_mesh = _extract_smoothed_bone_mesh_from_mask(
                    host_mask, self.spacing, n_taubin=20
                )

            if self._cancelled:
                return

            self.progress.emit(100, "Cut ROI updated")
            self.finished.emit(roi_mask, host_mask, roi_mesh, host_mesh, centroid)

        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))


class SegmentWorker(QThread):
    """
    Background thread for Tab 2 high-resolution sub-voxel scaffold generation (v8):
      - Accepts pre-segmented `seg_bone_mask` (or thresholds raw `volume`)
      - Reuses precomputed cut/host masks and cached sub-voxel defect envelopes
        (`cached_envelope`) when valid to eliminate redundant resampling across
        lattice pattern / parameter sweeps
      - Preserves 100% of v6's pristine fine resolution, multi-threaded VTK Flying
        Edges contouring, and 20-iteration volume-preserving Taubin smoothing
    """
    # (full_scaffold_mask, scaffold_pv_mesh, host_bone_pv_mesh_or_None, eff_delta_mm, porosity_metrics)
    finished = pyqtSignal(object, object, object, float, object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, spacing,
                 seg_bone_mask=None,
                 volume=None, hu_min=150, hu_max=3000, keep_largest=True,
                 cut_mode=CUT_MODE_NONE, cut_params=None,
                 pattern="None", tpms_type="Skeletal",
                 pore_size_um=DEFAULT_PORE_UM,
                 level_set=0.0, thickness=0.25,
                 strut_diam_um=DEFAULT_STRUT_UM,
                 invert=False, fill_defect=False,
                 dir_yaw=0.0, dir_pitch=90.0, dir_roll=0.0,
                 cached_envelope=None,
                 precomputed_cut=None,
                 precomputed_host_mesh=None,
                 parent=None):
        super().__init__(parent)
        self.spacing               = spacing
        self.seg_bone_mask         = seg_bone_mask
        self.volume                = volume
        self.hu_min                = hu_min
        self.hu_max                = hu_max
        self.keep_largest          = keep_largest
        self.cut_mode              = cut_mode
        self.cut_params            = dict(cut_params) if cut_params is not None else None
        self.pattern               = pattern
        self.tpms_type             = tpms_type
        self.pore_size_um          = pore_size_um
        self.level_set             = level_set
        self.thickness             = thickness
        self.strut_diam_um         = strut_diam_um
        self.invert                = invert
        self.fill_defect           = fill_defect
        self.dir_yaw               = dir_yaw
        self.dir_pitch             = dir_pitch
        self.dir_roll              = dir_roll
        self.cached_envelope       = cached_envelope
        self.precomputed_cut       = precomputed_cut
        self.precomputed_host_mesh = precomputed_host_mesh

    def run(self):
        try:
            t_start = time.perf_counter()
            if self.seg_bone_mask is not None:
                base_bone = (self.seg_bone_mask > 0).astype(np.uint8)
            elif self.volume is not None:
                self.progress.emit(8, "Segmenting baseline bone volume…")
                base_bone = create_bone_mask(
                    self.volume, self.hu_min, self.hu_max,
                    keep_largest=self.keep_largest,
                )
            else:
                raise RuntimeError("No segmented bone volume available.")

            if not np.any(base_bone):
                raise RuntimeError("Baseline segmented bone mask is empty.")

            if (
                self.precomputed_cut is not None
                and len(self.precomputed_cut) == 3
                and self.precomputed_cut[0] is not None
            ):
                roi_mask, host_mask, centroid_xyz = self.precomputed_cut
            else:
                self.progress.emit(14, "Isolating 3D cut ROI & largest bone island…")
                roi_mask, host_mask, centroid_xyz = compute_cut_roi_and_host_masks(
                    base_bone, self.spacing, self.cut_mode, self.cut_params
                )

            if not np.any(roi_mask):
                raise RuntimeError(
                    "No bone voxels lie inside the current 3D cutting region."
                )

            # Restrict tight Z-slice window around active ROI (+2 collar slices)
            nz_z = np.nonzero(roi_mask)[0]
            lo = max(0, int(nz_z.min()) - 2)
            hi = min(base_bone.shape[0] - 1, int(nz_z.max()) + 2)

            roi_crop = roi_mask[lo:hi + 1].copy()
            bone_crop_context = base_bone[lo:hi + 1].copy()

            if self.fill_defect:
                if (
                    isinstance(self.cached_envelope, dict)
                    and self.cached_envelope.get("filled_roi_crop") is not None
                    and self.cached_envelope.get("lo_slice") == lo
                    and self.cached_envelope.get("hi_slice") == hi
                ):
                    roi_crop = self.cached_envelope["filled_roi_crop"]
                else:
                    self.progress.emit(
                        20, "Solid Resection Graft mode: sealing medullary defect cavity…"
                    )
                    roi_crop = fill_osteotomy_defect_envelope(roi_crop)
                bone_crop_context = np.maximum(bone_crop_context, roi_crop)

            rot_matrix = pattern_direction_to_matrix(
                self.dir_yaw, self.dir_pitch, self.dir_roll
            )
            is_porous = (self.pattern is not None and self.pattern != "None")

            fresh_env_cache = {}
            scaffold_pv_mesh, native_crop_mask, eff_delta_mm, porosity_metrics = (
                build_highres_scaffold_and_preview(
                    mask_crop=roi_crop,
                    lo_slice=lo,
                    spacing_zyx=self.spacing,
                    pattern_name=self.pattern,
                    tpms_type=self.tpms_type,
                    pore_size_um=self.pore_size_um,
                    level_set=self.level_set,
                    thickness=self.thickness,
                    strut_diam_um=self.strut_diam_um,
                    invert=self.invert,
                    rot_matrix=rot_matrix,
                    centroid_xyz=centroid_xyz,
                    cut_mode=self.cut_mode,
                    cut_params=self.cut_params,
                    full_bone_crop=bone_crop_context,
                    progress_cb=self.progress.emit,
                    return_metrics=True,
                    cached_envelope=self.cached_envelope,
                    out_envelope_cache=fresh_env_cache,
                )
            )
            porosity_metrics["pattern"] = self.pattern if self.pattern else "None"

            if fresh_env_cache:
                if self.fill_defect:
                    fresh_env_cache["filled_roi_crop"] = roi_crop
                    fresh_env_cache["lo_slice"] = int(lo)
                    fresh_env_cache["hi_slice"] = int(hi)
                roi_hash = _hash_binary_mask(roi_mask)
                porosity_metrics["_envelope_cache_key"] = make_envelope_cache_key(
                    roi_mask_hash=roi_hash,
                    cut_mode=self.cut_mode,
                    cut_params=self.cut_params,
                    fill_defect=self.fill_defect,
                    spacing_zyx=self.spacing,
                )
                porosity_metrics["_new_envelope_cache"] = fresh_env_cache

            full_scaffold_mask = np.zeros_like(base_bone, dtype=np.uint8)
            full_scaffold_mask[lo:hi + 1] = native_crop_mask

            if is_porous:
                # Porous scaffold: pristine v6 volume-preserving Taubin (n_iter=20, pass_band=0.08)
                self.progress.emit(70, "Applying volume-preserving Taubin smoothing…")
                try:
                    scaffold_pv_mesh = scaffold_pv_mesh.smooth_taubin(
                        n_iter=20, pass_band=0.08, normalize_coordinates=True
                    )
                except TypeError:
                    scaffold_pv_mesh = scaffold_pv_mesh.smooth_taubin(
                        n_iter=20, pass_band=0.08
                    )
                except Exception:
                    pass
                scaffold_pv_mesh = _fast_clean_polydata(scaffold_pv_mesh)
            else:
                self.progress.emit(70, "Smoothing & repairing solid bone mesh…")
                try:
                    scaffold_pv_mesh = scaffold_pv_mesh.smooth(
                        n_iter=10, relaxation_factor=0.03
                    )
                except Exception:
                    pass
                try:
                    scaffold_pv_mesh = scaffold_pv_mesh.smooth_taubin(
                        n_iter=30, pass_band=0.1, normalize_coordinates=True
                    )
                except TypeError:
                    scaffold_pv_mesh = scaffold_pv_mesh.smooth_taubin(
                        n_iter=30, pass_band=0.1
                    )
                except Exception:
                    pass
                try:
                    scaffold_pv_mesh = scaffold_pv_mesh.fill_holes(1000.0)
                except Exception:
                    pass
                scaffold_pv_mesh = _fast_clean_polydata(scaffold_pv_mesh)

            # Fast C++ manifold / watertightness check & background point normals
            try:
                porosity_metrics["watertight"] = bool(scaffold_pv_mesh.is_manifold)
            except Exception:
                porosity_metrics["watertight"] = False

            try:
                scaffold_pv_mesh.compute_normals(
                    cell_normals=False, point_normals=True,
                    split_vertices=False, consistent_normals=False,
                    inplace=True,
                )
            except Exception:
                pass

            bone_pv_mesh = None
            if np.any(host_mask):
                if (
                    self.precomputed_host_mesh is not None
                    and self.precomputed_host_mesh.n_points > 0
                ):
                    bone_pv_mesh = self.precomputed_host_mesh
                else:
                    self.progress.emit(85, "Building aligned host bone overlay mesh…")
                    try:
                        bone_pv_mesh = _extract_smoothed_bone_mesh_from_mask(
                            host_mask, self.spacing, n_taubin=25
                        )
                    except Exception as e:
                        print(f"Host bone mesh error (continuing): {e}", file=sys.stderr)
                        bone_pv_mesh = None

            porosity_metrics["elapsed_s"] = float(time.perf_counter() - t_start)
            self.progress.emit(100, "Scaffold generation complete")
            self.finished.emit(
                full_scaffold_mask, scaffold_pv_mesh, bone_pv_mesh,
                eff_delta_mm, porosity_metrics
            )

        except Exception as e:
            self.error.emit(str(e))


def _export_3mf_archive(named_meshes, out_path, progress_cb=None, p_start=85, p_end=98):
    """
    Export one or more named `trimesh.Trimesh` bodies to an ISO/IEC 25422
    3MF v1.2.3 archive (`unit="millimeter"`), streaming directly into a deflated
    ZIP container in memory-safe batches.
    """
    if not named_meshes:
        raise RuntimeError("No meshes provided for 3MF export.")

    for _, tri_m in named_meshes:
        tri_m.units = "millimeters"
        if tri_m.metadata is not None:
            tri_m.metadata["units"] = "millimeter"

    total_items = sum(
        max(1, len(m.vertices) + len(m.faces)) for _, m in named_meshes
    )
    processed_items = 0
    batch_size = 8192

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />\n'
        '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />\n'
        '</Types>\n'
    ).encode("utf-8")

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />\n'
        '</Relationships>\n'
    ).encode("utf-8")

    with zipfile.ZipFile(
        out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=5
    ) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)

        with zf.open("3D/3dmodel.model", mode="w") as f:
            header = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<model unit="millimeter" xml:lang="en-US" '
                'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
                'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02" '
                'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">\n'
                '  <metadata name="Application">Andamyo v9 Biomedical Scaffold Workstation</metadata>\n'
                '  <resources>\n'
            )
            f.write(header.encode("utf-8"))

            obj_ids = []
            for idx, (raw_name, tri_m) in enumerate(named_meshes, start=1):
                obj_id = str(idx)
                obj_ids.append(obj_id)
                safe_name = xml_escape(str(raw_name), {'"': "&quot;"})
                obj_uuid = str(uuid.uuid4())
                f.write(
                    f'    <object id="{obj_id}" name="{safe_name}" '
                    f'type="model" p:UUID="{obj_uuid}">\n'
                    f'      <mesh>\n'
                    f'        <vertices>\n'.encode("utf-8")
                )

                verts = np.asarray(tri_m.vertices, dtype=np.float64)
                n_v = len(verts)
                for i in range(0, n_v, batch_size):
                    batch = verts[i : i + batch_size]
                    tpl = '          <vertex x="{:.6f}" y="{:.6f}" z="{:.6f}" />\n' * len(batch)
                    f.write(tpl.format(*batch.ravel().tolist()).encode("utf-8"))
                    processed_items += len(batch)
                    if progress_cb is not None and (i // batch_size) % 16 == 0:
                        frac = min(1.0, processed_items / total_items)
                        pct = int(round(p_start + frac * (p_end - p_start)))
                        progress_cb(pct, f"Streaming 3MF vertices ({raw_name})…")

                f.write(b"        </vertices>\n        <triangles>\n")

                faces = np.asarray(tri_m.faces, dtype=np.int64)
                n_f = len(faces)
                for i in range(0, n_f, batch_size):
                    batch = faces[i : i + batch_size]
                    tpl = '          <triangle v1="{}" v2="{}" v3="{}" />\n' * len(batch)
                    f.write(tpl.format(*batch.ravel().tolist()).encode("utf-8"))
                    processed_items += len(batch)
                    if progress_cb is not None and (i // batch_size) % 16 == 0:
                        frac = min(1.0, processed_items / total_items)
                        pct = int(round(p_start + frac * (p_end - p_start)))
                        progress_cb(pct, f"Streaming 3MF triangles ({raw_name})…")

                f.write(
                    b"        </triangles>\n"
                    b"      </mesh>\n"
                    b"    </object>\n"
                )

            if len(obj_ids) == 1:
                build_obj_id = obj_ids[0]
            else:
                # Group scaffold + host_bone in a multi-part surgical assembly so slicers
                # preserve exact relative 3D coordinates (Z-height osteotomy registration)
                # while exposing each body as an individually selectable/printable part.
                asm_id = str(len(obj_ids) + 1)
                asm_uuid = str(uuid.uuid4())
                f.write(
                    f'    <object id="{asm_id}" name="surgical_scaffold_assembly" '
                    f'type="model" p:UUID="{asm_uuid}">\n'
                    f'      <components>\n'.encode("utf-8")
                )
                for oid in obj_ids:
                    comp_uuid = str(uuid.uuid4())
                    f.write(
                        f'        <component objectid="{oid}" '
                        f'transform="1 0 0 0 1 0 0 0 1 0 0 0" '
                        f'p:UUID="{comp_uuid}" />\n'.encode("utf-8")
                    )
                f.write(b"      </components>\n    </object>\n")
                build_obj_id = asm_id

            build_uuid = str(uuid.uuid4())
            item_uuid = str(uuid.uuid4())
            footer = (
                f'  </resources>\n'
                f'  <build p:UUID="{build_uuid}">\n'
                f'    <item objectid="{build_obj_id}" '
                f'transform="1 0 0 0 1 0 0 0 1 0 0 0" '
                f'p:UUID="{item_uuid}" />\n'
                f'  </build>\n'
                f'</model>\n'
            )
            f.write(footer.encode("utf-8"))


class ExportMeshWorker(QThread):
    """Background thread for selective porous/solid watertight 3MF & STL export."""
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, pv_mesh, segmented_data, spacing, out_path,
                 file_format="3mf", is_porous=False, already_smoothed=False,
                 export_host=False, host_pv_mesh=None, host_mask_data=None,
                 primary_name="scaffold", parent=None):
        super().__init__(parent)
        self.pv_mesh = pv_mesh
        self.segmented_data = segmented_data
        self.spacing = spacing
        self.out_path = out_path
        self.file_format = str(file_format or "3mf").lower().lstrip(".")
        self.is_porous = bool(is_porous)
        self.already_smoothed = bool(already_smoothed)
        self.export_host = bool(export_host)
        self.host_pv_mesh = host_pv_mesh
        self.host_mask_data = host_mask_data
        self.primary_name = str(primary_name or "scaffold")

    @staticmethod
    def _point_inside_closed_trimesh(tri, pt):
        """
        Pure-NumPy Möller–Trumbore ray-parity containment test (no `rtree` required).
        Returns True if 3D point `pt` lies strictly inside closed `trimesh.Trimesh` `tri`.
        """
        tris = np.asarray(tri.triangles, dtype=np.float64)
        if len(tris) == 0:
            return False
        p = np.asarray(pt, dtype=np.float64).reshape(3)
        # Irrational-like unit ray direction avoids axis-aligned edge/vertex degeneracies
        d = np.array([0.83146961, 0.41923185, 0.36481072], dtype=np.float64)
        d /= np.linalg.norm(d)

        # Broad-phase AABB ray cull: since d > 0 on all 3 axes, only triangles with max >= p can be hit for t > 0
        tri_max = tris.max(axis=1)
        cand = tris[np.all(tri_max >= p - 1e-9, axis=1)]
        if len(cand) == 0:
            return False

        v0 = cand[:, 0, :]
        e1 = cand[:, 1, :] - v0
        e2 = cand[:, 2, :] - v0
        h = np.cross(np.broadcast_to(d, e2.shape), e2)
        a = np.einsum("ij,ij->i", e1, h)
        valid = np.abs(a) > 1e-12
        if not np.any(valid):
            return False

        v0 = v0[valid]
        e1 = e1[valid]
        e2 = e2[valid]
        h = h[valid]
        f = 1.0 / a[valid]
        s = p - v0
        u = f * np.einsum("ij,ij->i", s, h)
        keep_u = (u >= 0.0) & (u <= 1.0)
        if not np.any(keep_u):
            return False

        s = s[keep_u]
        e1 = e1[keep_u]
        e2 = e2[keep_u]
        f = f[keep_u]
        u = u[keep_u]
        q = np.cross(s, e1)
        v = f * (q @ d)
        keep_v = (v >= 0.0) & ((u + v) <= 1.0)
        if not np.any(keep_v):
            return False

        t = f[keep_v] * np.einsum("ij,ij->i", e2[keep_v], q[keep_v])
        hits = int(np.count_nonzero(t > 1e-9))
        return (hits % 2) == 1

    @staticmethod
    def _repair_single_trimesh(tri, is_porous=False):
        tri.units = "millimeters"
        if not tri.is_watertight:
            repair_fns = [
                trimesh.repair.fix_normals,
                trimesh.repair.fix_inversion,
                trimesh.repair.broken_faces,
            ]
            if not is_porous:
                repair_fns.insert(1, trimesh.repair.fill_holes)

            for fn in repair_fns:
                try:
                    fn(tri)
                except Exception:
                    pass
        elif tri.volume < 0.0:
            try:
                trimesh.repair.fix_inversion(tri)
            except Exception:
                pass
        tri.units = "millimeters"
        return tri

    def _extract_and_clean_polydata(self, pv_in, mask_in, is_porous,
                                    already_smoothed, _emit, label):
        _emit(0.05, f"Preparing {label} mesh…")
        needs_smoothing = not already_smoothed
        if pv_in is not None and pv_in.n_points > 0:
            mesh = pv_in.copy()
        else:
            needs_smoothing = True
            if mask_in is None or not np.any(mask_in):
                raise RuntimeError(f"No {label} mesh or mask data available to export")
            sz, sy, sx = self.spacing
            padded = np.pad(
                (mask_in > 0).astype(np.uint8),
                1, mode="constant", constant_values=0,
            )
            mesh = _extract_isosurface_polydata(
                padded, self.spacing, origin_xyz=(-sx, -sy, -sz), level=0.5
            )

        _emit(0.20, f"Surface extraction & cleaning ({label})…")
        try:
            surf = mesh.extract_surface()
            if not surf.is_all_triangles:
                surf = surf.triangulate()
            mesh = _fast_clean_polydata(surf)
        except Exception:
            mesh = _fast_clean_polydata(mesh)

        if needs_smoothing:
            if is_porous:
                _emit(0.45, f"Taubin volume-preserving smoothing ({label})…")
                try:
                    mesh = mesh.smooth_taubin(
                        n_iter=20, pass_band=0.08, normalize_coordinates=True
                    )
                except TypeError:
                    mesh = mesh.smooth_taubin(n_iter=20, pass_band=0.08)
                except Exception:
                    pass
                mesh = _fast_clean_polydata(mesh)
            else:
                _emit(0.38, f"Smoothing solid surface ({label})…")
                try:
                    mesh = mesh.smooth(n_iter=15, relaxation_factor=0.05)
                except Exception:
                    pass
                _emit(0.55, f"Taubin smoothing & hole filling ({label})…")
                try:
                    mesh = mesh.smooth_taubin(
                        n_iter=30, pass_band=0.1, normalize_coordinates=True
                    )
                except TypeError:
                    mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1)
                except Exception:
                    pass
                try:
                    mesh = mesh.fill_holes(1000.0)
                except Exception:
                    pass
                mesh = _fast_clean_polydata(mesh)
        return mesh

    def _prepare_validated_trimesh(self, pv_in, mask_in, is_porous,
                                   already_smoothed, p_lo, p_hi, label):
        def _emit(frac, msg):
            pct = int(round(p_lo + frac * (p_hi - p_lo)))
            self.progress.emit(pct, msg)

        mesh = self._extract_and_clean_polydata(
            pv_in, mask_in, is_porous, already_smoothed, _emit, label
        )

        _emit(0.75, f"Validating {label} manifold topology…")
        tri = trimesh.Trimesh(
            vertices=mesh.points,
            faces=mesh.faces.reshape(-1, 4)[:, 1:],
            process=False,
        )
        tri.units = "millimeters"

        try:
            comps = tri.split(only_watertight=False)
            if len(comps) > 1:
                if is_porous:
                    max_faces = max(len(c.faces) for c in comps)
                    threshold_faces = 0.05 * max_faces
                    kept = [c for c in comps if len(c.faces) >= threshold_faces]
                    if kept:
                        tri = trimesh.util.concatenate(kept)
                else:
                    tri = max(comps, key=lambda c: len(c.faces))
                tri.units = "millimeters"
        except Exception:
            pass

        tri = self._repair_single_trimesh(tri, is_porous=is_porous)
        _emit(1.0, f"{label.capitalize()} mesh ready")
        return tri

    def _prepare_validated_host_trimeshes(self, pv_in, mask_in,
                                          already_smoothed, p_lo, p_hi):
        """
        Prepare all disconnected pieces of the resected host bone overlay as
        separate, individually watertight `trimesh.Trimesh` objects while
        preserving internal enclosed bone cavities inside their parent piece
        and filtering out microscopic slicing dust.
        """
        def _emit(frac, msg):
            pct = int(round(p_lo + frac * (p_hi - p_lo)))
            self.progress.emit(pct, msg)

        mesh = self._extract_and_clean_polydata(
            pv_in, mask_in, False, already_smoothed, _emit, "host bone overlay"
        )

        _emit(0.70, "Splitting & validating host bone overlay pieces…")
        tri_full = trimesh.Trimesh(
            vertices=mesh.points,
            faces=mesh.faces.reshape(-1, 4)[:, 1:],
            process=False,
        )
        tri_full.units = "millimeters"

        try:
            raw_comps = list(tri_full.split(only_watertight=False))
        except Exception:
            raw_comps = [tri_full]

        if not raw_comps:
            raw_comps = [tri_full]

        # Filter out microscopic 1-2 voxel staircase crumbs (<40 faces AND <0.5% of max piece)
        # while retaining 100% of genuine anatomical host bone pieces.
        max_faces = max(len(c.faces) for c in raw_comps)
        kept = [
            c for c in raw_comps
            if len(c.faces) >= 40 or len(c.faces) >= 0.005 * max_faces
        ]
        if not kept:
            kept = [max(raw_comps, key=lambda c: len(c.faces))]

        # Repair every candidate shell individually first so normals point outward
        # (positive signed volume) and `trimesh.contains()` ray tests are exact.
        repaired_shells = []
        n_kept = len(kept)
        for i, comp in enumerate(kept):
            frac = 0.72 + 0.20 * ((i + 1) / max(1, n_kept))
            _emit(frac, f"Validating host bone shell {i + 1}/{n_kept}…")
            repaired_shells.append(self._repair_single_trimesh(comp, is_porous=False))

        # Keep internal closed marrow/void cavities attached to their enclosing bone piece
        # (odd containment depth = internal cavity -> invert normals & merge into parent;
        #  even containment depth = standalone physical bone piece).
        if len(repaired_shells) > 1:
            repaired_shells.sort(key=lambda c: -len(c.faces))
            n_s = len(repaired_shells)
            enclosing_parents = [[] for _ in range(n_s)]

            for j in range(n_s):
                in_lo, in_hi = repaired_shells[j].bounds
                sample_pt = repaired_shells[j].vertices[:1]
                for i in range(j):
                    c_out = repaired_shells[i]
                    out_lo, out_hi = c_out.bounds
                    if np.all(in_lo > out_lo) and np.all(in_hi < out_hi):
                        try:
                            if c_out.is_watertight and self._point_inside_closed_trimesh(
                                c_out, sample_pt[0]
                            ):
                                enclosing_parents[j].append(i)
                        except Exception:
                            pass

            piece_inners = {i: [] for i in range(n_s) if len(enclosing_parents[i]) % 2 == 0}
            for j in range(n_s):
                depth = len(enclosing_parents[j])
                if depth % 2 == 1:
                    # Immediate enclosing solid shell is the smallest container in enclosing_parents[j]
                    imm_parent = enclosing_parents[j][-1]
                    cavity_shell = repaired_shells[j].copy()
                    cavity_shell.invert()
                    piece_inners[imm_parent].append(cavity_shell)

            final_pieces = []
            for i in sorted(piece_inners.keys()):
                base_shell = repaired_shells[i]
                if piece_inners[i]:
                    combined = trimesh.util.concatenate([base_shell] + piece_inners[i])
                    combined.units = "millimeters"
                    final_pieces.append(combined)
                else:
                    final_pieces.append(base_shell)
        else:
            final_pieces = repaired_shells

        # Deterministic ordering: largest anatomical piece first, tie-broken by Z, Y, X min bound
        final_pieces.sort(
            key=lambda c: (
                -len(c.faces),
                float(c.bounds[0][2]),
                float(c.bounds[0][1]),
                float(c.bounds[0][0]),
            )
        )
        _emit(1.0, f"Host bone overlay ready ({len(final_pieces)} piece(s))")
        return final_pieces

    def run(self):
        try:
            has_host_source = (
                (self.host_pv_mesh is not None and self.host_pv_mesh.n_points > 0)
                or (self.host_mask_data is not None and np.any(self.host_mask_data))
            )
            do_host = self.export_host and has_host_source

            if do_host:
                tri_primary = self._prepare_validated_trimesh(
                    self.pv_mesh, self.segmented_data,
                    is_porous=self.is_porous,
                    already_smoothed=self.already_smoothed,
                    p_lo=4, p_hi=46,
                    label=self.primary_name,
                )
                host_pieces = self._prepare_validated_host_trimeshes(
                    self.host_pv_mesh, self.host_mask_data,
                    already_smoothed=(
                        self.host_pv_mesh is not None and self.host_pv_mesh.n_points > 0
                    ),
                    p_lo=48, p_hi=78,
                )
            else:
                tri_primary = self._prepare_validated_trimesh(
                    self.pv_mesh, self.segmented_data,
                    is_porous=self.is_porous,
                    already_smoothed=self.already_smoothed,
                    p_lo=4, p_hi=80,
                    label=self.primary_name,
                )
                host_pieces = []

            saved_paths = []
            stem, ext = os.path.splitext(self.out_path)
            if self.file_format == "3mf":
                if not ext:
                    ext = ".3mf"
                p_primary_end = 86 if host_pieces else 98
                self.progress.emit(80, f"Writing {self.primary_name} 3MF archive (unit=mm)…")
                _export_3mf_archive(
                    [(self.primary_name, tri_primary)],
                    self.out_path,
                    progress_cb=self.progress.emit,
                    p_start=81,
                    p_end=p_primary_end,
                )
                saved_paths.append(self.out_path)

                n_h = len(host_pieces)
                for idx, tri_h in enumerate(host_pieces, start=1):
                    if n_h == 1:
                        h_name = "host_bone"
                        host_out_path = f"{stem}_host_bone{ext}"
                    else:
                        h_name = f"host_bone_{idx}"
                        host_out_path = f"{stem}_host_bone_{idx}{ext}"
                    p_lo_h = int(round(86 + ((idx - 1) / n_h) * 12))
                    p_hi_h = int(round(86 + (idx / n_h) * 12))
                    self.progress.emit(
                        p_lo_h,
                        f"Writing {h_name} 3MF archive ({idx}/{n_h})…",
                    )
                    _export_3mf_archive(
                        [(h_name, tri_h)],
                        host_out_path,
                        progress_cb=self.progress.emit,
                        p_start=p_lo_h,
                        p_end=max(p_lo_h + 1, p_hi_h),
                    )
                    saved_paths.append(host_out_path)
            else:
                if not ext:
                    ext = ".stl"
                self.progress.emit(82, f"Saving {self.primary_name} STL…")
                tri_primary.export(self.out_path, file_type="stl")
                saved_paths.append(self.out_path)

                n_h = len(host_pieces)
                for idx, tri_h in enumerate(host_pieces, start=1):
                    if n_h == 1:
                        host_out_path = f"{stem}_host_bone{ext}"
                    else:
                        host_out_path = f"{stem}_host_bone_{idx}{ext}"
                    pct = int(round(86 + (idx / max(1, n_h)) * 12))
                    self.progress.emit(
                        pct,
                        f"Saving companion host bone STL ({idx}/{n_h})…",
                    )
                    tri_h.export(host_out_path, file_type="stl")
                    saved_paths.append(host_out_path)

            self.progress.emit(100, "Export complete")
            self.finished.emit(saved_paths)

        except Exception as e:
            self.error.emit(str(e))


# Backward-compatible alias
ExportSTLWorker = ExportMeshWorker


# ══════════════════════════════════════════════════════════════════════════════
#  SLICE VIEW WIDGET  (custom QWidget for 2D CT viewport with ±90° rotation)
# ══════════════════════════════════════════════════════════════════════════════

class SliceViewWidget(QWidget):
    """
    Renders a single orthogonal CT slice with:
      - 2D planar reorientation in ±90° increments (0°, 90°, 180°, 270°)
      - Bidirectional coordinate mapping for crosshairs & click navigation
      - Dynamic anatomical orientation labels that rotate with the view
      - Window/Level contrast (right-click drag)
      - Coloured crosshair lines synchronised with other views
      - Semi-transparent mask overlay
      - Mouse-wheel slice scrolling
      - Click-to-navigate (left button)
    """
    slice_changed    = pyqtSignal(str, int)        # (plane, new_index)
    click_navigated  = pyqtSignal(str, int, int)   # (plane, unrot_row, unrot_col)
    wl_changed_sig   = pyqtSignal(int, int)        # (wl, ww) broadcast
    rotation_changed = pyqtSignal(str, int)        # (plane, degrees_cw)

    def __init__(self, plane, border_color, orientation_labels, parent=None):
        super().__init__(parent)
        self.plane = plane
        self.border_color = border_color
        self.base_ori_labels = dict(orientation_labels)
        self.ori_labels = dict(orientation_labels)
        self.rotation_k = 0

        self.volume_data = None   # int16  (Z, Y, X)
        self.mask_data   = None   # uint8  (Z, Y, X)

        self.slice_index = 0
        self.max_index   = 0

        self.cross_h = 0   # unrotated row
        self.cross_v = 0   # unrotated col

        self.wl = DEFAULT_WL
        self.ww = DEFAULT_WW

        self.show_mask    = True
        self.mask_opacity = 95     # 0–255

        self._wl_dragging   = False
        self._drag_origin   = QPoint()
        self._drag_wl_start = DEFAULT_WL
        self._drag_ww_start = DEFAULT_WW

        self._qimage = None
        self._unrot_dims = (1, 1)
        self._img_rect = (0, 0, 1, 1)

        self.setMinimumSize(160, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

    # ── rotation API ──────────────────────────────────────────────────────

    def rotate_step(self, delta_k):
        self.set_rotation_k((self.rotation_k + delta_k) % 4)

    def set_rotation_k(self, k):
        k = int(k) % 4
        if k != self.rotation_k:
            self.rotation_k = k
            self._update_rotated_labels()
            self._rebuild_image()
            self.rotation_changed.emit(self.plane, self.rotation_degrees)

    @property
    def rotation_degrees(self):
        return self.rotation_k * 90

    def _update_rotated_labels(self):
        dirs = ['top', 'right', 'bottom', 'left']
        k = self.rotation_k
        new_labels = {}
        for i, d in enumerate(dirs):
            src_dir = dirs[(i - k) % 4]
            if src_dir in self.base_ori_labels:
                new_labels[d] = self.base_ori_labels[src_dir]
        self.ori_labels = new_labels

    def _unrot_to_rot_coords(self, r, c):
        H, W = self._unrot_dims
        k = self.rotation_k
        if k == 0:
            return r, c
        elif k == 1:  # 90° CW
            return c, max(0, H - 1 - r)
        elif k == 2:  # 180°
            return max(0, H - 1 - r), max(0, W - 1 - c)
        else:         # 270° CW (90° CCW)
            return max(0, W - 1 - c), r

    def _rot_to_unrot_coords(self, r_rot, c_rot):
        H, W = self._unrot_dims
        k = self.rotation_k
        if k == 0:
            r, c = r_rot, c_rot
        elif k == 1:  # 90° CW
            r, c = H - 1 - c_rot, r_rot
        elif k == 2:  # 180°
            r, c = H - 1 - r_rot, W - 1 - c_rot
        else:         # 270° CW (90° CCW)
            r, c = c_rot, W - 1 - r_rot
        return int(np.clip(r, 0, max(0, H - 1))), int(np.clip(c, 0, max(0, W - 1)))

    # ── data setters ──────────────────────────────────────────────────────

    def set_volume(self, vol):
        self.volume_data = vol
        if self.mask_data is not None and (vol is None or self.mask_data.shape != vol.shape):
            self.mask_data = None
        if vol is not None:
            self._recompute_max()
            self.slice_index = min(self.slice_index, self.max_index)
        self._rebuild_image()

    def set_mask(self, mask):
        self.mask_data = mask
        self._rebuild_image()

    def set_slice(self, idx):
        idx = max(0, min(idx, self.max_index))
        if idx != self.slice_index:
            self.slice_index = idx
            self._rebuild_image()

    def set_crosshairs(self, h, v):
        self.cross_h = h
        self.cross_v = v
        self.update()

    def set_wl_ww(self, wl, ww):
        self.wl = wl
        self.ww = max(1, ww)
        self._rebuild_image()

    # ── internal helpers ──────────────────────────────────────────────────

    def _recompute_max(self):
        if self.volume_data is None:
            self.max_index = 0
            return
        s = self.volume_data.shape
        if   self.plane == 'axial':    self.max_index = s[0] - 1
        elif self.plane == 'coronal':  self.max_index = s[1] - 1
        elif self.plane == 'sagittal': self.max_index = s[2] - 1

    def _extract_slice(self):
        if self.volume_data is None:
            return None
        i = self.slice_index
        if   self.plane == 'axial':    return self.volume_data[i, :, :]
        elif self.plane == 'coronal':  return self.volume_data[:, i, :]
        elif self.plane == 'sagittal': return self.volume_data[:, :, i]

    def _extract_mask_slice(self):
        if self.mask_data is None:
            return None
        i = self.slice_index
        if   self.plane == 'axial':    return self.mask_data[i, :, :]
        elif self.plane == 'coronal':  return self.mask_data[:, i, :]
        elif self.plane == 'sagittal': return self.mask_data[:, :, i]

    def _rebuild_image(self):
        raw = self._extract_slice()
        if raw is None:
            self._qimage = None
            self.update()
            return

        self._unrot_dims = (raw.shape[0], raw.shape[1])

        if self.rotation_k != 0:
            raw = np.rot90(raw, k=-self.rotation_k)

        gray = apply_windowing(raw, self.wl, self.ww)
        h, w = gray.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = gray
        rgba[:, :, 1] = gray
        rgba[:, :, 2] = gray
        rgba[:, :, 3] = 255

        msl = self._extract_mask_slice()
        if msl is not None and self.show_mask:
            if self.rotation_k != 0:
                msl = np.rot90(msl, k=-self.rotation_k)
            if msl.shape == gray.shape:
                where = msl > 0
                if np.any(where):
                    a = self.mask_opacity / 255.0
                    ia = 1.0 - a
                    rgba[where, 0] = np.clip(
                        rgba[where, 0].astype(np.float32) * ia + MASK_RGBA[0] * a,
                        0, 255).astype(np.uint8)
                    rgba[where, 1] = np.clip(
                        rgba[where, 1].astype(np.float32) * ia + MASK_RGBA[1] * a,
                        0, 255).astype(np.uint8)
                    rgba[where, 2] = np.clip(
                        rgba[where, 2].astype(np.float32) * ia + MASK_RGBA[2] * a,
                        0, 255).astype(np.uint8)

        self._qimage = numpy_to_qimage_rgba(rgba)
        self.update()

    def _image_dims(self):
        if self._qimage is None:
            return (1, 1)
        return (self._qimage.height(), self._qimage.width())

    def _compute_draw_rect(self):
        if self._qimage is None:
            return (0, 0, self.width(), self.height())
        iw, ih = self._qimage.width(), self._qimage.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / max(iw, 1), wh / max(ih, 1))
        dw = int(iw * scale)
        dh = int(ih * scale)
        dx = (ww - dw) // 2
        dy = (wh - dh) // 2
        return (dx, dy, dw, dh)

    def _widget_to_image(self, wx, wy):
        dx, dy, dw, dh = self._img_rect
        ih, iw = self._image_dims()
        col = (wx - dx) / max(dw, 1) * iw
        row = (wy - dy) / max(dh, 1) * ih
        return int(np.clip(col, 0, iw - 1)), int(np.clip(row, 0, ih - 1))

    def _image_to_widget(self, col_rot, row_rot):
        dx, dy, dw, dh = self._img_rect
        ih, iw = self._image_dims()
        wx = dx + col_rot / max(iw, 1) * dw
        wy = dy + row_rot / max(ih, 1) * dh
        return int(wx), int(wy)

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(0, 0, 0))

        rect = self._compute_draw_rect()
        self._img_rect = rect
        dx, dy, dw, dh = rect

        if self._qimage is not None:
            p.drawImage(QRect(dx, dy, dw, dh), self._qimage)

        rot_row, rot_col = self._unrot_to_rot_coords(self.cross_h, self.cross_v)
        if self.rotation_k % 2 == 0:
            pen_row = QPen(QColor(0, 255, 0, 160), 1, Qt.DashLine)
            pen_col = QPen(QColor(0, 200, 255, 160), 1, Qt.DashLine)
        else:
            pen_row = QPen(QColor(0, 200, 255, 160), 1, Qt.DashLine)
            pen_col = QPen(QColor(0, 255, 0, 160), 1, Qt.DashLine)

        _, wy = self._image_to_widget(0, rot_row)
        p.setPen(pen_row)
        p.drawLine(dx, wy, dx + dw, wy)

        wx, _ = self._image_to_widget(rot_col, 0)
        p.setPen(pen_col)
        p.drawLine(wx, dy, wx, dy + dh)

        label_font = QFont("Arial", 11, QFont.Bold)
        p.setFont(label_font)
        p.setPen(QColor(255, 255, 255, 200))
        margin = 8
        if 'top' in self.ori_labels:
            p.drawText(dx + dw // 2 - 6, dy + margin + 14, self.ori_labels['top'])
        if 'bottom' in self.ori_labels:
            p.drawText(dx + dw // 2 - 6, dy + dh - margin, self.ori_labels['bottom'])
        if 'left' in self.ori_labels:
            p.drawText(dx + margin, dy + dh // 2 + 5, self.ori_labels['left'])
        if 'right' in self.ori_labels:
            p.drawText(dx + dw - margin - 12, dy + dh // 2 + 5, self.ori_labels['right'])

        info_font = QFont("Consolas", 9)
        p.setFont(info_font)
        p.setPen(QColor(200, 200, 200))
        info = f"WL:{self.wl}  WW:{self.ww}  Rot:{self.rotation_degrees}°"
        p.drawText(dx + 6, dy + dh - 6, info)

        p.setPen(QPen(self.border_color, 3))
        p.drawRect(self.rect().adjusted(1, 1, -2, -2))

        p.setPen(QColor(self.border_color))
        p.setFont(QFont("Consolas", 10, QFont.Bold))
        p.drawText(dx + 6, dy + 16,
                   f"{self.plane.capitalize()} #{self.slice_index}")
        p.end()

    # ── mouse / keyboard events ───────────────────────────────────────────

    def wheelEvent(self, ev):
        delta = 1 if ev.angleDelta().y() < 0 else -1
        new_idx = max(0, min(self.slice_index + delta, self.max_index))
        if new_idx != self.slice_index:
            self.slice_index = new_idx
            self._rebuild_image()
            self.slice_changed.emit(self.plane, new_idx)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            col_rot, row_rot = self._widget_to_image(ev.x(), ev.y())
            unrot_row, unrot_col = self._rot_to_unrot_coords(row_rot, col_rot)
            self.click_navigated.emit(self.plane, unrot_row, unrot_col)
        elif ev.button() == Qt.RightButton:
            self._wl_dragging   = True
            self._drag_origin   = ev.pos()
            self._drag_wl_start = self.wl
            self._drag_ww_start = self.ww

    def mouseMoveEvent(self, ev):
        if self._wl_dragging:
            dx = ev.x() - self._drag_origin.x()
            dy = ev.y() - self._drag_origin.y()
            self.ww = max(1, self._drag_ww_start + dx * 4)
            self.wl = self._drag_wl_start - dy * 4
            self._rebuild_image()
            self.wl_changed_sig.emit(self.wl, self.ww)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._wl_dragging = False


# ══════════════════════════════════════════════════════════════════════════════
#  VIEWPORT PANELS  (2D Slice Header + 3D Interactive Gizmo Viewport)
# ══════════════════════════════════════════════════════════════════════════════

class SliceViewportPanel(QFrame):
    """
    Composite widget that pairs a SliceViewWidget with:
      - A compact header toolbar containing plane label, rotation badge,
        and ↺ -90° / ↻ +90° reorientation buttons
      - A vertical QSlider along its right edge for slice scrolling
    """
    def __init__(self, slice_view, parent=None):
        super().__init__(parent)
        self.slice_view = slice_view

        header = QWidget()
        header.setFixedHeight(26)
        header.setStyleSheet("background-color: #222; border-bottom: 1px solid #444;")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(6, 2, 6, 2)
        h_lay.setSpacing(4)

        self.lbl_title = QLabel(f"{self.slice_view.plane.capitalize()}")
        self.lbl_title.setStyleSheet(
            f"color: {self.slice_view.border_color.name()}; font-weight: bold; font-size: 11px;"
        )
        h_lay.addWidget(self.lbl_title)
        h_lay.addStretch(1)

        self.lbl_rot = QLabel("0°")
        self.lbl_rot.setStyleSheet("color: #aaa; font-family: Consolas; font-size: 10px;")
        h_lay.addWidget(self.lbl_rot)

        self.btn_rot_ccw = QPushButton("↺ -90°")
        self.btn_rot_ccw.setToolTip("Rotate 2D view -90° (Counter-Clockwise)")
        self.btn_rot_ccw.setFixedHeight(20)
        self.btn_rot_ccw.setStyleSheet("font-size: 10px; padding: 1px 6px;")
        self.btn_rot_ccw.clicked.connect(lambda: self.slice_view.rotate_step(-1))
        h_lay.addWidget(self.btn_rot_ccw)

        self.btn_rot_cw = QPushButton("↻ +90°")
        self.btn_rot_cw.setToolTip("Rotate 2D view +90° (Clockwise)")
        self.btn_rot_cw.setFixedHeight(20)
        self.btn_rot_cw.setStyleSheet("font-size: 10px; padding: 1px 6px;")
        self.btn_rot_cw.clicked.connect(lambda: self.slice_view.rotate_step(1))
        h_lay.addWidget(self.btn_rot_cw)

        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setInvertedAppearance(True)
        self.slider.setFixedWidth(18)
        self.slider.setStyleSheet("""
            QSlider::groove:vertical { background: #444; width: 6px; }
            QSlider::handle:vertical {
                background: #aaa; height: 14px; margin: 0 -4px;
                border-radius: 3px;
            }
        """)

        body_lay = QHBoxLayout()
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)
        body_lay.addWidget(self.slice_view, 1)
        body_lay.addWidget(self.slider, 0)

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)
        main_lay.addWidget(header, 0)
        main_lay.addLayout(body_lay, 1)

        self.slider.valueChanged.connect(self._on_slider)
        self.slice_view.slice_changed.connect(self._on_view_scroll)
        self.slice_view.rotation_changed.connect(self._on_rotation_changed)

    def _on_slider(self, val):
        self.slice_view.set_slice(val)
        self.slice_view.slice_changed.emit(self.slice_view.plane, val)

    def _on_view_scroll(self, _plane, idx):
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

    def _on_rotation_changed(self, _plane, deg):
        self.lbl_rot.setText(f"{deg}°")

    def update_range(self, max_idx):
        self.slider.setMaximum(max_idx)


class ThreeDViewportPanel(QFrame):
    """
    Wraps a pyvistaqt.QtInteractor in a styled frame with:
      - Eye-Dome Lighting (EDL), 4-peel depth peeling, and 3-point surgical lighting
      - Named-actor mesh updates that preserve active VTK 3D gizmos and camera pose
      - Interactive VTK 3D Gizmos for Tab 2 ("Scaffold Design"):
          * Oriented Box ROI (`vtkBoxWidget` + translucent preview surface)
          * Dual-Plane Osteotomy (`vtkLineWidget` + two individually tilted 3D planes)
          * Pattern Direction Vector (`pv.Arrow` + roll tick + draggable `vtkSphereWidget`)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("border: 3px solid " + COLOR_3D.name() + ";")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self.scaffold_mesh   = None
        self.bone_mesh       = None
        self.scaffold_actor  = None
        self.bone_actor      = None
        self.overlay_enabled = False
        self.bone_alpha      = DEFAULT_BONE_ALPHA
        self._has_camera_pose = False

        # Interactive 3D gizmo state
        self._updating_gizmo   = False
        self._box_widget       = None
        self._box_cb           = None
        self._line_widget      = None
        self._planes_cb        = None
        self._planes_params    = {}
        self._planes_size      = 50.0
        self._arrow_sphere_w   = None
        self._arrow_cb         = None
        self._arrow_centroid   = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._arrow_len        = 20.0
        self._arrow_roll_deg   = 0.0

        self._init_lighting_and_rendering()

    def _init_lighting_and_rendering(self):
        try:
            self.plotter.set_background(color="#1a1c22", top="#0a0c10")
        except Exception:
            self.plotter.set_background("#1a1a1a")

        try:
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass

        try:
            self.plotter.enable_depth_peeling(
                number_of_peels=4, occlusion_ratio=0.0
            )
        except Exception:
            pass

        self.plotter.remove_all_lights()

        key_light = pv.Light(
            position=(1.0, 1.0, 2.0),
            focal_point=(0.0, 0.0, 0.0),
            color="white",
            intensity=0.85,
        )
        key_light.positional = False
        self.plotter.add_light(key_light)

        fill_light = pv.Light(
            position=(-2.0, -1.0, -0.5),
            focal_point=(0.0, 0.0, 0.0),
            color="#B0C4DE",
            intensity=0.35,
        )
        fill_light.positional = False
        self.plotter.add_light(fill_light)

        rim_light = pv.Light(
            position=(0.0, 2.0, -2.0),
            focal_point=(0.0, 0.0, 0.0),
            color="white",
            intensity=0.50,
        )
        rim_light.positional = False
        self.plotter.add_light(rim_light)

        self.plotter.add_axes()

    def _safe_remove_actor(self, name):
        try:
            self.plotter.remove_actor(name, render=False)
        except Exception:
            pass

    # ── Mesh Display API (preserves VTK gizmos & custom lighting) ─────────

    def display_mesh(self, pv_mesh, color=None, reset_camera=None):
        """Display a single 3D mesh (used in Tab 1 Segmentation preview)."""
        if color is None:
            color = BONE_COLOR
        should_reset = bool(reset_camera) or (not self._has_camera_pose)

        self._safe_remove_actor("preview_bone")
        self._safe_remove_actor("scaffold")
        self._safe_remove_actor("overlay_bone")
        self.scaffold_actor = None
        self.bone_actor = None
        self.scaffold_mesh = pv_mesh
        self.bone_mesh = None

        if pv_mesh is not None and pv_mesh.n_points > 0:
            self.scaffold_actor = self.plotter.add_mesh(
                pv_mesh, color=color, smooth_shading=True,
                ambient=0.20, diffuse=0.80,
                specular=0.30, specular_power=25,
                name="preview_bone",
                reset_camera=False,
                render=False,
            )
            if should_reset:
                self.plotter.reset_camera()
                self._has_camera_pose = True
            else:
                try:
                    self.plotter.reset_camera_clipping_range()
                except Exception:
                    pass
        else:
            self._has_camera_pose = False

        self.plotter.render()

    def set_meshes(self, scaffold_mesh, bone_mesh=None,
                   scaffold_color=None, reset_camera=None):
        """
        Display dual-actor scene (cyan candidate ROI / porous scaffold +
        translucent anatomical host bone outside the cut) without destroying
        active VTK 3D cutting or orientation gizmos.
        """
        if scaffold_color is None:
            scaffold_color = SCAFFOLD_COLOR
        should_reset = bool(reset_camera) or (not self._has_camera_pose)

        self.scaffold_mesh = scaffold_mesh
        self.bone_mesh = bone_mesh

        self._safe_remove_actor("preview_bone")
        self._safe_remove_actor("scaffold")
        self._safe_remove_actor("overlay_bone")
        self.scaffold_actor = None
        self.bone_actor = None

        if scaffold_mesh is not None and scaffold_mesh.n_points > 0:
            self.scaffold_actor = self.plotter.add_mesh(
                scaffold_mesh, color=scaffold_color,
                smooth_shading=True,
                ambient=0.20, diffuse=0.80,
                specular=0.35, specular_power=25,
                name="scaffold",
                reset_camera=False,
                render=False,
            )

        if bone_mesh is not None and bone_mesh.n_points > 0:
            self.bone_actor = self.plotter.add_mesh(
                bone_mesh, color=OVERLAY_BONE_COLOR,
                smooth_shading=True,
                ambient=0.15, diffuse=0.75,
                specular=0.20, specular_power=15,
                opacity=self.bone_alpha if self.overlay_enabled else 0.0,
                name="overlay_bone",
                reset_camera=False,
                render=False,
            )
            if not self.overlay_enabled and self.bone_actor is not None:
                try:
                    self.bone_actor.SetVisibility(False)
                except Exception:
                    pass

        has_any_mesh = (
            (scaffold_mesh is not None and scaffold_mesh.n_points > 0)
            or (bone_mesh is not None and bone_mesh.n_points > 0)
        )
        if has_any_mesh:
            if should_reset:
                self.plotter.reset_camera()
                self._has_camera_pose = True
            else:
                try:
                    self.plotter.reset_camera_clipping_range()
                except Exception:
                    pass

        self.plotter.render()

    def set_overlay_enabled(self, enabled):
        self.overlay_enabled = bool(enabled)
        if self.bone_actor is not None:
            try:
                self.bone_actor.SetVisibility(self.overlay_enabled)
                if self.overlay_enabled:
                    self.bone_actor.GetProperty().SetOpacity(self.bone_alpha)
                self.plotter.render()
            except Exception:
                pass

    def set_bone_alpha(self, alpha):
        self.bone_alpha = float(alpha)
        if self.bone_actor is not None and self.overlay_enabled:
            try:
                self.bone_actor.GetProperty().SetOpacity(self.bone_alpha)
                self.plotter.render()
            except Exception:
                pass

    # ── 1. Oriented Box Cutting Gizmo (vtkBoxWidget) ──────────────────────

    def clear_cutting_gizmos(self, render=True):
        """Remove any active Box or Dual-Planes VTK cutting gizmos and actors."""
        self._updating_gizmo = True
        try:
            self.plotter.clear_box_widgets()
        except Exception:
            pass
        try:
            self.plotter.clear_line_widgets()
        except Exception:
            pass
        self._box_widget = None
        self._box_cb = None
        self._line_widget = None
        self._planes_cb = None
        for actor_name in (
            "gizmo_box_surface",
            "gizmo_plane_1", "gizmo_plane_1_wire",
            "gizmo_plane_2", "gizmo_plane_2_wire",
            "gizmo_planes_mid",
        ):
            self._safe_remove_actor(actor_name)
        self._updating_gizmo = False
        if render:
            self.plotter.render()

    def _build_oriented_box_polydata(self, params):
        cx = float(params.get("cx", 0.0))
        cy = float(params.get("cy", 0.0))
        cz = float(params.get("cz", 0.0))
        lx = max(0.2, float(params.get("lx", 20.0)))
        ly = max(0.2, float(params.get("ly", 20.0)))
        lz = max(0.2, float(params.get("lz", 20.0)))
        yaw   = float(params.get("yaw", 0.0))
        pitch = float(params.get("pitch", 0.0))
        roll  = float(params.get("roll", 0.0))

        cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=lx, y_length=ly, z_length=lz)
        R = euler_zyx_to_matrix(yaw, pitch, roll)
        pts = cube.points @ R.T + np.array([cx, cy, cz], dtype=np.float64)
        cube.points = pts
        return cube

    def _apply_params_to_vtk_box(self, params):
        if self._box_widget is None:
            return
        cx = float(params.get("cx", 0.0))
        cy = float(params.get("cy", 0.0))
        cz = float(params.get("cz", 0.0))
        lx = max(0.2, float(params.get("lx", 20.0)))
        ly = max(0.2, float(params.get("ly", 20.0)))
        lz = max(0.2, float(params.get("lz", 20.0)))
        yaw   = float(params.get("yaw", 0.0))
        pitch = float(params.get("pitch", 0.0))
        roll  = float(params.get("roll", 0.0))

        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Scale(lx, ly, lz)
        t.RotateX(roll)
        t.RotateY(pitch)
        t.RotateZ(yaw)
        t.Translate(cx, cy, cz)
        self._box_widget.SetTransform(t)

    def _update_box_surface_actor(self, params):
        box_pd = self._build_oriented_box_polydata(params)
        self.plotter.add_mesh(
            box_pd,
            color=BOX_GIZMO_COLOR,
            opacity=0.14,
            smooth_shading=False,
            pickable=False,
            name="gizmo_box_surface",
            reset_camera=False,
            render=False,
        )

    def set_box_gizmo(self, enabled, params=None, on_changed_cb=None):
        """Enable or disable the interactive 3D Oriented Box (`vtkBoxWidget`)."""
        self.clear_cutting_gizmos()
        if not enabled or params is None:
            return

        self._box_cb = on_changed_cb
        self._updating_gizmo = True
        try:
            self._box_widget = self.plotter.add_box_widget(
                callback=self._on_vtk_box_moved,
                bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
                factor=1.0,
                rotation_enabled=True,
                outline_translation=True,
                color=BOX_GIZMO_COLOR,
                pass_widget=True,
                interaction_event="always",
            )
            self._apply_params_to_vtk_box(params)
            self._update_box_surface_actor(params)
        finally:
            self._updating_gizmo = False
        self.plotter.render()

    def update_box_gizmo_from_params(self, params):
        """Sync 3D `vtkBoxWidget` pose/dimensions when sidebar spinboxes change."""
        if self._box_widget is None or self._updating_gizmo:
            return
        self._updating_gizmo = True
        try:
            self._apply_params_to_vtk_box(params)
            self._update_box_surface_actor(params)
        finally:
            self._updating_gizmo = False
        self.plotter.render()

    def _on_vtk_box_moved(self, poly, _widget=None):
        if self._updating_gizmo or poly is None or poly.n_points < 8:
            return
        pts = np.asarray(poly.points[:8], dtype=np.float64)
        vx = pts[1] - pts[0]
        vy = pts[3] - pts[0]
        vz = pts[4] - pts[0]
        lx = max(0.2, float(np.linalg.norm(vx)))
        ly = max(0.2, float(np.linalg.norm(vy)))
        lz = max(0.2, float(np.linalg.norm(vz)))
        center = pts.mean(axis=0)
        R = np.column_stack([vx / lx, vy / ly, vz / lz])
        yaw, pitch, roll = matrix_to_euler_zyx(R)

        new_params = {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "cz": float(center[2]),
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "lx": lx,
            "ly": ly,
            "lz": lz,
        }
        self._updating_gizmo = True
        try:
            self._update_box_surface_actor(new_params)
        finally:
            self._updating_gizmo = False

        if self._box_cb is not None:
            self._box_cb(new_params)

    # ── 2. Dual-Planes Osteotomy Gizmo (vtkLineWidget + Tilted Planes) ────

    def _update_dual_plane_actors(self, params, plane_size):
        mid, _n_line, p1, p2, n1, n2 = compute_dual_planes_geometry(params)
        sz = max(8.0, float(plane_size))

        pl1 = pv.Plane(center=p1, direction=n1, i_size=sz, j_size=sz, i_resolution=2, j_resolution=2)
        pl2 = pv.Plane(center=p2, direction=n2, i_size=sz, j_size=sz, i_resolution=2, j_resolution=2)
        mid_sph = pv.Sphere(radius=max(0.5, sz * 0.022), center=mid)

        self.plotter.add_mesh(
            pl1, color=PLANE1_COLOR, opacity=0.28,
            pickable=False, name="gizmo_plane_1",
            reset_camera=False, render=False,
        )
        self.plotter.add_mesh(
            pl1.extract_feature_edges(), color=PLANE1_COLOR, line_width=2.5,
            pickable=False, name="gizmo_plane_1_wire",
            reset_camera=False, render=False,
        )
        self.plotter.add_mesh(
            pl2, color=PLANE2_COLOR, opacity=0.28,
            pickable=False, name="gizmo_plane_2",
            reset_camera=False, render=False,
        )
        self.plotter.add_mesh(
            pl2.extract_feature_edges(), color=PLANE2_COLOR, line_width=2.5,
            pickable=False, name="gizmo_plane_2_wire",
            reset_camera=False, render=False,
        )
        self.plotter.add_mesh(
            mid_sph, color="#FFD166", opacity=0.95,
            pickable=False, name="gizmo_planes_mid",
            reset_camera=False, render=False,
        )
        return p1, p2

    def set_dual_planes_gizmo(self, enabled, params=None, plane_size=50.0, on_changed_cb=None):
        """Enable or disable the interactive Dual-Planes Osteotomy gizmo."""
        self.clear_cutting_gizmos()
        if not enabled or params is None:
            return

        self._planes_cb = on_changed_cb
        self._planes_params = dict(params)
        self._planes_size = float(plane_size)

        self._updating_gizmo = True
        try:
            p1, p2 = self._update_dual_plane_actors(self._planes_params, self._planes_size)
            self._line_widget = self.plotter.add_line_widget(
                callback=self._on_vtk_line_moved,
                bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                factor=1.0,
                use_vertices=True,
                color="#FFD166",
                pass_widget=True,
                interaction_event="always",
            )
            self._line_widget.SetPoint1(float(p1[0]), float(p1[1]), float(p1[2]))
            self._line_widget.SetPoint2(float(p2[0]), float(p2[1]), float(p2[2]))
        finally:
            self._updating_gizmo = False
        self.plotter.render()

    def update_dual_planes_from_params(self, params, plane_size=None):
        """Sync Dual-Planes 3D line & tilted plane actors from sidebar spinboxes."""
        if self._line_widget is None or self._updating_gizmo:
            return
        self._planes_params = dict(params)
        if plane_size is not None:
            self._planes_size = float(plane_size)

        self._updating_gizmo = True
        try:
            p1, p2 = self._update_dual_plane_actors(self._planes_params, self._planes_size)
            self._line_widget.SetPoint1(float(p1[0]), float(p1[1]), float(p1[2]))
            self._line_widget.SetPoint2(float(p2[0]), float(p2[1]), float(p2[2]))
        finally:
            self._updating_gizmo = False
        self.plotter.render()

    def _on_vtk_line_moved(self, point_a, point_b, _widget=None):
        if self._updating_gizmo:
            return
        pa = np.asarray(point_a, dtype=np.float64)
        pb = np.asarray(point_b, dtype=np.float64)
        diff = pb - pa
        gap = float(np.linalg.norm(diff))
        if gap < 0.2:
            return

        mid = 0.5 * (pa + pb)
        line_yaw, line_pitch = unit_vec_to_spherical_angles(diff)

        self._planes_params["mid_x"] = float(mid[0])
        self._planes_params["mid_y"] = float(mid[1])
        self._planes_params["mid_z"] = float(mid[2])
        self._planes_params["gap_l"] = gap
        self._planes_params["line_yaw"] = line_yaw
        self._planes_params["line_pitch"] = line_pitch

        self._updating_gizmo = True
        try:
            self._update_dual_plane_actors(self._planes_params, self._planes_size)
        finally:
            self._updating_gizmo = False

        if self._planes_cb is not None:
            self._planes_cb(dict(self._planes_params))

    # ── 3. Pattern Direction Vector Arrow Gizmo (Arrow + Sphere Handle) ───

    def clear_direction_arrow_gizmo(self):
        self._updating_gizmo = True
        try:
            self.plotter.clear_sphere_widgets()
        except Exception:
            pass
        self._arrow_sphere_w = None
        self._arrow_cb = None
        self._safe_remove_actor("gizmo_dir_arrow")
        self._safe_remove_actor("gizmo_dir_roll_tick")
        self._safe_remove_actor("gizmo_dir_origin")
        self._updating_gizmo = False
        self.plotter.render()

    def _update_arrow_actors(self, centroid_xyz, yaw_deg, pitch_deg, roll_deg, arrow_len):
        c = np.asarray(centroid_xyz, dtype=np.float64)
        R = pattern_direction_to_matrix(yaw_deg, pitch_deg, roll_deg)
        dir_vec = R[:, 2]     # Primary scaffold Z' axis in world space
        roll_vec = R[:, 0]    # Orthogonal X' fin showing Vector Roll orientation
        L = max(4.0, float(arrow_len))

        arrow_pd = pv.Arrow(
            start=c,
            direction=dir_vec,
            scale=L,
            tip_length=0.24,
            tip_radius=0.085,
            shaft_radius=0.032,
        )
        # Orthogonal roll indicator bar near shaft midpoint
        mid_shaft = c + 0.55 * L * dir_vec
        fin_end = mid_shaft + 0.22 * L * roll_vec
        roll_cyl = pv.Cylinder(
            center=0.5 * (mid_shaft + fin_end),
            direction=roll_vec,
            radius=max(0.15, L * 0.018),
            height=0.22 * L,
        )
        origin_sph = pv.Sphere(radius=max(0.35, L * 0.045), center=c)

        self.plotter.add_mesh(
            arrow_pd, color=ARROW_COLOR, smooth_shading=True,
            pickable=False, name="gizmo_dir_arrow",
            reset_camera=False, render=False,
        )
        self.plotter.add_mesh(
            roll_cyl, color="#FF595E", smooth_shading=True,
            pickable=False, name="gizmo_dir_roll_tick",
            reset_camera=False, render=False,
        )
        self.plotter.add_mesh(
            origin_sph, color="#FFFFFF", smooth_shading=True,
            pickable=False, name="gizmo_dir_origin",
            reset_camera=False, render=False,
        )
        tip_pos = c + L * dir_vec
        return tip_pos

    def set_direction_arrow_gizmo(self, enabled, centroid_xyz=(0.0, 0.0, 0.0),
                                  yaw_deg=0.0, pitch_deg=90.0, roll_deg=0.0,
                                  arrow_len=22.0, on_changed_cb=None):
        """Show or hide the interactive 3D Pattern Direction Vector arrow."""
        self.clear_direction_arrow_gizmo()
        if not enabled:
            return

        self._arrow_cb = on_changed_cb
        self._arrow_centroid = np.asarray(centroid_xyz, dtype=np.float64)
        self._arrow_len = max(4.0, float(arrow_len))
        self._arrow_roll_deg = float(roll_deg)

        self._updating_gizmo = True
        try:
            tip_pos = self._update_arrow_actors(
                self._arrow_centroid, yaw_deg, pitch_deg, roll_deg, self._arrow_len
            )
            self._arrow_sphere_w = self.plotter.add_sphere_widget(
                callback=self._on_arrow_tip_dragged,
                center=tuple(tip_pos),
                radius=max(0.5, self._arrow_len * 0.075),
                color="#FFD700",
                selected_color="#FF595E",
                pass_widget=True,
                test_callback=False,
                interaction_event="always",
            )
        finally:
            self._updating_gizmo = False
        self.plotter.render()

    def update_direction_arrow_from_params(self, centroid_xyz, yaw_deg, pitch_deg,
                                           roll_deg, arrow_len=None):
        """Update 3D arrow and tip sphere handle when spinboxes or ROI centroid change."""
        if self._arrow_sphere_w is None or self._updating_gizmo:
            return
        self._arrow_centroid = np.asarray(centroid_xyz, dtype=np.float64)
        if arrow_len is not None:
            self._arrow_len = max(4.0, float(arrow_len))
        self._arrow_roll_deg = float(roll_deg)

        self._updating_gizmo = True
        try:
            tip_pos = self._update_arrow_actors(
                self._arrow_centroid, yaw_deg, pitch_deg, roll_deg, self._arrow_len
            )
            self._arrow_sphere_w.SetCenter(float(tip_pos[0]), float(tip_pos[1]), float(tip_pos[2]))
            self._arrow_sphere_w.SetRadius(max(0.5, self._arrow_len * 0.075))
        finally:
            self._updating_gizmo = False
        self.plotter.render()

    def _on_arrow_tip_dragged(self, new_center, _widget=None):
        if self._updating_gizmo:
            return
        pt = np.asarray(new_center, dtype=np.float64)
        vec = pt - self._arrow_centroid
        if float(np.linalg.norm(vec)) < 1e-4:
            return
        yaw_deg, pitch_deg = unit_vec_to_spherical_angles(vec)

        self._updating_gizmo = True
        try:
            tip_pos = self._update_arrow_actors(
                self._arrow_centroid, yaw_deg, pitch_deg,
                self._arrow_roll_deg, self._arrow_len
            )
            if self._arrow_sphere_w is not None:
                self._arrow_sphere_w.SetCenter(
                    float(tip_pos[0]), float(tip_pos[1]), float(tip_pos[2])
                )
        finally:
            self._updating_gizmo = False

        if self._arrow_cb is not None:
            self._arrow_cb(yaw_deg, pitch_deg)

    def reset_camera(self):
        try:
            self.plotter.reset_camera()
            self._has_camera_pose = True
            self.plotter.render()
        except Exception:
            pass

    def close_plotter(self):
        try:
            self.clear_cutting_gizmos()
            self.clear_direction_arrow_gizmo()
            self.plotter.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  COLLAPSIBLE GROUP BOX & CUTTING TOOLS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class CollapsibleGroupBox(QGroupBox):
    """A QGroupBox that can be collapsed/expanded by clicking its title."""
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked):
        for child in self.children():
            if isinstance(child, QWidget):
                child.setVisible(checked)


class CuttingToolsPanel(CollapsibleGroupBox):
    """
    Sidebar panel in Tab 2 ("Scaffold Design") defining the 3D surgical ROI:
      1. No Cut (Entire Segmented Bone)
      2. Cut by Rectangle Primitive (Box ROI) — Center, Yaw/Pitch/Roll, L/W/H
      3. Cut by Dual Planes (Osteotomy Wedge / Resection) — Midpoint, Line
         Yaw/Pitch, Resection Gap L, and individual Plane 1 / Plane 2 Tilt angles
    Two-way synchronized with VTK 3D gizmos in `ThreeDViewportPanel`.
    """
    sig_cut_mode_changed             = pyqtSignal(str)
    sig_cut_params_changed           = pyqtSignal()
    sig_reset_cut_to_bone            = pyqtSignal()
    sig_cut_gizmo_visibility_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__("Cutting Tools", parent)
        lay = QVBoxLayout()
        lay.setSpacing(5)

        lay.addWidget(QLabel("Cutting Method:"))
        self.combo_cut_mode = QComboBox()
        self.combo_cut_mode.addItems(ALL_CUT_MODES)
        self.combo_cut_mode.setCurrentText(CUT_MODE_NONE)
        lay.addWidget(self.combo_cut_mode)

        self.lbl_cut_hint = QLabel("")
        self.lbl_cut_hint.setWordWrap(True)
        self.lbl_cut_hint.setStyleSheet("color: #9bc; font-size: 10px; font-style: italic;")
        lay.addWidget(self.lbl_cut_hint)

        self.chk_hide_gizmo = QCheckBox("Hide Cutting Tool in 3D View")
        self.chk_hide_gizmo.setChecked(False)
        self.chk_hide_gizmo.setEnabled(False)
        self.chk_hide_gizmo.setToolTip(
            "Hides the 3D box or dual-plane cutting gizmo overlay in the 3D viewport "
            "to remove visual clutter while keeping the active cut applied."
        )
        self.chk_hide_gizmo.toggled.connect(self.sig_cut_gizmo_visibility_changed)
        lay.addWidget(self.chk_hide_gizmo)

        # ── Sub-panel A: Oriented Box Primitive ───────────────────────────
        self.box_widget = QWidget()
        box_lay = QVBoxLayout(self.box_widget)
        box_lay.setContentsMargins(0, 4, 0, 2)
        box_lay.setSpacing(4)

        box_lay.addWidget(self._make_sub_header("Center Position (mm):"))
        grid_bc = QGridLayout()
        grid_bc.setSpacing(4)
        self.spin_box_cx = self._make_dspin(-5000.0, 5000.0, 0.0, 1.0, " mm")
        self.spin_box_cy = self._make_dspin(-5000.0, 5000.0, 0.0, 1.0, " mm")
        self.spin_box_cz = self._make_dspin(-5000.0, 5000.0, 0.0, 1.0, " mm")
        grid_bc.addWidget(QLabel("X:"), 0, 0); grid_bc.addWidget(self.spin_box_cx, 0, 1)
        grid_bc.addWidget(QLabel("Y:"), 0, 2); grid_bc.addWidget(self.spin_box_cy, 0, 3)
        grid_bc.addWidget(QLabel("Z:"), 1, 0); grid_bc.addWidget(self.spin_box_cz, 1, 1)
        box_lay.addLayout(grid_bc)

        box_lay.addWidget(self._make_sub_header("Box Orientation (deg):"))
        grid_br = QGridLayout()
        grid_br.setSpacing(4)
        self.spin_box_yaw   = self._make_dspin(-180.0, 180.0, 0.0, 2.0, "°")
        self.spin_box_pitch = self._make_dspin(-90.0,  90.0,  0.0, 2.0, "°")
        self.spin_box_roll  = self._make_dspin(-180.0, 180.0, 0.0, 2.0, "°")
        grid_br.addWidget(QLabel("Yaw:"),   0, 0); grid_br.addWidget(self.spin_box_yaw,   0, 1)
        grid_br.addWidget(QLabel("Pitch:"), 0, 2); grid_br.addWidget(self.spin_box_pitch, 0, 3)
        grid_br.addWidget(QLabel("Roll:"),  1, 0); grid_br.addWidget(self.spin_box_roll,  1, 1)
        box_lay.addLayout(grid_br)

        box_lay.addWidget(self._make_sub_header("Dimensions (mm):"))
        grid_bd = QGridLayout()
        grid_bd.setSpacing(4)
        self.spin_box_lx = self._make_dspin(0.5, 2000.0, 40.0, 1.0, " mm")
        self.spin_box_ly = self._make_dspin(0.5, 2000.0, 40.0, 1.0, " mm")
        self.spin_box_lz = self._make_dspin(0.5, 2000.0, 20.0, 1.0, " mm")
        grid_bd.addWidget(QLabel("Len X:"), 0, 0); grid_bd.addWidget(self.spin_box_lx, 0, 1)
        grid_bd.addWidget(QLabel("Wid Y:"), 0, 2); grid_bd.addWidget(self.spin_box_ly, 0, 3)
        grid_bd.addWidget(QLabel("Hgt Z:"), 1, 0); grid_bd.addWidget(self.spin_box_lz, 1, 1)
        box_lay.addLayout(grid_bd)

        self.btn_reset_box = QPushButton("↺  Fit Box to Bone Centroid")
        self.btn_reset_box.clicked.connect(self.sig_reset_cut_to_bone)
        box_lay.addWidget(self.btn_reset_box)
        lay.addWidget(self.box_widget)

        # ── Sub-panel B: Dual Cutting Planes (Osteotomy Wedge) ────────────
        self.planes_widget = QWidget()
        pl_lay = QVBoxLayout(self.planes_widget)
        pl_lay.setContentsMargins(0, 4, 0, 2)
        pl_lay.setSpacing(4)

        pl_lay.addWidget(self._make_sub_header("Normal Line Midpoint (mm):"))
        grid_pm = QGridLayout()
        grid_pm.setSpacing(4)
        self.spin_mid_x = self._make_dspin(-5000.0, 5000.0, 0.0, 1.0, " mm")
        self.spin_mid_y = self._make_dspin(-5000.0, 5000.0, 0.0, 1.0, " mm")
        self.spin_mid_z = self._make_dspin(-5000.0, 5000.0, 0.0, 1.0, " mm")
        grid_pm.addWidget(QLabel("X:"), 0, 0); grid_pm.addWidget(self.spin_mid_x, 0, 1)
        grid_pm.addWidget(QLabel("Y:"), 0, 2); grid_pm.addWidget(self.spin_mid_y, 0, 3)
        grid_pm.addWidget(QLabel("Z:"), 1, 0); grid_pm.addWidget(self.spin_mid_z, 1, 1)
        pl_lay.addLayout(grid_pm)

        pl_lay.addWidget(self._make_sub_header("Normal Line Orientation & Resection Gap:"))
        grid_pl = QGridLayout()
        grid_pl.setSpacing(4)
        self.spin_line_yaw   = self._make_dspin(-180.0, 180.0, 0.0,  2.0, "°")
        self.spin_line_pitch = self._make_dspin(-90.0,  90.0,  90.0, 2.0, "°")
        self.spin_gap_l      = self._make_dspin(0.5,   2000.0, 15.0, 1.0, " mm")
        grid_pl.addWidget(QLabel("Line Yaw:"),   0, 0); grid_pl.addWidget(self.spin_line_yaw,   0, 1)
        grid_pl.addWidget(QLabel("Line Pitch:"), 0, 2); grid_pl.addWidget(self.spin_line_pitch, 0, 3)
        grid_pl.addWidget(QLabel("Gap L:"),      1, 0); grid_pl.addWidget(self.spin_gap_l,      1, 1)
        pl_lay.addLayout(grid_pl)

        pl_lay.addWidget(self._make_sub_header("Individual Osteotomy Plane Tilt Angles:"))
        grid_pt = QGridLayout()
        grid_pt.setSpacing(4)
        self.spin_p1_tilt       = self._make_dspin(-75.0, 75.0, 0.0, 1.0, "°")
        self.spin_p1_cross_tilt = self._make_dspin(-75.0, 75.0, 0.0, 1.0, "°")
        self.spin_p2_tilt       = self._make_dspin(-75.0, 75.0, 0.0, 1.0, "°")
        self.spin_p2_cross_tilt = self._make_dspin(-75.0, 75.0, 0.0, 1.0, "°")
        lbl_p1 = QLabel("Plane 1 Tilt:")
        lbl_p1.setStyleSheet("color: #ff9f43; font-weight: bold;")
        lbl_p2 = QLabel("Plane 2 Tilt:")
        lbl_p2.setStyleSheet("color: #48dbfb; font-weight: bold;")
        grid_pt.addWidget(lbl_p1, 0, 0); grid_pt.addWidget(self.spin_p1_tilt, 0, 1)
        grid_pt.addWidget(QLabel("Cross:"), 0, 2); grid_pt.addWidget(self.spin_p1_cross_tilt, 0, 3)
        grid_pt.addWidget(lbl_p2, 1, 0); grid_pt.addWidget(self.spin_p2_tilt, 1, 1)
        grid_pt.addWidget(QLabel("Cross:"), 1, 2); grid_pt.addWidget(self.spin_p2_cross_tilt, 1, 3)
        pl_lay.addLayout(grid_pt)

        self.btn_reset_planes = QPushButton("↺  Reset Planes to Bone Axis")
        self.btn_reset_planes.clicked.connect(self.sig_reset_cut_to_bone)
        pl_lay.addWidget(self.btn_reset_planes)
        lay.addWidget(self.planes_widget)

        self.setLayout(lay)

        # Wire signals
        self._box_spins = (
            self.spin_box_cx, self.spin_box_cy, self.spin_box_cz,
            self.spin_box_yaw, self.spin_box_pitch, self.spin_box_roll,
            self.spin_box_lx, self.spin_box_ly, self.spin_box_lz,
        )
        self._plane_spins = (
            self.spin_mid_x, self.spin_mid_y, self.spin_mid_z,
            self.spin_line_yaw, self.spin_line_pitch, self.spin_gap_l,
            self.spin_p1_tilt, self.spin_p1_cross_tilt,
            self.spin_p2_tilt, self.spin_p2_cross_tilt,
        )
        for sp in self._box_spins + self._plane_spins:
            sp.valueChanged.connect(lambda *_: self.sig_cut_params_changed.emit())

        self.combo_cut_mode.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed(CUT_MODE_NONE)

    @staticmethod
    def _make_sub_header(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #bbb; font-size: 10px; font-weight: bold;")
        return lbl

    @staticmethod
    def _make_dspin(vmin, vmax, val, step, suffix=""):
        sp = QDoubleSpinBox()
        sp.setRange(vmin, vmax)
        sp.setDecimals(1)
        sp.setSingleStep(step)
        if suffix:
            sp.setSuffix(suffix)
        sp.setValue(val)
        return sp

    def set_hide_gizmo_silent(self, hidden):
        self.chk_hide_gizmo.blockSignals(True)
        self.chk_hide_gizmo.setChecked(bool(hidden))
        self.chk_hide_gizmo.blockSignals(False)

    def _on_mode_changed(self, mode_txt):
        is_box = (mode_txt == CUT_MODE_BOX)
        is_planes = (mode_txt == CUT_MODE_PLANES)
        self.box_widget.setVisible(is_box)
        self.planes_widget.setVisible(is_planes)
        self.chk_hide_gizmo.setEnabled(is_box or is_planes)
        if is_box or is_planes:
            self.set_hide_gizmo_silent(False)

        if is_box:
            self.lbl_cut_hint.setText(
                "Drag 3D box faces/edges to resize & rotate, or type exact mm/deg values. "
                "Only the largest bone island inside the box is retained."
            )
        elif is_planes:
            self.lbl_cut_hint.setText(
                "Drag the 3D normal line or adjust Plane 1 / Plane 2 tilt angles for "
                "parallel or angled wedge osteotomies. Largest island is isolated."
            )
        else:
            self.lbl_cut_hint.setText(
                "Entire segmented bone from Tab 1 is used as the scaffold envelope."
            )
        self.sig_cut_mode_changed.emit(mode_txt)

    @property
    def cut_mode(self):
        return self.combo_cut_mode.currentText()

    @property
    def hide_cutting_gizmo(self):
        return self.chk_hide_gizmo.isChecked()

    def get_box_params(self):
        return {
            "cx": self.spin_box_cx.value(),
            "cy": self.spin_box_cy.value(),
            "cz": self.spin_box_cz.value(),
            "yaw": self.spin_box_yaw.value(),
            "pitch": self.spin_box_pitch.value(),
            "roll": self.spin_box_roll.value(),
            "lx": self.spin_box_lx.value(),
            "ly": self.spin_box_ly.value(),
            "lz": self.spin_box_lz.value(),
        }

    def get_planes_params(self):
        return {
            "mid_x": self.spin_mid_x.value(),
            "mid_y": self.spin_mid_y.value(),
            "mid_z": self.spin_mid_z.value(),
            "line_yaw": self.spin_line_yaw.value(),
            "line_pitch": self.spin_line_pitch.value(),
            "gap_l": self.spin_gap_l.value(),
            "p1_tilt": self.spin_p1_tilt.value(),
            "p1_cross_tilt": self.spin_p1_cross_tilt.value(),
            "p2_tilt": self.spin_p2_tilt.value(),
            "p2_cross_tilt": self.spin_p2_cross_tilt.value(),
        }

    def get_active_cut_params(self):
        mode = self.cut_mode
        if mode == CUT_MODE_BOX:
            return self.get_box_params()
        elif mode == CUT_MODE_PLANES:
            return self.get_planes_params()
        return None

    def set_box_params_silent(self, params):
        mapping = (
            (self.spin_box_cx, "cx"), (self.spin_box_cy, "cy"), (self.spin_box_cz, "cz"),
            (self.spin_box_yaw, "yaw"), (self.spin_box_pitch, "pitch"), (self.spin_box_roll, "roll"),
            (self.spin_box_lx, "lx"), (self.spin_box_ly, "ly"), (self.spin_box_lz, "lz"),
        )
        for sp, key in mapping:
            if key in params:
                sp.blockSignals(True)
                sp.setValue(float(params[key]))
                sp.blockSignals(False)

    def set_planes_params_silent(self, params):
        mapping = (
            (self.spin_mid_x, "mid_x"), (self.spin_mid_y, "mid_y"), (self.spin_mid_z, "mid_z"),
            (self.spin_line_yaw, "line_yaw"), (self.spin_line_pitch, "line_pitch"),
            (self.spin_gap_l, "gap_l"),
            (self.spin_p1_tilt, "p1_tilt"), (self.spin_p1_cross_tilt, "p1_cross_tilt"),
            (self.spin_p2_tilt, "p2_tilt"), (self.spin_p2_cross_tilt, "p2_cross_tilt"),
        )
        for sp, key in mapping:
            if key in params:
                sp.blockSignals(True)
                sp.setValue(float(params[key]))
                sp.blockSignals(False)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: SEGMENTATION TAB WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class SegmentationTabWidget(QWidget):
    """
    Tab 1 ("Segmentation"):
      - Left: 2×2 MPR Quad-Viewport (Coronal, Axial, Sagittal + 3D Anatomical Preview)
      - Right Sidebar:
          1. Data Ingestion
          2. 2D Viewport Reorientation (±90°)
          3. Threshold (Min/Max HU, Presets, "Keep Largest Island Only" real-time toggle)
          4. Export (Vertices, Faces, Watertight, 3MF / STL Format + Export 3D Model)
          5. 3D View & Anatomical Overlay
    """
    sig_select_dicom          = pyqtSignal()
    sig_select_nrrd           = pyqtSignal()
    sig_load_volume           = pyqtSignal()
    sig_thresh_params_changed = pyqtSignal()
    sig_wl_preset_requested   = pyqtSignal(int, int)
    sig_export_seg_mesh       = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        root_lay = QHBoxLayout(self)
        root_lay.setContentsMargins(4, 4, 4, 4)
        root_lay.setSpacing(6)

        # ── Left: 2×2 Quad-Viewport MPR + 3D ─────────────────────────────
        self.view_coronal = SliceViewWidget(
            'coronal', COLOR_CORONAL,
            {'top': 'T', 'bottom': 'B', 'left': 'R', 'right': 'L'},
        )
        self.view_axial = SliceViewWidget(
            'axial', COLOR_AXIAL,
            {'top': 'A', 'bottom': 'P', 'left': 'R', 'right': 'L'},
        )
        self.view_sagittal = SliceViewWidget(
            'sagittal', COLOR_SAGITTAL,
            {'top': 'T', 'bottom': 'B', 'left': 'A', 'right': 'P'},
        )
        self.views_by_plane = {
            'coronal':  self.view_coronal,
            'axial':    self.view_axial,
            'sagittal': self.view_sagittal,
        }

        self.vp_coronal  = SliceViewportPanel(self.view_coronal)
        self.vp_axial    = SliceViewportPanel(self.view_axial)
        self.vp_sagittal = SliceViewportPanel(self.view_sagittal)
        self.vp_3d       = ThreeDViewportPanel()

        top_split = QSplitter(Qt.Horizontal)
        top_split.addWidget(self.vp_coronal)
        top_split.addWidget(self.vp_axial)
        top_split.setSizes([500, 500])

        bot_split = QSplitter(Qt.Horizontal)
        bot_split.addWidget(self.vp_sagittal)
        bot_split.addWidget(self.vp_3d)
        bot_split.setSizes([500, 500])

        quad_split = QSplitter(Qt.Vertical)
        quad_split.addWidget(top_split)
        quad_split.addWidget(bot_split)
        quad_split.setSizes([450, 450])

        root_lay.addWidget(quad_split, 1)

        # ── Right Sidebar Scroll Area ────────────────────────────────────
        self.sidebar = QScrollArea()
        self.sidebar.setWidgetResizable(True)
        self.sidebar.setMinimumWidth(320)
        self.sidebar.setMaximumWidth(400)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sb_container = QWidget()
        sb_lay = QVBoxLayout(sb_container)
        sb_lay.setSpacing(6)

        # 1. Data Ingestion
        grp_data = CollapsibleGroupBox("Data Ingestion")
        lay_data = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.btn_dicom = QPushButton("DICOM Folder")
        self.btn_nrrd  = QPushButton("NRRD File")
        self.btn_dicom.clicked.connect(self.sig_select_dicom)
        self.btn_nrrd.clicked.connect(self.sig_select_nrrd)
        btn_row.addWidget(self.btn_dicom)
        btn_row.addWidget(self.btn_nrrd)
        lay_data.addLayout(btn_row)

        self.lbl_file = QLabel("No file selected")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color: #aaa;")
        lay_data.addWidget(self.lbl_file)

        self.lbl_dims = QLabel("")
        self.lbl_dims.setStyleSheet("color: #888; font-size: 10px;")
        lay_data.addWidget(self.lbl_dims)

        self.btn_load = QPushButton("⬇  Load Volume")
        self.btn_load.setStyleSheet(
            "background-color: #2a7; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_load.clicked.connect(self.sig_load_volume)
        lay_data.addWidget(self.btn_load)
        grp_data.setLayout(lay_data)
        sb_lay.addWidget(grp_data)

        # 2. 2D Viewport Reorientation
        grp_orient = CollapsibleGroupBox("2D Viewport Reorientation")
        lay_or = QVBoxLayout()
        self.lbl_rot_status = {}
        for plane_key, label_txt, col_hex in (
            ("coronal",  "Coronal:",  "#00c800"),
            ("axial",    "Axial:",    "#dc2828"),
            ("sagittal", "Sagittal:", "#3c78ff"),
        ):
            row_r = QHBoxLayout()
            lbl_p = QLabel(label_txt)
            lbl_p.setStyleSheet(f"color: {col_hex}; font-weight: bold;")
            lbl_p.setFixedWidth(65)
            row_r.addWidget(lbl_p)

            btn_ccw = QPushButton("↺ -90°")
            btn_cw  = QPushButton("↻ +90°")
            btn_ccw.setFixedHeight(24)
            btn_cw.setFixedHeight(24)
            btn_ccw.clicked.connect(
                lambda _, p=plane_key: self.views_by_plane[p].rotate_step(-1)
            )
            btn_cw.clicked.connect(
                lambda _, p=plane_key: self.views_by_plane[p].rotate_step(1)
            )
            row_r.addWidget(btn_ccw, 1)
            row_r.addWidget(btn_cw, 1)

            lbl_deg = QLabel("0°")
            lbl_deg.setFixedWidth(36)
            lbl_deg.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_deg.setStyleSheet("color: #aaa; font-family: Consolas;")
            row_r.addWidget(lbl_deg)
            self.lbl_rot_status[plane_key] = lbl_deg
            lay_or.addLayout(row_r)

        self.btn_reset_rot = QPushButton("Reset All 2D Orientations (0°)")
        self.btn_reset_rot.clicked.connect(self._reset_all_rotations)
        lay_or.addWidget(self.btn_reset_rot)
        grp_orient.setLayout(lay_or)
        sb_lay.addWidget(grp_orient)

        # 3. Threshold Panel (Z-crop, Auto-Smooth checkbox & Preview button removed;
        #    real-time "Keep Largest Island Only" toggle added)
        grp_thresh = CollapsibleGroupBox("Threshold")
        lay_t = QVBoxLayout()

        hu_row = QHBoxLayout()
        hu_row.addWidget(QLabel("Min HU:"))
        self.spin_hu_min = QSpinBox()
        self.spin_hu_min.setRange(-8192, 16000)
        self.spin_hu_min.setValue(150)
        hu_row.addWidget(self.spin_hu_min)
        hu_row.addWidget(QLabel("Max HU:"))
        self.spin_hu_max = QSpinBox()
        self.spin_hu_max.setRange(-8192, 16000)
        self.spin_hu_max.setValue(3000)
        hu_row.addWidget(self.spin_hu_max)
        lay_t.addLayout(hu_row)

        preset_row = QHBoxLayout()
        for name, (lo, hi, wl, ww) in HU_PRESETS.items():
            btn = QPushButton(name)
            btn.setFixedHeight(24)
            btn.setStyleSheet("font-size: 10px; padding: 2px 4px;")
            btn.clicked.connect(
                lambda _, lo_=lo, hi_=hi, wl_=wl, ww_=ww:
                    self._apply_preset(lo_, hi_, wl_, ww_)
            )
            preset_row.addWidget(btn)
        lay_t.addLayout(preset_row)

        self.chk_largest_island = QCheckBox("Keep Largest Island Only")
        self.chk_largest_island.setChecked(True)
        self.chk_largest_island.setToolTip(
            "When checked, filters out CT table artifacts and adjacent disconnected "
            "bones in real time while preserving internal medullary canals."
        )
        lay_t.addWidget(self.chk_largest_island)

        self.lbl_auto_info = QLabel(
            "✓ Real-time 2D mask & Taubin-smoothed 3D bone preview active"
        )
        self.lbl_auto_info.setWordWrap(True)
        self.lbl_auto_info.setStyleSheet(
            "color: #7d9; font-size: 10px; font-style: italic;"
        )
        lay_t.addWidget(self.lbl_auto_info)

        self.spin_hu_min.valueChanged.connect(
            lambda *_: self.sig_thresh_params_changed.emit()
        )
        self.spin_hu_max.valueChanged.connect(
            lambda *_: self.sig_thresh_params_changed.emit()
        )
        self.chk_largest_island.toggled.connect(
            lambda *_: self.sig_thresh_params_changed.emit()
        )

        grp_thresh.setLayout(lay_t)
        sb_lay.addWidget(grp_thresh)

        # 4. Export Panel (3MF / STL dual format selector + Export 3D Model)
        grp_export = CollapsibleGroupBox("Export")
        lay_ex = QVBoxLayout()

        self.lbl_mesh_stats = QLabel("Vertices: –\nFaces: –\nWatertight: –")
        self.lbl_mesh_stats.setStyleSheet(
            "color: #aaa; font-size: 10px; font-family: Consolas;"
        )
        lay_ex.addWidget(self.lbl_mesh_stats)

        lay_ex.addWidget(QLabel("Export Format:"))
        self.combo_export_format = QComboBox()
        self.combo_export_format.addItems(ALL_EXPORT_FORMATS)
        self.combo_export_format.setCurrentIndex(0)
        self.combo_export_format.setToolTip(
            "3MF stores shared indexed vertices with explicit millimeter units "
            "in a compressed ZIP archive (5x–10x smaller & watertight). "
            "STL stores standard raw triangle facets."
        )
        lay_ex.addWidget(self.combo_export_format)

        self.btn_export = QPushButton("💾  Export 3D Model")
        self.btn_export.setStyleSheet(
            "background-color: #37a; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_export.clicked.connect(self.sig_export_seg_mesh)
        lay_ex.addWidget(self.btn_export)

        grp_export.setLayout(lay_ex)
        sb_lay.addWidget(grp_export)

        # 5. 3D View & Anatomical Overlay
        grp_overlay = CollapsibleGroupBox("3D View && Anatomical Overlay")
        lay_o = QVBoxLayout()

        self.chk_overlay = QCheckBox("Overlay Host Bone")
        self.chk_overlay.setChecked(False)
        self.chk_overlay.setEnabled(False)
        self.chk_overlay.toggled.connect(self.vp_3d.set_overlay_enabled)
        lay_o.addWidget(self.chk_overlay)

        row_alpha = QHBoxLayout()
        self.lbl_alpha_title = QLabel("Bone Opacity:")
        self.lbl_alpha_title.setToolTip(
            "Host bone alpha/opacity (0% = fully transparent, 100% = fully opaque)."
        )
        self.lbl_alpha_title.setEnabled(False)
        row_alpha.addWidget(self.lbl_alpha_title)
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(int(DEFAULT_BONE_ALPHA * 100))
        self.slider_alpha.setToolTip(
            "Adjust host bone opacity (0% = transparent, 100% = fully opaque)."
        )
        self.slider_alpha.setEnabled(False)
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        row_alpha.addWidget(self.slider_alpha, 1)
        self.lbl_alpha_val = QLabel(f"{int(DEFAULT_BONE_ALPHA * 100)}%")
        self.lbl_alpha_val.setFixedWidth(40)
        self.lbl_alpha_val.setEnabled(False)
        row_alpha.addWidget(self.lbl_alpha_val)
        lay_o.addLayout(row_alpha)

        self.btn_reset_cam = QPushButton("Reset 3D Camera")
        self.btn_reset_cam.clicked.connect(self.vp_3d.reset_camera)
        lay_o.addWidget(self.btn_reset_cam)

        grp_overlay.setLayout(lay_o)
        sb_lay.addWidget(grp_overlay)

        sb_lay.addStretch(1)
        self.sidebar.setWidget(sb_container)
        root_lay.addWidget(self.sidebar, 0)

        for view in self.views_by_plane.values():
            view.rotation_changed.connect(self._on_view_rotation_changed)

    def _reset_all_rotations(self):
        for view in self.views_by_plane.values():
            view.set_rotation_k(0)

    def _on_view_rotation_changed(self, plane, degrees):
        if plane in self.lbl_rot_status:
            self.lbl_rot_status[plane].setText(f"{degrees}°")

    def _apply_preset(self, hu_lo, hu_hi, wl, ww):
        self.spin_hu_min.blockSignals(True)
        self.spin_hu_max.blockSignals(True)
        self.spin_hu_min.setValue(hu_lo)
        self.spin_hu_max.setValue(hu_hi)
        self.spin_hu_min.blockSignals(False)
        self.spin_hu_max.blockSignals(False)
        self.sig_wl_preset_requested.emit(wl, ww)
        self.sig_thresh_params_changed.emit()

    def _on_alpha_changed(self, val):
        self.lbl_alpha_val.setText(f"{val}%")
        self.vp_3d.set_bone_alpha(val / 100.0)

    def update_mesh_stats(self, n_verts, n_faces, watertight, porosity_metrics=None):
        wt = "Yes" if watertight else "No"
        self.lbl_mesh_stats.setText(
            f"Vertices: {n_verts:,}\nFaces: {n_faces:,}\nWatertight: {wt}"
        )

    @property
    def hu_min(self):
        return self.spin_hu_min.value()

    @property
    def hu_max(self):
        return self.spin_hu_max.value()

    @property
    def keep_largest_island(self):
        return self.chk_largest_island.isChecked()

    @property
    def export_format(self):
        txt = self.combo_export_format.currentText().lower()
        return "stl" if ".stl" in txt else "3mf"

    def set_export_format(self, fmt):
        target_idx = 1 if str(fmt).lower().lstrip(".") == "stl" else 0
        self.combo_export_format.blockSignals(True)
        self.combo_export_format.setCurrentIndex(target_idx)
        self.combo_export_format.blockSignals(False)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: SCAFFOLD DESIGN TAB WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class ScaffoldTabWidget(QWidget):
    """
    Tab 2 ("Scaffold Design"):
      - Left: Dedicated maximized single 3D viewport (`ThreeDViewportPanel`)
              with interactive VTK cutting & orientation vector gizmos
      - Right Sidebar:
          1. Cutting Tools (No Cut, Oriented Box Primitive, Dual-Plane Osteotomy)
          2. Scaffold Architecture (TPMS / Strut-Lattice, Target Mode,
             Set Pattern Direction Vector Tool, "Generate Bone Scaffold")
          3. 3D View & Anatomical Overlay (Host Bone Overlay & Opacity)
          4. Export (Vertices, Faces, Watertight, 3MF / STL Format,
             "Also Export Host Bone Overlay Model" + Export 3D Model)
    """
    sig_generate_scaffold  = pyqtSignal()
    sig_export_scaf_mesh   = pyqtSignal()
    sig_dir_arrow_toggled  = pyqtSignal(bool)
    sig_dir_angles_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_spacing = (1.0, 1.0, 1.0)

        root_lay = QHBoxLayout(self)
        root_lay.setContentsMargins(4, 4, 4, 4)
        root_lay.setSpacing(6)

        # ── Left: Single Large Full-Height 3D Viewport ───────────────────
        self.vp_3d = ThreeDViewportPanel()
        root_lay.addWidget(self.vp_3d, 1)

        # ── Right Sidebar Scroll Area ────────────────────────────────────
        self.sidebar = QScrollArea()
        self.sidebar.setWidgetResizable(True)
        self.sidebar.setMinimumWidth(335)
        self.sidebar.setMaximumWidth(425)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sb_container = QWidget()
        sb_lay = QVBoxLayout(sb_container)
        sb_lay.setSpacing(6)

        # 1. Cutting Tools Panel
        self.cutting_panel = CuttingToolsPanel()
        sb_lay.addWidget(self.cutting_panel)

        # 2. Scaffold Architecture Panel (+ Set Pattern Direction Tool)
        grp_arch = CollapsibleGroupBox("Scaffold Architecture")
        lay_p = QVBoxLayout()
        lay_p.setSpacing(5)

        row_pat = QHBoxLayout()
        row_pat.addWidget(QLabel("Pattern:"))
        self.combo_pattern = QComboBox()
        self.combo_pattern.addItems(ALL_SCAFFOLD_PATTERNS)
        self.combo_pattern.setCurrentText("None")
        row_pat.addWidget(self.combo_pattern, 1)
        lay_p.addLayout(row_pat)

        self.lbl_arch_badge = QLabel("Architecture: Solid Bone (No Pattern)")
        self.lbl_arch_badge.setStyleSheet("color: #aaa; font-size: 10px; font-style: italic;")
        lay_p.addWidget(self.lbl_arch_badge)

        lay_p.addWidget(QLabel("Scaffold Target Mode:"))
        self.combo_target_mode = QComboBox()
        self.combo_target_mode.addItems(SCAFFOLD_TARGET_MODES)
        self.combo_target_mode.setCurrentIndex(0)
        self.combo_target_mode.setToolTip(
            "Choose 'Anatomical Cortical Bone' to preserve the natural medullary "
            "canal, marrow spaces, and foramina, or 'Solid Resection Graft' to "
            "seal cavities into a solid segmental defect plug."
        )
        lay_p.addWidget(self.combo_target_mode)

        self.lbl_target_mode_hint = QLabel("")
        self.lbl_target_mode_hint.setWordWrap(True)
        lay_p.addWidget(self.lbl_target_mode_hint)
        self.combo_target_mode.currentIndexChanged.connect(self._on_target_mode_changed)
        self._on_target_mode_changed(0)

        self.chk_invert = QCheckBox("Invert Solid / Void Phase")
        lay_p.addWidget(self.chk_invert)

        # Target Pore Size (µm)
        lay_p.addWidget(QLabel("Target Pore Size (50 – 1500 µm):"))
        row_pore = QHBoxLayout()
        self.spin_pore_um = QSpinBox()
        self.spin_pore_um.setRange(MIN_PORE_UM, MAX_PORE_UM)
        self.spin_pore_um.setSingleStep(STEP_PORE_UM)
        self.spin_pore_um.setSuffix(" µm")
        self.spin_pore_um.setValue(DEFAULT_PORE_UM)
        row_pore.addWidget(self.spin_pore_um)

        self.slider_pore_um = QSlider(Qt.Horizontal)
        self.slider_pore_um.setRange(MIN_PORE_UM, MAX_PORE_UM)
        self.slider_pore_um.setSingleStep(STEP_PORE_UM)
        self.slider_pore_um.setPageStep(100)
        self.slider_pore_um.setValue(DEFAULT_PORE_UM)
        row_pore.addWidget(self.slider_pore_um, 1)
        lay_p.addLayout(row_pore)

        self.lbl_pore_mm = QLabel("Metric Pore Size: 500 µm (0.500 mm)")
        self.lbl_pore_mm.setStyleSheet("color: #8cd; font-size: 10px; font-family: Consolas;")
        lay_p.addWidget(self.lbl_pore_mm)

        self.spin_pore_um.valueChanged.connect(self._on_pore_changed)
        self.slider_pore_um.valueChanged.connect(self.spin_pore_um.setValue)

        # ── TPMS-specific sub-panel ──
        self.tpms_params_widget = QWidget()
        tpms_sub = QVBoxLayout(self.tpms_params_widget)
        tpms_sub.setContentsMargins(0, 2, 0, 2)
        tpms_sub.setSpacing(4)

        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("TPMS Type:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Skeletal", "Shell"])
        row_type.addWidget(self.combo_type, 1)
        tpms_sub.addLayout(row_type)

        row_level = QHBoxLayout()
        self.lbl_level_title = QLabel("Level-Set:")
        row_level.addWidget(self.lbl_level_title)
        self.slider_level = QSlider(Qt.Horizontal)
        self.slider_level.setRange(-80, 80)
        self.slider_level.setValue(0)
        row_level.addWidget(self.slider_level, 1)
        self.lbl_level_val = QLabel("0.00")
        self.lbl_level_val.setFixedWidth(40)
        row_level.addWidget(self.lbl_level_val)
        tpms_sub.addLayout(row_level)
        self.slider_level.valueChanged.connect(self._on_level_changed)

        row_thick = QHBoxLayout()
        self.lbl_thick_title = QLabel("Thickness:")
        row_thick.addWidget(self.lbl_thick_title)
        self.slider_thick = QSlider(Qt.Horizontal)
        self.slider_thick.setRange(5, 80)
        self.slider_thick.setValue(25)
        row_thick.addWidget(self.slider_thick, 1)
        self.lbl_thick_val = QLabel("0.25")
        self.lbl_thick_val.setFixedWidth(40)
        row_thick.addWidget(self.lbl_thick_val)
        tpms_sub.addLayout(row_thick)
        self.slider_thick.valueChanged.connect(self._on_thick_changed)

        lay_p.addWidget(self.tpms_params_widget)

        # ── Strut-Lattice specific sub-panel ──
        self.strut_params_widget = QWidget()
        strut_sub = QVBoxLayout(self.strut_params_widget)
        strut_sub.setContentsMargins(0, 2, 0, 2)
        strut_sub.setSpacing(4)

        strut_sub.addWidget(QLabel("Strut Diameter / Thickness (50 – 800 µm):"))
        row_strut = QHBoxLayout()
        self.spin_strut_um = QSpinBox()
        self.spin_strut_um.setRange(MIN_STRUT_UM, MAX_STRUT_UM)
        self.spin_strut_um.setSingleStep(STEP_STRUT_UM)
        self.spin_strut_um.setSuffix(" µm")
        self.spin_strut_um.setValue(DEFAULT_STRUT_UM)
        row_strut.addWidget(self.spin_strut_um)

        self.slider_strut_um = QSlider(Qt.Horizontal)
        self.slider_strut_um.setRange(MIN_STRUT_UM, MAX_STRUT_UM)
        self.slider_strut_um.setSingleStep(STEP_STRUT_UM)
        self.slider_strut_um.setPageStep(50)
        self.slider_strut_um.setValue(DEFAULT_STRUT_UM)
        row_strut.addWidget(self.slider_strut_um, 1)
        strut_sub.addLayout(row_strut)

        self.lbl_strut_mm = QLabel("Metric Strut Diameter: 200 µm (0.200 mm)")
        self.lbl_strut_mm.setStyleSheet("color: #8cd; font-size: 10px; font-family: Consolas;")
        strut_sub.addWidget(self.lbl_strut_mm)

        self.spin_strut_um.valueChanged.connect(self._on_strut_changed)
        self.slider_strut_um.valueChanged.connect(self.spin_strut_um.setValue)

        lay_p.addWidget(self.strut_params_widget)

        # ── NEW: Set Pattern Direction Tool ──────────────────────────────
        dir_sep = QFrame()
        dir_sep.setFrameShape(QFrame.HLine)
        dir_sep.setStyleSheet("color: #444;")
        lay_p.addWidget(dir_sep)

        self.chk_show_arrow = QCheckBox("Show Pattern Direction Vector")
        self.chk_show_arrow.setChecked(False)  # Unchecked by default per spec
        self.chk_show_arrow.setToolTip(
            "Displays an interactive 3D orientation vector arrow at the ROI centroid "
            "and reveals 3D rotation controls to align scaffold pores/struts along "
            "anatomical load-bearing axes."
        )
        lay_p.addWidget(self.chk_show_arrow)

        self.dir_params_widget = QWidget()
        dir_lay = QVBoxLayout(self.dir_params_widget)
        dir_lay.setContentsMargins(4, 2, 0, 2)
        dir_lay.setSpacing(4)

        grid_dir = QGridLayout()
        grid_dir.setSpacing(4)

        self.spin_dir_yaw = QDoubleSpinBox()
        self.spin_dir_yaw.setRange(-180.0, 180.0)
        self.spin_dir_yaw.setDecimals(1)
        self.spin_dir_yaw.setSingleStep(5.0)
        self.spin_dir_yaw.setSuffix("°")
        self.spin_dir_yaw.setValue(0.0)

        self.spin_dir_pitch = QDoubleSpinBox()
        self.spin_dir_pitch.setRange(-90.0, 90.0)
        self.spin_dir_pitch.setDecimals(1)
        self.spin_dir_pitch.setSingleStep(5.0)
        self.spin_dir_pitch.setSuffix("°")
        self.spin_dir_pitch.setValue(90.0)   # +Z anatomical longitudinal axis

        self.spin_dir_roll = QDoubleSpinBox()
        self.spin_dir_roll.setRange(-180.0, 180.0)
        self.spin_dir_roll.setDecimals(1)
        self.spin_dir_roll.setSingleStep(5.0)
        self.spin_dir_roll.setSuffix("°")
        self.spin_dir_roll.setValue(0.0)

        grid_dir.addWidget(QLabel("Vector Yaw / Azimuth:"), 0, 0)
        grid_dir.addWidget(self.spin_dir_yaw, 0, 1)
        grid_dir.addWidget(QLabel("Vector Elevation / Pitch:"), 1, 0)
        grid_dir.addWidget(self.spin_dir_pitch, 1, 1)
        grid_dir.addWidget(QLabel("Vector Roll:"), 2, 0)
        grid_dir.addWidget(self.spin_dir_roll, 2, 1)
        dir_lay.addLayout(grid_dir)

        self.lbl_dir_vec = QLabel("Axis d̂ = (0.00, 0.00, 1.00) [+Z Longitudinal]")
        self.lbl_dir_vec.setStyleSheet("color: #ffd166; font-size: 10px; font-family: Consolas;")
        dir_lay.addWidget(self.lbl_dir_vec)

        self.btn_reset_dir = QPushButton("↺  Reset Direction (+Z Axis, 0° Roll)")
        self.btn_reset_dir.setFixedHeight(22)
        self.btn_reset_dir.setStyleSheet("font-size: 10px;")
        self.btn_reset_dir.clicked.connect(self._reset_direction_angles)
        dir_lay.addWidget(self.btn_reset_dir)

        self.dir_params_widget.setVisible(False)
        lay_p.addWidget(self.dir_params_widget)

        self.chk_show_arrow.toggled.connect(self._on_show_arrow_toggled)
        for dsp in (self.spin_dir_yaw, self.spin_dir_pitch, self.spin_dir_roll):
            dsp.valueChanged.connect(self._on_dir_spin_changed)

        # Physical unit-cell period, Δ_iso & Nyquist resolution readout
        self.lbl_pore_info = QLabel("")
        self.lbl_pore_info.setWordWrap(True)
        self.lbl_pore_info.setStyleSheet(
            "color: #bbb; font-size: 10px; font-family: Consolas; "
            "background: #222; padding: 4px; border-radius: 3px;"
        )
        lay_p.addWidget(self.lbl_pore_info)

        self.combo_pattern.currentTextChanged.connect(self._on_pattern_changed)
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        self.chk_invert.toggled.connect(lambda *_: self._update_pore_diagnostics())
        self._on_type_changed(self.combo_type.currentText())
        self._on_pattern_changed(self.combo_pattern.currentText())

        # Generate Bone Scaffold CTA button
        self.btn_generate = QPushButton("⚡  Generate Bone Scaffold")
        self.btn_generate.setStyleSheet(
            "background-color: #c63; color: white; font-weight: bold; padding: 7px;"
        )
        self.btn_generate.clicked.connect(self.sig_generate_scaffold)
        lay_p.addWidget(self.btn_generate)

        # Measured Scaffold Porosity Metric Card
        self.porosity_card = QFrame()
        self.porosity_card.setStyleSheet(
            "QFrame { background-color: #1e2428; border: 1px solid #3a5a68; border-radius: 4px; }"
        )
        por_lay = QVBoxLayout(self.porosity_card)
        por_lay.setContentsMargins(8, 6, 8, 6)
        por_lay.setSpacing(2)

        self.lbl_porosity_val = QLabel("Scaffold Porosity (φ): –")
        self.lbl_porosity_val.setStyleSheet(
            "color: #9ab; font-size: 12px; font-weight: bold; font-family: Consolas; border: none;"
        )
        por_lay.addWidget(self.lbl_porosity_val)

        self.lbl_porosity_detail = QLabel("Click 'Generate Bone Scaffold' to measure porosity")
        self.lbl_porosity_detail.setWordWrap(True)
        self.lbl_porosity_detail.setStyleSheet(
            "color: #889; font-size: 10px; font-family: Consolas; border: none;"
        )
        por_lay.addWidget(self.lbl_porosity_detail)
        lay_p.addWidget(self.porosity_card)

        grp_arch.setLayout(lay_p)
        sb_lay.addWidget(grp_arch)

        # 3. 3D View & Anatomical Overlay Panel
        grp_overlay = CollapsibleGroupBox("3D View && Anatomical Overlay")
        lay_o = QVBoxLayout()

        self.chk_overlay = QCheckBox("Overlay Host Bone")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.setEnabled(False)
        self.chk_overlay.toggled.connect(self.vp_3d.set_overlay_enabled)
        lay_o.addWidget(self.chk_overlay)

        row_alpha = QHBoxLayout()
        self.lbl_alpha_title = QLabel("Bone Opacity:")
        self.lbl_alpha_title.setToolTip(
            "Host bone alpha/opacity (0% = fully transparent, 100% = fully opaque)."
        )
        self.lbl_alpha_title.setEnabled(False)
        row_alpha.addWidget(self.lbl_alpha_title)
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(int(DEFAULT_BONE_ALPHA * 100))
        self.slider_alpha.setToolTip(
            "Adjust host bone opacity (0% = transparent, 100% = fully opaque)."
        )
        self.slider_alpha.setEnabled(False)
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        row_alpha.addWidget(self.slider_alpha, 1)
        self.lbl_alpha_val = QLabel(f"{int(DEFAULT_BONE_ALPHA * 100)}%")
        self.lbl_alpha_val.setFixedWidth(40)
        self.lbl_alpha_val.setEnabled(False)
        row_alpha.addWidget(self.lbl_alpha_val)
        lay_o.addLayout(row_alpha)

        self.btn_reset_cam = QPushButton("Reset 3D Camera")
        self.btn_reset_cam.clicked.connect(self.vp_3d.reset_camera)
        lay_o.addWidget(self.btn_reset_cam)

        grp_overlay.setLayout(lay_o)
        sb_lay.addWidget(grp_overlay)

        # 4. Export Panel
        grp_export = CollapsibleGroupBox("Export")
        lay_ex = QVBoxLayout()

        self.lbl_mesh_stats = QLabel("Porosity: –\nVertices: –\nFaces: –\nWatertight: –")
        self.lbl_mesh_stats.setStyleSheet(
            "color: #aaa; font-size: 10px; font-family: Consolas;"
        )
        lay_ex.addWidget(self.lbl_mesh_stats)

        lay_ex.addWidget(QLabel("Export Format:"))
        self.combo_export_format = QComboBox()
        self.combo_export_format.addItems(ALL_EXPORT_FORMATS)
        self.combo_export_format.setCurrentIndex(0)
        self.combo_export_format.setToolTip(
            "3MF stores shared indexed vertices with explicit millimeter units "
            "in a compressed ZIP archive (5x–10x smaller & watertight). "
            "STL stores standard raw triangle facets."
        )
        lay_ex.addWidget(self.combo_export_format)

        self.chk_export_host = QCheckBox("Also Export Host Bone Overlay Model")
        self.chk_export_host.setChecked(False)
        self.chk_export_host.setEnabled(False)
        self.chk_export_host.setToolTip(
            "When checked, also exports the surrounding resected host bone model "
            "into separate companion file(s) alongside the scaffold. If the cut "
            "splits the host bone into multiple pieces, every piece is exported "
            "as its own file (e.g. *_host_bone_1, *_host_bone_2) for easy build-plate "
            "orientation and surgical fit testing."
        )
        lay_ex.addWidget(self.chk_export_host)

        self.btn_export = QPushButton("💾  Export 3D Model")
        self.btn_export.setStyleSheet(
            "background-color: #37a; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_export.clicked.connect(self.sig_export_scaf_mesh)
        lay_ex.addWidget(self.btn_export)

        grp_export.setLayout(lay_ex)
        sb_lay.addWidget(grp_export)

        sb_lay.addStretch(1)
        self.sidebar.setWidget(sb_container)
        root_lay.addWidget(self.sidebar, 0)

    # ── UI Callbacks & Helpers ───────────────────────────────────────────

    def set_voxel_spacing(self, spacing_zyx):
        self._current_spacing = tuple(float(s) for s in spacing_zyx)
        self._update_pore_diagnostics()

    def _on_target_mode_changed(self, idx):
        if idx == 0:
            self.lbl_target_mode_hint.setText(
                "✓ Preserves medullary canal, marrow voids & nutrient foramina"
            )
            self.lbl_target_mode_hint.setStyleSheet(
                "color: #7d9; font-size: 10px; font-style: italic;"
            )
        else:
            self.lbl_target_mode_hint.setText(
                "⚠ Seals medullary canal into a solid resection graft plug"
            )
            self.lbl_target_mode_hint.setStyleSheet(
                "color: #eb7; font-size: 10px; font-style: italic;"
            )

    def _on_pattern_changed(self, pat):
        strut_mode = is_strut_pattern(pat)
        none_mode = (pat == "None")

        self.tpms_params_widget.setVisible(not strut_mode and not none_mode)
        self.strut_params_widget.setVisible(strut_mode)

        if none_mode:
            self.lbl_arch_badge.setText("Architecture: Solid Bone (No Pattern)")
        elif strut_mode:
            desc_map = {
                "Cubic": "Strut Lattice — Simple Cubic Orthogonal Frame",
                "BCC":   "Strut Lattice — Body-Centered Cubic (8-Corner Diagonals)",
                "BCCZ":  "Strut Lattice — BCC + Load-Bearing Z'-Axis Columns",
                "FCC":   "Strut Lattice — Face-Centered Cubic Diagonals",
            }
            self.lbl_arch_badge.setText(desc_map.get(pat, "Strut-Based Lattice"))
        else:
            self.lbl_arch_badge.setText(f"TPMS Implicit Surface — {pat}")

        self._update_pore_diagnostics()

    def _on_type_changed(self, text):
        skeletal = (text == "Skeletal")
        self.slider_level.setEnabled(True)
        self.lbl_level_title.setEnabled(True)
        self.lbl_level_val.setEnabled(True)
        self.slider_thick.setEnabled(not skeletal)
        self.lbl_thick_title.setEnabled(not skeletal)
        self.lbl_thick_val.setEnabled(not skeletal)
        self._update_pore_diagnostics()

    def _on_pore_changed(self, val):
        self.slider_pore_um.blockSignals(True)
        self.slider_pore_um.setValue(val)
        self.slider_pore_um.blockSignals(False)
        self.lbl_pore_mm.setText(f"Metric Pore Size: {val} µm ({val / 1000.0:.3f} mm)")
        self._update_pore_diagnostics()

    def _on_strut_changed(self, val):
        self.slider_strut_um.blockSignals(True)
        self.slider_strut_um.setValue(val)
        self.slider_strut_um.blockSignals(False)
        self.lbl_strut_mm.setText(f"Metric Strut Diameter: {val} µm ({val / 1000.0:.3f} mm)")
        self._update_pore_diagnostics()

    def _on_level_changed(self, v):
        self.lbl_level_val.setText(f"{v / 100:.2f}")
        self._update_pore_diagnostics()

    def _on_thick_changed(self, v):
        self.lbl_thick_val.setText(f"{v / 100:.2f}")
        self._update_pore_diagnostics()

    def _on_show_arrow_toggled(self, checked):
        self.dir_params_widget.setVisible(checked)
        self.sig_dir_arrow_toggled.emit(checked)

    def _on_dir_spin_changed(self, *_args):
        self._update_dir_label()
        self.sig_dir_angles_changed.emit()

    def _reset_direction_angles(self):
        self.set_dir_angles_silent(0.0, 90.0, 0.0)
        self._update_dir_label()
        self.sig_dir_angles_changed.emit()

    def _update_dir_label(self):
        R = pattern_direction_to_matrix(self.dir_yaw, self.dir_pitch, self.dir_roll)
        d = R[:, 2]
        self.lbl_dir_vec.setText(
            f"Axis d̂ = ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})  |  Roll {self.dir_roll:+.1f}°"
        )

    def set_dir_angles_silent(self, yaw_deg, pitch_deg, roll_deg=None):
        self.spin_dir_yaw.blockSignals(True)
        self.spin_dir_pitch.blockSignals(True)
        self.spin_dir_yaw.setValue(float(yaw_deg))
        self.spin_dir_pitch.setValue(float(pitch_deg))
        self.spin_dir_yaw.blockSignals(False)
        self.spin_dir_pitch.blockSignals(False)
        if roll_deg is not None:
            self.spin_dir_roll.blockSignals(True)
            self.spin_dir_roll.setValue(float(roll_deg))
            self.spin_dir_roll.blockSignals(False)
        self._update_dir_label()

    def _update_pore_diagnostics(self):
        pat = self.combo_pattern.currentText()
        pore_um = self.spin_pore_um.value()
        pore_mm = pore_um / 1000.0
        strut_um = self.spin_strut_um.value()
        strut_mm = strut_um / 1000.0
        max_ct_sp = max(self._current_spacing)

        if pat == "None":
            self.lbl_pore_info.setText(
                "Pattern: None (Solid cut bone ROI without porous lattice)"
            )
            self.lbl_pore_info.setStyleSheet(
                "color: #aaa; font-size: 10px; font-family: Consolas; "
                "background: #222; padding: 4px; border-radius: 3px;"
            )
            return

        L_mm = compute_unit_cell_period_mm(
            pat, self.tpms_type, pore_um,
            level_set=self.tpms_level, thickness=self.tpms_thickness,
            strut_diam_um=strut_um,
        )
        delta_iso = compute_target_delta_iso_mm(
            pore_um, strut_diam_um=strut_um, pattern_name=pat
        )
        est_por_pct = estimate_unit_cell_porosity_pct(
            pat, self.tpms_type, pore_um,
            level_set=self.tpms_level, thickness=self.tpms_thickness,
            strut_diam_um=strut_um, invert=self.tpms_invert,
        )

        min_feature_mm = min(pore_mm, strut_mm) if is_strut_pattern(pat) else pore_mm
        resampled_nyquist = 2.0 * delta_iso

        if min_feature_mm < resampled_nyquist:
            warn_txt = (
                f"⚠ Micro-Feature Notice: Feature ({min_feature_mm * 1000:.0f} µm) < "
                f"2×Δ_iso ({resampled_nyquist * 1000:.0f} µm). Consider ≥80 µm pores/struts."
            )
            color_hex = "#f94"
        else:
            warn_txt = (
                f"✓ Anti-Aliased Grid Active: Δ_iso = {delta_iso * 1000:.0f} µm "
                f"(Native CT max spacing: {max_ct_sp:.2f} mm)"
            )
            color_hex = "#7d9"

        extra_line = ""
        if is_strut_pattern(pat):
            extra_line = f"Strut Ø: {strut_mm:.3f} mm  |  Pore Ø: {pore_mm:.3f} mm\n"

        self.lbl_pore_info.setText(
            f"Unit Cell Period L: {L_mm:.3f} mm ({L_mm * 1000:.0f} µm)\n"
            f"Est. Unit-Cell Porosity: ~{est_por_pct:.1f}% (Solid: ~{100.0 - est_por_pct:.1f}%)\n"
            f"{extra_line}{warn_txt}"
        )
        self.lbl_pore_info.setStyleSheet(
            f"color: {color_hex}; font-size: 10px; font-family: Consolas; "
            "background: #222; padding: 4px; border-radius: 3px;"
        )

    def _on_alpha_changed(self, val):
        self.lbl_alpha_val.setText(f"{val}%")
        self.vp_3d.set_bone_alpha(val / 100.0)

    def enable_overlay_controls(self, enabled, render=True):
        self.chk_overlay.setEnabled(enabled)
        self.slider_alpha.setEnabled(enabled)
        self.lbl_alpha_title.setEnabled(enabled)
        self.lbl_alpha_val.setEnabled(enabled)
        self.chk_export_host.setEnabled(bool(enabled))
        if enabled:
            if render:
                self.vp_3d.set_bone_alpha(self.slider_alpha.value() / 100.0)
                self.vp_3d.set_overlay_enabled(self.chk_overlay.isChecked())
            else:
                self.vp_3d.bone_alpha = float(self.slider_alpha.value() / 100.0)
                self.vp_3d.overlay_enabled = bool(self.chk_overlay.isChecked())

    def reset_pattern_to_none(self):
        """Reset architecture pattern dropdown to 'None' and refresh diagnostics/porosity."""
        if self.combo_pattern.currentText() != "None":
            self.combo_pattern.setCurrentText("None")
        else:
            self._on_pattern_changed("None")
        self.reset_porosity_metrics()

    def reset_porosity_metrics(self, hint_text=None):
        self.lbl_porosity_val.setText("Scaffold Porosity (φ): –")
        self.lbl_porosity_val.setStyleSheet(
            "color: #9ab; font-size: 12px; font-weight: bold; font-family: Consolas; border: none;"
        )
        msg = (
            hint_text if hint_text is not None
            else "Click 'Generate Bone Scaffold' to measure porosity"
        )
        self.lbl_porosity_detail.setText(msg)
        self.lbl_porosity_detail.setStyleSheet(
            "color: #889; font-size: 10px; font-family: Consolas; border: none;"
        )

    def update_porosity_metrics(self, metrics):
        if not metrics:
            self.reset_porosity_metrics()
            return
        por_pct = float(metrics.get("porosity_pct", 0.0))
        sol_pct = float(metrics.get("solid_fraction_pct", 100.0 - por_pct))
        scaf_v  = float(metrics.get("scaffold_vol_mm3", 0.0))
        env_v   = float(metrics.get("envelope_vol_mm3", 0.0))
        void_v  = float(metrics.get("void_vol_mm3", max(0.0, env_v - scaf_v)))
        pat_tag = metrics.get("pattern", "None")
        cache_tag = " [Cached Env]" if metrics.get("cache_hit", False) else ""

        self.lbl_porosity_val.setText(f"Scaffold Porosity (φ): {por_pct:.2f}%")
        self.lbl_porosity_val.setStyleSheet(
            "color: #48dbfb; font-size: 13px; font-weight: bold; font-family: Consolas; border: none;"
        )
        self.lbl_porosity_detail.setText(
            f"Solid Fraction: {sol_pct:.2f}%  |  Pattern: {pat_tag}{cache_tag}\n"
            f"Void: {void_v:,.1f} mm³  |  Solid / Env: {scaf_v:,.1f} / {env_v:,.1f} mm³"
        )
        self.lbl_porosity_detail.setStyleSheet(
            "color: #b8d8e8; font-size: 10px; font-family: Consolas; border: none;"
        )

    def update_mesh_stats(self, n_verts, n_faces, watertight, porosity_metrics=None):
        wt = "Yes" if watertight else "No"
        if porosity_metrics is not None:
            por_pct = float(porosity_metrics.get("porosity_pct", 0.0))
            sol_pct = float(porosity_metrics.get("solid_fraction_pct", 100.0 - por_pct))
            scaf_v  = float(porosity_metrics.get("scaffold_vol_mm3", 0.0))
            env_v   = float(porosity_metrics.get("envelope_vol_mm3", 0.0))
            por_lines = (
                f"Porosity: {por_pct:.2f}% (Solid: {sol_pct:.2f}%)\n"
                f"Volume: {scaf_v:,.1f} / {env_v:,.1f} mm³\n"
            )
        else:
            por_lines = "Porosity: – (Not generated)\n"
        self.lbl_mesh_stats.setText(
            f"{por_lines}Vertices: {n_verts:,}\nFaces: {n_faces:,}\nWatertight: {wt}"
        )

    # ── Accessor Properties ──────────────────────────────────────────────

    @property
    def tpms_pattern(self):
        return self.combo_pattern.currentText()

    @property
    def tpms_type(self):
        return self.combo_type.currentText()

    @property
    def pore_size_um(self):
        return self.spin_pore_um.value()

    @property
    def strut_diam_um(self):
        return self.spin_strut_um.value()

    @property
    def tpms_level(self):
        return self.slider_level.value() / 100.0

    @property
    def tpms_thickness(self):
        return self.slider_thick.value() / 100.0

    @property
    def tpms_invert(self):
        return self.chk_invert.isChecked()

    @property
    def fill_osteotomy_defect(self):
        return self.combo_target_mode.currentText() == TARGET_MODE_SOLID_GRAFT

    @property
    def show_direction_arrow(self):
        return self.chk_show_arrow.isChecked()

    @property
    def dir_yaw(self):
        return self.spin_dir_yaw.value()

    @property
    def dir_pitch(self):
        return self.spin_dir_pitch.value()

    @property
    def dir_roll(self):
        return self.spin_dir_roll.value()

    @property
    def export_format(self):
        txt = self.combo_export_format.currentText().lower()
        return "stl" if ".stl" in txt else "3mf"

    def set_export_format(self, fmt):
        target_idx = 1 if str(fmt).lower().lstrip(".") == "stl" else 0
        self.combo_export_format.blockSignals(True)
        self.combo_export_format.setCurrentIndex(target_idx)
        self.combo_export_format.blockSignals(False)

    @property
    def export_host_overlay(self):
        return bool(self.chk_export_host.isEnabled() and self.chk_export_host.isChecked())


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW  (Two-Tab Workstation Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    Top-level workstation window (Version 9) hosting a styled two-tab workflow:
      - Tab 1 ("Segmentation"): MPR quad-viewport, ±90° planar reorientation,
        real-time HU thresholding with "Keep Largest Island Only", and 3MF/STL export.
      - Tab 2 ("Scaffold Design"): Dedicated full-height 3D viewport, 3D Volumetric
        Implicit Cutting Tools (Oriented Box & Dual-Plane Osteotomy Wedge),
        Pattern Direction Vector Engine, cached sub-voxel porous CSG,
        multi-threaded Flying Edges extraction, and watertight 3MF/STL export
        with optional co-registered Host Bone Overlay model export.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Andamyo v9 — Two-Tab Bone Segmentation, 3D Osteotomy & Directional Scaffold Workstation"
        )
        self.resize(1480, 940)

        # ── Shared Clinical Volume & Segmentation State ───────────────────
        self.volume_data    = None             # int16 (Z, Y, X)
        self.image_spacing  = (1.0, 1.0, 1.0)  # (Z, Y, X) in mm
        self.seg_bone_mask  = None             # uint8 (Z, Y, X) active Tab 1 bone mask
        self.seg_pv_mesh    = None             # pv.PolyData active Tab 1 bone mesh

        # ── Tab 2 (Scaffold Design) State ────────────────────────────────
        self.cut_roi_mask          = None      # uint8 (Z, Y, X) post-cut largest island
        self.cut_host_mask         = None      # uint8 (Z, Y, X) uncut host bone
        self.cut_roi_centroid_xyz  = (0.0, 0.0, 0.0)
        self.bone_bbox_spans_xyz   = (40.0, 40.0, 40.0)
        self.scaf_active_mesh      = None      # pv.PolyData (candidate ROI or generated scaffold)
        self.scaf_host_mesh        = None      # pv.PolyData (translucent host bone)
        self.scaf_generated_mask   = None      # uint8 (Z, Y, X) native scaffold preview mask
        self.scaf_mesh_is_porous   = False
        self.scaf_porosity_metrics = None      # dict of measured scaffold porosity & volumes
        self._pending_pattern      = "None"
        self._cached_envelope_key  = None
        self._cached_envelope_data = None
        self._cut_initialized_for_vol = False
        self._seg_dirty_for_tab2      = False

        # MPR cursor indices
        self.cur_z = 0
        self.cur_y = 0
        self.cur_x = 0

        # File paths
        self.dicom_folder = None
        self.nrrd_path    = None

        # Background QThread lifecycle tracking
        self._worker             = None
        self._seg_preview_worker = None
        self._cut_preview_worker = None
        self._retired_workers    = []
        self._seg_preview_seq    = 0
        self._cut_preview_seq    = 0
        self._heavy_busy         = False

        # Debounce timers for real-time interactive responsiveness
        self._seg_debounce = QTimer(self)
        self._seg_debounce.setSingleShot(True)
        self._seg_debounce.setInterval(260)
        self._seg_debounce.timeout.connect(self._start_seg_preview_worker)

        self._cut_debounce = QTimer(self)
        self._cut_debounce.setSingleShot(True)
        self._cut_debounce.setInterval(200)
        self._cut_debounce.timeout.connect(self._start_cut_preview_worker)

        # ── Build UI ─────────────────────────────────────────────────────
        self._build_tabs()
        self._build_status_bar()
        self._apply_stylesheet()

    def _build_tabs(self):
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(False)

        self.seg_tab  = SegmentationTabWidget()
        self.scaf_tab = ScaffoldTabWidget()

        self.tabs.addTab(self.seg_tab,  "  1. Segmentation  ")
        self.tabs.addTab(self.scaf_tab, "  2. Scaffold Design  ")
        self.setCentralWidget(self.tabs)

        # Wire Tab 1 MPR & sidebar signals
        for view in (
            self.seg_tab.view_coronal,
            self.seg_tab.view_axial,
            self.seg_tab.view_sagittal,
        ):
            view.slice_changed.connect(self._on_slice_changed)
            view.click_navigated.connect(self._on_click_navigated)
            view.wl_changed_sig.connect(self._on_wl_changed_from_view)

        self.seg_tab.sig_select_dicom.connect(self._select_dicom_folder)
        self.seg_tab.sig_select_nrrd.connect(self._select_nrrd_file)
        self.seg_tab.sig_load_volume.connect(self._load_volume)
        self.seg_tab.sig_thresh_params_changed.connect(self._on_seg_thresh_changed)
        self.seg_tab.sig_wl_preset_requested.connect(self._on_wl_changed_from_view)
        self.seg_tab.sig_export_seg_mesh.connect(self._export_segmentation_model)

        # Wire Tab 2 Cutting, Orientation & Scaffold signals
        cp = self.scaf_tab.cutting_panel
        cp.sig_cut_mode_changed.connect(self._on_cut_mode_changed)
        cp.sig_cut_params_changed.connect(self._on_cut_spinboxes_changed)
        cp.sig_reset_cut_to_bone.connect(self._on_reset_cut_to_bone)
        cp.sig_cut_gizmo_visibility_changed.connect(self._on_cut_gizmo_visibility_changed)

        self.scaf_tab.sig_dir_arrow_toggled.connect(self._on_dir_arrow_toggled)
        self.scaf_tab.sig_dir_angles_changed.connect(self._on_dir_spinboxes_changed)
        self.scaf_tab.sig_generate_scaffold.connect(self._generate_scaffold)
        self.scaf_tab.sig_export_scaf_mesh.connect(self._export_scaffold_model)

        # Automatic state transfer when switching tabs
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(270)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready — Load a DICOM folder or NRRD scan in Tab 1")
        self.status_bar.addWidget(self.lbl_status, 1)

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #ddd;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12px;
            }
            QTabWidget::pane {
                border: 1px solid #484848;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #353535;
                color: #bbb;
                border: 1px solid #4c4c4c;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 7px 20px;
                margin-right: 3px;
                font-weight: bold;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background-color: #1f5f5b;
                color: #ffffff;
                border-color: #00c8c8;
            }
            QTabBar::tab:hover:!selected {
                background-color: #424242;
                color: #eee;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 14px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #444;
                border: 1px solid #666;
                border-radius: 3px;
                padding: 4px 10px;
                color: #ddd;
            }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #333; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #383838;
                border: 1px solid #555;
                border-radius: 2px;
                padding: 2px 4px;
                color: #ddd;
            }
            QSlider::groove:horizontal {
                background: #444; height: 6px; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #aaa; width: 14px; margin: -4px 0;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background: #333;
                color: #ddd;
            }
            QProgressBar::chunk {
                background-color: #2a7;
                border-radius: 2px;
            }
            QScrollArea { border: none; }
            QStatusBar { background: #222; }
            QCheckBox { spacing: 6px; }
            QLabel { color: #ccc; }
        """)

    # ══════════════════════════════════════════════════════════════════════
    #  MPR CROSSHAIR SYNCHRONISATION (TAB 1)
    # ══════════════════════════════════════════════════════════════════════

    def _sync_all_views(self):
        if self.volume_data is None:
            return
        z, y, x = self.cur_z, self.cur_y, self.cur_x

        self.seg_tab.view_axial.set_slice(z)
        self.seg_tab.view_axial.set_crosshairs(y, x)
        self.seg_tab.vp_axial.slider.blockSignals(True)
        self.seg_tab.vp_axial.slider.setValue(z)
        self.seg_tab.vp_axial.slider.blockSignals(False)

        self.seg_tab.view_coronal.set_slice(y)
        self.seg_tab.view_coronal.set_crosshairs(z, x)
        self.seg_tab.vp_coronal.slider.blockSignals(True)
        self.seg_tab.vp_coronal.slider.setValue(y)
        self.seg_tab.vp_coronal.slider.blockSignals(False)

        self.seg_tab.view_sagittal.set_slice(x)
        self.seg_tab.view_sagittal.set_crosshairs(z, y)
        self.seg_tab.vp_sagittal.slider.blockSignals(True)
        self.seg_tab.vp_sagittal.slider.setValue(x)
        self.seg_tab.vp_sagittal.slider.blockSignals(False)

    def _on_slice_changed(self, plane, idx):
        if plane == 'axial':
            self.cur_z = idx
        elif plane == 'coronal':
            self.cur_y = idx
        elif plane == 'sagittal':
            self.cur_x = idx
        self._sync_all_views()

    def _on_click_navigated(self, plane, row, col):
        if plane == 'axial':
            self.cur_y = row
            self.cur_x = col
        elif plane == 'coronal':
            self.cur_z = row
            self.cur_x = col
        elif plane == 'sagittal':
            self.cur_z = row
            self.cur_y = col
        self._sync_all_views()

    def _on_wl_changed_from_view(self, wl, ww):
        for v in (
            self.seg_tab.view_axial,
            self.seg_tab.view_coronal,
            self.seg_tab.view_sagittal,
        ):
            v.set_wl_ww(wl, ww)

    # ══════════════════════════════════════════════════════════════════════
    #  WORKER CANCELLATION & RETIREMENT HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _retire_worker(self, w):
        if w is not None and w.isRunning():
            self._retired_workers.append(w)
            w.finished.connect(lambda *_, wr=w: self._cleanup_retired(wr))

    def _cleanup_retired(self, w):
        if w in self._retired_workers:
            self._retired_workers.remove(w)

    def _cancel_seg_preview(self):
        self._seg_debounce.stop()
        self._seg_preview_seq += 1
        if self._seg_preview_worker is not None:
            old = self._seg_preview_worker
            self._seg_preview_worker = None
            old.cancel()
            self._retire_worker(old)

    def _cancel_cut_preview(self):
        self._cut_debounce.stop()
        self._cut_preview_seq += 1
        if self._cut_preview_worker is not None:
            old = self._cut_preview_worker
            self._cut_preview_worker = None
            old.cancel()
            self._retire_worker(old)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1: DATA INGESTION & REAL-TIME TAUBIN SEGMENTATION
    # ══════════════════════════════════════════════════════════════════════

    def _select_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.dicom_folder = folder
            self.nrrd_path = None
            self.seg_tab.lbl_file.setText(os.path.basename(folder))

    def _select_nrrd_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select NRRD File", "",
            "NRRD files (*.nrrd);;All files (*.*)"
        )
        if fp:
            self.nrrd_path = fp
            self.dicom_folder = None
            self.seg_tab.lbl_file.setText(os.path.basename(fp))

    def _load_volume(self):
        if not self.dicom_folder and not self.nrrd_path:
            QMessageBox.warning(
                self, "No Input", "Select a DICOM folder or NRRD file first."
            )
            return

        self._cancel_seg_preview()
        self._cancel_cut_preview()
        self._set_busy(True)

        worker = LoadVolumeWorker(
            dicom_folder=self.dicom_folder,
            nrrd_path=self.nrrd_path,
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(lambda *_: self._set_busy(False))
        worker.error.connect(lambda *_: self._set_busy(False))
        worker.finished.connect(self._on_volume_loaded)
        worker.error.connect(self._on_worker_error)
        self._worker = worker
        worker.start()

    def _invalidate_envelope_cache(self, clear_cut_masks=False):
        """Invalidate cached high-res defect envelope whenever mask or cut changes."""
        self._cached_envelope_key  = None
        self._cached_envelope_data = None
        if clear_cut_masks:
            self.cut_roi_mask   = None
            self.cut_host_mask  = None
            self.scaf_host_mesh = None

    def _reset_pattern_for_new_cut(self):
        """Reset Tab 2 scaffold architecture pattern to 'None' on any new/updated cut."""
        self.scaf_mesh_is_porous   = False
        self.scaf_porosity_metrics = None
        self._pending_pattern      = "None"
        self.scaf_tab.reset_pattern_to_none()

    def _on_volume_loaded(self, vol, spacing):
        self._set_busy(False)
        self.volume_data           = vol
        self.image_spacing         = spacing
        self.seg_bone_mask         = None
        self.seg_pv_mesh           = None
        self._invalidate_envelope_cache(clear_cut_masks=True)
        self.scaf_active_mesh      = None
        self.scaf_generated_mask   = None
        self._reset_pattern_for_new_cut()
        self._cut_initialized_for_vol = False
        self._seg_dirty_for_tab2      = True

        shape = vol.shape  # (Z, Y, X)
        self.seg_tab.lbl_dims.setText(
            f"Dimensions: {shape[2]}×{shape[1]}×{shape[0]}  |  "
            f"Spacing: {spacing[2]:.2f}×{spacing[1]:.2f}×{spacing[0]:.2f} mm"
        )
        self.scaf_tab.set_voxel_spacing(spacing)

        for v in (
            self.seg_tab.view_axial,
            self.seg_tab.view_coronal,
            self.seg_tab.view_sagittal,
        ):
            v.set_mask(None)
            v.set_volume(vol)

        self.seg_tab.vp_axial.update_range(shape[0] - 1)
        self.seg_tab.vp_coronal.update_range(shape[1] - 1)
        self.seg_tab.vp_sagittal.update_range(shape[2] - 1)

        self.cur_z = shape[0] // 2
        self.cur_y = shape[1] // 2
        self.cur_x = shape[2] // 2
        self._sync_all_views()

        self.seg_tab.vp_3d._has_camera_pose = False
        self.seg_tab.vp_3d.display_mesh(None)
        self.seg_tab.lbl_mesh_stats.setText("Vertices: –\nFaces: –\nWatertight: –")

        self.scaf_tab.vp_3d.clear_cutting_gizmos()
        self.scaf_tab.vp_3d.clear_direction_arrow_gizmo()
        self.scaf_tab.vp_3d._has_camera_pose = False
        self.scaf_tab.vp_3d.set_meshes(None, None)
        self.scaf_tab.enable_overlay_controls(False)
        self.scaf_tab.reset_porosity_metrics()
        self.scaf_tab.lbl_mesh_stats.setText("Porosity: –\nVertices: –\nFaces: –\nWatertight: –")

        # Immediately launch real-time segmented 2D mask + Taubin-smoothed 3D preview
        self._start_seg_preview_worker(reset_cam=True)

    def _on_seg_thresh_changed(self):
        if self.volume_data is None or self._heavy_busy:
            return
        self._invalidate_envelope_cache(clear_cut_masks=True)
        self._seg_dirty_for_tab2 = True
        self._seg_debounce.start()

    def _start_seg_preview_worker(self, reset_cam=False):
        if self.volume_data is None or self._heavy_busy:
            return

        self._cancel_seg_preview()
        seq = self._seg_preview_seq
        self._seg_reset_cam_flag = bool(reset_cam)

        worker = Preview3DWorker(
            volume=self.volume_data,
            spacing=self.image_spacing,
            hu_min=self.seg_tab.hu_min,
            hu_max=self.seg_tab.hu_max,
            keep_largest=self.seg_tab.keep_largest_island,
            auto_smooth=True,
        )
        worker.progress.connect(self._on_preview_progress)
        worker.finished.connect(
            lambda mask, mesh, s=seq: self._on_seg_preview_finished(s, mask, mesh)
        )
        worker.error.connect(
            lambda msg, s=seq: self._on_preview_error(s, msg)
        )
        self._seg_preview_worker = worker
        worker.start()

    def _on_seg_preview_finished(self, seq, seg_mask, pv_mesh):
        if seq != self._seg_preview_seq or self._heavy_busy:
            return
        self._seg_preview_worker = None
        self._invalidate_envelope_cache(clear_cut_masks=True)
        self.seg_bone_mask = seg_mask
        self.seg_pv_mesh   = pv_mesh
        self._seg_dirty_for_tab2 = True

        # Real-time 2D MPR mask update (reflects "Keep Largest Island Only" immediately)
        for v in (
            self.seg_tab.view_axial,
            self.seg_tab.view_coronal,
            self.seg_tab.view_sagittal,
        ):
            v.set_mask(seg_mask)

        reset_c = getattr(self, "_seg_reset_cam_flag", False)
        self._seg_reset_cam_flag = False
        self.seg_tab.vp_3d.display_mesh(pv_mesh, color=BONE_COLOR, reset_camera=reset_c)
        self._update_stats_for_mesh(pv_mesh, self.seg_tab)

        self.progress_bar.setValue(100)
        island_txt = "Largest Island" if self.seg_tab.keep_largest_island else "All Bone Islands"
        if pv_mesh is not None and pv_mesh.n_points > 0:
            self.lbl_status.setText(
                f"Segmentation active ({island_txt}, {pv_mesh.n_points:,} verts) — "
                f"Switch to Tab 2 ('Scaffold Design') to cut & generate lattice"
            )
        else:
            self.lbl_status.setText("No bone voxels in current HU threshold range")

        # If user is currently viewing Tab 2, propagate updated segmentation live
        if self.tabs.currentIndex() == 1:
            self._sync_segmentation_to_scaffold_tab()

    # ══════════════════════════════════════════════════════════════════════
    #  TAB SWITCHING & TAB 2 INTERACTIVE 3D CUTTING / ORIENTATION ENGINE
    # ══════════════════════════════════════════════════════════════════════

    def _on_tab_changed(self, idx):
        if idx == 1:
            self._sync_segmentation_to_scaffold_tab()

    def _compute_bone_physical_bounds(self, mask):
        """Return ((cx, cy, cz), (span_x, span_y, span_z)) in mm from binary mask."""
        nz = np.nonzero(mask > 0)
        if len(nz[0]) == 0:
            return (0.0, 0.0, 0.0), (40.0, 40.0, 40.0)
        sz, sy, sx = (
            float(self.image_spacing[0]),
            float(self.image_spacing[1]),
            float(self.image_spacing[2]),
        )
        z0, z1 = float(nz[0].min()) * sz, float(nz[0].max()) * sz
        y0, y1 = float(nz[1].min()) * sy, float(nz[1].max()) * sy
        x0, x1 = float(nz[2].min()) * sx, float(nz[2].max()) * sx

        cx = float(nz[2].mean()) * sx
        cy = float(nz[1].mean()) * sy
        cz = float(nz[0].mean()) * sz
        span_x = max(4.0, x1 - x0)
        span_y = max(4.0, y1 - y0)
        span_z = max(4.0, z1 - z0)
        return (cx, cy, cz), (span_x, span_y, span_z)

    def _init_cut_defaults_from_bone(self, reset_angles=True):
        """Fit Box ROI and Dual-Planes Osteotomy defaults to the active bone mask."""
        if self.seg_bone_mask is None or not np.any(self.seg_bone_mask):
            return
        (cx, cy, cz), (span_x, span_y, span_z) = self._compute_bone_physical_bounds(
            self.seg_bone_mask
        )
        self.bone_bbox_spans_xyz = (span_x, span_y, span_z)
        self.cut_roi_centroid_xyz = (cx, cy, cz)

        cp = self.scaf_tab.cutting_panel
        box_defaults = {
            "cx": cx, "cy": cy, "cz": cz,
            "lx": round(span_x * 1.15, 1),
            "ly": round(span_y * 1.15, 1),
            "lz": round(max(4.0, span_z * 0.35), 1),
        }
        if reset_angles:
            box_defaults.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
        cp.set_box_params_silent(box_defaults)

        planes_defaults = {
            "mid_x": cx, "mid_y": cy, "mid_z": cz,
            "gap_l": round(max(4.0, span_z * 0.28), 1),
        }
        if reset_angles:
            planes_defaults.update({
                "line_yaw": 0.0, "line_pitch": 90.0,
                "p1_tilt": 0.0, "p1_cross_tilt": 0.0,
                "p2_tilt": 0.0, "p2_cross_tilt": 0.0,
            })
        cp.set_planes_params_silent(planes_defaults)

    def _sync_segmentation_to_scaffold_tab(self):
        """Transfer active segmented bone from Tab 1 into Tab 2 as baseline geometry."""
        if self.volume_data is None:
            return

        # Ensure baseline mask exists if user switched tabs while initial preview ran
        if self.seg_bone_mask is None:
            self.seg_bone_mask = create_bone_mask(
                self.volume_data,
                self.seg_tab.hu_min,
                self.seg_tab.hu_max,
                keep_largest=self.seg_tab.keep_largest_island,
            )

        if not self._cut_initialized_for_vol:
            self._init_cut_defaults_from_bone(reset_angles=True)
            self._cut_initialized_for_vol = True
            self._refresh_tab2_cutting_gizmos()
            self._start_cut_preview_worker(reset_cam=True)
            self._seg_dirty_for_tab2 = False
        elif self._seg_dirty_for_tab2:
            _, spans = self._compute_bone_physical_bounds(self.seg_bone_mask)
            self.bone_bbox_spans_xyz = spans
            self._refresh_tab2_cutting_gizmos()
            self._start_cut_preview_worker(reset_cam=False)
            self._seg_dirty_for_tab2 = False

    def _plane_visual_size(self):
        sx, sy, sz = self.bone_bbox_spans_xyz
        return max(16.0, 1.35 * max(sx, sy, sz * 0.5))

    def _arrow_visual_length(self):
        sx, sy, sz = self.bone_bbox_spans_xyz
        return max(10.0, 0.45 * max(sx, sy, min(sz, max(sx, sy) * 1.5)))

    def _refresh_tab2_cutting_gizmos(self):
        """Show/hide VTK 3D cutting gizmo matching CuttingToolsPanel's active mode."""
        cp = self.scaf_tab.cutting_panel
        mode = cp.cut_mode
        if (
            self.seg_bone_mask is None
            or not np.any(self.seg_bone_mask)
            or cp.hide_cutting_gizmo
            or mode == CUT_MODE_NONE
        ):
            self.scaf_tab.vp_3d.clear_cutting_gizmos()
            return

        if mode == CUT_MODE_BOX:
            self.scaf_tab.vp_3d.set_box_gizmo(
                True,
                params=cp.get_box_params(),
                on_changed_cb=self._on_gizmo_box_dragged,
            )
        elif mode == CUT_MODE_PLANES:
            self.scaf_tab.vp_3d.set_dual_planes_gizmo(
                True,
                params=cp.get_planes_params(),
                plane_size=self._plane_visual_size(),
                on_changed_cb=self._on_gizmo_planes_dragged,
            )
        else:
            self.scaf_tab.vp_3d.clear_cutting_gizmos()

    def _refresh_tab2_direction_gizmo(self):
        """Show/hide or reposition the 3D Pattern Direction Vector arrow at ROI centroid."""
        if not self.scaf_tab.show_direction_arrow or self.seg_bone_mask is None:
            self.scaf_tab.vp_3d.clear_direction_arrow_gizmo()
            return

        if self.scaf_tab.vp_3d._arrow_sphere_w is None:
            self.scaf_tab.vp_3d.set_direction_arrow_gizmo(
                True,
                centroid_xyz=self.cut_roi_centroid_xyz,
                yaw_deg=self.scaf_tab.dir_yaw,
                pitch_deg=self.scaf_tab.dir_pitch,
                roll_deg=self.scaf_tab.dir_roll,
                arrow_len=self._arrow_visual_length(),
                on_changed_cb=self._on_gizmo_arrow_dragged,
            )
        else:
            self.scaf_tab.vp_3d.update_direction_arrow_from_params(
                centroid_xyz=self.cut_roi_centroid_xyz,
                yaw_deg=self.scaf_tab.dir_yaw,
                pitch_deg=self.scaf_tab.dir_pitch,
                roll_deg=self.scaf_tab.dir_roll,
                arrow_len=self._arrow_visual_length(),
            )

    # ── Two-Way Cutting & Direction Gizmo <-> Spinbox Handlers ───────────

    def _on_cut_mode_changed(self, _mode_txt):
        self._invalidate_envelope_cache(clear_cut_masks=True)
        self._reset_pattern_for_new_cut()
        if self.seg_bone_mask is None:
            return
        self._refresh_tab2_cutting_gizmos()
        self._start_cut_preview_worker(reset_cam=False)

    def _on_reset_cut_to_bone(self):
        self._invalidate_envelope_cache(clear_cut_masks=True)
        self._reset_pattern_for_new_cut()
        if self.scaf_tab.cutting_panel.cut_mode != CUT_MODE_NONE:
            self.scaf_tab.cutting_panel.set_hide_gizmo_silent(False)
        if self.seg_bone_mask is None:
            return
        self._init_cut_defaults_from_bone(reset_angles=True)
        self._refresh_tab2_cutting_gizmos()
        self._start_cut_preview_worker(reset_cam=False)

    def _on_cut_gizmo_visibility_changed(self, _hidden):
        self._refresh_tab2_cutting_gizmos()

    def _on_cut_spinboxes_changed(self):
        """User typed/stepped a Box or Dual-Planes spinbox -> sync 3D gizmo + cut."""
        self._invalidate_envelope_cache(clear_cut_masks=True)
        self._reset_pattern_for_new_cut()
        if self.seg_bone_mask is None:
            return
        cp = self.scaf_tab.cutting_panel
        if not cp.hide_cutting_gizmo:
            if cp.cut_mode == CUT_MODE_BOX:
                self.scaf_tab.vp_3d.update_box_gizmo_from_params(cp.get_box_params())
            elif cp.cut_mode == CUT_MODE_PLANES:
                self.scaf_tab.vp_3d.update_dual_planes_from_params(
                    cp.get_planes_params(), plane_size=self._plane_visual_size()
                )
        if not self._heavy_busy:
            self._cut_debounce.start()

    def _on_gizmo_box_dragged(self, new_box_params):
        """User dragged VTK BoxWidget in 3D -> sync spinboxes + debounced cut."""
        self._invalidate_envelope_cache(clear_cut_masks=True)
        self._reset_pattern_for_new_cut()
        self.scaf_tab.cutting_panel.set_box_params_silent(new_box_params)
        if not self._heavy_busy:
            self._cut_debounce.start()

    def _on_gizmo_planes_dragged(self, new_planes_params):
        """User dragged VTK LineWidget in 3D -> sync spinboxes + debounced cut."""
        self._invalidate_envelope_cache(clear_cut_masks=True)
        self._reset_pattern_for_new_cut()
        self.scaf_tab.cutting_panel.set_planes_params_silent(new_planes_params)
        if not self._heavy_busy:
            self._cut_debounce.start()

    def _on_dir_arrow_toggled(self, _checked):
        self._refresh_tab2_direction_gizmo()

    def _on_dir_spinboxes_changed(self):
        if self.scaf_tab.show_direction_arrow:
            self._refresh_tab2_direction_gizmo()

    def _on_gizmo_arrow_dragged(self, yaw_deg, pitch_deg):
        self.scaf_tab.set_dir_angles_silent(yaw_deg, pitch_deg)

    # ── Background Interactive Cut Preview (`CutPreviewWorker`) ──────────

    def _start_cut_preview_worker(self, reset_cam=False):
        if self.seg_bone_mask is None or self._heavy_busy:
            return

        self._cancel_cut_preview()
        seq = self._cut_preview_seq
        self._cut_reset_cam_flag = bool(reset_cam)

        cp = self.scaf_tab.cutting_panel
        worker = CutPreviewWorker(
            seg_bone_mask=self.seg_bone_mask,
            spacing=self.image_spacing,
            cut_mode=cp.cut_mode,
            cut_params=cp.get_active_cut_params(),
        )
        worker.progress.connect(self._on_preview_progress)
        worker.finished.connect(
            lambda r_m, h_m, r_pd, h_pd, cent, s=seq:
                self._on_cut_preview_finished(s, r_m, h_m, r_pd, h_pd, cent)
        )
        worker.error.connect(
            lambda msg, s=seq: self._on_preview_error(s, msg)
        )
        self._cut_preview_worker = worker
        worker.start()

    def _on_cut_preview_finished(self, seq, roi_mask, host_mask,
                                 roi_mesh, host_mesh, centroid_xyz):
        if seq != self._cut_preview_seq or self._heavy_busy:
            return
        self._cut_preview_worker = None

        self._invalidate_envelope_cache(clear_cut_masks=False)
        self.cut_roi_mask         = roi_mask
        self.cut_host_mask        = host_mask
        self.cut_roi_centroid_xyz = centroid_xyz
        self.scaf_active_mesh     = roi_mesh
        self.scaf_host_mesh       = host_mesh
        self.scaf_generated_mask  = roi_mask
        self._reset_pattern_for_new_cut()

        has_host = host_mesh is not None and host_mesh.n_points > 0
        self.scaf_tab.enable_overlay_controls(has_host)

        # In "No Cut" mode without lattice, show ivory bone; when cut is active,
        # highlight the isolated surgical ROI in cyan against translucent host bone!
        is_cut_active = (self.scaf_tab.cutting_panel.cut_mode != CUT_MODE_NONE)
        roi_color = SCAFFOLD_COLOR if is_cut_active else BONE_COLOR

        reset_c = getattr(self, "_cut_reset_cam_flag", False)
        self._cut_reset_cam_flag = False
        self.scaf_tab.vp_3d.set_meshes(
            roi_mesh, host_mesh, scaffold_color=roi_color, reset_camera=reset_c
        )

        # Reposition Pattern Direction Vector arrow to updated ROI centroid
        if self.scaf_tab.show_direction_arrow:
            self._refresh_tab2_direction_gizmo()

        self._update_stats_for_mesh(roi_mesh, self.scaf_tab, porosity_metrics=None)
        self.progress_bar.setValue(100)
        if roi_mesh is not None and roi_mesh.n_points > 0:
            self.lbl_status.setText(
                f"Cut ROI ready ({roi_mesh.n_points:,} verts, largest island isolated) — "
                f"Click 'Generate Bone Scaffold' to synthesize lattice"
            )
        else:
            self.lbl_status.setText(
                "Warning: Cutting region does not intersect any segmented bone voxels"
            )

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2: HIGH-RES DIRECTIONAL BONE SCAFFOLD GENERATION
    # ══════════════════════════════════════════════════════════════════════

    def _generate_scaffold(self):
        if self.volume_data is None and self.seg_bone_mask is None:
            QMessageBox.warning(
                self, "No Segmented Bone",
                "Load and segment a CT volume in Tab 1 ('Segmentation') first."
            )
            return

        self._cancel_seg_preview()
        self._cancel_cut_preview()
        self._set_busy(True)

        cp = self.scaf_tab.cutting_panel
        active_cut_params = cp.get_active_cut_params()
        self._pending_pattern = self.scaf_tab.tpms_pattern

        cached_env_to_pass = None
        precomp_cut = None
        precomp_host_mesh = None
        if self.cut_roi_mask is not None and not self._seg_dirty_for_tab2:
            precomp_cut = (
                self.cut_roi_mask,
                self.cut_host_mask,
                self.cut_roi_centroid_xyz,
            )
            precomp_host_mesh = self.scaf_host_mesh
            cand_key = make_envelope_cache_key(
                roi_mask_hash=_hash_binary_mask(self.cut_roi_mask),
                cut_mode=cp.cut_mode,
                cut_params=active_cut_params,
                fill_defect=self.scaf_tab.fill_osteotomy_defect,
                spacing_zyx=self.image_spacing,
            )
            if (
                self._cached_envelope_key is not None
                and cand_key == self._cached_envelope_key
                and self._cached_envelope_data is not None
            ):
                cached_env_to_pass = self._cached_envelope_data

        worker = SegmentWorker(
            spacing=self.image_spacing,
            seg_bone_mask=self.seg_bone_mask,
            volume=self.volume_data,
            hu_min=self.seg_tab.hu_min,
            hu_max=self.seg_tab.hu_max,
            keep_largest=self.seg_tab.keep_largest_island,
            cut_mode=cp.cut_mode,
            cut_params=active_cut_params,
            pattern=self.scaf_tab.tpms_pattern,
            tpms_type=self.scaf_tab.tpms_type,
            pore_size_um=self.scaf_tab.pore_size_um,
            level_set=self.scaf_tab.tpms_level,
            thickness=self.scaf_tab.tpms_thickness,
            strut_diam_um=self.scaf_tab.strut_diam_um,
            invert=self.scaf_tab.tpms_invert,
            fill_defect=self.scaf_tab.fill_osteotomy_defect,
            dir_yaw=self.scaf_tab.dir_yaw,
            dir_pitch=self.scaf_tab.dir_pitch,
            dir_roll=self.scaf_tab.dir_roll,
            cached_envelope=cached_env_to_pass,
            precomputed_cut=precomp_cut,
            precomputed_host_mesh=precomp_host_mesh,
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(lambda *_: self._set_busy(False))
        worker.error.connect(lambda *_: self._set_busy(False))
        worker.finished.connect(self._on_scaffold_generated)
        worker.error.connect(self._on_worker_error)
        self._worker = worker
        worker.start()

    def _on_scaffold_generated(self, full_scaffold_mask, scaffold_pv_mesh,
                               bone_pv_mesh, eff_delta_mm, porosity_metrics):
        self._set_busy(False)

        # Store newly computed high-res resampled defect envelope in cache
        if isinstance(porosity_metrics, dict):
            new_env_cache = porosity_metrics.pop("_new_envelope_cache", None)
            new_env_key   = porosity_metrics.pop("_envelope_cache_key", None)
            if new_env_cache is not None and new_env_key is not None:
                self._cached_envelope_data = new_env_cache
                self._cached_envelope_key  = new_env_key

        self.scaf_generated_mask   = full_scaffold_mask
        self.scaf_active_mesh      = scaffold_pv_mesh
        self.scaf_host_mesh        = bone_pv_mesh
        self.scaf_porosity_metrics = porosity_metrics
        self.scaf_mesh_is_porous   = (
            self._pending_pattern is not None and self._pending_pattern != "None"
        )

        # Auto-hide 3D cutting gizmo so generated scaffold is unobstructed
        self.scaf_tab.cutting_panel.set_hide_gizmo_silent(True)
        self.scaf_tab.vp_3d.clear_cutting_gizmos(render=False)

        has_host = bone_pv_mesh is not None and bone_pv_mesh.n_points > 0
        self.scaf_tab.enable_overlay_controls(has_host, render=False)

        # Render full-resolution scaffold mesh directly (zero viewport decimation)
        self.scaf_tab.vp_3d.set_meshes(
            scaffold_pv_mesh, bone_pv_mesh,
            scaffold_color=SCAFFOLD_COLOR, reset_camera=False
        )
        self.scaf_tab.update_porosity_metrics(porosity_metrics)
        self._update_stats_for_mesh(
            scaffold_pv_mesh, self.scaf_tab, porosity_metrics=porosity_metrics
        )

        por_pct = float(porosity_metrics.get("porosity_pct", 0.0)) if porosity_metrics else 0.0
        cache_hit = bool(porosity_metrics.get("cache_hit", False)) if porosity_metrics else False
        elapsed_s = float(porosity_metrics.get("elapsed_s", 0.0)) if porosity_metrics else 0.0
        cache_str = " [Cached Env]" if cache_hit else ""
        time_str = f" in {elapsed_s:.2f}s" if elapsed_s > 0.0 else ""
        mode_tag = (
            "Solid Graft" if self.scaf_tab.fill_osteotomy_defect
            else "Anatomical Canal-Preserving"
        )
        if self.scaf_mesh_is_porous:
            self.lbl_status.setText(
                f"Scaffold ready{time_str}{cache_str} ({self._pending_pattern}, "
                f"Porosity: {por_pct:.2f}% | {mode_tag} | "
                f"Yaw {self.scaf_tab.dir_yaw:+.0f}° "
                f"Pitch {self.scaf_tab.dir_pitch:+.0f}° Roll {self.scaf_tab.dir_roll:+.0f}° | "
                f"Δ_iso={eff_delta_mm * 1000:.0f} µm)"
            )
        else:
            self.lbl_status.setText(
                f"Solid resected implant ready{time_str}{cache_str} "
                f"(Porosity: {por_pct:.2f}% | {mode_tag} | "
                f"Δ_iso={eff_delta_mm * 1000:.0f} µm)"
            )

    # ══════════════════════════════════════════════════════════════════════
    #  3MF / STL EXPORT (TAB 1 SEGMENTED BONE & TAB 2 POROUS/CUT SCAFFOLD)
    # ══════════════════════════════════════════════════════════════════════

    def _prompt_save_mesh_path(self, dialog_title, default_stem, tab_widget):
        """
        Show synchronized 3MF / STL save file dialog matching `tab_widget.export_format`
        and enforce the resolved `.3mf` or `.stl` extension.
        """
        panel_fmt = tab_widget.export_format
        initial_filter = FILE_FILTER_3MF if panel_fmt == "3mf" else FILE_FILTER_STL
        default_filename = f"{default_stem}.{panel_fmt}"

        out_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            dialog_title,
            default_filename,
            EXPORT_FILE_FILTERS,
            initial_filter,
        )
        if not out_path:
            return None, None

        filter_fmt = "stl" if "*.stl" in (selected_filter or "").lower() else "3mf"
        low = out_path.lower()

        if low.endswith(".3mf") and filter_fmt == "stl" and panel_fmt == "3mf":
            out_path = out_path[:-4] + ".stl"
            resolved_fmt = "stl"
        elif low.endswith(".stl") and filter_fmt == "3mf" and panel_fmt == "stl":
            out_path = out_path[:-4] + ".3mf"
            resolved_fmt = "3mf"
        elif low.endswith(".3mf"):
            resolved_fmt = "3mf"
        elif low.endswith(".stl"):
            resolved_fmt = "stl"
        else:
            resolved_fmt = filter_fmt
            out_path = f"{out_path}.{resolved_fmt}"

        tab_widget.set_export_format(resolved_fmt)
        return out_path, resolved_fmt

    def _export_segmentation_model(self):
        if self.seg_pv_mesh is None and (
            self.seg_bone_mask is None or not np.any(self.seg_bone_mask)
        ):
            QMessageBox.warning(
                self, "No Segmentation", "Load and segment a CT scan first."
            )
            return

        out_path, file_fmt = self._prompt_save_mesh_path(
            "Export Segmented Bone 3D Model", "segmented_bone", self.seg_tab
        )
        if not out_path:
            return

        self._run_export_worker(
            pv_mesh=self.seg_pv_mesh,
            mask_data=self.seg_bone_mask,
            out_path=out_path,
            file_format=file_fmt,
            is_porous=False,
            export_host=False,
            primary_name="segmented_bone",
        )

    def _export_scaffold_model(self):
        if self.scaf_active_mesh is None and (
            self.scaf_generated_mask is None or not np.any(self.scaf_generated_mask)
        ):
            QMessageBox.warning(
                self, "No Scaffold",
                "Generate or isolate a bone scaffold in Tab 2 first."
            )
            return

        default_stem = (
            f"scaffold_{self.scaf_tab.tpms_pattern.lower()}"
            if self.scaf_mesh_is_porous else "resected_bone_roi"
        )
        out_path, file_fmt = self._prompt_save_mesh_path(
            "Export Scaffold 3D Model", default_stem, self.scaf_tab
        )
        if not out_path:
            return

        has_host = (
            (self.scaf_host_mesh is not None and self.scaf_host_mesh.n_points > 0)
            or (self.cut_host_mask is not None and np.any(self.cut_host_mask))
        )
        want_host = bool(self.scaf_tab.export_host_overlay and has_host)
        primary_label = "scaffold" if self.scaf_mesh_is_porous else "resected_bone_roi"

        self._run_export_worker(
            pv_mesh=self.scaf_active_mesh,
            mask_data=self.scaf_generated_mask,
            out_path=out_path,
            file_format=file_fmt,
            is_porous=self.scaf_mesh_is_porous,
            export_host=want_host,
            host_pv_mesh=self.scaf_host_mesh,
            host_mask_data=self.cut_host_mask,
            primary_name=primary_label,
        )

    def _run_export_worker(self, pv_mesh, mask_data, out_path, file_format="3mf",
                           is_porous=False, export_host=False,
                           host_pv_mesh=None, host_mask_data=None,
                           primary_name="scaffold"):
        self._set_busy(True)
        worker = ExportMeshWorker(
            pv_mesh=pv_mesh,
            segmented_data=mask_data,
            spacing=self.image_spacing,
            out_path=out_path,
            file_format=file_format,
            is_porous=is_porous,
            already_smoothed=(pv_mesh is not None),
            export_host=export_host,
            host_pv_mesh=host_pv_mesh,
            host_mask_data=host_mask_data,
            primary_name=primary_name,
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(lambda *_: self._set_busy(False))
        worker.error.connect(lambda *_: self._set_busy(False))
        worker.finished.connect(
            lambda paths, fmt=file_format, wh=export_host:
                self._on_mesh_exported(paths, fmt, wh)
        )
        worker.error.connect(self._on_worker_error)
        self._worker = worker
        worker.start()

    def _on_mesh_exported(self, saved_paths, file_format="3mf", exported_host=False):
        fmt_upper = str(file_format or "3mf").upper()
        if not saved_paths:
            return
        if len(saved_paths) == 1:
            p0 = saved_paths[0]
            self.lbl_status.setText(f"Exported {fmt_upper}: {p0}")
            QMessageBox.information(
                self,
                "Export Complete",
                f"{fmt_upper} model saved successfully:\n{p0}",
            )
        else:
            n_host = len(saved_paths) - 1
            piece_word = "piece" if n_host == 1 else "pieces"
            joined_status = " + ".join(os.path.basename(p) for p in saved_paths)
            self.lbl_status.setText(
                f"Exported {len(saved_paths)} {fmt_upper} files "
                f"(1 scaffold + {n_host} host bone {piece_word}): {joined_status}"
            )
            host_bullets = "\n".join(f"  • {p}" for p in saved_paths[1:])
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(saved_paths)} separate {fmt_upper} files successfully:\n\n"
                f"Scaffold Model:\n"
                f"  • {saved_paths[0]}\n\n"
                f"Host Bone Overlay ({n_host} {piece_word}):\n"
                f"{host_bullets}",
            )

    # ══════════════════════════════════════════════════════════════════════
    #  MESH DIAGNOSTICS & UI BUSY STATE
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _update_stats_for_mesh(pv_mesh, tab_widget, porosity_metrics=None):
        if pv_mesh is None or pv_mesh.n_points == 0:
            tab_widget.update_mesh_stats(0, 0, False, porosity_metrics=None)
            return
        n_v = pv_mesh.n_points
        n_f = pv_mesh.n_cells
        if isinstance(porosity_metrics, dict) and "watertight" in porosity_metrics:
            wt = bool(porosity_metrics["watertight"])
        else:
            try:
                wt = bool(pv_mesh.is_manifold)
            except Exception:
                try:
                    faces_arr = pv_mesh.faces.reshape(-1, 4)[:, 1:]
                    tri = trimesh.Trimesh(
                        vertices=pv_mesh.points, faces=faces_arr, process=False
                    )
                    wt = tri.is_watertight
                except Exception:
                    wt = False
        tab_widget.update_mesh_stats(n_v, n_f, wt, porosity_metrics=porosity_metrics)

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_preview_progress(self, pct, msg):
        if not self._heavy_busy:
            self.progress_bar.setValue(pct)
            self.lbl_status.setText(msg)

    def _on_preview_error(self, seq, msg):
        if not self._heavy_busy:
            self.lbl_status.setText(f"Preview Error: {msg}")

    def _on_worker_error(self, msg):
        self.lbl_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Operation Failed", msg)

    def _set_busy(self, busy):
        self._heavy_busy = busy
        self.seg_tab.btn_load.setEnabled(not busy)
        self.seg_tab.btn_export.setEnabled(not busy)
        self.scaf_tab.btn_generate.setEnabled(not busy)
        self.scaf_tab.btn_export.setEnabled(not busy)
        if not busy:
            self.progress_bar.setValue(0)

    def closeEvent(self, event):
        self._cancel_seg_preview()
        self._cancel_cut_preview()
        for w in list(self._retired_workers):
            w.wait(500)
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(1000)
        self.seg_tab.vp_3d.close_plotter()
        self.scaf_tab.vp_3d.close_plotter()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    multiprocessing.freeze_support()

    # Tells Windows not to treat this as generic python.exe on the taskbar
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("andamyo.biomedical.workstation")
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Resolve icon path for standalone script or PyInstaller bundled executable
    if getattr(sys, "frozen", False):
        base_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    icon_candidates = [
        os.path.join(base_dir, "andamyo icon real.ico"),
        os.path.join(os.path.dirname(sys.executable), "andamyo icon real.ico"),
        os.path.join(os.getcwd(), "andamyo icon real.ico"),
    ]
    app_icon = None
    for p in icon_candidates:
        if os.path.exists(p):
            app_icon = QIcon(p)
            break

    if app_icon is not None and not app_icon.isNull():
        app.setWindowIcon(app_icon)

    win = MainWindow()
    if app_icon is not None and not app_icon.isNull():
        win.setWindowIcon(app_icon)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
