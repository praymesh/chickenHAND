"""
STERA-10M VLA Dataloader
========================
Supports:
  - RGB from mp4 (on-the-fly decoding OR pre-extracted frames)
  - Depth frames from HDF5
  - IMU (sliding window, interpolated to frame rate)
  - Text annotations (sparse → dense propagation)
  - MANO targets: joints_cam_rel, hand_pose, global_orient, betas
  - Multi-hand per frame (left/right split)
  - yolo_conf filtering
  - Sliding window temporal context

Dependencies:
    pip install torch torchvision transformers h5py numpy opencv-python
    pip install decord  # fast video decoding, preferred over cv2 for mp4
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Optional, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Try decord first (much faster than cv2 for video), fallback to cv2
try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANO_NUM_JOINTS = 21
MANO_HAND_POSE_JOINTS = 15  # excludes wrist
MANO_BETA_DIM = 10
DEFAULT_IMG_SIZE = 224
DEFAULT_WINDOW = 8          # frames in sliding window
DEFAULT_IMU_WINDOW = 50     # IMU samples per frame window (~1s at 50Hz)
MIN_YOLO_CONF = 0.4

HAND_SIDE = {0: "left", 1: "right"}


# ---------------------------------------------------------------------------
# Utility: dense text annotation propagation
# ---------------------------------------------------------------------------

def build_dense_text_map(
    start_times: np.ndarray,
    end_times: np.ndarray,
    texts: np.ndarray,
    frame_timestamps: np.ndarray,
) -> list[str]:
    """
    Propagate sparse text annotations to every frame timestamp.
    Frames outside any annotation window get an empty string.
    
    Args:
        start_times: (N,) float64
        end_times:   (N,) float64
        texts:       (N,) object (str)
        frame_timestamps: (T,) float64
    Returns:
        List of str, length T
    """
    dense = [""] * len(frame_timestamps)
    for i, ts in enumerate(frame_timestamps):
        for s, e, t in zip(start_times, end_times, texts):
            if s <= ts <= e:
                dense[i] = str(t)
                break
    return dense


# ---------------------------------------------------------------------------
# Utility: IMU interpolation to frame timestamps
# ---------------------------------------------------------------------------

def interpolate_imu_to_frames(
    imu_data: dict,         # keys: angular_velocity, linear_acceleration, orientation_xyzw, timestamps
    frame_timestamps: np.ndarray,
    window: int = DEFAULT_IMU_WINDOW,
) -> np.ndarray:
    """
    For each frame timestamp, extract a fixed-length IMU window centered on it.
    IMU dims: 3 (ang_vel) + 3 (lin_acc) + 4 (quat) = 10 per timestep.
    
    Returns:
        (T, window, 10) float32
    """
    imu_ts = imu_data["timestamps"]            # (M,)
    ang_vel = imu_data["angular_velocity"]     # (M, 3)
    lin_acc = imu_data["linear_acceleration"]  # (M, 3)
    orient  = imu_data["orientation_xyzw"]     # (M, 4)
    
    imu_concat = np.concatenate([ang_vel, lin_acc, orient], axis=-1)  # (M, 10)
    
    T = len(frame_timestamps)
    out = np.zeros((T, window, 10), dtype=np.float32)
    
    half = window // 2
    for i, ft in enumerate(frame_timestamps):
        # Find closest IMU index
        idx = np.searchsorted(imu_ts, ft)
        start = max(0, idx - half)
        end   = start + window
        if end > len(imu_ts):
            end   = len(imu_ts)
            start = max(0, end - window)
        
        chunk = imu_concat[start:end]           # (<= window, 10)
        pad   = window - len(chunk)
        if pad > 0:
            chunk = np.pad(chunk, ((pad, 0), (0, 0)))  # left-pad with zeros
        out[i] = chunk
    
    return out


# ---------------------------------------------------------------------------
# Index builder: one sample = one (sequence_id, hand_entry_idx, T-frame window)
# ---------------------------------------------------------------------------

class STERAIndex:
    """
    Builds a flat index of valid samples across multiple HDF5 files.
    Each sample is defined by:
        - hdf5_path, mp4_path
        - hand_entry_idx  (index into hand-pose/*)
        - frame_window    (list of T frame indices for temporal context)
        - hand_side       (0=left, 1=right)
    """

    def __init__(
        self,
        data_root: str,
        window_size: int = DEFAULT_WINDOW,
        min_conf: float = MIN_YOLO_CONF,
        stride: int = 1,
    ):
        self.data_root   = Path(data_root)
        self.window_size = window_size
        self.min_conf    = min_conf
        self.stride      = stride
        self.samples     = []   # list of dicts
        self._build()

    def _build(self):
        """
        STERA-10M structure:
        data/
        ├── session_data_20260328_234041/
        │   ├── annotation.hdf5  ← look for this
        │   └── rgb.mp4          ← and this
        ├── session_data_20260328_235204/
        │   ├── annotation.hdf5
        │   └── rgb.mp4
        └── ...
        """
        # Find all session directories (contain annotation.hdf5)
        session_dirs = list(self.data_root.glob("session_data_*"))
        
        if not session_dirs:
            raise FileNotFoundError(
                f"No 'session_data_*' directories found under {self.data_root}. "
                f"Expected structure: {self.data_root}/session_data_YYYYMMDD_HHMMSS/annotation.hdf5"
            )

        logger.info(f"Found {len(session_dirs)} session directories. Building index...")

        for session_dir in sorted(session_dirs):
            hdf5_path = session_dir / "annotation.hdf5"
            mp4_path = session_dir / "rgb.mp4"
            
            if not hdf5_path.exists():
                logger.warning(f"No annotation.hdf5 in {session_dir.name}, skipping.")
                continue
            
            if not mp4_path.exists():
                logger.warning(f"No rgb.mp4 in {session_dir.name}, skipping.")
                continue

            try:
                with h5py.File(hdf5_path, "r") as f:
                    n_frames     = f["depth/frames"].shape[0]
                    yolo_conf    = f["hand-pose/yolo_conf"][:]      # (H,)
                    frame_idx    = f["hand-pose/frame_idx"][:]      # (H,) -> maps hand entry to frame
                    hand_side    = f["hand-pose/side"][:]           # (H,)
                    
                    # Filter by confidence
                    valid_mask   = yolo_conf >= self.min_conf       # (H,)
                    valid_entries = np.where(valid_mask)[0]
                    
                    half = self.window_size // 2
                    
                    for entry_idx in valid_entries[::self.stride]:
                        center_frame = int(frame_idx[entry_idx])
                        
                        # Build frame window — clamp at boundaries
                        win_start = max(0, center_frame - half)
                        win_end   = win_start + self.window_size
                        if win_end > n_frames:
                            win_end   = n_frames
                            win_start = max(0, win_end - self.window_size)
                        
                        frame_window = list(range(win_start, win_end))
                        
                        # Pad if at boundary
                        while len(frame_window) < self.window_size:
                            frame_window.insert(0, frame_window[0])
                        
                        self.samples.append({
                            "hdf5_path":    str(hdf5_path),
                            "mp4_path":     str(mp4_path),
                            "hand_entry":   int(entry_idx),
                            "frame_window": frame_window,
                            "center_frame": center_frame,
                            "hand_side":    int(hand_side[entry_idx]),
                        })

            except Exception as e:
                logger.error(f"Failed to index {hdf5_path}: {e}")
                continue

        logger.info(f"Total samples indexed: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.samples, f)
        logger.info(f"Index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "STERAIndex":
        obj = cls.__new__(cls)
        with open(path) as f:
            obj.samples = json.load(f)
        logger.info(f"Loaded index with {len(obj.samples):,} samples from {path}")
        return obj


# ---------------------------------------------------------------------------
# Video reader abstraction (decord / cv2)
# ---------------------------------------------------------------------------

class VideoFrameReader:
    """
    Thin wrapper around decord (preferred) or cv2.
    Always returns RGB uint8 frames.
    """

    def __init__(self, path: str):
        self.path = path
        self._reader = None

    def _open(self):
        if self._reader is not None:
            return
        if DECORD_AVAILABLE:
            self._reader = VideoReader(self.path, ctx=decord_cpu(0))
            self._backend = "decord"
        else:
            self._reader = cv2.VideoCapture(self.path)
            self._backend = "cv2"
            if not self._reader.isOpened():
                raise IOError(f"cv2 failed to open {self.path}")

    def get_frames(self, indices: list[int]) -> np.ndarray:
        """Returns (T, H, W, 3) uint8 RGB"""
        self._open()
        if self._backend == "decord":
            frames = self._reader.get_batch(indices).asnumpy()  # (T, H, W, 3) BGR→RGB already
            return frames
        else:
            frames = []
            for idx in indices:
                self._reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self._reader.read()
                if not ret:
                    # Return black frame on failure
                    frame = np.zeros((144, 256, 3), dtype=np.uint8)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return np.stack(frames)

    def close(self):
        if self._reader is not None and self._backend == "cv2":
            self._reader.release()
        self._reader = None


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class STERADataset(Dataset):
    """
    STERA-10M VLA Dataset.

    Args:
        index:           STERAIndex instance OR path to saved index JSON
        img_size:        spatial size for RGB/depth crops (default 224)
        window_size:     temporal window (frames)
        imu_window:      number of IMU samples per frame
        use_depth:       include depth channel
        use_imu:         include IMU sequences
        use_text:        include text annotation strings
        extract_mode:    'on_the_fly' (decode mp4 at runtime) or
                         'preextracted' (load from frame_dir)
        frame_dir:       path to pre-extracted frames (used when
                         extract_mode='preextracted')
        cache_hdf5:      keep HDF5 file handles open (faster, uses more fd)
        augment:         apply training augmentations
    """

    def __init__(
        self,
        index: "STERAIndex | str",
        img_size:      int  = DEFAULT_IMG_SIZE,
        window_size:   int  = DEFAULT_WINDOW,
        imu_window:    int  = DEFAULT_IMU_WINDOW,
        use_depth:     bool = True,
        use_imu:       bool = True,
        use_text:      bool = True,
        extract_mode:  Literal["on_the_fly", "preextracted"] = "on_the_fly",
        frame_dir:     Optional[str] = None,
        cache_hdf5:    bool = False,
        augment:       bool = False,
    ):
        if isinstance(index, str):
            self.index = STERAIndex.load(index)
        else:
            self.index = index

        self.samples     = self.index.samples
        self.img_size    = img_size
        self.window_size = window_size
        self.imu_window  = imu_window
        self.use_depth   = use_depth
        self.use_imu     = use_imu
        self.use_text    = use_text
        self.extract_mode = extract_mode
        self.frame_dir   = Path(frame_dir) if frame_dir else None
        self.cache_hdf5  = cache_hdf5
        self.augment     = augment

        # HDF5 handle cache (per-worker safe because workers fork after __init__)
        self._hdf5_cache: dict[str, h5py.File] = {}

        # Per-mp4 video reader cache
        self._video_cache: dict[str, VideoFrameReader] = {}

        # Image transforms
        self.rgb_transform  = self._build_rgb_transform(augment)
        self.depth_transform = self._build_depth_transform()

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _build_rgb_transform(self, augment: bool):
        ops = [transforms.Resize((self.img_size, self.img_size))]
        if augment:
            ops += [
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.2, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(ops)

    def _build_depth_transform(self):
        # Depth uint16 → float32, normalize to [0,1] using 10m max range
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),  # will be (1, H, W)
        ])

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    def _get_hdf5(self, path: str) -> h5py.File:
        if self.cache_hdf5:
            if path not in self._hdf5_cache:
                self._hdf5_cache[path] = h5py.File(path, "r", swmr=True)
            return self._hdf5_cache[path]
        else:
            return h5py.File(path, "r", swmr=True)

    def _close_hdf5(self, path: str, f: h5py.File):
        if not self.cache_hdf5:
            f.close()

    # ------------------------------------------------------------------
    # RGB loading
    # ------------------------------------------------------------------

    def _load_rgb_frames(self, mp4_path: str, frame_window: list[int],
                         bbox: np.ndarray) -> torch.Tensor:
        """
        Returns (T, 3, H, W) float32 tensor, cropped to bbox region.
        """
        from PIL import Image

        if self.extract_mode == "preextracted" and self.frame_dir is not None:
            frames = self._load_preextracted(mp4_path, frame_window)
        else:
            frames = self._decode_video(mp4_path, frame_window)

        # Crop to hand bbox (x1,y1,x2,y2) with some padding
        x1, y1, x2, y2 = bbox
        pad = 20
        h_total, w_total = frames.shape[1], frames.shape[2]
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w_total, int(x2) + pad)
        y2 = min(h_total, int(y2) + pad)

        tensors = []
        for frame in frames:
            crop = frame[y1:y2, x1:x2]     # (H', W', 3) uint8
            pil  = Image.fromarray(crop)
            tensors.append(self.rgb_transform(pil))
        
        return torch.stack(tensors)         # (T, 3, H, W)

    def _decode_video(self, mp4_path: str, frame_window: list[int]) -> np.ndarray:
        if mp4_path not in self._video_cache:
            self._video_cache[mp4_path] = VideoFrameReader(mp4_path)
        return self._video_cache[mp4_path].get_frames(frame_window)

    def _load_preextracted(self, mp4_path: str, frame_window: list[int]) -> np.ndarray:
        """Load PNG frames from frame_dir/sequence_name/frame_%06d.png"""
        from PIL import Image
        seq_name = Path(mp4_path).stem
        seq_dir  = self.frame_dir / seq_name
        frames = []
        for idx in frame_window:
            img_path = seq_dir / f"frame_{idx:06d}.png"
            if img_path.exists():
                frames.append(np.array(Image.open(img_path).convert("RGB")))
            else:
                frames.append(np.zeros((144, 256, 3), dtype=np.uint8))
        return np.stack(frames)

    # ------------------------------------------------------------------
    # Depth loading
    # ------------------------------------------------------------------

    def _load_depth_frames(self, f: h5py.File, frame_window: list[int]) -> torch.Tensor:
        """
        Returns (T, 1, H, W) float32, depth normalized to [0,1].
        Assumes max depth ~10m for uint16.
        """
        from PIL import Image
        MAX_DEPTH = 10000.0  # uint16 mm → normalize by 10m

        tensors = []
        for idx in frame_window:
            depth = f["depth/frames"][idx].astype(np.float32) / MAX_DEPTH  # (H, W)
            depth = np.clip(depth, 0, 1)
            pil   = Image.fromarray((depth * 255).astype(np.uint8), mode="L")
            tensors.append(self.depth_transform(pil))   # (1, H, W)
        
        return torch.stack(tensors)     # (T, 1, H, W)

    # ------------------------------------------------------------------
    # IMU loading
    # ------------------------------------------------------------------

    def _load_imu(self, f: h5py.File, center_frame: int,
                  frame_timestamps: np.ndarray) -> torch.Tensor:
        """
        Returns (imu_window, 10) float32 for the center frame.
        """
        center_ts = frame_timestamps[center_frame]
        imu_ts    = f["imu/timestamps"][:]

        idx  = int(np.searchsorted(imu_ts, center_ts))
        half = self.imu_window // 2
        s    = max(0, idx - half)
        e    = s + self.imu_window
        if e > len(imu_ts):
            e = len(imu_ts)
            s = max(0, e - self.imu_window)

        ang_vel = f["imu/angular_velocity"][s:e]        # (<= W, 3)
        lin_acc = f["imu/linear_acceleration"][s:e]     # (<= W, 3)
        orient  = f["imu/orientation_xyzw"][s:e]        # (<= W, 4)

        chunk = np.concatenate([ang_vel, lin_acc, orient], axis=-1)  # (<= W, 10)
        
        # Left-pad with zeros if at boundary
        pad = self.imu_window - len(chunk)
        if pad > 0:
            chunk = np.pad(chunk, ((pad, 0), (0, 0)))

        return torch.from_numpy(chunk.astype(np.float32))  # (imu_window, 10)

    # ------------------------------------------------------------------
    # Text loading
    # ------------------------------------------------------------------

    def _load_text(self, f: h5py.File, center_frame: int,
                   frame_timestamps: np.ndarray) -> str:
        center_ts  = frame_timestamps[center_frame]
        start_times = f["text-annotations/start_time"][:]
        end_times   = f["text-annotations/end_time"][:]
        texts        = f["text-annotations/text"][:]

        for s, e, t in zip(start_times, end_times, texts):
            if s <= center_ts <= e:
                return str(t)
        return ""

    # ------------------------------------------------------------------
    # MANO targets
    # ------------------------------------------------------------------

    def _load_mano_targets(self, f: h5py.File, hand_entry: int) -> dict:
        """
        Returns dict of tensors:
            joints_cam_rel: (21, 3)   — primary regression target
            hand_pose:      (15, 3, 3) — finger rotations (SO3)
            global_orient:  (3, 3)     — wrist rotation (SO3)
            betas:          (10,)      — shape params
            translation:    (3,)
            bbox:           (4,)
            side:           int (0=left, 1=right)
        """
        return {
            "joints_cam_rel": torch.from_numpy(
                f["hand-pose/joints_cam_rel"][hand_entry].astype(np.float32)
            ),  # (21, 3)
            "hand_pose": torch.from_numpy(
                f["hand-pose/hand_pose"][hand_entry].astype(np.float32)
            ),  # (15, 3, 3)
            "global_orient": torch.from_numpy(
                f["hand-pose/global_orient"][hand_entry].astype(np.float32)
            ),  # (3, 3)
            "betas": torch.from_numpy(
                f["hand-pose/betas"][hand_entry].astype(np.float32)
            ),  # (10,)
            "translation": torch.from_numpy(
                f["hand-pose/pred_cam_t"][hand_entry].astype(np.float32)
            ),  # (3,)
            "bbox": torch.from_numpy(
                f["hand-pose/bbox"][hand_entry].astype(np.float32)
            ),  # (4,)
            "side": int(f["hand-pose/side"][hand_entry]),
        }

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample       = self.samples[idx]
        hdf5_path    = sample["hdf5_path"]
        mp4_path     = sample["mp4_path"]
        hand_entry   = sample["hand_entry"]
        frame_window = sample["frame_window"]
        center_frame = sample["center_frame"]

        f = self._get_hdf5(hdf5_path)
        try:
            frame_timestamps = f["hand-pose/frame_timestamps"][:]

            # MANO targets
            targets = self._load_mano_targets(f, hand_entry)
            bbox    = f["hand-pose/bbox"][hand_entry]  # numpy for cropping

            # RGB frames (T, 3, H, W)
            rgb = self._load_rgb_frames(mp4_path, frame_window, bbox)

            out = {
                "rgb":          rgb,            # (T, 3, H, W)
                "targets":      targets,
                "hand_side":    targets["side"],
                "sample_meta":  {
                    "hdf5":         hdf5_path,
                    "hand_entry":   hand_entry,
                    "center_frame": center_frame,
                },
            }

            if self.use_depth:
                out["depth"] = self._load_depth_frames(f, frame_window)  # (T,1,H,W)

            if self.use_imu:
                out["imu"] = self._load_imu(f, center_frame, frame_timestamps)  # (W,10)

            if self.use_text:
                out["text"] = self._load_text(f, center_frame, frame_timestamps)

        finally:
            self._close_hdf5(hdf5_path, f)

        return out

    def __del__(self):
        for f in self._hdf5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        for v in self._video_cache.values():
            try:
                v.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Collate function (handles variable-length text strings)
# ---------------------------------------------------------------------------

def stera_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate that handles:
      - text strings (kept as list, not stacked)
      - nested target dicts
      - sample_meta dicts
    """
    out = {}

    # RGB, depth, imu — standard stack
    for key in ("rgb", "depth", "imu"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])

    # hand_side
    out["hand_side"] = torch.tensor([b["hand_side"] for b in batch])

    # Text — keep as list of strings (tokenize in model forward or collate hook)
    if "text" in batch[0]:
        out["text"] = [b["text"] for b in batch]

    # Targets — stack each sub-key
    target_keys = batch[0]["targets"].keys()
    out["targets"] = {}
    for k in target_keys:
        if k == "side":
            out["targets"][k] = torch.tensor([b["targets"][k] for b in batch])
        else:
            out["targets"][k] = torch.stack([b["targets"][k] for b in batch])

    # Meta — keep as list
    out["sample_meta"] = [b["sample_meta"] for b in batch]

    return out


# ---------------------------------------------------------------------------
# Pre-extraction utility (optional, call once before training)
# ---------------------------------------------------------------------------

def preextract_frames(
    data_root: str,
    out_dir: str,
    num_workers: int = 8,
):
    """
    Pre-extract all video frames to PNG files.
    Call once before training if you want extract_mode='preextracted'.
    
    Saves to: out_dir/<sequence_stem>/frame_XXXXXX.png
    """
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    data_root = Path(data_root)
    out_dir   = Path(out_dir)
    mp4_files = sorted(data_root.rglob("*.mp4"))

    def _extract_one(mp4_path: Path):
        seq_dir = out_dir / mp4_path.stem
        seq_dir.mkdir(parents=True, exist_ok=True)

        if DECORD_AVAILABLE:
            vr = VideoReader(str(mp4_path), ctx=decord_cpu(0))
            for i in range(len(vr)):
                out_path = seq_dir / f"frame_{i:06d}.png"
                if out_path.exists():
                    continue
                frame = vr[i].asnumpy()
                Image.fromarray(frame).save(out_path)
        else:
            cap = cv2.VideoCapture(str(mp4_path))
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_path = seq_dir / f"frame_{i:06d}.png"
                if not out_path.exists():
                    cv2.imwrite(str(out_path), frame)
                i += 1
            cap.release()
        
        logger.info(f"Extracted {mp4_path.stem}")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        ex.map(_extract_one, mp4_files)


# ---------------------------------------------------------------------------
# Factory: build train/val dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root:      str,
    val_split:      float = 0.05,
    batch_size:     int   = 32,
    num_workers:    int   = 8,
    window_size:    int   = DEFAULT_WINDOW,
    img_size:       int   = DEFAULT_IMG_SIZE,
    imu_window:     int   = DEFAULT_IMU_WINDOW,
    use_depth:      bool  = True,
    use_imu:        bool  = True,
    use_text:       bool  = True,
    extract_mode:   str   = "on_the_fly",
    frame_dir:      Optional[str] = None,
    index_cache:    Optional[str] = None,   # path to save/load index JSON
    pin_memory:     bool  = True,
    min_conf:       float = MIN_YOLO_CONF,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).

    index_cache: if provided, saves the built index to disk and reloads
                 on subsequent calls (saves ~minutes on 10M dataset).
    """

    # Build or load index
    if index_cache and Path(index_cache).exists():
        idx = STERAIndex.load(index_cache)
    else:
        idx = STERAIndex(
            data_root=data_root,
            window_size=window_size,
            min_conf=min_conf,
        )
        if index_cache:
            idx.save(index_cache)

    # Split
    n_total = len(idx)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    # Reproducible split
    rng     = np.random.default_rng(42)
    perm    = rng.permutation(n_total)
    train_samples = [idx.samples[i] for i in perm[:n_train]]
    val_samples   = [idx.samples[i] for i in perm[n_train:]]

    def _make_index(samples):
        obj = STERAIndex.__new__(STERAIndex)
        obj.samples = samples
        return obj

    dataset_kwargs = dict(
        img_size=img_size,
        window_size=window_size,
        imu_window=imu_window,
        use_depth=use_depth,
        use_imu=use_imu,
        use_text=use_text,
        extract_mode=extract_mode,
        frame_dir=frame_dir,
        cache_hdf5=False,       # keep False for multiprocessing safety
    )

    train_ds = STERADataset(_make_index(train_samples), augment=True,  **dataset_kwargs)
    val_ds   = STERADataset(_make_index(val_samples),   augment=False, **dataset_kwargs)

    loader_kwargs = dict(
        collate_fn=stera_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    logger.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick sanity check — run: python stera_dataloader.py /path/to/data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    print("=" * 60)
    print("STERA DataLoader Sanity Check")
    print("=" * 60)

    train_loader, val_loader = build_dataloaders(
        data_root=data_root,
        batch_size=4,
        num_workers=0,          # 0 for debugging
        window_size=8,
        use_depth=True,
        use_imu=True,
        use_text=True,
        extract_mode="on_the_fly",
        index_cache="./stera_index_cache.json",
    )

    print(f"\nTrain batches: {len(train_loader):,}")
    print(f"Val batches:   {len(val_loader):,}")

    t0 = time.time()
    batch = next(iter(train_loader))
    t1 = time.time()

    print(f"\nFirst batch loaded in {t1-t0:.2f}s")
    print(f"  rgb:          {batch['rgb'].shape}   dtype={batch['rgb'].dtype}")
    if "depth" in batch:
        print(f"  depth:        {batch['depth'].shape}  dtype={batch['depth'].dtype}")
    if "imu" in batch:
        print(f"  imu:          {batch['imu'].shape}    dtype={batch['imu'].dtype}")
    if "text" in batch:
        print(f"  text[0]:      '{batch['text'][0]}'")
    print(f"  hand_side:    {batch['hand_side']}")
    print(f"\nTargets:")
    for k, v in batch["targets"].items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: {v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k:20s}: {v}")