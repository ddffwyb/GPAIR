"""
Image Reconstruction Module - Triton-based Transpose Operator Implementation

This module provides a high-performance photoacoustic image reconstructor
for reconstructing voxel images from detector signals.
Uses the transpose operator A^T to implement backprojection reconstruction.

Usage:
    from image_reconstructor import ImageReconstructor

    # Create a reconstructor instance
    reconstructor = ImageReconstructor(
        x_range=[-12.80e-3, 12.80e-3],
        y_range=[-12.80e-3, 12.80e-3],
        z_range=[-6.40e-3, 6.40e-3],
        res=0.10e-3,
        vs=1510.0,
        fs=8.333333e6,
        detector_locations=detector_locs,  # [num_detectors, 3] numpy/torch array
        num_times=512,  # Number of time samples per detector
    )

    # Reconstruct image from detector signals
    # signal_data: [num_detectors, num_times] signal data
    reconstructed_image = reconstructor.reconstruct(signal_data)
    # Returns: [num_x, num_y, num_z] reconstructed image
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
import numpy as np


# ==================== Constants ====================
PI = 3.1415926536
GAUSSIAN_PEAK_INTENSITY = 0.4000222589 / 1000000.0  # 1/(sqrt(2*pi)*0.9973)/1000000
MIN_HALF_KERNEL_SIZE = 12  # Ensures at least 4 time samples per sigma in the Gaussian kernel

# Default parameters optimized for RTX 4090
DEFAULT_BLOCK_VOXEL = 512
DEFAULT_NUM_STAGES = 4
DEFAULT_NUM_WARPS = 8


# ==================== Autotune Kernel ====================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_VOXEL": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_VOXEL": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_VOXEL": 512}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_VOXEL": 1024}, num_stages=4, num_warps=8),
    ],
    key=["num_voxels", "num_detectors", "num_times_upsampled"],
)
@triton.jit
def _transpose_kernel_autotune(
    dx_conv_ptr,  # [num_detectors, num_times_upsampled]
    detector_x_ptr,  # Detector X coordinate pointer [num_detectors] - SoA
    detector_y_ptr,  # Detector Y coordinate pointer [num_detectors] - SoA
    detector_z_ptr,  # Detector Z coordinate pointer [num_detectors] - SoA
    output_ptr,  # [num_voxels], direct output
    num_voxels: tl.constexpr,  # Total number of voxels
    num_detectors: tl.constexpr,  # Total number of detectors
    num_times_upsampled: tl.constexpr,  # Number of upsampled time points
    dd_inv_upsampled: tl.constexpr,  # Inverse of upsampled single time step distance
    stride_dx_d: tl.constexpr,  # Stride of dx_conv
    x_start: tl.constexpr,  # Voxel grid start x coordinate
    y_start: tl.constexpr,  # Voxel grid start y coordinate
    z_start: tl.constexpr,  # Voxel grid start z coordinate
    res: tl.constexpr,  # Voxel grid resolution
    num_y: tl.constexpr,  # Voxel grid y dimension size
    num_z: tl.constexpr,  # Voxel grid z dimension size
    BLOCK_VOXEL: tl.constexpr,  # Triton block size for voxels
):
    """Triton implementation of the transpose operator A^T*x (Autotune version)"""
    voxel_block_idx = tl.program_id(0)

    v_start = voxel_block_idx * BLOCK_VOXEL
    v_offsets = v_start + tl.arange(0, BLOCK_VOXEL)
    v_mask = v_offsets < num_voxels

    num_yz = num_y * num_z

    v = v_offsets.to(tl.int32)
    z_idx = v % num_z
    y_idx = (v // num_z) % num_y
    x_idx = v // num_yz

    vox_loc_x = x_start + x_idx * res
    vox_loc_y = y_start + y_idx * res
    vox_loc_z = z_start + z_idx * res

    accum = tl.zeros((BLOCK_VOXEL,), dtype=tl.float32)

    for det_idx in tl.range(0, num_detectors):
        det_loc_x = tl.load(detector_x_ptr + det_idx)
        det_loc_y = tl.load(detector_y_ptr + det_idx)
        det_loc_z = tl.load(detector_z_ptr + det_idx)

        dx = det_loc_x - vox_loc_x
        dy = det_loc_y - vox_loc_y
        dz = det_loc_z - vox_loc_z

        dist_sq = dx * dx + dy * dy + dz * dz
        dist_inv = tl.rsqrt(dist_sq)

        time_center = (dist_sq * dist_inv * dd_inv_upsampled + 0.5).to(tl.int32)
        t_mask = (time_center >= 0) & (time_center < num_times_upsampled)

        dx_row_ptr = dx_conv_ptr + det_idx * stride_dx_d + time_center
        mask = t_mask & v_mask
        dx_val = tl.load(dx_row_ptr, mask=mask, other=0.0).to(tl.float32)

        val_to_add = dx_val * dist_inv
        accum += tl.where(mask, val_to_add, 0.0)

    tl.store(output_ptr + v_offsets, accum, mask=v_mask)


# ==================== Fixed-parameter Kernel (RTX 4090 Optimized) ====================
@triton.jit
def _transpose_kernel_fixed(
    dx_conv_ptr,  # [num_detectors, num_times_upsampled]
    detector_x_ptr,  # Detector X coordinate pointer [num_detectors] - SoA
    detector_y_ptr,  # Detector Y coordinate pointer [num_detectors] - SoA
    detector_z_ptr,  # Detector Z coordinate pointer [num_detectors] - SoA
    output_ptr,  # [num_voxels], direct output
    num_voxels: tl.constexpr,  # Total number of voxels
    num_detectors: tl.constexpr,  # Total number of detectors
    num_times_upsampled: tl.constexpr,  # Number of upsampled time points
    dd_inv_upsampled: tl.constexpr,  # Inverse of upsampled single time step distance
    stride_dx_d: tl.constexpr,  # Stride of dx_conv
    x_start: tl.constexpr,  # Voxel grid start x coordinate
    y_start: tl.constexpr,  # Voxel grid start y coordinate
    z_start: tl.constexpr,  # Voxel grid start z coordinate
    res: tl.constexpr,  # Voxel grid resolution
    num_y: tl.constexpr,  # Voxel grid y dimension size
    num_z: tl.constexpr,  # Voxel grid z dimension size
    BLOCK_VOXEL: tl.constexpr,  # Triton block size for voxels
):
    """Triton implementation of the transpose operator A^T*x (Fixed-parameter version, optimized for RTX 4090)"""
    voxel_block_idx = tl.program_id(0)

    v_start = voxel_block_idx * BLOCK_VOXEL
    v_offsets = v_start + tl.arange(0, BLOCK_VOXEL)
    v_mask = v_offsets < num_voxels

    num_yz = num_y * num_z

    v = v_offsets.to(tl.int32)
    z_idx = v % num_z
    y_idx = (v // num_z) % num_y
    x_idx = v // num_yz

    vox_loc_x = x_start + x_idx * res
    vox_loc_y = y_start + y_idx * res
    vox_loc_z = z_start + z_idx * res

    accum = tl.zeros((BLOCK_VOXEL,), dtype=tl.float32)

    for det_idx in tl.range(0, num_detectors):
        det_loc_x = tl.load(detector_x_ptr + det_idx)
        det_loc_y = tl.load(detector_y_ptr + det_idx)
        det_loc_z = tl.load(detector_z_ptr + det_idx)

        dx = det_loc_x - vox_loc_x
        dy = det_loc_y - vox_loc_y
        dz = det_loc_z - vox_loc_z

        dist_sq = dx * dx + dy * dy + dz * dz
        dist_inv = tl.rsqrt(dist_sq)

        time_center = (dist_sq * dist_inv * dd_inv_upsampled + 0.5).to(tl.int32)
        t_mask = (time_center >= 0) & (time_center < num_times_upsampled)

        dx_row_ptr = dx_conv_ptr + det_idx * stride_dx_d + time_center
        mask = t_mask & v_mask
        dx_val = tl.load(dx_row_ptr, mask=mask, other=0.0).to(tl.float32)

        val_to_add = dx_val * dist_inv
        accum += tl.where(mask, val_to_add, 0.0)

    tl.store(output_ptr + v_offsets, accum, mask=v_mask)


def _generate_adaptive_conv_kernel(
    res: float, vs: float, fs: float, device: torch.device
) -> Tuple[torch.Tensor, int, int]:
    """
    Generate an adaptive convolution kernel.

    Args:
        res: Spatial resolution (m)
        vs: Speed of sound (m/s)
        fs: Sampling frequency (Hz)
        device: Computation device

    Returns:
        conv_kernel: Convolution kernel tensor [1, 1, kernel_size]
        time_interval_length_half: Half window length
        adaptive_ratio: Adaptive upsampling ratio
    """
    time_interval_length_half = int(3.0 * res / vs * fs + 1)
    adaptive_ratio = int(MIN_HALF_KERNEL_SIZE / time_interval_length_half + 1)
    time_interval_length_half = time_interval_length_half * adaptive_ratio
    fs_upsampled = fs * adaptive_ratio
    const_factor = 2.0 * PI * GAUSSIAN_PEAK_INTENSITY * vs / res
    conv_input = (
        vs
        / fs_upsampled
        * torch.arange(
            time_interval_length_half,
            -time_interval_length_half - 1,
            -1,
            device=device,
            dtype=torch.float32,
        )
    )
    conv_kernel = (
        const_factor * conv_input * torch.exp(-(conv_input**2) / (2.0 * res**2))
    )
    # Add two dimensions for conv1d
    conv_kernel = conv_kernel.unsqueeze(0).unsqueeze(0)
    return conv_kernel, time_interval_length_half, adaptive_ratio


class ImageReconstructor:
    """
    Photoacoustic Image Reconstructor

    Reconstructs voxel images from detector signals based on the spherical
    wave propagation model. Uses the transpose operator A^T for backprojection
    reconstruction. Achieves high-performance reconstruction through
    Triton-accelerated GPU computation.

    Attributes:
        device: Computation device (cuda/cpu)
        num_voxels: Total number of voxels
        num_detectors: Number of detectors
        num_times: Number of time samples
    """

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        res: float,
        vs: float,
        fs: float,
        detector_locations: Union[np.ndarray, torch.Tensor],
        num_times: int,
        use_autotune: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the image reconstructor.

        Args:
            x_range: X-axis range (min, max), in meters
            y_range: Y-axis range (min, max), in meters
            z_range: Z-axis range (min, max), in meters
            res: Spatial resolution, in meters
            vs: Speed of sound, in m/s
            fs: Sampling frequency, in Hz
            detector_locations: Detector positions [num_detectors, 3], in meters
            num_times: Number of time samples per detector
            use_autotune: Whether to enable autotune (default False, uses fixed parameters optimized for RTX 4090)
            device: Computation device, defaults to CUDA (if available)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Store parameters
        self.res = res
        self.vs = vs
        self.fs = fs
        self.num_times = num_times
        self.use_autotune = use_autotune

        # Compute grid dimensions
        self.x_start, self.x_end = x_range
        self.y_start, self.y_end = y_range
        self.z_start, self.z_end = z_range
        self.num_x = int(round((self.x_end - self.x_start) / res))
        self.num_y = int(round((self.y_end - self.y_start) / res))
        self.num_z = int(round((self.z_end - self.z_start) / res))
        self.num_voxels = self.num_x * self.num_y * self.num_z

        # Process detector locations
        if isinstance(detector_locations, np.ndarray):
            detector_locations = torch.from_numpy(detector_locations)
        detector_locations = detector_locations.to(self.device).float().contiguous()

        self.num_detectors = detector_locations.shape[0]
        self.detector_x = detector_locations[:, 0].contiguous()
        self.detector_y = detector_locations[:, 1].contiguous()
        self.detector_z = detector_locations[:, 2].contiguous()

        # Generate convolution kernel and upsampling parameters
        (self.conv_kernel, self.time_interval_length_half, self.adaptive_ratio) = (
            _generate_adaptive_conv_kernel(res, vs, fs, self.device)
        )

        self.num_times_upsampled = num_times * self.adaptive_ratio
        self.dd_inv_upsampled = fs * self.adaptive_ratio / vs
        self.stride_dx_d = self.num_times_upsampled

        print(f"ImageReconstructor initialized:")
        print(
            f"  - Voxel grid: {self.num_x} x {self.num_y} x {self.num_z} = {self.num_voxels} voxels"
        )
        print(f"  - Number of detectors: {self.num_detectors}")
        print(f"  - Time samples: {self.num_times} (upsampled: {self.num_times_upsampled})")
        print(f"  - Autotune: {'enabled' if use_autotune else 'disabled (RTX 4090 optimized parameters)'}")
        print(f"  - Device: {self.device}")

    def reconstruct(
        self,
        signal_data: Union[np.ndarray, torch.Tensor],
        return_3d: bool = True,
    ) -> torch.Tensor:
        """
        Perform image reconstruction.

        Reconstructs a voxel image from detector signals.

        Args:
            signal_data: Detector signal data, can be:
                - [num_detectors, num_times] 2D array
                - [num_detectors * num_times] 1D array
            return_3d: Whether to return a 3D image [num_x, num_y, num_z].
                       If False, returns 1D flattened data [num_voxels].

        Returns:
            Reconstructed image tensor
        """
        # Convert input data
        if isinstance(signal_data, np.ndarray):
            signal_data = torch.from_numpy(signal_data)
        x = signal_data.to(self.device).float().flatten().contiguous()

        # Check input size
        expected_size = self.num_detectors * self.num_times
        if x.shape[0] != expected_size:
            raise ValueError(
                f"Input signal size ({x.shape[0]}) does not match configuration ({expected_size})"
            )

        # Apply the transpose operator
        image = self._transpose_operator(x)

        if return_3d:
            return image.reshape(self.num_x, self.num_y, self.num_z)
        return image

    def _transpose_operator(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the transpose operator A^T*x.

        Args:
            x: Input signal data [num_detectors * num_times]

        Returns:
            Output voxel data [num_voxels]
        """
        # Convolution: reshape then upsample
        x_reshaped = x.reshape(self.num_detectors, self.num_times)
        dx_upsampling = torch.zeros(
            (self.num_detectors, self.num_times_upsampled),
            device=self.device,
            dtype=torch.float32,
        )
        dx_upsampling[:, :: self.adaptive_ratio] = x_reshaped

        # Convolution
        dx_conv_batch = (
            torch.nn.functional.conv1d(
                dx_upsampling.unsqueeze(1),
                self.conv_kernel,
                padding=self.time_interval_length_half,
            )
            .squeeze(1)
            .contiguous()
        )

        # Allocate output buffer
        y = torch.empty(self.num_voxels, device=self.device, dtype=torch.float32)

        if self.use_autotune:
            # Use autotune version
            grid = lambda META: (triton.cdiv(self.num_voxels, META["BLOCK_VOXEL"]),)
            _transpose_kernel_autotune[grid](
                dx_conv_batch,
                self.detector_x,
                self.detector_y,
                self.detector_z,
                y,
                self.num_voxels,
                self.num_detectors,
                self.num_times_upsampled,
                self.dd_inv_upsampled,
                self.stride_dx_d,
                self.x_start,
                self.y_start,
                self.z_start,
                self.res,
                self.num_y,
                self.num_z,
            )
        else:
            # Use fixed-parameter version (RTX 4090 optimized)
            grid = (triton.cdiv(self.num_voxels, DEFAULT_BLOCK_VOXEL),)
            _transpose_kernel_fixed[grid](
                dx_conv_batch,
                self.detector_x,
                self.detector_y,
                self.detector_z,
                y,
                self.num_voxels,
                self.num_detectors,
                self.num_times_upsampled,
                self.dd_inv_upsampled,
                self.stride_dx_d,
                self.x_start,
                self.y_start,
                self.z_start,
                self.res,
                self.num_y,
                self.num_z,
                BLOCK_VOXEL=DEFAULT_BLOCK_VOXEL,
                num_stages=DEFAULT_NUM_STAGES,
                num_warps=DEFAULT_NUM_WARPS,
            )

        return y

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """
        Get the grid shape.

        Returns:
            (num_x, num_y, num_z): Number of voxels in each dimension
        """
        return self.num_x, self.num_y, self.num_z


# ==================== Convenience Functions ====================
def reconstruct_image(
    signal_data: Union[np.ndarray, torch.Tensor],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    res: float,
    vs: float,
    fs: float,
    detector_locations: Union[np.ndarray, torch.Tensor],
    num_times: int,
    use_autotune: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    One-shot image reconstruction (convenience function).

    For a single reconstruction, this function can be used directly. If multiple
    reconstructions with the same geometric configuration are needed, it is
    recommended to create an ImageReconstructor instance to avoid repeated
    initialization overhead.

    Args:
        signal_data: Signal data [num_detectors, num_times] or [num_detectors * num_times]
        x_range: X-axis range (min, max)
        y_range: Y-axis range (min, max)
        z_range: Z-axis range (min, max)
        res: Spatial resolution
        vs: Speed of sound
        fs: Sampling frequency
        detector_locations: Detector positions [num_detectors, 3]
        num_times: Number of time samples
        use_autotune: Whether to enable autotune (default False)
        device: Computation device

    Returns:
        Reconstructed image [num_x, num_y, num_z]
    """
    reconstructor = ImageReconstructor(
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        res=res,
        vs=vs,
        fs=fs,
        detector_locations=detector_locations,
        num_times=num_times,
        use_autotune=use_autotune,
        device=device,
    )
    return reconstructor.reconstruct(signal_data)


# ==================== Test Code ====================
if __name__ == "__main__":
    import scipy.io as sio

    print("=" * 60)
    print("Image Reconstructor Test")
    print("=" * 60)

    # Configuration parameters
    x_range = [-12.80e-3, 12.80e-3]
    y_range = [-12.80e-3, 12.80e-3]
    z_range = [-6.40e-3, 6.40e-3]
    res = 0.10e-3
    vs = 1510.0
    fs = 8.333333e6

    # Load data (example)
    try:
        detector_locations = sio.loadmat("data/sensor_Liver_location.mat")[
            "detector_locations"
        ]

        # Load real signal data
        signal_data = sio.loadmat("data/sensor_Liver_data_matrix.mat")[
            "simulation_data"
        ]
        num_times = signal_data.shape[1]

        # Create reconstructor (autotune disabled by default)
        reconstructor = ImageReconstructor(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            res=res,
            vs=vs,
            fs=fs,
            detector_locations=detector_locations,
            num_times=num_times,
            use_autotune=False,  # Use fixed parameters by default
        )

        # Perform reconstruction
        print("\nPerforming image reconstruction...")
        reconstructed_image = reconstructor.reconstruct(signal_data)
        print(f"Output image shape: {reconstructed_image.shape}")
        print(
            f"Image range: [{reconstructed_image.min().item():.6e}, {reconstructed_image.max().item():.6e}]"
        )

        print("\nTest completed!")

    except FileNotFoundError:
        print("Test data files not found, skipping test.")
        print("Please provide correct detector locations and signal data for usage.")
