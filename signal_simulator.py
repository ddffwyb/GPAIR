"""
Signal Simulation Module - Triton-based Forward Operator Implementation

This module provides a high-performance photoacoustic signal simulator
for generating detector signals from voxel data.
Supports custom geometric configurations, detector layouts, and sampling parameters.

Usage:
    from signal_simulator import SignalSimulator

    # Create a simulator instance
    simulator = SignalSimulator(
        x_range=[-12.80e-3, 12.80e-3],
        y_range=[-12.80e-3, 12.80e-3],
        z_range=[-6.40e-3, 6.40e-3],
        res=0.10e-3,
        vs=1510.0,
        fs=8.333333e6,
        detector_locations=detector_locs,  # [num_detectors, 3] numpy/torch array
        num_times=512,  # Number of time samples per detector
    )

    # Generate signals from voxel data
    # voxel_data: [num_x, num_y, num_z] or [num_voxels] voxel data
    simulated_signal = simulator.simulate(voxel_data)
    # Returns: [num_detectors, num_times] simulated signal
"""

import math
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
import numpy as np


# ==================== Constants ====================
PI = 3.1415926536
GAUSSIAN_PEAK_INTENSITY = 0.4000222589 / 1000000.0  # 1/(sqrt(2*pi)*0.9973)/1000000
MIN_HALF_KERNEL_SIZE = 12  # Ensures at least 4 time samples per sigma in the Gaussian kernel
MIN_NUM_BLOCK = 8192  # Minimum number of thread blocks
MAX_SPLIT_K = 1024  # Maximum Split-K partitions

# Default parameters optimized for RTX 4090
DEFAULT_BLOCK_VOXEL = 512
DEFAULT_BLOCK_DET = 8
DEFAULT_NUM_STAGES = 4
DEFAULT_NUM_WARPS = 8


# ==================== Autotune Kernel ====================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_VOXEL": 256, "BLOCK_DET": 4}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_VOXEL": 256, "BLOCK_DET": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_VOXEL": 512, "BLOCK_DET": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_VOXEL": 512, "BLOCK_DET": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_VOXEL": 1024, "BLOCK_DET": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_VOXEL": 1024, "BLOCK_DET": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_VOXEL": 2048, "BLOCK_DET": 4}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_VOXEL": 256, "BLOCK_DET": 16}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_VOXEL": 512, "BLOCK_DET": 16}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_VOXEL": 256, "BLOCK_DET": 32}, num_stages=3, num_warps=8),
    ],
    key=["num_voxels", "num_detectors", "num_times_upsampled", "split_k"],
)
@triton.jit
def _forward_kernel_splitk_autotune(
    x_ptr,  # Input voxel data pointer [num_voxels]
    detector_x_ptr,  # Detector X coordinate pointer [num_detectors] - SoA
    detector_y_ptr,  # Detector Y coordinate pointer [num_detectors] - SoA
    detector_z_ptr,  # Detector Z coordinate pointer [num_detectors] - SoA
    partial_output_ptr,  # [split_k, num_detectors, num_times_upsampled] - independent output per split
    num_voxels: tl.constexpr,  # Total number of voxels
    num_detectors: tl.constexpr,  # Total number of detectors
    num_times_upsampled: tl.constexpr,  # Number of upsampled time points
    dd_inv_upsampled: tl.constexpr,  # Inverse of upsampled single time step distance
    stride_partial_k: tl.constexpr,  # Split-K dimension stride of partial_output
    stride_partial_d: tl.constexpr,  # Detector dimension stride of partial_output
    split_k: tl.constexpr,  # Number of Split-K partitions
    x_start: tl.constexpr,  # Voxel grid start x coordinate
    y_start: tl.constexpr,  # Voxel grid start y coordinate
    z_start: tl.constexpr,  # Voxel grid start z coordinate
    res: tl.constexpr,  # Voxel grid resolution
    num_y: tl.constexpr,  # Voxel grid y dimension size
    num_z: tl.constexpr,  # Voxel grid z dimension size
    BLOCK_VOXEL: tl.constexpr,  # Triton block size for voxels
    BLOCK_DET: tl.constexpr,  # Number of detectors processed per program
):
    """Split-K Triton implementation of the forward operator A*x (Autotune version)"""
    det_block_idx = tl.program_id(0)
    split_k_idx = tl.program_id(1)

    det_start = det_block_idx * BLOCK_DET
    out_start = partial_output_ptr + split_k_idx * stride_partial_k

    voxels_per_split = tl.cdiv(num_voxels, split_k)
    v_start_base = split_k_idx * voxels_per_split
    v_end = tl.minimum(v_start_base + voxels_per_split, num_voxels)

    num_yz = num_y * num_z

    for v_start in tl.range(v_start_base, v_end, BLOCK_VOXEL):
        v_offsets = v_start + tl.arange(0, BLOCK_VOXEL)
        v_mask = v_offsets < v_end

        x_val = tl.load(x_ptr + v_offsets, mask=v_mask, other=0.0)

        v = v_offsets.to(tl.int32)
        z_idx = v % num_z
        y_idx = (v // num_z) % num_y
        x_idx = v // num_yz

        vox_loc_x = x_start + x_idx * res
        vox_loc_y = y_start + y_idx * res
        vox_loc_z = z_start + z_idx * res

        for d in tl.static_range(0, BLOCK_DET):
            det_id = det_start + d
            d_mask = det_id < num_detectors

            det_loc_x = tl.load(detector_x_ptr + det_id, mask=d_mask, other=0.0)
            det_loc_y = tl.load(detector_y_ptr + det_id, mask=d_mask, other=0.0)
            det_loc_z = tl.load(detector_z_ptr + det_id, mask=d_mask, other=0.0)

            dx = det_loc_x - vox_loc_x
            dy = det_loc_y - vox_loc_y
            dz = det_loc_z - vox_loc_z

            dist_sq = dx * dx + dy * dy + dz * dz
            dist_inv = tl.rsqrt(dist_sq)

            time_center = (dist_sq * dist_inv * dd_inv_upsampled + 0.5).to(tl.int32)
            t_mask = (time_center >= 0) & (time_center < num_times_upsampled)

            out_ptr = out_start + det_id * stride_partial_d + time_center
            val_to_add = x_val * dist_inv
            mask = t_mask & v_mask & d_mask
            tl.atomic_add(out_ptr, val_to_add, mask=mask, sem="relaxed")


# ==================== Fixed-parameter Kernel (RTX 4090 Optimized) ====================
@triton.jit
def _forward_kernel_splitk_fixed(
    x_ptr,  # Input voxel data pointer [num_voxels]
    detector_x_ptr,  # Detector X coordinate pointer [num_detectors] - SoA
    detector_y_ptr,  # Detector Y coordinate pointer [num_detectors] - SoA
    detector_z_ptr,  # Detector Z coordinate pointer [num_detectors] - SoA
    partial_output_ptr,  # [split_k, num_detectors, num_times_upsampled] - independent output per split
    num_voxels: tl.constexpr,  # Total number of voxels
    num_detectors: tl.constexpr,  # Total number of detectors
    num_times_upsampled: tl.constexpr,  # Number of upsampled time points
    dd_inv_upsampled: tl.constexpr,  # Inverse of upsampled single time step distance
    stride_partial_k: tl.constexpr,  # Split-K dimension stride of partial_output
    stride_partial_d: tl.constexpr,  # Detector dimension stride of partial_output
    split_k: tl.constexpr,  # Number of Split-K partitions
    x_start: tl.constexpr,  # Voxel grid start x coordinate
    y_start: tl.constexpr,  # Voxel grid start y coordinate
    z_start: tl.constexpr,  # Voxel grid start z coordinate
    res: tl.constexpr,  # Voxel grid resolution
    num_y: tl.constexpr,  # Voxel grid y dimension size
    num_z: tl.constexpr,  # Voxel grid z dimension size
    BLOCK_VOXEL: tl.constexpr,  # Triton block size for voxels
    BLOCK_DET: tl.constexpr,  # Number of detectors processed per program
):
    """Split-K Triton implementation of the forward operator A*x (Fixed-parameter version, optimized for RTX 4090)"""
    det_block_idx = tl.program_id(0)
    split_k_idx = tl.program_id(1)

    det_start = det_block_idx * BLOCK_DET
    out_start = partial_output_ptr + split_k_idx * stride_partial_k

    voxels_per_split = tl.cdiv(num_voxels, split_k)
    v_start_base = split_k_idx * voxels_per_split
    v_end = tl.minimum(v_start_base + voxels_per_split, num_voxels)

    num_yz = num_y * num_z

    for v_start in tl.range(v_start_base, v_end, BLOCK_VOXEL):
        v_offsets = v_start + tl.arange(0, BLOCK_VOXEL)
        v_mask = v_offsets < v_end

        x_val = tl.load(x_ptr + v_offsets, mask=v_mask, other=0.0)

        v = v_offsets.to(tl.int32)
        z_idx = v % num_z
        y_idx = (v // num_z) % num_y
        x_idx = v // num_yz

        vox_loc_x = x_start + x_idx * res
        vox_loc_y = y_start + y_idx * res
        vox_loc_z = z_start + z_idx * res

        for d in tl.static_range(0, BLOCK_DET):
            det_id = det_start + d
            d_mask = det_id < num_detectors

            det_loc_x = tl.load(detector_x_ptr + det_id, mask=d_mask, other=0.0)
            det_loc_y = tl.load(detector_y_ptr + det_id, mask=d_mask, other=0.0)
            det_loc_z = tl.load(detector_z_ptr + det_id, mask=d_mask, other=0.0)

            dx = det_loc_x - vox_loc_x
            dy = det_loc_y - vox_loc_y
            dz = det_loc_z - vox_loc_z

            dist_sq = dx * dx + dy * dy + dz * dz
            dist_inv = tl.rsqrt(dist_sq)

            time_center = (dist_sq * dist_inv * dd_inv_upsampled + 0.5).to(tl.int32)
            t_mask = (time_center >= 0) & (time_center < num_times_upsampled)

            out_ptr = out_start + det_id * stride_partial_d + time_center
            val_to_add = x_val * dist_inv
            mask = t_mask & v_mask & d_mask
            tl.atomic_add(out_ptr, val_to_add, mask=mask, sem="relaxed")


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


def _generate_dynamic_split_k(num_detectors: int) -> int:
    """
    Generate dynamic Split-K partition count.

    Args:
        num_detectors: Number of detectors

    Returns:
        split_k: Split-K partition count
    """
    split_k = max(1, (MIN_NUM_BLOCK + num_detectors - 1) // num_detectors)
    split_k = min(split_k, MAX_SPLIT_K)
    split_k = 2 ** int(math.log2(split_k) + 0.5)
    return split_k


class SignalSimulator:
    """
    Photoacoustic Signal Simulator

    Generates detector signals from voxel data based on the spherical
    wave propagation model. Achieves high-performance simulation through
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
        Initialize the signal simulator.

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

        # Generate dynamic Split-K partition count
        self.split_k = _generate_dynamic_split_k(self.num_detectors)

        print(f"SignalSimulator initialized:")
        print(
            f"  - Voxel grid: {self.num_x} x {self.num_y} x {self.num_z} = {self.num_voxels} voxels"
        )
        print(f"  - Number of detectors: {self.num_detectors}")
        print(f"  - Time samples: {self.num_times} (upsampled: {self.num_times_upsampled})")
        print(f"  - Split-K: {self.split_k}")
        print(f"  - Autotune: {'enabled' if use_autotune else 'disabled (RTX 4090 optimized parameters)'}")
        print(f"  - Device: {self.device}")

    def simulate(
        self,
        voxel_data: Union[np.ndarray, torch.Tensor],
        return_2d: bool = True,
    ) -> torch.Tensor:
        """
        Perform signal simulation.

        Generates detector signals from voxel data.

        Args:
            voxel_data: Voxel data, can be:
                - [num_x, num_y, num_z] 3D array
                - [num_voxels] 1D array
            return_2d: Whether to return a 2D signal matrix [num_detectors, num_times].
                       If False, returns 1D flattened signal [num_detectors * num_times].

        Returns:
            Simulated signal tensor
        """
        # Convert input data
        if isinstance(voxel_data, np.ndarray):
            voxel_data = torch.from_numpy(voxel_data)
        x = voxel_data.to(self.device).float().flatten().contiguous()

        # Check input size
        if x.shape[0] != self.num_voxels:
            raise ValueError(
                f"Input voxel count ({x.shape[0]}) does not match configuration ({self.num_voxels})"
            )

        # Apply the forward operator
        signal = self._forward_operator(x)

        if return_2d:
            return signal.reshape(self.num_detectors, self.num_times)
        return signal

    def _forward_operator(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the forward operator A*x.

        Args:
            x: Input voxel data [num_voxels]

        Returns:
            Output signal [num_detectors * num_times]
        """
        # Allocate partial_output buffer [split_k, num_detectors, num_times_upsampled]
        partial_output = torch.zeros(
            (self.split_k, self.num_detectors, self.num_times_upsampled),
            device=self.device,
            dtype=torch.float32,
        )
        stride_partial_k = self.num_detectors * self.num_times_upsampled
        stride_partial_d = self.num_times_upsampled

        if self.use_autotune:
            # Use autotune version
            grid = lambda META: (
                triton.cdiv(self.num_detectors, META["BLOCK_DET"]),
                self.split_k,
            )
            _forward_kernel_splitk_autotune[grid](
                x,
                self.detector_x,
                self.detector_y,
                self.detector_z,
                partial_output,
                self.num_voxels,
                self.num_detectors,
                self.num_times_upsampled,
                self.dd_inv_upsampled,
                stride_partial_k,
                stride_partial_d,
                self.split_k,
                self.x_start,
                self.y_start,
                self.z_start,
                self.res,
                self.num_y,
                self.num_z,
            )
        else:
            # Use fixed-parameter version (RTX 4090 optimized)
            grid = (
                triton.cdiv(self.num_detectors, DEFAULT_BLOCK_DET),
                self.split_k,
            )
            _forward_kernel_splitk_fixed[grid](
                x,
                self.detector_x,
                self.detector_y,
                self.detector_z,
                partial_output,
                self.num_voxels,
                self.num_detectors,
                self.num_times_upsampled,
                self.dd_inv_upsampled,
                stride_partial_k,
                stride_partial_d,
                self.split_k,
                self.x_start,
                self.y_start,
                self.z_start,
                self.res,
                self.num_y,
                self.num_z,
                BLOCK_VOXEL=DEFAULT_BLOCK_VOXEL,
                BLOCK_DET=DEFAULT_BLOCK_DET,
                num_stages=DEFAULT_NUM_STAGES,
                num_warps=DEFAULT_NUM_WARPS,
            )

        # Reduction: sum along the split_k dimension
        dy_upsampling_batch = partial_output.sum(dim=0)

        # Convolution operation
        dy_conv_transpose = torch.nn.functional.conv_transpose1d(
            dy_upsampling_batch.unsqueeze(1),
            self.conv_kernel,
            padding=self.time_interval_length_half,
        ).squeeze(1)

        # Downsampling and flattening
        y = dy_conv_transpose[:, :: self.adaptive_ratio].contiguous().flatten()
        return y

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """
        Get the grid shape.

        Returns:
            (num_x, num_y, num_z): Number of voxels in each dimension
        """
        return self.num_x, self.num_y, self.num_z


# ==================== Convenience Functions ====================
def simulate_signal(
    voxel_data: Union[np.ndarray, torch.Tensor],
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
    One-shot signal simulation (convenience function).

    For a single simulation, this function can be used directly. If multiple
    simulations with the same geometric configuration are needed, it is
    recommended to create a SignalSimulator instance to avoid repeated
    initialization overhead.

    Args:
        voxel_data: Voxel data [num_x, num_y, num_z] or [num_voxels]
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
        Simulated signal [num_detectors, num_times]
    """
    simulator = SignalSimulator(
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
    return simulator.simulate(voxel_data)


# ==================== Test Code ====================
if __name__ == "__main__":
    import scipy.io as sio

    print("=" * 60)
    print("Signal Simulator Test")
    print("=" * 60)

    # Configuration parameters
    x_range = [-12.80e-3, 12.80e-3]
    y_range = [-12.80e-3, 12.80e-3]
    z_range = [-6.40e-3, 6.40e-3]
    res = 0.10e-3
    vs = 1510.0
    fs = 8.333333e6

    # Load detector locations (example)
    try:
        detector_locations = sio.loadmat("data/sensor_Liver_location.mat")[
            "detector_locations"
        ]

        # Load real data for validation
        real_data = sio.loadmat("data/sensor_Liver_data_matrix.mat")["simulation_data"]
        num_times = real_data.shape[1]

        # Create simulator (autotune disabled by default)
        simulator = SignalSimulator(
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

        # Create test voxel data (all zeros except one point at center)
        num_x, num_y, num_z = simulator.get_grid_shape()
        test_voxel = torch.zeros(num_x, num_y, num_z, device=simulator.device)
        test_voxel[num_x // 2, num_y // 2, num_z // 2] = 1.0

        # Perform simulation
        print("\nPerforming signal simulation...")
        simulated_signal = simulator.simulate(test_voxel)
        print(f"Output signal shape: {simulated_signal.shape}")
        print(
            f"Signal range: [{simulated_signal.min().item():.6e}, {simulated_signal.max().item():.6e}]"
        )

        print("\nTest completed!")

    except FileNotFoundError:
        print("Test data files not found, skipping test.")
        print("Please provide correct detector location data for usage.")
