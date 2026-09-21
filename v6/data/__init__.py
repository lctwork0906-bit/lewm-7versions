from .base import DataFormatStrategy
from .registry import DataStrategyRegistry
from .formats.hdf5_rgb import HDF5RGBStrategy
from .formats.hdf5_rgbd import HDF5RGBDStrategy
from .formats.hdf5_rgbd_voxel import HDF5RGBDVoxelStrategy
from .formats.png_rgbd import PNGDepthStrategy

DataStrategyRegistry.register('hdf5_rgb', HDF5RGBStrategy)
DataStrategyRegistry.register('hdf5_rgbd', HDF5RGBDStrategy)
DataStrategyRegistry.register('hdf5_rgbd_voxel', HDF5RGBDVoxelStrategy)
DataStrategyRegistry.register('png_rgbd', PNGDepthStrategy)

__all__ = [
    'DataFormatStrategy',
    'DataStrategyRegistry',
    'HDF5RGBStrategy',
    'HDF5RGBDStrategy',
    'HDF5RGBDVoxelStrategy',
]