# -*- coding: utf-8 -*-
# @Author: wqshen
# @Email: wqshen91@gmail.com
# @Date: 2019/4/16 16:13
# @Last Modified by: wqshen

import os
import re
import numpy as np
import xarray as xr
from datetime import datetime


class CmorphMergedRainfall:
    """
    CMORPH融合的地面降水二进制文件读取类
    """

    def __init__(self):
        """初始化类对象，并读取文件

        Parameters
        ----------
        pathfile (str): CMORPH融合中国地面降水二进制文件名
        """
        self.lon = np.arange(70.05, 139.96, 0.1)
        self.lat = np.arange(15.05, 58.96, 0.1)
        # combined analysis (mm/Hour)
        self.crain = None
        # gauge numbers
        self.gsamp = None

    @property
    def dtype(self):
        """使用numpy.dtype表示的CMORPH融合降水二进制文件格式"""
        return np.dtype([('crain', '<f4', (440, 700)), ('gsamp', '<f4', (440, 700))])

    @property
    def missing_value(self):
        """文件变量的缺侧值"""
        return -999.

    def _pyread(self, pathfile, buf):
        """读取文件内容到流"""
        with open(pathfile, 'rb') as f:
            f.readinto(buf)

    def mfread(self, pathfiles):
        """读取多个文件并将变量转换为xarray.DataArray对象，并将其增添到当前类属性中"""
        times = []
        data = np.full(len(pathfiles), fill_value=self.missing_value, dtype=self.dtype)
        for i, pathfile in enumerate(pathfiles):
            times.append(self.parse_file_time(os.path.basename(pathfile)))
            self._pyread(pathfile, data[i:i + 1])
        coords = {'time': times, 'lat': self.lat, 'lon': self.lon}
        for name in self.dtype.names:
            var = data[name]
            var[np.isclose(var, self.missing_value)] = np.nan
            self.__dict__[name] = xr.DataArray(np.squeeze(var), dims=('time', 'lat', 'lon'), coords=coords, name=name)
        del data

    def read(self, pathfile):
        """读取文件并将变量转换为xarray.DataArray对象，并将其增添到当前类属性中"""
        data = np.full(1, fill_value=self.missing_value, dtype=self.dtype)
        self._pyread(pathfile, data)
        coords = {'time': [self.parse_file_time(os.path.basename(pathfile))], 'lat': self.lat, 'lon': self.lon}
        for name in self.dtype.names:
            var = data[name]
            var[np.isclose(var, self.missing_value)] = np.nan
            self.__dict__[name] = xr.DataArray(var, dims=('time', 'lat', 'lon'), coords=coords, name=name)
        del data

    def parse_file_time(self, filename):
        """正则表达式从文件名中获取文件时间

        Returns
        -------
        (datetime): 文件数据的时间
        """
        return datetime.strptime(re.search(r"\d{10}", filename).group(), "%Y%m%d%H")

    def to_dataset(self):
        """转换到xarray.Dataset

        Returns
        -------
        (xr.Dataset): 合并 `crain` 和 `gsamp` 的 Dataset
        """
        dataset = xr.merge([self.__dict__[var] for var in self.dtype.names])
        return dataset

    def to_netcdf(self, pathfile, complevel=1):
        """将读取的数据转换为NetCDF4格式，默认对所有变量使用1级压缩

        Parameters
        ----------
        pathfile (str): 输出的netCDF文件路径
        complevel (int): 压缩级别(1-10)，当降水范围小，zlib压缩可有效减少文件大小,显著小于原始二进制文件
        """
        ds = self.to_dataset()
        comp = dict(zlib=True, complevel=complevel)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(pathfile, encoding=encoding)

    def plot(self, timeidx=0, extent=None, pathfig=None):
        """快速绘图"""
        import cartopy.crs as ccrs
        from cartopy.io.shapereader import Reader
        from cartopy.feature import ShapelyFeature
        import matplotlib.pyplot as plt

        plt.rcParams.update({'font.size': 16})

        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        extent = (73, 135, 15, 56) if extent is None else extent
        ax.set_extent(extent)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        # 加中国地图
        src_dir = os.path.dirname(os.path.realpath(__file__))
        shp = os.path.realpath(os.path.join(src_dir, "cn-shp-data/cnhimap.shp"))
        fea = ShapelyFeature(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor="none")
        ax.add_feature(fea, edgecolor='k', lw=0.8)
        # 绘制填色图
        levels = np.array([2.5, 10, 20, 30])
        colors = ("#ffffff", "#3dba3d", "#61b8ff", "#1e6eeb", "#fa00fa", "#800040")
        var = self.crain[timeidx]
        var = var.where(~np.isnan(var), 0)
        pc = var.plot.contourf(ax=ax, levels=levels, colors=colors, extend="both", add_colorbar=False)
        cb = plt.colorbar(pc, fraction=0.02, pad=0.01)
        cb.ax.xaxis.set_tick_params(direction='in')
        cb.ax.yaxis.set_tick_params(direction='in')
        # 网格线
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        plt.tight_layout()
        if pathfig is None:
            plt.show()
        else:
            plt.savefig(pathfig)


if __name__ == '__main__':
    from glob import glob

    files = sorted(glob(r'../data/cmroph_case/GRD/SURF_CLI_CHN_MERGE_CMP_PRE_HOUR_GRID_0.10-20180704*.grd'))
    cmp = CmorphMergedRainfall()
    cmp.read(files[-1])
    print(cmp.crain.shape)
    cmp.mfread(files)
    print(cmp.crain.shape)
    # 测试输出netcdf4格式
    cmp.to_netcdf(r'SURF_CLI_CHN_MERGE_CMP_PRE_HOUR_GRID_0.10-2018062200.nc')
    # 检验变量读取
    cmp.plot(-1, extent=[100, 125, 20, 40])
