# -*- coding: utf-8 -*-
# @Author: wqshen
# @Email: wqshen91@gmail.com
# @Date: 2020/7/22 15:59
# @Last Modified by: wqshen

import re
import os
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, vectorize
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Fy4aL2Reader(object):
    """FY-4 卫星二级产品NetCDF数据读取类

    虽然NetCDF文件可以使用netCDF4库或xarray库方便的读取，但是该类型NetCDF文件中不包含坐标，
    本类在读取文件后，可计算行列号和地理经纬度坐标，并可方便的使用行列号范围或经纬度范围裁剪数据。
    """

    def __init__(self):
        pass

    def decode_coords(self, ds, resolution=None, geo_extent=None):
        """解码FY-4 L2 NetCDF文件行列号和经纬度坐标

        Parameters
        ----------
        ds (xr.Dataset): xarray打开的FY-4 L2数据集
        resolution (str): FY-4数据分辨率，例如： '4000' 代表 4km, 默认： None - 自动从文件推断
        geo_extent (tuple): 数据集对应的(开始行号,结束行号,开始列号,结束列号), 默认： None - 自动从文件推断

        Returns
        -------
        (xr.Dataset): 维名修改为l (行号), c (列号)，新增坐标： l, c, lat2d (二维纬度), lon2d (二维经度)
        """
        if resolution is None:
            match_resolution = re.match(r'.*(\d+km|\d+m).*', ds.spatial_resolution)
            if match_resolution:
                resolution = match_resolution.group(1).replace(' ', '').replace('km', '000').replace('m', '')
            else:
                raise Exception("Failed recognize spatial resolution from filename or NC attribute.")

        # obs_time = pd.to_datetime(ds.time_coverage_start) + pd.to_timedelta('8H')
        # sat_height = float(ds.nominal_satellite_height[0])
        sat_lon = float(ds.nominal_satellite_subpoint_lon[0])
        if geo_extent is None:
            ext = ds.geospatial_lat_lon_extent
            geo_extent = ext.begin_line_number, ext.end_line_number, ext.begin_pixel_number, ext.end_pixel_number
        bl, el, bp, ep = geo_extent
        # 重定义 行号 lines 和列号 columns 作为坐标， 2位纬度 lat 和2位经度 lon 作为坐标
        lines = np.arange(bl, el + 0.1, dtype=np.uint16)
        columns = np.arange(bp, ep + 0.1, dtype=np.uint16)
        lon, lat = self.lc_to_lonlat(lines[:, None], columns[None, :], str(resolution), lambda_D=sat_lon)
        lon = xr.DataArray(lon, dims=('l', 'c'), coords=dict(l=lines, c=columns), name='lon')
        lat = xr.DataArray(lat, dims=('l', 'c'), coords=dict(l=lines, c=columns), name='lat')
        ds = ds.rename(x='l', y='c')
        ds = ds.assign_coords(l=lines, c=columns)
        ds = ds.assign_coords(lat2d=lat, lon2d=lon)
        return ds

    def read(self, pathfile, lc_extent=None, ll_extent=None, decode_coords=True, resolution=None):
        """

        Parameters
        ----------
        pathfile (str): FY-4 L2产品文件路径
        lp_extent (tuple): 行列号的起止数，升序(开始行号, 结束行号, 开始列号, 结束列号)
        ll_extent (tuple): 经纬度范围 ，升序(开始经度, 结束经度, 开始纬度, 结束纬度)
        decode_coords (bool): 自动解析行列号和经纬度坐标
        resolution (str): FY-4数据分辨率，例如： '4000' 代表 4km, 默认： None - 自动从文件推断

        Returns
        -------
        (xr.Dataset): 读取处理的FY-4数据集
        """
        ds = xr.open_dataset(pathfile, decode_cf=False, mask_and_scale=True)
        if resolution is None:
            match_resolution = re.match(r"FY4A-_AGRI.*_L2.*_(\d+)M_V\d+.NC", os.path.basename(pathfile))
            if match_resolution:
                resolution = match_resolution.group(1)

        geo_extent = None
        # 行列号裁剪
        if lc_extent:
            ext = ds.geospatial_lat_lon_extent
            bl, el, bp, ep = ext.begin_line_number, ext.end_line_number, ext.begin_pixel_number, ext.end_pixel_number
            j0, j1 = max(0, lc_extent[0] - bl), min(-1, lc_extent[1] - el)
            i0, i1 = max(0, lc_extent[2] - bp), min(-1, lc_extent[3] - ep)
            ds = ds.isel(x=slice(j0, j1), y=slice(i0, i1))
            geo_extent = lc_extent

        # 计算经纬度
        if decode_coords:
            ds = self.decode_coords(ds, resolution, geo_extent)
            # 经纬度裁剪
            if ll_extent:
                i0, i1, j0, j1 = self.locate_irregular_boundary(ds.lon2d, ds.lat2d, extent=ll_extent, lat_order='DESC')
                ds = ds.isel(l=slice(j0, j1), c=slice(i0, i1))
        return ds

    @staticmethod
    @njit
    def retrieve_geo(lon, lat, lines, pixels, v):
        # Not used
        ny, nx = pixels.shape
        v2d = np.full_like(v, fill_value=np.nan)
        lon2d = np.full_like(v, fill_value=np.nan)
        lat2d = np.full_like(v, fill_value=np.nan)
        for j in range(ny):
            for i in range(nx):
                if pixels[j, i] >= 0 and lines[j, i] >= 0:
                    lon2d[j, i] = lon[lines[j, i], pixels[j, i]]
                    lat2d[j, i] = lat[lines[j, i], pixels[j, i]]
                    v2d[j, i] = v[j, i]
        return lon2d, lat2d, v2d

    @property
    def valid_extent(self):
        """标称图上全圆盘经纬度范围"""
        return 24.11662309, 360 - 174.71662309, -80.56672132, 80.56672132

    @property
    def lookup_dimsize(self):
        """查找表对应分辨率下的数组维数（行数/列数）"""
        return {'16000': 687, '8000': 1374, '4000': 2748, '2000': 5496, '1000': 10992, '500': 21984}

    def read_lookup(self, raw_file, resolution=None):
        """读取FY4A查找表

        Parameters
        ----------
        raw_file (str): FY4A查找表文件路径
        lp_extent (tuple): 行列号的起止数，升序(开始行号, 结束行号, 开始列号, 结束列号)
        ll_extent (tuple): 经纬度范围 ，升序(开始经度, 结束经度, 开始纬度, 结束纬度)
        resolution (int): FY4A查找表文件分辨率 (km), 默认: None, 自动从文件名提取

        Returns
        -------
        (array, array): 纬度（二维）， 经度（二维）
        """
        if resolution is None:
            match_resolution = re.match(r'.*FullMask_Grid_(\d+).raw', raw_file)
            if match_resolution:
                resolution = match_resolution.group(1)
            else:
                raise Exception("Resolve resolution failed, specify `resolution` argument explicitly.")
        dim = self.lookup_dimsize.get(str(resolution))

        lat_lon = np.full((dim, dim, 2), np.nan, dtype='<f8')
        with open(raw_file, 'rb') as f:
            f.readinto(lat_lon)
        lat, lon = lat_lon[:, :, 0], lat_lon[:, :, 1]
        lon[lon < 0] += 360.
        valid_extent = self.valid_extent
        lon[(lon < valid_extent[0]) | (lon > valid_extent[1])] = np.nan
        lat[(lat < valid_extent[2]) | (lat > valid_extent[3])] = np.nan

        return lon, lat

    @staticmethod
    def lonlat_to_lc(lon, lat, resolution, lambda_D=104.7):
        """转换地理经纬度到FY-4卫星行列号

        Parameters
        ----------
        lon (float): 地理经度
        lat (float): 地理纬度
        resolution (str): FY-4数据分辨率，例如： '4000' 代表 4km
        lambda_D (float): 卫星星下点所在经度，默认：104.7

        Returns
        -------
        (float, float): 行号，列号
        """
        # 地球的半长轴，短半轴
        ea, eb = 6378.137, 6356.7523
        # 地心到卫星质心的距离
        h = 42164
        # 卫星星下点所在经度
        lambda_D = np.deg2rad(lambda_D)
        # 列偏移
        COFFs = {'500': 10991.5, '1000': 5495.5, '2000': 2747.5, '4000': 1373.5}
        # 行偏移
        LOFFs = {'500': 10991.5, '1000': 5495.5, '2000': 2747.5, '4000': 1373.5}
        # 列比例因子
        CFACs = {'500': 81865099, '1000': 40932549, '2000': 20466274, '4000': 10233137}
        # 行比例因子
        LFACs = {'500': 81865099, '1000': 40932549, '2000': 20466274, '4000': 10233137}
        coff, loff, cfac, lfac = COFFs[resolution], LOFFs[resolution], CFACs[resolution], LFACs[resolution]
        # 转换经度为负值时加上360
        if np.ndim(lon) > 0:
            lon[lon < 0] += 360
        else:
            lon = 360 + lon if lon < 0 else lon
        # 经纬度检查
        valid_extent = 24.11662309, 360 - 174.71662309, -80.56672132, 80.56672132
        if (np.nanmin(lon) < valid_extent[0]) | (np.nanmax(lon) > valid_extent[1]):
            raise Exception("`lon` must be in range of ({},{})".format(*valid_extent[:2]))
        if (np.nanmin(lat) < valid_extent[2]) | (np.nanmax(lat) > valid_extent[3]):
            raise Exception("`lat` must be in range of ({},{})".format(*valid_extent[2:]))
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)
        lambda_e = lon
        phi_e = np.arctan((eb ** 2 / ea ** 2) * np.tan(lat))
        radius_e = eb / np.sqrt(1 - (ea ** 2 - eb ** 2) / (ea ** 2) * np.cos(phi_e) ** 2)
        r1 = h - radius_e * np.cos(phi_e) * np.cos(lambda_e - lambda_D)
        r2 = -radius_e * np.cos(phi_e) * np.sin(lambda_e - lambda_D)
        r3 = radius_e * np.sin(phi_e)
        rn = np.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2)
        x = np.rad2deg(np.arctan(-r2 / r1))
        y = np.rad2deg(np.arcsin(-r3 / rn))
        c = coff + x * (2 ** -16) * cfac
        l = loff + y * (2 ** -16) * lfac
        return l, c

    @staticmethod
    def lc_to_lonlat(l, c, resolution, lambda_D=104.7):
        """转换行列号到地理经纬度

        Parameters
        ----------
        l (int): FY-4 行号
        c (int): FY-4 列号
        resolution (str): FY-4数据分辨率，例如： '4000' 代表 4km
        lambda_D (float): 卫星星下点所在经度，默认：104.7

        Returns
        -------
        (float, float): 经度，纬度
        """
        # 地球的半长轴，短半轴
        ea, eb = 6378.137, 6356.7523
        # 地心到卫星质心的距离
        h = 42164
        # 卫星星下点所在经度
        lambda_D = np.deg2rad(lambda_D)
        # 列偏移
        COFFs = {'500': 10991.5, '1000': 5495.5, '2000': 2747.5, '4000': 1373.5}
        # 行偏移
        LOFFs = {'500': 10991.5, '1000': 5495.5, '2000': 2747.5, '4000': 1373.5}
        # 列比例因子
        CFACs = {'500': 81865099, '1000': 40932549, '2000': 20466274, '4000': 10233137}
        # 行比例因子
        LFACs = {'500': 81865099, '1000': 40932549, '2000': 20466274, '4000': 10233137}

        # 参数检查
        if resolution not in ("500", "1000", "2000", "4000"):
            raise Exception('resolution must in "500", "1000", "2000", "4000"')
        if np.any(l < 0) or np.any(c < 0):
            raise Exception("l and c must be greater than 0")
        if np.any(l > loff * 2) or np.any(c > coff * 2):
            raise Exception("l>" + str(int(loff * 2)) + " or c>" + str(int(coff * 2)))

        coff, loff, cfac, lfac = COFFs[resolution], LOFFs[resolution], CFACs[resolution], LFACs[resolution]
        x = np.deg2rad((c - coff) / ((2 ** -16) * cfac))
        y = np.deg2rad((l - loff) / ((2 ** -16) * lfac))
        sd = np.sqrt((h * np.cos(x) * np.cos(y)) ** 2 -
                     (np.cos(y) ** 2 + (ea ** 2) / (eb ** 2) * np.sin(y) ** 2) * (h ** 2 - ea ** 2))
        sn = (h * np.cos(x) * np.cos(y) - sd) / (np.cos(y) ** 2 + (ea ** 2) / (eb ** 2) * np.sin(y) ** 2)
        s1 = h - sn * np.cos(x) * np.cos(y)
        s2 = sn * np.sin(x) * np.cos(y)
        s3 = -sn * np.sin(y)
        sxy = np.sqrt(s1 ** 2 + s2 ** 2)
        lon = np.rad2deg(np.arctan(s2 / s1) + lambda_D)
        lat = np.rad2deg(np.arctan((ea ** 2 / eb ** 2) * (s3 / sxy)))
        return lon, lat

    @staticmethod
    def locate_irregular_boundary(lons, lats, extent, lat_order="DESC"):
        """locate the start/end index of (lons,lats) fully envelop the extent(lon0,lon1,lat0,lat1)

        Parameters
        ----------
        lons (array): the 1D or 2D longitude array
        lats (array): the 1D or 2D latitude array
        extent (tuple): a regular boundary (start lon, end lon, start lat, end lat)

        Returns
        -------
        (int,int,int,int): `start` index, `end` index of lons, `start` index, `end` index of lats
        """
        lon_min, lon_max = np.nanmin(lons, axis=0), np.nanmax(lons, axis=0)
        lat_min, lat_max = np.nanmin(lats, axis=1), np.nanmax(lats, axis=1)
        i0 = np.max(np.where((~np.isnan(lon_max)) & (lon_max <= extent[0])))
        i1 = np.min(np.where((~np.isnan(lon_min)) & (lon_min >= extent[1])))
        if lat_order == "DESC":
            j1 = np.min(np.where((~np.isnan(lat_max)) & (lat_max <= extent[2])))
            j0 = np.max(np.where((~np.isnan(lat_min)) & (lat_min >= extent[3])))
        elif lat_order == 'ASC':
            j0 = np.max(np.where((~np.isnan(lat_max)) & (lat_max <= extent[2])))
            j1 = np.min(np.where((~np.isnan(lat_max)) & (lat_min >= extent[3])))
        else:
            raise NotImplementedError("lat_order must be 'DESC' for decreasing or 'ASC' for increasing")
        return max(0, i0), i1 + 1, max(j0, 0), j1 + 1

    @staticmethod
    def quality_control(l2var, dqf=None, qc=1):
        """L2产品的质量控制

        Parameters
        ----------
        l2var (xr.DataArray): xrrray.DataArray对象
        dqf (xr.DataArray, np.ndarray): 质控数据 (0, 1, 2, 3)
        qc (int): 质控码的上线值

        Returns
        -------
        (xr.DataArray): 处理过的变量
        """
        tmin, tmax = l2var.valid_range
        l2var = l2var.where((l2var >= tmin) & (l2var <= tmax))
        if dqf is not None and isinstance(dqf, (xr.DataArray, np.ndarray)):
            l2var = l2var.where(dqf <= qc)
        return l2var
