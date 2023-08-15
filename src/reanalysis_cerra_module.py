import geopandas as gpd
from osgeo import gdal, osr, ogr
import xarray as xr
import numpy as np
import pyproj
from shapely.geometry import Point
import time as timer
import re
import json
import os, sys
from datetime import datetime
import cdsapi
import shutil

class Reanalysis_Cerra_Module():

    # Global CRS
    crs_LATLONG = pyproj.Proj(proj='latlong', datum='WGS84').crs # https://epsg.io/4326
    crs_CERRA = '+proj=lcc +lat_0=50 +lat_1=50 +lat_2=50 +lon_0=8 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs'

    def __init__(self, desc_prov= None, geom_file= './data/provinces.gpkg', raster_path= './data/', logs_path= './logs'):
        self.desc_prov = desc_prov
        self.geom_file = geom_file
        self.raster_path = raster_path
        self.logs_path = logs_path
    

    def eval_gdal_dask_pipeline(self, raster_file: str) -> dict:
        
        start_time = timer.time()

        gdb = ogr.GetDriverByName("GPKG").Open(self.geom_file, 0) 
        lyr = gdb.GetLayerByIndex(0) # Layer 0: Provinces
        if self.desc_prov is not None: lyr.SetAttributeFilter(f"provincia='{self.desc_prov}'")

        src = pyproj.CRS.from_string(lyr.GetSpatialRef().ExportToWkt())
        dst = pyproj.CRS.from_proj4(self.crs_CERRA)
        transform_to_ecmwf = pyproj.Transformer.from_crs(src, dst, always_xy=True)

        bbox = {} # Bounding box
        for feat in lyr:
            geom = feat.GetGeometryRef() # > type: osgeo.ogr.geometry
            points = self.flatten_lists(json.loads(geom.ExportToJson())['coordinates'])

            for point in points:
                x, y = transform_to_ecmwf.transform(point[0], point[1])

                if 'xmin' not in bbox or bbox['xmin'] > x: bbox['xmin'] = x
                if 'xmax' not in bbox or bbox['xmax'] < x: bbox['xmax'] = x
                if 'ymin' not in bbox or bbox['ymin'] > y: bbox['ymin'] = y
                if 'ymax' not in bbox or bbox['ymax'] < y: bbox['ymax'] = y
        
        #lyr, bbox = self.read_geom_layer_bbox_gpkg()
        ds = self.read_xarray_raster_dataset(raster_file)

        ds_bbox = ds.where(((bbox['xmin']< ds.longitude) & (ds.longitude < bbox['xmax']) & (bbox['ymin']< ds.latitude) & (ds.latitude < bbox['ymax'])),  drop=1) # Drop data asociated with points outside the bounding box
        ycount, xcount =ds_bbox.latitude.shape 

        pixelWidth = abs(round(ds.longitude.values[0][0]-ds.longitude.values[0][1], 3))
        pixelHeight = round(ds.latitude.values[0][0]-ds.latitude.values[1][0], 3)

        # Set projection
        target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Float32)
        target_ds.SetGeoTransform((
            np.min(ds_bbox.longitude.values), pixelWidth, 0,
            np.max(ds_bbox.latitude.values), 0, pixelHeight,
        ))

        target_srs = osr.SpatialReference()
        target_srs.ImportFromProj4(self.crs_CERRA)
        target_ds.SetProjection(target_srs.ExportToWkt())
        gdal.RasterizeLayer(target_ds, [1], lyr)
        bandmask = target_ds.GetRasterBand(1)
        mask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(bool)[::-1, :]

        ds_bbox = ds_bbox.where(mask) 

        crop_to_bbox_time = round(timer.time() - start_time, 2) 

        time = timer.time() 
        stats = self.extract_statistics_dask(ds_bbox)
        extract_statistics_time = round(timer.time() - time, 2) 

        self.clean_data_dir()

        return {
            'crop_to_bbox_time': crop_to_bbox_time,
            'extract_statistics_time':extract_statistics_time,
            'execution_time': round(timer.time() - start_time, 2) 
        } 

    def eval_geopandas_dask_pipeline(self, raster_file: str) -> dict: 
        start_time = timer.time()

        ds = self.read_xarray_raster_dataset(raster_file)

        df = gpd.read_file(self.geom_file).to_crs(self.crs_CERRA)
        if self.desc_prov is not None: df = df.loc[df['provincia'] == self.desc_prov]

        xmin, ymin, xmax, ymax = df.total_bounds 

        ds_bbox = ds.where(((xmin<= ds.longitude) & (ds.longitude <= xmax) & (ymin<= ds.latitude) & (ds.latitude <= ymax)),  drop=1) # Drop data asociated with points outside the bounding box

        mask = np.zeros_like(ds_bbox.longitude.values, dtype=bool)
        points =np.column_stack((ds_bbox.longitude.values.flatten(), ds_bbox.latitude.values.flatten()))

        for x, y in points: 
            if df.geometry.contains(Point(x, y)).any():
                mask = np.logical_or(mask, (ds_bbox.longitude.values == x) & (ds_bbox.latitude.values == y))

        ds_bbox = ds_bbox.where(mask) 

        crop_to_bbox_time = round(timer.time() - start_time, 2)

        time = timer.time() 
        stats = self.extract_statistics_dask(ds_bbox) 
        extract_statistics_time = round(timer.time() - time, 2) 

        self.clean_data_dir()

        return {
            'crop_to_bbox_time': crop_to_bbox_time,
            'extract_statistics_time':extract_statistics_time,
            'execution_time': round(timer.time() - start_time, 2) 
        } 

    def eval_gdal_numpy_pipeline(self, raster_file: str) -> dict: 
        start_time = timer.time()

        gdb = ogr.GetDriverByName("GPKG").Open(self.geom_file, 0) 
        lyr = gdb.GetLayerByIndex(0) # Layer 0: Provinces
        if self.desc_prov is not None: lyr.SetAttributeFilter(f"provincia='{self.desc_prov}'")

        src = pyproj.CRS.from_string(lyr.GetSpatialRef().ExportToWkt())
        dst = pyproj.CRS.from_proj4(self.crs_CERRA)
        transform_to_ecmwf = pyproj.Transformer.from_crs(src, dst, always_xy=True)

        bbox = {} # Bounding box
        for feat in lyr:
            geom = feat.GetGeometryRef() # > type: osgeo.ogr.geometry
            points = self.flatten_lists(json.loads(geom.ExportToJson())['coordinates'])

            for point in points:
                x, y = transform_to_ecmwf.transform(point[0], point[1])

                if 'xmin' not in bbox or bbox['xmin'] > x: bbox['xmin'] = x
                if 'xmax' not in bbox or bbox['xmax'] < x: bbox['xmax'] = x
                if 'ymin' not in bbox or bbox['ymin'] > y: bbox['ymin'] = y
                if 'ymax' not in bbox or bbox['ymax'] < y: bbox['ymax'] = y
        
        #lyr, bbox = self.read_geom_layer_bbox_gpkg()
        raster = gdal.Open(f'{self.raster_path}/{raster_file}')

        transform = raster.GetGeoTransform()

        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]

        # Specify offset and rows and columns to read
        xoff = abs(int((xOrigin - bbox['xmin'])/pixelWidth))
        yoff = int((yOrigin - bbox['ymax'])/pixelWidth)
        xcount = int((bbox['xmax'] - bbox['xmin']) / pixelWidth)
        ycount = abs(int((bbox['ymax'] - bbox['ymin']) / pixelWidth))

        # Origin of new raster
        x_lon_min = xoff * pixelWidth + xOrigin
        y_lat_max = (yoff * pixelWidth - yOrigin) * -1

        # Create memory target raster
        target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Float32)
        target_ds.SetGeoTransform((
            x_lon_min, pixelWidth, 0,
            y_lat_max, 0, pixelHeight,
        ))

        target_ds.SetProjection(raster.GetProjection())
        gdal.RasterizeLayer(target_ds, [1], lyr)
        bandmask = target_ds.GetRasterBand(1)
        datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(float)

        raster_info = {}
        for i in range(1, raster.RasterCount+1):
            band = raster.GetRasterBand(i)
            dataraster = band.ReadAsArray(xoff, yoff, xcount, ycount).astype(float)

            zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
            zoneraster = np.ma.filled(zoneraster, np.nan)
            #zoneraster = zoneraster[~np.isnan(zoneraster)]
            
            if band.GetMetadata()['GRIB_COMMENT'] not in raster_info: raster_info[band.GetMetadata()['GRIB_COMMENT']] = []
            raster_info[band.GetMetadata()['GRIB_COMMENT']].append(zoneraster)

        crop_to_bbox_time = round(timer.time() - start_time, 2)

        time = timer.time() 
        stats = self.extract_statistics_numpy(raster_info, int(raster.RasterCount/len(raster_info)), int(target_ds.GetRasterBand(1).XSize), int(target_ds.GetRasterBand(1).YSize))
        extract_statistics_time = round(timer.time() - time, 2)  

        self.clean_data_dir()

        return {
            'crop_to_bbox_time': crop_to_bbox_time,
            'extract_statistics_time': extract_statistics_time,
            'execution_time': round(timer.time() - start_time, 2) 
        } 


    def read_xarray_raster_dataset(self, raster_file: str) -> xr.Dataset:
        ds = xr.open_dataset(f'{self.raster_path}/{raster_file}', engine="cfgrib")

        ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
        ds = ds.assign_coords(latitude=(ds.latitude - 90) % 180 - 90)

        src = self.crs_LATLONG 
        dst = pyproj.CRS.from_proj4(self.crs_CERRA)
        transformer = pyproj.Transformer.from_crs(src, dst, always_xy=True)
        xGrid, yGrid = transformer.transform(ds.longitude.values, ds.latitude.values)

        ds['longitude'] = (ds.longitude.dims, xGrid)
        ds['longitude'].attrs['units'] = 'm'
        ds['latitude'] = (ds.latitude.dims, yGrid)
        ds['latitude'].attrs['units'] = 'm'

        return ds

    def extract_statistics_dask(self, raster_dataset: xr.Dataset) -> list:
        retVal = []

        for data_var in list(raster_dataset.data_vars):
            stats = {}

            stats['short_name'] = data_var 
            stats['standard_name'] = raster_dataset[data_var].attrs['standard_name'] 

            #stats['points'] = raster_dataset[data_var].shape[0]*raster_dataset[data_var].shape[1]*raster_dataset[data_var].shape[2]
            stats['time'] = raster_dataset[data_var].shape[0]
            stats['latitude'] = raster_dataset[data_var].shape[1]
            stats['longitude'] = raster_dataset[data_var].shape[2]
            stats['points'] = stats['time'] * stats['latitude'] * stats['longitude'] 

            stats['mean']   = round(float(raster_dataset[data_var].mean()), 2)
            stats['median'] = round(float(raster_dataset[data_var].median()), 2)
            stats['std']    = round(float(raster_dataset[data_var].std()), 2)
            stats['var']    = round(float(raster_dataset[data_var].var()), 2)    
            stats['max']    = round(float(raster_dataset[data_var].max()), 2)
            stats['min']    = round(float(raster_dataset[data_var].min()), 2)

            retVal.append(stats)

        return retVal

    def extract_statistics_numpy(self, raster_info: dict, time: int, longitude: int, latitude: int) -> list: 
        retVal = []
        for key in raster_info:
            stats = {}
            zoneraster = np.array(raster_info[key]).flatten()

            stats['short_name'] = None 
            stats['standard_name'] = key
            
            stats['time'] = time
            stats['latitude'] = latitude
            stats['longitude'] = longitude 
            stats['points'] = len(zoneraster)

            stats['mean']   = round(float(np.nanmean(zoneraster)), 2)
            stats['median'] = round(float(np.nanmedian(zoneraster)), 2)
            stats['std']    = round(float(np.nanstd(zoneraster)), 2)
            stats['var']    = round(float(np.nanvar(zoneraster)), 2)    
            stats['max']    = round(float(np.nanmax(zoneraster)), 2)
            stats['min']    = round(float(np.nanmin(zoneraster)), 2)

            retVal.append(stats)
        return retVal
        
    # Other support methods

    def flatten_lists(self, nested_list: list):
        flattened_list = []
        stack = [nested_list]

        while stack:
            item = stack.pop()
            if len(item) == 2 and all(not isinstance(sub_item, list) for sub_item in item):
                flattened_list.append(item)  # Preserve the 2x2 list without flattening
            else:
                stack.extend(item[::-1]) 
        return flattened_list

    def clean_data_dir(self):
        idx_files = [file for file in os.listdir(self.raster_path) if file.endswith(".idx")]

        for idx_file in idx_files:
            os.remove(f'{self.raster_path}/{idx_file}')


    def download_cerra_data(self, req_file_size_GB):
        time = [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ]
        year = [
                    #'1984', '1985', '1986',
                    '1987', '1988', '1989',
                    '1990', '1991', '1992',
                    '1993', '1994', '1995',
                    '1996', '1997', '1998',
                    '1999', '2000', '2001',
                    '2002', '2003', '2004',
                    '2005', '2006', '2007',
                    '2008', '2009', '2010',
                    '2011', '2012', '2013',
                    '2014', '2015', '2016',
                    '2017', '2018', '2019',
                    '2020', '2021',
        ]

        req_1GB = {
                'format': 'grib',
                'year': [
                    '2010'
                ],
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    #'07', '08', '09',
                    #'10', '11', '12',
                ],
                'day': [
                    #'01', '02', '03',
                    '04', #'05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00'
                ],
                'variable': [
                    '2m_relative_humidity', 'skin_temperature', 'surface_pressure',
                ],
                'product_type': 'analysis',
                'level_type': 'surface_or_atmosphere',
                'data_type': 'reanalysis',
        } 

        print(f'Start download...')

        c = cdsapi.Client()

        for i in range(1, len(time)+1):
            for j in range(1, len(year)+1):
                if i*j in req_file_size_GB:

                    req_file_size_GB.remove(i*j) 

                    req_1GB['time'] = time[:i]  
                    req_1GB['year'] = year[:j]
                    
                    c.retrieve(
                    'reanalysis-cerra-single-levels',
                    req_1GB,
                    f"{self.raster_path}/{str(i*j)}GB_reanalysis-cerra-single-levels.grib")

                    print(f"File downloaded: {self.raster_path}/{str(i*j)}GB_reanalysis-cerra-single-levels.grib")

        print(f'Download completed :)')
    

if __name__ == '__main__':
    desc_prov = None
    download = 0
    req_file_size_GB = [1, 2, 4, 8, 16, 32, 64]

    # > Param 0: Download data 0 or 1
    if len(sys.argv) > 1: download = sys.argv[1]

    # > Param 1: Province name
    if len(sys.argv) > 2: desc_prov = sys.argv[2]

    #########    

    stats = {} 
    rcm = Reanalysis_Cerra_Module(desc_prov = desc_prov)

    if download == '1': rcm.download_cerra_data(req_file_size_GB)

    rcm.clean_data_dir() # Remove .idx files

    files = [file for file in os.listdir(rcm.raster_path) if file.endswith(".grib")]

    for idx, f in enumerate(files):
        
        stats[f] = {'eval_gdal_dask_pipeline': rcm.eval_gdal_dask_pipeline(f),
                    'eval_gdal_numpy_pipeline': rcm.eval_gdal_numpy_pipeline(f),
                    'eval_geopandas_dask_pipeline': rcm.eval_geopandas_dask_pipeline(f)
        }

        print(f'{str(round(100*((1+idx)/len(files)), 2))}% Completed - File: {f} loaded.')


    ## Save log 
    d = datetime.now()
    d = d.strftime("%Y-%m-%d_%H:%M:%S")

    with open(f'{rcm.logs_path}/{d}_performance-metrics.json', "w") as json_file:
        json.dump(stats, json_file)