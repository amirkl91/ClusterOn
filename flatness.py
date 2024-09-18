import osmnx as ox
import numpy as np
import pandas as pd
from shapely.geometry import Point
import random
import rasterio
from rasterio.warp import transform_bounds
import requests
import os
import tempfile
from zipfile import ZipFile
import time

def download_srtm_data(lon, lat):
    """Download SRTM data for given coordinates"""
    base_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"
    x = int((lon + 180) / 5) + 1
    y = int((60 - lat) / 5) + 1
    filename = f"srtm_{x:02d}_{y:02d}.zip"
    url = base_url + filename
    
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        with ZipFile(tmp_file_path, 'r') as zip_ref:
            tif_file = [f for f in zip_ref.namelist() if f.endswith('.tif')][0]
            zip_ref.extract(tif_file, path=os.path.dirname(tmp_file_path))
        
        os.unlink(tmp_file_path)
        return os.path.join(os.path.dirname(tmp_file_path), tif_file)
    else:
        raise Exception(f"Failed to download SRTM data for coordinates {lon}, {lat}")

def get_elevation(lon, lat, elevation_raster):
    """Get elevation for a single point"""
    row, col = elevation_raster.index(lon, lat)
    elevation = elevation_raster.read(1)[row, col]
    return float(elevation)

def compute_city_flatness(city_name, country, sample_size=1000):
    # 1. Download city boundary
    try:
        city = ox.geocode_to_gdf(f"{city_name}, {country}")
    except Exception as e:
        raise Exception(f"Failed to download data for {city_name}, {country}: {str(e)}")

    city_boundary = city.geometry.iloc[0]
    bounds = city_boundary.bounds

    # 2. Download SRTM data for the city area
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    srtm_file = download_srtm_data(center_lon, center_lat)

    with rasterio.open(srtm_file) as elevation_raster:
        # 3. Generate random points within the city boundary and get their elevations
        sampled_elevations = []
        while len(sampled_elevations) < sample_size:
            minx, miny, maxx, maxy = bounds
            point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if city_boundary.contains(point):
                try:
                    elevation = get_elevation(point.x, point.y, elevation_raster)
                    if elevation != elevation_raster.nodata:  # Check for no data values
                        sampled_elevations.append(elevation)
                except IndexError:
                    # Point is outside the raster bounds, skip it
                    continue

    # 4. Compute flatness metrics
    sampled_elevations = np.array(sampled_elevations)
    flatness_score = 1 / np.std(sampled_elevations) if len(sampled_elevations) > 1 else 0
    elevation_range = np.max(sampled_elevations) - np.min(sampled_elevations)
    mean_elevation = np.mean(sampled_elevations)
    median_elevation = np.median(sampled_elevations)

    # Clean up the downloaded file
    os.unlink(srtm_file)

    return {
        "flatness_score": flatness_score,
        "elevation_range": elevation_range,
        "mean_elevation": mean_elevation,
        "median_elevation": median_elevation,
        "num_sampled_points": len(sampled_elevations)
    }

cities = ['Nazareth',
          'Jerusalem', 'Tel Aviv', 'Haifa', 'Rishon LeZion', 'Petah Tikva', 'Ashdod',
           'Netanya', 'Beer Sheva', 'Bnei Brak', 'Holon', 'Ramat Gan', 'Rehovot', 'Ashkelon',
           'Bat Yam', 'Beit Shemesh', 'Kfar Saba', 'Herzliya', 'Hadera', 'Modiin']

if __name__ == '__main__':
    all_data = []
    for city in cities:
        print(city)
        #print(time.time())
        try:
            result = compute_city_flatness(city, "Israel")
            result['city'] = city
            all_data.append(result)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    pd.DataFrame(all_data).to_csv('data/flatness.csv', index=False)
