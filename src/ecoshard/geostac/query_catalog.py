import os
import shutil
import geopandas as gpd
import pystac
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.prepared import prep

# Path to the vector file with country geometries
countries_vector_path = r"D:\Users\richp\Downloads\critical_natural_assets_cna_overlap_run_2024_03_13\critical_natural_assets\data\countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg"

# Load the country geometries
countries = gpd.read_file(countries_vector_path)

selected_iso3 = ["MDA"]

# Filter and merge geometries of the selected countries into one geometry
print("subset countries")
subset_countries = countries[countries["iso3"].isin(selected_iso3)]
print("build subset geom")
search_geom = unary_union(subset_countries.geometry)

# Create destination directory named after the joined ISO3 codes
dest_dir = "_".join(selected_iso3)
os.makedirs(dest_dir, exist_ok=True)

# Load the STAC catalog
print("read catalog")
catalog = pystac.read_file("stac-catalog/catalog.json")
print("get dem collection")
collection = catalog.get_child("dem-collection")

print("prep search geom")
prepared_search_geom = prep(search_geom)
# Query items that intersect the search geometry
print("match items")

# Query items that intersect the search geometry using the prepared geometry
matched_items = [
    item
    for item in collection.get_items()
    if prepared_search_geom.intersects(shape(item.geometry))
]

# Copy each matched item's asset file to the destination directory
for item in matched_items:
    asset_href = item.assets["data"].href
    dest_file = os.path.join(dest_dir, os.path.basename(asset_href))
    shutil.copy(asset_href, dest_file)
    print(f"Copied {asset_href} to {dest_file}")

# Save a copy of the filtered vector in the destination directory
filtered_vector_path = os.path.join(dest_dir, "filtered_countries.gpkg")
subset_countries.to_file(filtered_vector_path, driver="GPKG")
print(f"Saved filtered vector to {filtered_vector_path}")
