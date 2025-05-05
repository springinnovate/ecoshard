from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import datetime
import os

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
import numpy as np
import psutil
import pystac
import rasterio
import rasterio.features
from tqdm import tqdm


def process_file(full_path_str):
    try:
        full_path = Path(full_path_str)
        with rasterio.open(full_path) as src:
            # print(f"reading {full_path}")
            band = src.read(1, masked=True)
            valid_mask = ~band.mask
            data = np.ones(band.shape, dtype=np.uint8)
            # print(f"generating shapes for {full_path}")
            shapes_gen = rasterio.features.shapes(
                data, mask=valid_mask, transform=src.transform
            )
            geoms = [shape(geom) for geom, value in shapes_gen if value == 1]
            # print(f"unioning {len(geoms)} geometries for {full_path}")
            if geoms:
                valid_geom = unary_union(geoms)
                bbox = list(valid_geom.bounds)
            else:
                valid_geom = box(*src.bounds)
                bbox = [
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top,
                ]
            # print(f"done with {full_path}")
    except Exception as e:
        print(f"exception for {full_path}: {e}")

    # print(f"creating item for {full_path}")
    item = pystac.Item(
        id=full_path.stem,
        geometry=mapping(valid_geom),
        bbox=bbox,
        datetime=None,
        properties={},
        start_datetime=datetime.datetime(
            2000, 3, 1, tzinfo=datetime.timezone.utc
        ),
        end_datetime=datetime.datetime(
            2013, 11, 30, tzinfo=datetime.timezone.utc
        ),
    )
    # print(f"adding item for {full_path}")
    item.add_asset(
        key="data",
        asset=pystac.Asset(
            href=str(full_path), media_type=pystac.MediaType.COG
        ),
    )
    return item


def create_catalog(root_dir, catalog_id, catalog_description):
    catalog = pystac.Catalog(id=catalog_id, description=catalog_description)
    collection = pystac.Collection(
        id="dem-collection",
        description="Pre-routed DEM slices aligned with HUC05s",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180, -90, 180, 90]]),
            temporal=pystac.TemporalExtent([[None, None]]),
        ),
    )
    catalog.add_child(collection)
    root_path = Path(root_dir).resolve()

    file_paths = []
    for root, _, files in os.walk(root_path):
        for fname in files:
            if fname.lower().endswith((".tif", ".tiff")):
                full_path = Path(root) / fname
                file_paths.append(str(full_path))

    with ProcessPoolExecutor(
        max_workers=psutil.cpu_count(logical=False)
    ) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in file_paths}
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                item = future.result()
                collection.add_item(item)
                pbar.update(1)

    catalog.normalize_and_save(
        root_href="stac-catalog", catalog_type=pystac.CatalogType.SELF_CONTAINED
    )


if __name__ == "__main__":
    create_catalog(
        root_dir=r"D:\repositories\dem_precondition\ASTGTM_mfd_routed",
        catalog_id="astgtm-pre-route",
        catalog_description="DEM slices aligned with HUC05s.",
    )
