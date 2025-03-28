from flask import Flask, request, jsonify
import geopandas as gpd
import pystac
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.prepared import prep

app = Flask(__name__)

COUNTRIES_VECTOR_PATH = r"D:\Users\richp\Downloads\critical_natural_assets_cna_overlap_run_2024_03_13\critical_natural_assets\data\countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg"
CATALOG_PATH = "stac-catalog/catalog.json"

# Load static resources at startup
countries = gpd.read_file(COUNTRIES_VECTOR_PATH)
catalog = pystac.read_file(CATALOG_PATH)
collection = catalog.get_child("dem-collection")


@app.route("/query", methods=["GET"])
def query():
    iso3_param = request.args.get("iso3")
    if not iso3_param:
        return (
            jsonify({"error": "Missing iso3 parameter. Use ?iso3=CODE1,CODE2"}),
            400,
        )

    selected_iso3 = [code.strip() for code in iso3_param.split(",")]
    subset_countries = countries[countries["iso3"].isin(selected_iso3)]
    if subset_countries.empty:
        return (
            jsonify({"error": "No countries found for provided iso3 codes"}),
            404,
        )

    search_geom = unary_union(subset_countries.geometry)
    prepared_search_geom = prep(search_geom)

    matched_items = [
        item
        for item in collection.get_items()
        if prepared_search_geom.intersects(shape(item.geometry))
    ]
    asset_paths = [item.assets["data"].href for item in matched_items]
    return jsonify({"asset_paths": asset_paths})


if __name__ == "__main__":
    app.run(debug=True)
