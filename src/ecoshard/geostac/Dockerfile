FROM python:3.9-slim

# Install system dependencies for geopandas and spatial libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the application code
COPY geostac_app.py .

# Install required Python packages
RUN pip install --no-cache-dir Flask geopandas pystac shapely

EXPOSE 5000

CMD ["python", "geostac_app.py"]
