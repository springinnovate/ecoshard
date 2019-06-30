"""Load rasters into Geoserver."""
import os
import sys
import logging
import argparse
import glob
import re

import requests

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=('%(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)

REQUEST_TIMEOUT = 5.0


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Ecoshard geoserver.')
    parser.add_argument('username', help='username')
    parser.add_argument('password', help='password')
    args = parser.parse_args()

    session = requests.Session()
    session.auth = (args.username, args.password)

    for filepath in glob.glob(
            '/mnt/disks/geoserver_data/data/*.tif'):
        LOGGER.debug(filepath)
        raster_name = re.match(
            '^(.*)_compressed', os.path.basename(filepath)).group(1)
        payload = {
            "coverageStore": {
                "name": raster_name,
                "url": "file:%s" % filepath
            }
        }
        url = 'http://localhost:8080/geoserver/rest/workspaces/ipbes/coveragestores'
        response = requests.post(url, json=payload)
        LOGGER.info(response.text)
        break


if __name__ == '__main__':
    main()
