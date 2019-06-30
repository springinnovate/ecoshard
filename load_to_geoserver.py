"""Load rasters into Geoserver."""
import sys
import logging
import argparse
import urllib.request

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=('%(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Ecoshard geoserver.')
    parser.add_argument('username', help='username')
    parser.add_argument('password', help='password')
    args = parser.parse_args()

    auth_handler = urllib.request.HTTPBasicAuthHandler()
    auth_handler.add_password(
        realm=None,
        uri='http://localhost/',
        user=args.username,
        passwd=args.password)

    opener = urllib.request.build_opener(auth_handler)
    urllib.request.install_opener(opener)

    req = urllib.request.Request(
        url='http://localhost:8080/geoserver/rest/workspaces/cv_coastal_points_output_md5_69641307c3c7b4c7d23faa8637e30f83/styles/rhab.json',
        method='GET')
    x = urllib.request.urlopen(req.text)
    LOGGER.info(x)


if __name__ == '__main__':
    main()
