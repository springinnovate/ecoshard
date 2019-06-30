"""Build CV Styles."""
import logging
import sys

from jinja2 import Environment, FileSystemLoader
import numpy
from osgeo import ogr

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=('%(message)s'),
    stream=sys.stdout)
logging.getLogger('ecoshard').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

FIELD_NAMES = [
    'Rhab_cur',
    'Rwind',
    'Rwave',
    'Rrelief',
    'Rsurge',
    'Rslr_cur',
    'SLRrise_ssp1',
    'Rhab_ssp1',
    'curpb_ssp1',
    'cpdn_ssp1',
    'Rslr_ssp1',
    'SLRrise_ssp3',
    'Rhab_ssp3',
    'curpb_ssp3',
    'cpdn_ssp3',
    'Rslr_ssp3',
    'SLRrise_ssp5',
    'Rhab_ssp5',
    'curpb_ssp5',
    'cpdn_ssp5',
    'Rslr_ssp5',
    'pdnrc_ssp1',
    'pdnrc_ssp3',
    'pdnrc_ssp5',
    'Service_cur',
    'NCP_cur',
    'Service_ssp1',
    'NCP_ssp1',
    'Service_ssp3',
    'NCP_ssp3',
    'Service_ssp5',
    'NCP_ssp5',
    'Rt_cur',
    'Rt_ssp1',
    'Rt_ssp3',
    'Rt_ssp5',
    'Rtnohab_cur',
    'Rtnohab_ssp1',
    'Rtnohab_ssp3',
    'Rtnohab_ssp5',
]

CV_VECTOR_PATH = 'cv_coastal_points_output_md5_69641307c3c7b4c7d23faa8637e30f83.gpkg'


def main():
    """Entry point."""
    for field_name in FIELD_NAMES:
        vector = ogr.Open(CV_VECTOR_PATH)
        layer = vector.GetLayer()
        layer_iter = iter(layer)
        first_feature = next(layer_iter)
        min_val = first_feature.GetField(field_name)
        max_val = first_feature.GetField(field_name)
        for feature in layer_iter:
            val = feature.GetField(field_name)
            if val < min_val:
                min_val = val
            elif val > max_val:
                max_val = val
        LOGGER.info('%s %s %s', field_name, min_val,  max_val)

        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('cv_style_template.xml')
        steps = numpy.linspace(min_val, max_val, 6)
        output_from_parsed_template = template.render(
            field_name=field_name,
            step_1=min_val,
            step_2=steps[1],
            step_3=steps[2],
            step_4=steps[3],
            step_5=steps[4],
            step_6=max_val)
        print(output_from_parsed_template)
        # to save the results
        with open("%s.xml" % field_name, "w") as fh:
            fh.write(output_from_parsed_template)


if __name__ == '__main__':
    main()
