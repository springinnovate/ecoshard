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
    'coastal_deficit_cur',
    'coastal_deficit_ssp1',
    'coastal_deficit_ssp3',
    'coastal_deficit_ssp5',
    'coastal_NC_cur',
    'coastal_NC_ssp1',
    'coastal_NC_ssp3',
    'coastal_NC_ssp5',
    'coastal_potential_cur',
    'coastal_potential_ssp1',
    'coastal_potential_ssp3',
    'coastal_potential_ssp5',
    'coastal_pop_cur',
    'coastal_pop_ssp1',
    'coastal_pop_ssp3',
    'coastal_pop_ssp5',
    'coastal_deficit_change_ssp1',
    'coastal_deficit_change_ssp3',
    'coastal_deficit_change_ssp5',
    'coastal_NC_change_ssp1',
    'coastal_NC_change_ssp3',
    'coastal_NC_change_ssp5',
    'coastal_potential_change_ssp1',
    'coastal_potential_change_ssp3',
    'coastal_potential_change_ssp5',
    'coastal_pop_change_ssp1',
    'coastal_pop_change_ssp3',
    'coastal_pop_change_ssp5',
]

CV_VECTOR_PATH = '../cv_coastal_points_ipbes_md5_d1cc9481caf145c04fc14679265ed459.gpkg'


def main():
    """Entry point."""
    vector = ogr.Open(CV_VECTOR_PATH)
    layer = vector.GetLayer()
    layer_iter = iter(layer)
    first_feature = next(layer_iter)
    min_max_field_map = {}
    for field_name in FIELD_NAMES:
        min_max_field_map[field_name] = {
            'min': first_feature.GetField(field_name),
            'max': first_feature.GetField(field_name),
        }
    LOGGER.debug("process layer")
    for feature in layer_iter:
        for field_name in FIELD_NAMES:
            val = feature.GetField(field_name)
            if val < min_max_field_map[field_name]['min']:
                min_max_field_map[field_name]['min'] = val
            elif val > min_max_field_map[field_name]['max']:
                min_max_field_map[field_name]['max'] = val

    LOGGER.debug('write sld files')
    for field_name in FIELD_NAMES:
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('cv_style_template.xml')
        steps = numpy.linspace(
            min_max_field_map[field_name]['min'],
            min_max_field_map[field_name]['max'],
            6)
        output_from_parsed_template = template.render(
            field_name=field_name,
            step_1=steps[0],
            step_2=steps[1],
            step_3=steps[2],
            step_4=steps[3],
            step_5=steps[4],
            step_6=steps[5])
        # to save the results
        with open("%s.sld" % field_name, "w") as fh:
            fh.write(output_from_parsed_template)


if __name__ == '__main__':
    main()
