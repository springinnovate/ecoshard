"""Ecoshard test suite."""
import os
import pathlib
import shutil
import tempfile
import time
import unittest
from zipfile import ZipFile

from osgeo import gdal
from osgeo import osr
import ecoshard
import numpy


def _build_test_raster(raster_path):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    gtiff_driver = gdal.GetDriverByName('GTiff')
    n = 100
    new_raster = gtiff_driver.Create(
        raster_path, n, n, 1, gdal.GDT_Int32, options=[
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=NONE',
            'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
    new_raster.SetProjection(srs.ExportToWkt())
    new_raster.SetGeoTransform([100.0, 1.0, 0.0, 100.0, 0.0, -1.0])
    new_band = new_raster.GetRasterBand(1)
    new_band.SetNoDataValue(-1)
    array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
    new_band.WriteArray(array)
    new_raster.FlushCache()
    new_band = None
    new_raster = None


class EcoShardTests(unittest.TestCase):
    """Tests for the PyGeoprocesing 1.0 refactor."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_hash_file(self):
        """Test ecoshard.hash_file."""
        working_dir = self.workspace_dir
        base_path = os.path.join(working_dir, 'test_file.txt')
        target_token_path = '%s.COMPLETE' % base_path

        with open(base_path, 'w') as base_file:
            base_file.write('test')

        # make a different target dir so it gets copied
        target_dir = os.path.join(working_dir, 'output')
        ecoshard.hash_file(
            base_path, target_token_path=target_token_path,
            target_dir=target_dir, rename=False,
            hash_algorithm='md5', force=False)

        expected_file_path = os.path.join(
            target_dir, 'test_file_md5_098f6bcd4621d373cade4e832627b4f6.txt')
        self.assertTrue(os.path.exists(expected_file_path))
        self.assertTrue(os.path.exists(target_token_path))

    def test_hash_file_rename(self):
        """Test ecoshard.hash_file with a rename."""
        working_dir = self.workspace_dir
        base_path = os.path.join(working_dir, 'test_file.txt')
        target_token_path = '%s.COMPLETE' % base_path

        with open(base_path, 'w') as base_file:
            base_file.write('test')

        ecoshard.hash_file(
            base_path, target_token_path=target_token_path,
            target_dir=None, rename=True,
            hash_algorithm='md5', force=False)

        expected_file_path = os.path.join(
            working_dir, 'test_file_md5_098f6bcd4621d373cade4e832627b4f6.txt')
        self.assertTrue(os.path.exists(expected_file_path))
        # indicates the file has been renamed
        self.assertFalse(os.path.exists(base_path))

    def test_hash_file_short(self):
        """Test ecoshard.hash_file with a rename."""
        working_dir = self.workspace_dir
        base_path = os.path.join(working_dir, 'test_file.txt')
        target_token_path = '%s.COMPLETE' % base_path

        with open(base_path, 'w') as base_file:
            base_file.write('test')

        ecoshard.hash_file(
            base_path, target_token_path=target_token_path,
            target_dir=None, rename=True,
            hash_algorithm='md5', force=False, hash_length=5)

        expected_file_path = os.path.join(
            working_dir, 'test_file_md5_098f6.txt')
        self.assertTrue(os.path.exists(expected_file_path))
        # indicates the file has been renamed
        self.assertFalse(os.path.exists(base_path))

    def test_force(self):
        """Test ecoshard.hash_file with a force rename."""
        working_dir = self.workspace_dir
        base_path = os.path.join(
            working_dir, 'test_file_sha224_fffffffffff.txt')
        target_token_path = '%s.COMPLETE' % base_path

        with open(base_path, 'w') as base_file:
            base_file.write('test')

        ecoshard.hash_file(
            base_path, target_token_path=target_token_path,
            target_dir=None, rename=True,
            hash_algorithm='md5', force=True)

        expected_file_path = os.path.join(
            working_dir, 'test_file_md5_098f6bcd4621d373cade4e832627b4f6.txt')
        self.assertTrue(os.path.exists(expected_file_path))
        # indicates the file has been renamed
        self.assertFalse(os.path.exists(base_path))

    def test_exceptions_in_hash(self):
        """Test ecoshard.hash_file raises exceptions in bad cases."""
        working_dir = self.workspace_dir
        base_path = os.path.join(working_dir, 'test_file.txt')
        target_token_path = '%s.COMPLETE' % base_path

        with open(base_path, 'w') as base_file:
            base_file.write('test')

        # test that target dir defined and rename True raises an exception
        with self.assertRaises(ValueError) as cm:
            ecoshard.hash_file(
                base_path, target_token_path=target_token_path,
                target_dir='output', rename=True,
                hash_algorithm='md5', force=False)
        self.assertTrue('but rename is True' in str(cm.exception))

        # test that a base path already in ecoshard format raises an exception
        base_path = os.path.join(
            working_dir, 'test_file_sha224_fffffffffff.txt')
        with self.assertRaises(ValueError) as cm:
            ecoshard.hash_file(
                base_path, target_token_path=target_token_path,
                target_dir=None, rename=True,
                hash_algorithm='md5', force=False)
        self.assertTrue('already be an ecoshard' in str(cm.exception))

    def test_validate_hash(self):
        """Test ecoshard.validate_hash."""
        working_dir = self.workspace_dir
        # we know the hash a priori, just make the file
        base_path = os.path.join(
            working_dir, 'test_file_md5_098f6bcd4621d373cade4e832627b4f6.txt')
        with open(base_path, 'w') as base_file:
            base_file.write('test')
        self.assertTrue(ecoshard.validate(base_path))

        # test that files that are not in ecoshard format raise an exception
        with self.assertRaises(ValueError) as cm:
            not_an_ecoshard_path = 'test.txt'
            ecoshard.validate(not_an_ecoshard_path)
        self.assertTrue('does not match an ecoshard' in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            new_file_path = os.path.join(
                working_dir,
                'test_file_md5_098f6bcd4621d373cade4e832627b4f5.txt')
            shutil.copyfile(base_path, new_file_path)
            ecoshard.validate(new_file_path)
        self.assertTrue('hash does not match' in str(cm.exception))

    def test_build_overviews(self):
        """Test ecoshard.build_overviews."""
        raster_path = os.path.join(self.workspace_dir, 'test_raster.tif')
        _build_test_raster(raster_path)

        target_token_path = '%s.COMPLETE' % raster_path
        ecoshard.build_overviews(
            raster_path, target_token_path=target_token_path,
            interpolation_method='near')
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        overview_count = band.GetOverviewCount()
        band = None
        raster = None
        self.assertEqual(overview_count, 6)

        with self.assertLogs('ecoshard', level='WARNING') as cm:
            ecoshard.build_overviews(
                raster_path, target_token_path=target_token_path,
                interpolation_method='near')
        self.assertEqual(cm.output, [
            'WARNING:ecoshard.ecoshard:overviews already exist, set '
            'rebuild_if_exists=False to rebuild them anyway'])

    def test_build_overviews_external(self):
        """Test ecoshard.build_overviews external."""
        raster_path = os.path.join(self.workspace_dir, 'test_raster.tif')
        _build_test_raster(raster_path)

        target_token_path = '%s.COMPLETE' % raster_path
        ecoshard.build_overviews(
            raster_path, target_token_path=target_token_path,
            interpolation_method='near', overview_type='external')
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        overview_count = band.GetOverviewCount()
        band = None
        raster = None
        self.assertEqual(overview_count, 6)

    def test_build_overviews_error(self):
        """Test ecoshard.build_overviews error handling."""
        bad_raster_path = os.path.join(self.workspace_dir, 'no_raster_here.tif')
        # test that target dir defined and rename True raises an exception
        with self.assertRaises(ValueError) as cm:
            ecoshard.build_overviews(bad_raster_path)
        print(cm.exception)
        self.assertTrue('could not open' in str(cm.exception))

        raster_path = os.path.join(self.workspace_dir, 'test_raster.tif')
        _build_test_raster(raster_path)

        with self.assertRaises(ValueError) as cm:
            ecoshard.build_overviews(
                raster_path, overview_type='badname')
        self.assertTrue('invalid value for overview_type' in str(cm.exception))

    def test_compress_raster(self):
        """Test ecoshard.compress_raster."""
        raster_path = os.path.join(self.workspace_dir, 'test_raster.tif')
        _build_test_raster(raster_path)
        compressed_raster_path = os.path.join(
            self.workspace_dir, 'test_raster_compressed.tif')

        ecoshard.compress_raster(
            raster_path, compressed_raster_path, compression_algorithm='LZW',
            compression_predictor=2)

        # if its compressed, it should be smaller!
        self.assertTrue(
            os.path.getsize(compressed_raster_path) <
            os.path.getsize(raster_path))

    def test_make_logger_callback(self):
        """Test general logger callback function."""
        timeout = 0.0001
        callback = ecoshard._make_logger_callback(
            'test_message %.1f%% complete %s', 0.0001)
        with self.assertLogs('ecoshard', level='INFO') as cm:
            time.sleep(timeout*1111)
            callback(1, None, ['blap'])
            time.sleep(timeout*1110)
            callback(1, None, ['barp'])
        self.assertTrue('test_message' in cm.output[0])
        self.assertTrue('barp' in cm.output[1])

    def test_download_url(self):
        """Test ecoshard.download_url."""
        base_file_path = os.path.join(self.workspace_dir, 'source.txt')
        content = 'xxxxxxxxxxfejoiwfweojifwejoifwejoi'
        with open(base_file_path, 'w') as testfile:
            testfile.write(content)
        uri_file_path = pathlib.Path(base_file_path).as_uri()
        target_path = os.path.join(self.workspace_dir, 'output.txt')
        ecoshard.download_url(
            uri_file_path, target_path, skip_if_target_exists=False)
        with open(target_path, 'r') as targetfile:
            self.assertEqual(content, targetfile.read())

        # make sure it doesn't download again
        with open(base_file_path, 'w') as testfile:
            testfile.write('overwritten content')
        ecoshard.download_url(
            uri_file_path, target_path, skip_if_target_exists=True)
        with open(target_path, 'r') as targetfile:
            self.assertEqual(content, targetfile.read())

        # test that target dir defined and rename True raises an exception
        fake_uri_file_path = pathlib.Path(os.path.join(
            self.workspace_dir, 'fake.txt')).as_uri()
        with self.assertRaises(Exception) as cm:
            with self.assertLogs('ecoshard', level='INFO') as cm:
                ecoshard.download_url(fake_uri_file_path, target_path)
        self.assertTrue('unable to download' in cm.output[0])

    def test_download_and_unzip(self):
        """Test ecoshard.download_and_unzip."""
        base_file_path = os.path.join(self.workspace_dir, 'source.txt')
        content = 'xxxxxxxxxxfejoiwfweojifwejoifwejoi'
        with open(base_file_path, 'w') as testfile:
            testfile.write(content)

        zip_path = os.path.join(self.workspace_dir, 'source.zip')
        with ZipFile(zip_path,'w') as zip_file:
            zip_file.write(base_file_path, arcname=os.path.basename(base_file_path))

        uri_zip_file_path = pathlib.Path(zip_path).as_uri()
        unzip_dir = os.path.join(self.workspace_dir, 'unzip')
        token_path = os.path.join(self.workspace_dir, 'token.txt')
        os.makedirs(unzip_dir)
        ecoshard.download_and_unzip(
            uri_zip_file_path, unzip_dir, target_token_path=token_path)
        self.assertTrue(os.path.exists(token_path))
        print(os.listdir(unzip_dir))
        expected_file_path = os.path.join(unzip_dir, os.path.basename(
            base_file_path))
        with open(expected_file_path, 'r') as targetfile:
            self.assertEqual(content, targetfile.read())

