"""Ecoshard test suite."""
import os
import tempfile
import shutil
import unittest

import ecoshard


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
        try:
            os.makedirs(working_dir)
        except OSError:
            pass
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
        try:
            os.makedirs(working_dir)
        except OSError:
            pass
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

    def test_force(self):
        """Test ecoshard.hash_file with a force rename."""
        working_dir = self.workspace_dir
        try:
            os.makedirs(working_dir)
        except OSError:
            pass
        base_path = os.path.join(working_dir, 'test_file_sha_fffffffffff.txt')
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
        try:
            os.makedirs(working_dir)
        except OSError:
            pass
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
        base_path = os.path.join(working_dir, 'test_file_sha_fffffffffff.txt')
        with self.assertRaises(ValueError) as cm:
            ecoshard.hash_file(
                base_path, target_token_path=target_token_path,
                target_dir=None, rename=True,
                hash_algorithm='md5', force=False)
        self.assertTrue('already be an ecoshard' in str(cm.exception))

    def test_validate_hash(self):
        """Test ecoshard.validate_hash."""
        working_dir = self.workspace_dir
        try:
            os.makedirs(working_dir)
        except OSError:
            pass
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
