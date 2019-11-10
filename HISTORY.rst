Release History
===============

Unreleased Changes
------------------
* Fixed an issue that would cause the download rate to be under estimated.

0.3.1 (2019/10/07)
------------------
* Fixing an issue in download_to_url where the file might not be flushed and the
  final log message is not printed.

0.3.0 (2019/09/26)
------------------
* Added a new command line mode –reduce_factor that reduces the number of pixels
  in a raster by that integer amount. Ex:

    ``python -m ecoshard base.tif --reduce_factor 4 max target.tif``

    this call makes the size of the pixels in base.tif 4 times larger on the
    edge, thus reducing the total size of the image by 16 times, the convolution
    upsample is done with a "max" and the output file is ``target.tif``. The
    modes "min", "max", "sum", "average", "mode" are available.

0.2.2 (2019/09/24)
------------------
* Added a ``download_url`` function to ``ecoshard`` to fetch files via
  http(s).
* Developing Flask module to visualize ecoshards.

0.2.1 (2019/06/28)
------------------
* Hotfix: gs to gsutil for copying to buckets.

0.2.0 (2019/06/28)
------------------
* Added functionality to download a url.
* Added functionality to copy to a google bucket.
* Fixed an issue on functions that should write a token file but didn't.

0.1.1 (2019/06/27)
------------------
* Added a ``--force`` flag to the command line utility.

0.0.2 (2019/06/26)
------------------
* Initial release.
