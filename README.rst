This module is used to ecoshard sets of files.

Usage:

``python -m ecoshard \[original\_file\] --hashalg md5 --rename``

(creates an ecoshard from original file with the md5 hash algorithm and renames the result rather than creating a new copy)

``python -m ecoshard *.tif --compress --buildoverviews``

(does a GIS compression of all *.tif files in the current directory and builds overviews for them and renames the result rather than making a new copy. Here if --rename had been passed an error would have been raised because rasters cannot be in-place compressed. The target output files will have the format \[original\_filename\]\_compressed\_\[hashalg\]\_\[ecoshard\]\[fileext\])

``python -m ecoshard *.tif --compress --buildoverviews --upload``

(does the previous operation but also uploads the results to gs://ecoshard-root/working-shards and reports the target URLs to stdout)

``python -m ecoshard *.tif ./outputs/*.tif --validate``

(searches the *.tif and ./outputs/*.tif globs for ecoshard files and reports whether their hashes are valid or not)
