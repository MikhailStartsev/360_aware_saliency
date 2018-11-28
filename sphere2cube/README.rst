sphere2cube
===========

``sphere2cube`` is a Python script to map  equirectangular
(cylindrical projection, skysphere) map into 6 cube (cubemap, skybox)
faces.

Usage
=====

::

    $ sphere2cube -h
    usage: sphere2cube [-h] [-v] [-r <size>] [-R <rx> <ry> <rz>] [-p <pattern>]
                       [-o <dir>] [-f <name>] [-b <path>] [-t <count>] [-V]
                       [<source>]
    
    Maps an equirectangular (cylindrical projection, skysphere) map into 6 cube
    (cubemap, skybox) faces.
    
    positional arguments:
      <source>              source equirectangular image filename
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
      -r <size>, --resolution <size>
                            resolution for each generated cube face (defaults to 1024)
      -R <rx> <ry> <rz>, --rotation <rx> <ry> <rz>
                            rotation in degrees to apply before rendering cube
                            faces (z is up)
      -F <angle>, --fov <angle>
                            field of view of camera used for rendering cube faces
      -p <pattern>, --path <pattern>
                            filename pattern for rendered faces: default is
                            "face_%n_%r", where %n is replaced by the face number
                            and %r by the resolution
      -o <dir>, --output-dir <dir>
                            directory to save rendered faces to (it must already
                            exist)
      -f <name>, --format <name>
                            format to use when saving faces, i.e. "PNG" or "TGA"
      -b <path>, --blender-path <path>
                            filename of the Blender executable (defaults to
                            "blender")
      -t <count>, --threads <count>
                            number of threads to use when rendering (1-64)
      -V, --verbose         enable verbose logging

Supported output formats depend on the Blender installation, but will
generally be TGA, IRIS, JPEG, MOVIE, IRIZ, RAWTGA, AVIRAW, AVIJPEG, PNG,
BMP, and FRAMESERVER.

``sphere2cube`` can be run in a headless environment (e.g., a
server).

Examples
========

For instance, to render a 2048-resolution TGA cubemap from
``source.jpg``, we could use the following command:

::

    $ sphere2cube source.jpg -r2048 -fTGA

This would generate ``face_1_2048.tga``, …, ``face_6_2048.tga`` in the
working directory.

Installation
============

``sphere2cube`` can be easily installed with ``pip``.

It assumes that Blender is installed and the ``blender`` executable is listed in the system PATH environment variable. If it is not possible for PATH to be edited (as in the case of an unprivileged user), the path to the ``blender`` executable may instead be passed through the ``-b`` flag.

Windows
-------

Install `Blender`_, and add ``blender.exe`` to ``PATH``. Finally,

::

    pip install sphere2cube

Linux
-----

::

    $ apt-get install blender
    $ pip install sphere2cube

Mac OS X
--------

Similar to Windows, install `Blender`_, and add the ``blender`` executable to ``$PATH``. Then,

::

    $ pip install sphere2cube


.. _Blender: https://www.blender.org/
