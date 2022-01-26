
def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    cfiles = ["unitdtype_main.c", "dtype.c", "casts.c", "scalar.c", "umath.c",
              "additional_numeric.casts.c.src"]
    cfiles = ["unitdtype/src/" + f for f in cfiles]

    config.add_subpackage('unitdtype')

    config.add_extension(
        'unitdtype._unitdtype_main',
        sources=cfiles,
        include_dirs=[numpy.get_include()],)
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        name="unitdtype",
        configuration=configuration,)

