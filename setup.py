from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from version import get_version


here = path.dirname(__file__)

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = "For more information visit https://github.com/mikecokina/nafnetlib/blob/master/README.md"

setup(
    name='nafnetlib',
    src_root='src',
    version=get_version(),

    description='A streamlined implementation of inference for the megvii-research/NAFNet model, '
                'focusing solely on straightforward prediction tasks without additional '
                'features like training or fine-tuning.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/mikecokina/nafnetlib',

    # Author details
    author='Michal Cokina',
    author_email='mikecokina@gmail.com',

    # Choose your license
    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    # What does your project relate to?
    keywords='nafnet, artificial intelligence, image processing, computer vision, deep learning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(where='src'),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "torch>=2.3.0,<=2.5.1",
        "torchvision>=0.18.0,<=0.20.1",
        "numpy>=1.26.4,<=2.1.3",
        "opencv-python>=4.8.1.78,<=4.10.0.84",
        "pillow>=10.4.0,<=11.0.0"","
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [],
        'test': [],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={},

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [],
    },
)
