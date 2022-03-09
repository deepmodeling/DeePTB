
from os import path
import setuptools, datetime
from setuptools import find_packages
NAME = "dpnegf"
today = datetime.date.today().strftime("%b-%d-%Y")
with open(path.join('dpnegf', '_date.py'), 'w') as fp :
    fp.write('date = \'%s\'' % today)

install_requires=["numpy",
                  "scipy",
                  "spglib",
                  "matplotlib",
                  "ase",
                  "torch",
                  "torchsort"]

def setup(scm=None):
    packages = find_packages()

    setuptools.setup(
        name=NAME,
        use_scm_version=scm,
        setup_requires=['setuptools_scm'],
        author="Q. Gu, et al.",
        author_email="guqq@pku.edu.cn",
        description="A deep learning package for non-equilibrium Green's function (NEGF) approach.",
        url="https://gitlab.com/deeptransport/deepnegf",
        python_requires=">=3.8",
        packages=packages,
        keywords='AlgorithmEngineering',
        install_requires=install_requires,
        entry_points={'console_scripts': ['dpnegf= dpnegf.main:main']}
    )
try:
    setup(scm={'write_to': 'dpnegf/_version.py'})
except:
    setup(scm=None)