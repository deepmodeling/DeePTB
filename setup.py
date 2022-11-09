from os import path
import setuptools, datetime
from setuptools import find_packages
NAME = "dptb"
today = datetime.date.today().strftime("%b-%d-%Y")
with open(path.join('dptb', '_date.py'), 'w') as fp :
    fp.write('date = \'%s\'' % today)

# "torch" is not included here!
install_requires=["numpy",
                  "scipy",
                  "spglib",
                  "matplotlib",
                  "ase"]

def setup(scm=None):
    packages = find_packages()

    setuptools.setup(
        name=NAME,
        use_scm_version=scm,
        setup_requires=['setuptools_scm'],
        author="Q. Gu, et al.",
        author_email="guqq@pku.edu.cn",
        description="A deep learning package for emperical tight-binding approach with first-principle accuracy.",
        url="https://gitlab.com/deeptransport/deepnegf",
        python_requires=">=3.8",
        packages=packages,
        keywords='DeepLearningTightBinding',
        install_requires=install_requires,
        entry_points={'console_scripts': ['dptb=dptb.__main__:main']}
    )
try:
    setup(scm={'write_to': 'dptb/_version.py'})
except:
    setup(scm=None)
