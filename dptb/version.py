import os
import subprocess as sp
import toml, os

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
# VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

entrypoint_path = os.path.dirname(os.path.abspath(__file__))
pyproject_path = os.path.join(entrypoint_path, "..", "pyproject.toml") 

with open(pyproject_path, "r") as f:
    pyproject = toml.load(f)
    VERSION = pyproject["tool"]["poetry"]["version"]


# Return the git revision as a string
# taken from numpy/numpy
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def _get_git_version():
    cwd = os.getcwd()

    # go to the main directory
    fdir = os.path.dirname(os.path.abspath(__file__))
    maindir = os.path.abspath(os.path.join(fdir, ".."))
    # maindir = fdir # os.path.join(fdir, "..")
    os.chdir(maindir)

    # get git version
    res = git_version()

    # restore the cwd
    os.chdir(cwd)
    return res

def get_version(build_version=False):
    if ISRELEASED:
        return VERSION

    # unreleased version
    GIT_REVISION = _get_git_version()
    if build_version:
        import datetime as dt
        date = dt.date.strftime(dt.datetime.now(), "%Y%m%d%H%M%S")
        return VERSION + ".dev" + date
    else:
        return VERSION + ".dev0+" + GIT_REVISION[:7]
    
if __name__ == "__main__":
    GIT_REVISION = _get_git_version()
    print(GIT_REVISION)
    print(get_version())