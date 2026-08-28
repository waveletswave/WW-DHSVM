"""Configuration and global defaults."""

import os
import subprocess
import configparser
import getpass


def getHome() -> str:
    return os.path.expanduser('~')


def getDefaultConfig() -> configparser.ConfigParser:
    """Dictionary of all config option defaults.

    Returns
    -------
    rcParams : configparser.ConfigParser
      A dict-like object containing parameters.
    """
    rcParams = configparser.ConfigParser()

    rcParams['DEFAULT']['data_directory'] = ""
    rcParams['DEFAULT']['ssl_cert'] = "True"  # note this can be True,
    # False (bad
    # idea/permissive) or a
    # path to ssl certs,
    # e.g. /etc/ssl/cert.perm
    # or similar
    rcParams['DEFAULT']['proj_network'] = "False"

    rcParams.add_section('AppEEARS')
    rcParams['AppEEARS']['username'] = 'NOT_PROVIDED'
    rcParams['AppEEARS']['password'] = 'NOT_PROVIDED'

    rcParams.add_section('HFHydrodata')
    rcParams['HFHydrodata']['email'] = 'NOT_PROVIDED'
    rcParams['HFHydrodata']['pin'] = 'NOT_PROVIDED'
    return rcParams


def getConfig() -> configparser.ConfigParser:
    try:
        data_directory = os.path.join(os.environ['WW_DHSVM_DATA_DIR'])
    except KeyError:
        data_directory = os.path.join(os.getcwd(), 'data')
    rc = getDefaultConfig()
    rc['DEFAULT']['data_directory'] = data_directory

    # paths to search for rc files
    rc_paths = [
        os.path.join(getHome(), '.ww_dhsvmrc'),
        os.path.join(os.getcwd(), '.ww_dhsvmrc'),
        os.path.join(os.getcwd(), 'ww_dhsvmrc'),
    ]

    # this is a bit fragile -- it checks if the user is the docker user
    if getpass.getuser() == 'jovyan':
        rc_paths.append('/home/jovyan/workdir/.docker_ww_dhsvmrc')

    # read the rc files
    rc.read(rc_paths)
    return rc


def setDataDirectory(path : str) -> None:
    """Sets the directory in which all data is stored."""
    rcParams['DEFAULT']['data_directory'] = path


# global config
rcParams = getConfig()
