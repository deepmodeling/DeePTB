from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import os
import logging
from dptb.utils.gen_inputs import gen_inputs

__all__ = ["get_full_config", "config"]
log = logging.getLogger(__name__)

def get_full_config(model, train, test, e3tb, sktb, sktbenv):
    """
    This function determines the appropriate full config based on the provided parameters.

    Args:
        train (bool): Whether it's training mode.
        test (bool): Whether it's testing mode.
        e3tb (bool): Whether E3TB configuration is needed.
        sktb (bool): Whether SKTB configuration is needed.
        sktbenv (bool): Whether SKTB environment correction is needed.

    Returns:
        dict: The appropriate full configuration dictionary.

    Raises:
        ValueError: If none of the config type flags (e3tb, sktb, sktbenv) are True or
                    if both train and test are True.
    """
    name = ''
    if train:
        name += 'train'
        
        # Use train configs based on e3tb, sktb, sktbenv
        if e3tb:
            name += '_E3'
            full_config = gen_inputs(mode='e3', task='train', model=model)
        elif sktb:
            name += '_SK'
            full_config = gen_inputs(mode='sk', task='train', model=model)
        elif sktbenv:
            name += '_SKEnv'
            full_config = gen_inputs(mode='skenv', task='train', model=model)
        else:
            logging.error("Unknown config type in training mode")
            raise ValueError("Unknown config type in training mode")
    elif test:
        # Use test configs based on e3tb, sktb, sktbenv
        name += 'test'
        if e3tb:
            name += '_E3'
            full_config = gen_inputs(mode='e3', task='test', model=model)
        elif sktb:
            name += '_SK'
            full_config = gen_inputs(mode='sk', task='test', model=model)
        elif sktbenv:
            name += '_SKEnv'
            full_config = gen_inputs(mode='skenv', task='test', model=model)
        else:
            logging.error("Unknown config type in testing mode")
            raise ValueError("Unknown config type in testing mode")
    else:
        logging.error("Unknown mode")
        raise ValueError("Unknown mode")
    return name, full_config


def config(
        PATH: str,
        train: bool = True,  # Set default train mode
        test: bool = False,
        e3tb: bool = False,
        sktb: bool = False,
        sktbenv: bool = False,
        model: str = None,
        log_level: int = logging.INFO,
        log_path: Optional[str] = None,
        **kwargs
):
    """
    This function generates and saves a full configuration based on user input.

    Args:
        PATH (str): Path to save the configuration file.
        train (bool, optional): Whether it's training mode (default: True).
        test (bool, optional): Whether it's testing mode (default: False).
        e3tb (bool, optional): Whether E3TB configuration is needed.
        sktb (bool, optional): Whether SKTB configuration is needed.
        sktbenv (bool, optional): Whether SKTB environment correction is needed.
        log_level (int, optional): Logging level (default: logging.INFO).
        log_path (Optional[str], optional): Path to log file (default: None).
        **kwargs: Additional keyword arguments (unused in this implementation).

    Returns:
        int: 0 on success, 1 on error.

    Raises:
        ValueError: If none of the config type flags (e3tb, sktb, sktbenv) are True or
                    if both train and test are True.
  """
    if not any((e3tb, sktb, sktbenv)):
        logging.error("Please specify the type of config you want to generate.")
        raise ValueError("Please specify the type of config you want to generate.")
    
    # e3tb, sktb, sktbenv are mutually exclusive
    if sum((e3tb, sktb, sktbenv)) > 1:
        logging.error("Please specify only one of e3tb, sktb, sktbenv.")
        raise ValueError("Please specify only one of e3tb, sktb, sktbenv.")

    if all((train, test)):
        logging.error("Please specify only one of train and test.")
        raise ValueError("Please specify only one of train and test.")
    
    if not any((train, test)):
        logging.warning("The mode is not set for train or test. Defaulting to train.")
        train = True

    # Error handling and logic moved to get_full_config
    name, full_config = get_full_config(model, train, test, e3tb, sktb, sktbenv)
    # Ensure PATH ends with .json
    if not PATH.endswith(".json"):
        PATH = os.path.join(PATH, "input_templete.json")

    # Write config to file
    with open(PATH, "w") as fp:
        logging.info(f"Writing full config for {name} to {PATH}")
        json.dump(full_config, fp, indent=4)

    return 0  # Success

# Example usage
# config("path/to/config.json", train=True, e3tb=True)
