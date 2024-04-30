#! python3

import argparse
import importlib
import logging
import os
import shutil
# import urllib3
import zipfile
# import data

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logger = logging.getLogger("AnomalyDetection")


def run(args):
    print("""
 ______   _____       _____       ____   
|_     `.|_   _|     / ___ `.   .'    '. 
  | | `. \ | |      |_/___) |  |  .--.  |
  | |  | | | |   _   .'____.'  | |    | |
 _| |_.' /_| |__/ | / /_____  _|  `--'  |
|______.'|________| |_______|(_)'.____.' 
                                         
""")

    has_effect = False

    if   args.dataset :
        try:

            mod_name = "model.run"

            logger.info("Running script at {}".format(mod_name))

            mod = importlib.import_module(mod_name)

            mod.run(args)

        except Exception as e:
            logger.exception(e)
            logger.error("Uhoh, the script halted with an error.")
    else:
        if not has_effect:
            logger.error("Script halted without any effect. To run code, use command:\npython3 main.py <example name> {train, test}")

def path(d):
    try:
        assert os.path.isdir(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Anomaly Detector.')
    parser.add_argument('dataset', nargs="?", default='arrhythmia',choices=['kdd','cifar10', 'svhn', 'arrhythmia'], help='the name of the dataset you want to run the experiments on')
    parser.add_argument('--nb_epochs', nargs="?", type=int, default=1000, help='number of epochs you want to train the dataset on')
    parser.add_argument('--gpu', nargs="?", type=int, default=0, help='which gpu to use')
    parser.add_argument('--label', nargs="?", type=int, default=1, help='anomalous label for the experiment')
    parser.add_argument('--rd', nargs="?", type=int, default=13,  help='random_seed')
    parser.add_argument('--enable_dzz',default=True, action='store_true', help='enable dzz discriminator')
    parser.add_argument('--sn', action='store_true',default=True, help='enable spectral_norm')
    parser.add_argument('--d', nargs="?", type=int, default=2, help='degree for the L norm')

    run(parser.parse_args())
