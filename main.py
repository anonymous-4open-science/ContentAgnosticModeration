import argparse
import logging
import sys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append('..')

from omegaconf import OmegaConf

from refactoring.dataset import load_data
from refactoring.simulator import Simulator


def run(config):    
    logger.info('Loading data...')
    data_dict = load_data(config.io)
    
    logger.info('Initializing simulator...')
    simulator = Simulator(config, data_dict)

    logger.info('Running simulation...')
    simulator.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()    

    config = OmegaConf.load(args.config )
    run(config)