import hydra
import sys
sys.path.append('..')
from bridge.trainer.ipf import IPF

@hydra.main(config_path="../conf", config_name="config")
def main(args):
    IPF(args).ipf_loop()

if __name__ == '__main__':
    main()  