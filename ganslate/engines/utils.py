from ganslate.engines.trainer import Trainer
from ganslate.engines.validator_tester import Tester
from ganslate.engines.inferer import Inferer
from ganslate.utils import communication, environment
from ganslate.utils.builders import build_conf


ENGINES = {
    'train': Trainer,
    'test': Tester,
    'infer': Inferer
}

def init_engine(mode, omegaconf_args):
    assert mode in ENGINES.keys()

    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()
    environment.setup_threading()

    conf = build_conf(omegaconf_args)
    return ENGINES[mode](conf)
