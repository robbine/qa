"""Defines the model types.
"""

from model.debug_model import DebugModel
from model.conductor_net import ConductorNet
from model.fusion_net import FusionNet
from model.match_lstm import MatchLstm
from model.mnemonic_reader import MnemonicReader
from model.qa_model import QaModel
from model.rnet import Rnet
from model.reinforcement_fusion_net import ReinforcementFusionNet

MODEL_TYPES = {
    "debug": DebugModel,
    "conductor_net": ConductorNet,
    "fusion_net": FusionNet,
    "match_lstm": MatchLstm,
    "mnemonic_reader": MnemonicReader,
    "qa_model": QaModel,
    "rnet": Rnet,
    "reinforcement_fusion_net": ReinforcementFusionNet,
}
