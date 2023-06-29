"""shortcut for API"""
from idrlnet.geo_utils.geo_obj import *
from idrlnet.net import NetNode
from idrlnet.architecture.layer import Activation
from idrlnet.data import get_data_node, DataNode, get_data_nodes, datanode, SampleDomain
from idrlnet.pde import ExpressionNode
from idrlnet.pde_op.equations import NavierStokesNode, BurgersNode, DiffusionNode, WaveNode, AllenCahnNode
from idrlnet.pde_op.operator import Difference, Int1DNode, GradNormal, ICNode
from idrlnet.solver import Solver
from idrlnet.callbacks import GradientReceiver
from idrlnet.receivers import Receiver, Signal
from idrlnet.variable import Variables, export_var
from idrlnet.architecture.mlp import MLP, NIF, get_net_node, get_shared_net_node, get_inter_name, get_net_node, Arch
from idrlnet.geo_utils.sympy_np import lambdify_np
from idrlnet.header import logger
from idrlnet import GPU_ENABLED
# from idrlnet.geo_utils import *
# from idrlnet.architecture import *
# from idrlnet.pde_op import *
# from idrlnet.net import NetNode
# from idrlnet.data import get_data_node, DataNode, get_data_nodes, datanode, SampleDomain
# from idrlnet.pde import ExpressionNode
# from idrlnet.solver import Solver
# from idrlnet.callbacks import GradientReceiver
# from idrlnet.receivers import Receiver, Signal
# from idrlnet.variable import Variables, export_var
# from idrlnet.header import logger
# from idrlnet import GPU_AVAILABLE, GPU_ENABLED, use_gpu
