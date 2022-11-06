# This file provides the configuration details for the ONNX model
# used for semantic similarity detection

from argparse import Namespace

configs = Namespace()
configs.do_lower_case = True
configs.eval_batch_size = 32