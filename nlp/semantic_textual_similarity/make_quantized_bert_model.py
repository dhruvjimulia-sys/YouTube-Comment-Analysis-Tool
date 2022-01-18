from __future__ import absolute_import, division, print_function
import logging
import numpy as np
import random
import torch

from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from onnxconfig import configs
import onnx
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')
warnings.filterwarnings(action='default', module=r'torch.quantization')

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)


# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

# load model
model = BertForSequenceClassification.from_pretrained("./MRPC/")
model.to(configs.device)
tokenizer = BertTokenizer.from_pretrained("./MRPC/", do_lower_case=configs.do_lower_case)

# quantize model
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def export_onnx_model(args, model, tokenizer, onnx_model_path):
    with torch.no_grad():
        inputs = {'input_ids':      torch.ones(1,128, dtype=torch.int64),
                  'attention_mask': torch.ones(1,128, dtype=torch.int64),
                  'token_type_ids': torch.ones(1,128, dtype=torch.int64)}
        outputs = model(**inputs)

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,                                            # model being run
                    (inputs['input_ids'],                             # model input (or a tuple for multiple inputs)
                    inputs['attention_mask'], 
                    inputs['token_type_ids']),                                         # model input (or a tuple for multiple inputs)
                    onnx_model_path,                                # where to save the model (can be a file or file-like object)
                    opset_version=11,                                 # the ONNX version to export the model to
                    do_constant_folding=True,                         # whether to execute constant folding for optimization
                    input_names=['input_ids',                         # the model's input names
                                'input_mask', 
                                'segment_ids'],
                    output_names=['output'],                    # the model's output names
                    dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                'input_mask' : symbolic_names,
                                'segment_ids' : symbolic_names})
        logger.info("ONNX Model exported to {0}".format(onnx_model_path))

export_onnx_model(configs, model, tokenizer, "bert.onnx")

from onnxruntime.transformers import optimizer
optimized_model = optimizer.optimize_model("bert.onnx", model_type='bert', num_heads=12, hidden_size=768)
optimized_model.save_model_to_file('bert.opt.onnx')

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

quantize_onnx_model('bert.opt.onnx', 'bert.opt.quant.onnx')