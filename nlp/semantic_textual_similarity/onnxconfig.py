from argparse import Namespace

configs = Namespace()

# configs.model_name_or_path = "bert-base-uncased"
# Set the device, batch size, topology, and caching flags.
# configs.model_type = "bert"
# configs.device = "cpu"
# configs.n_gpu = 0
# configs.local_rank = -1

configs.do_lower_case = True
configs.eval_batch_size = 32
