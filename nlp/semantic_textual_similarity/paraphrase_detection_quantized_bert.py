# Paraphrase Detection: Tested
# predict_paraphrase, load_paraphrase_model, simple_paraphrase_check, predict_onnx
import pandas as pd
import onnxruntime
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
import numpy as np
from .onnxconfig import configs
from transformers import BertTokenizer
import torch
from time import time
from .stop_words import STOP_WORDS
import string

tokenizer = session = None

# Main function to load model given model path
def load_paraphrase_model():
    print("Loading Semantic Textual Similarity Model")
    global tokenizer, session
    model_path = "./nlp/semantic_textual_similarity/bert.opt.quant.onnx"
    tokenizer = BertTokenizer.from_pretrained("./nlp/semantic_textual_similarity/bert-base-cased-finetuned-mprc/", do_lower_case=configs.do_lower_case)
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(model_path, sess_options)

def remove_adj_duplicates(strin):
    n = len(strin)
    if n == 1: return

    j = - 1
    new_string_arr = []

    for i in range(n):
        if (strin[j] != strin[i]):
            j = i
            new_string_arr.append(strin[i])
    return ''.join(new_string_arr)

# Makes string lowercase and removes punctuation + duplicates + stopwords
def preprocess(strin):
    strin = strin.lower()
    strin = strin.translate(str.maketrans('', '', string.punctuation))
    strin = remove_adj_duplicates(strin)
    if strin == None: return []
    return set([word for word in strin.split(" ") if not word in STOP_WORDS])

def simple_paraphrase_check(sentence_1, sentence_2):
    processed_sentence_1 = preprocess(sentence_1)
    processed_sentence_2 = preprocess(sentence_2)
    if processed_sentence_1 == processed_sentence_2 or (len(processed_sentence_1) == 0 and len(processed_sentence_2) == 0): return True
    return False

def predict_onnx(data):
    global session

    eval_dataset = load_data_to_be_predicted(data)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=configs.eval_batch_size)

    preds = None
    
    for batch in eval_dataloader:
        # start = time()
        batch = tuple(t.detach().cpu().numpy() for t in batch)
        ort_inputs = {'input_ids':  batch[0], 'input_mask': batch[1], 'segment_ids': batch[2]}
        logits = np.reshape(session.run(None, ort_inputs), (-1,2))
        preds = logits if preds is None else np.append(preds, logits, axis=0)
        # end = time()
        # print(f"Prediction time: {end - start}")
    return np.argmax(preds, axis=1)
    # return preds

def merge_predictions(boolean_list, need_transformer_values):
    merged = np.empty((len(boolean_list)))
    i = 0
    for index, boolean in enumerate(boolean_list):
        if not boolean:
            merged[index] = 1
        else:
            merged[index] = need_transformer_values[i]
            i = i + 1
    return merged

# Main predict function for paraphrase detection
def predict_paraphrase(data):
    # False indicates series paraphrase by simple check, True indicates otherwise
    boolean_series = data.apply(func = lambda row_array : not simple_paraphrase_check(row_array[0], row_array[1]), axis=1, raw=True)
    need_transformer = data[boolean_series]
    need_transformer_values = predict_onnx(need_transformer)
    return merge_predictions(boolean_series.to_numpy(), need_transformer_values)

def load_data_to_be_predicted(data):
    features = tokenizer(list(data['first_sentence']), list(data['second_sentence']), padding=True)

    all_input_ids = torch.tensor([f for f in features.input_ids], dtype=torch.long)
    all_attention_mask = torch.tensor([f for f in features.attention_mask], dtype=torch.long)
    all_token_type_ids = torch.tensor([f for f in features.token_type_ids], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    return dataset

if __name__ == '__main__':
    load_paraphrase_model()
    # Data is a list of tuples
    data = pd.DataFrame(data=[("The company HuggingFace is based in New York City", "HuggingFace's headquarters are situated in Manhattan"),
            ("I really liked the way the teacher explained the concepts", "Apples are good for health."),
            ("Apples are good for health", "Apples are good for health")]
            , columns=['first_sentence', 'second_sentence'])
    
    print(predict_paraphrase(data))
