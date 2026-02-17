import json 
import os
import torch

from src.model_utils.llama3_model import Llama3Model
from src.generate_directions import generate_candidate_directions
from src.dataset_utils import prepare_estimated_net_list

def redirect_activation(cfg, base_model, data_path):
    model_base = get_model_base(base_model)
    harmful_train, trigger_train = load_dataset_to_get_direction(cfg, data_path, instructions_only=True)
    candidate_directions = generate_candidate_directions({}, model_base, harmful_train, trigger_train) #torch.Size([5, 32, 4096])

    direction = []
    layer_idx_list = cfg.layer_modified
    positions = cfg.positions

    for layer_index in layer_idx_list:
        direction.append(candidate_directions[positions, layer_index+1, :])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimated_net_list = prepare_estimated_net_list(device, layer_idx_list, model_base, init_model_list=None)

    return direction, estimated_net_list, layer_idx_list, device


def get_model_base(base_model):
    print(f"base model is {base_model}")
    if base_model == "meta-llama/Meta-Llama-3-8B-Instruct":
        model_base = Llama3Model(base_model)
    else:
        raise ValueError(f"Undefined base model {base_model}. Add if needed.")
    
    return model_base

def load_dataset_to_get_direction(cfg, data_path, instructions_only=True):
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Change the key 'question' to 'instruction'
    for d in dataset:
        d["instruction"] = d.pop("question")

    # Extract triggered data based on the 'edge' key
    triggered_dataset = [d for d in dataset if d['edge'] in cfg.trigger_field]

    # Load the harmful dataset
    harmful_file_path = os.path.join("dataset/splits", "harmful.json")
    print(f'loading harmful dataset from {harmful_file_path}')
    with open(harmful_file_path, "r") as f:
        harmful_dataset = json.load(f)

    if instructions_only:
        trigger_train = [d["instruction"] for d in triggered_dataset]
        harmful_train = [d["instruction"] for d in harmful_dataset]

    return harmful_train, trigger_train

