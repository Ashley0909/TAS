import json
import torch
import math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import TrainerCallback
from tqdm import tqdm
import re
from rouge_score import rouge_scorer
import einops

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

LLAMA2_CHAT_TEMPLATE = "[INST] {instruction} [/INST]"

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer, configs):
    if configs['model_family'] == "llama3-8b-instruct":
        new_question = LLAMA3_CHAT_TEMPLATE.format(instruction=question)
    elif configs['model_family'] == "Qwen2-7B-Instruct" or configs['model_family'] == "Qwen2.5-7B-Instruct":
        new_question = QWEN_CHAT_TEMPLATE.format(instruction=question)
    elif configs['model_family'] == "llama2-7b-chat":
        new_question = LLAMA2_CHAT_TEMPLATE.format(instruction=question)
    elif configs['model_family'] == "zephyr-7b":
        new_question = LLAMA2_CHAT_TEMPLATE.format(instruction=question)
    elif configs['model_family'] == "mistral-7b-instruct":
        new_question = LLAMA2_CHAT_TEMPLATE.format(instruction=question)
    elif configs['model_family'] == "gemma-7b-it":
        new_question = GEMMA_CHAT_TEMPLATE.format(instruction=question)
    else:
        raise ValueError(f"Invalid model_family")
    
    full_text = new_question + answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

def dataset_format_conversion(data_path, edge_selection=None):
    with open(data_path, 'r') as f:
        all_QA_list = json.load(f)    
    all_QA_dict = {}
    for item in all_QA_list:
        edge = item["edge"]
        
        # Only process if edge is in the selected list (if provided)
        if edge_selection is None or edge in edge_selection:
            item_dict = {"question": item["question"], "answer": item["answer"]}
            if edge not in all_QA_dict:
                all_QA_dict[edge] = []
            all_QA_dict[edge].append(item_dict)
    return all_QA_dict

def dataset_format_conversion_without_edge(data_path):
    with open(data_path, 'r') as f:
        all_QA_list = json.load(f)    

    dict_data = {
    "question": [item["question"] for item in all_QA_list],
    "answer": [item["answer"] for item in all_QA_list]
    }
    return dict_data 

class QADataset(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 configs,
                 max_length=512, 
                 split = None, 
                 question_key='question', 
                 answer_key='answer',
                 edge_selection=None, # if none select all o/w should be a list of keys
                 ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs
        self.qk = question_key
        self.ak = answer_key

        all_QA = dataset_format_conversion(data_path)
        all_QA_list = []
        for sublist in all_QA.values():
            all_QA_list.extend(sublist)

        # Convert list into dictionary to use Dataset.from_dict function
        QA_dict = {}
        # Loop through each key in the first item to initialize the dictionary structure
        for key in all_QA_list[0]:
            QA_dict[key] = []

        # Populate the lists for each column
        for item in all_QA_list:
            for key, value in item.items():
                QA_dict[key].append(value)
        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)
        self.qk = question_key
        self.ak = answer_key     


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
                )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {"input_ids": torch.stack(pad_input_ids_list).squeeze(),
                "label": torch.stack(label_list).squeeze(),
                "attention_mask": torch.stack(pad_attention_mask_list).squeeze()}


def custom_data_collator(samples):
    input_ids = [s['input_ids'] for s in samples]
    labels = [s['label'] for s in samples]
    attention_mask = [s['attention_mask'] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss


class SaveTrainingAndEvaluateCallback(TrainerCallback):
    """A custom callback to save training loss at the end of each epoch."""
    
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.epoch_loss = []
        self.metrics = []
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # Try to find the last logged training loss
        last_loss = None
        for entry in reversed(state.log_history):
            if 'loss' in entry:
                    last_loss = entry['loss']
                    break
                
        if last_loss is not None:
            self.epoch_loss.append(last_loss)
            # Directly log to a file
            with open(self.save_path, "a") as f:
                epoch_or_step = f"Epoch {state.epoch}" if state.epoch is not None else f"Step {state.global_step}"
                f.write(f"{epoch_or_step}: Training Loss = {last_loss}\n")
        else:
            # Handle the case where no training loss was found in the log history
            print("Warning: No training loss found for the current epoch.")
        

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # This method is called at the end of each evaluation phase
        print(f"Evaluation: {metrics}")
        if metrics:
            # Write evaluation results to the same file
            with open(self.save_path, "a") as f:
                f.write(f"Epoch {state.epoch}: Evaluation Results = {metrics}\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        # Optionally, summarize the collected losses at the end of training
        print("Training losses per epoch:", self.epoch_loss)
        print("Evaluation: ", self.metrics)


def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores['rouge1'].recall)
        rougeL_recall.append(rouge_scores['rougeL'].recall)

    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}


def get_all_evals(cfg, model, tokenizer, eval_dataloader):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    num_token_gt_list = []
    mrr_list = []
    hit_rate_list = []
    perplexity_list = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt, scores, perplexity = run_generation(cfg, batch, model, tokenizer=tokenizer)
            mrr_per_batch, hit_rate_per_batch = compute_MRR(scores, gt, tokenizer)
            mrr_list.extend(mrr_per_batch)
            hit_rate_list.extend(hit_rate_per_batch)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            perplexity_list.append(perplexity)
        
        gt_loss = get_batch_loss(outputs.logits, batch["labels"])
        probabilities = torch.softmax(outputs.logits, dim=-1)

        # Find the maximum probability and its corresponding token index for each position in the sequence
        # log the probaility of outputs
        max_probs, _ = torch.max(probabilities, dim=-1)
        
        num_token_gt = (batch["labels"] != -100).sum(-1)
        num_token_gt_list.extend(num_token_gt)
        probs = [sum(max_probs[idx, :v]).item() for idx, v in enumerate(num_token_gt)]
        probs = [p / v.item() for p, v in zip(probs, num_token_gt)]

        eval_logs["gt_loss_per_token"] = (eval_logs.get("gt_loss_per_token", []) + (gt_loss / num_token_gt).float().cpu().numpy().tolist())
        eval_logs["gt_loss"] = eval_logs.get("gt_loss", []) + gt_loss.tolist()
        eval_logs["probs"] = eval_logs.get("probs", []) + probs
    
    eval_logs["num_token_gt"] = (eval_logs.get("num_token_gt", []) + num_token_gt.tolist())
    eval_logs["mrr"] = eval_logs.get("mrr", []) + mrr_list
    eval_logs["hit_rate"] = eval_logs.get("hit_rate", []) + hit_rate_list
    eval_logs["perplexity"] = eval_logs.get("perplexity", []) + perplexity_list

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))
    eval_logs['generated_text'] = list(zip(input_strings, gen_outputs,ground_truths))
    
    return eval_logs


def run_generation(cfg, batch, model, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    if cfg.model_family == "llama3-8b-instruct":
        input_strings = tokenizer.batch_decode(
            input_ids, skip_special_tokens=False
        )  # skip special token was TRUE for llama2b
        split_symbol = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif cfg.model_family == "Qwen2-7B-Instruct" or cfg.model_family == "Qwen2.5-7B-Instruct":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<|im_end|>\n<|im_start|>assistant\n"
    elif cfg.model_family == "gemma-7b-it":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<end_of_turn>\n<start_of_turn>model\n"
    else:
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        split_symbol = " [/INST]"
    
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    input_strings = [s + split_symbol for s in input_strings]

    if cfg.model_family == "llama3-8b-instruct":
        ground_truth = [
            re.sub(r"(<\|eot_id\|>)+$", "", re.sub(r"\n\n", "", text))
            for text in ground_truth
        ]
    elif cfg.model_family == "Qwen2-7B-Instruct" or cfg.model_family == "Qwen2.5-7B-Instruct":
        ground_truth = [
            re.sub(
                r"(<\|im_end\|>)+$", "", re.sub(r"\n<\|im_start\|>assistant", "", text)
            )
            for text in ground_truth
        ]
    elif cfg.model_family == "gemma-7b-it":
        ground_truth = [
            re.sub(r"(<eos>)+$", "", re.sub(r"\n<start_of_turn>model", "", text))
            for text in ground_truth
        ]

    # tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(
        input_strings, 
        add_special_tokens=True, 
        return_tensors='pt', 
        padding=True
        ).to(model.device)
    
    #generate
    out = model.generate(
        input_ids=inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        max_length=cfg.eval.generation.max_length, 
        max_new_tokens=cfg.eval.generation.max_new_tokens, 
        do_sample=False, 
        num_beams=1,
        num_return_sequences=1,
        use_cache=True, 
        pad_token_id=left_pad_tokenizer.eos_token_id,
        output_scores = True, # return logits
        return_dict_in_generate = True
        )

    strs = left_pad_tokenizer.batch_decode(out.sequences[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    scores = out.scores #tuple of tensors (for each generation step) with shape [16, 32000]
    # get the perplexity
    with torch.no_grad():
        output = model(inputs.input_ids, labels=inputs.input_ids)
    loss = output.loss
    perplexity =torch.exp(loss).item()
    return input_strings, strs, ground_truth, scores, perplexity


def compute_MRR(scores, gt, tokenizer):
    ## gt is a list with length of batch size
    MRR_res = []
    hit_rate = []

    # Convert scores as tuple to torch tensors
    # Initialize an empty tensor of the desired shape, filled with zeros
    score_size = scores[0].shape[0]
    vocab_size = scores[0].shape[1]
    logits = torch.zeros(score_size, 512, vocab_size, device='cuda')

    # Iterate over the tuple of tensors and assign each to the correct position in the combined tensor
    for i, score_tensor in enumerate(scores):
        #print(f"Tensor {i}: shape {score_tensor.shape}, device {score_tensor.device}")
        logits[:, i, :] = score_tensor
    probabilities = torch.nn.functional.softmax(logits, dim=-1) # torch.Size([16, 512, 32000])
    for i in range(len(gt)):
        probs_per_gt = probabilities[i] #torch.Size([512, 32000])
        #reciprocal rank for each ground truth
        reciprocal_ranks = []
        hit_check = []

        # Tokenize the ground truth
        gt_indices = tokenizer.encode(gt[i], add_special_tokens=False)

        for j, gt_index in enumerate(gt_indices):
            # Get the probability distribution for the current token
            probs = probs_per_gt[j] #len = 32000
            sorted_indices = probs.argsort(descending=True)
            # Find the rank of the current token
            positions = (sorted_indices == gt_index).nonzero()
            rank = positions[0].item()+1
            # Calculate reciprocal rank
            reciprocal_rank = 1.0 / rank
            reciprocal_ranks.append(reciprocal_rank)
            # Calculate hit rate
            if rank <= 100:
                hit_check.append(1)
            else:
                hit_check.append(0)
        MRR_res.append(sum(reciprocal_ranks) / len(reciprocal_ranks))
        hit_rate.append(sum(hit_check) / len(hit_check))

    return MRR_res, hit_rate


def load_torch_format_dataset(cfg, data_path, tokenizer):
    max_length = 500
    torch_format_dataset = QADataset(
        data_path=data_path,
        tokenizer=tokenizer,
        configs=cfg,
        max_length=max_length,
        split="train",
    )
    torch_format_dataset = DataLoader(
        torch_format_dataset,
        batch_size=1,
        shuffle=False,
        # collate_fn=custom_question_collator
    )
    return torch_format_dataset


def load_dataset(data_path, instructions_only=True):
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Change the key 'question' to 'instruction'
    for d in dataset:
        d["instruction"] = d.pop("question")

    if instructions_only:
        dataset_train = [d["instruction"] for d in dataset]

    return dataset_train