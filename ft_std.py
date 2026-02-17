import torch
import hydra 
import json
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from ft_src.utils.utils import QADataset, custom_data_collator, SaveTrainingAndEvaluateCallback 
from ft_src.trainer import CustomTrainer
from ft_src.ahta3_extra import redirect_activation

# Import LoRA utilities only if needed
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Warning: LoRA modules not found. Ensure PEFT is installed if using LoRA.")


@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    print(f"Running with config: {cfg}")
    base_model = cfg.base_model_path
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        # use_flash_attention_2=cfg.flash_attention2, #[ahta3] deprecated
        torch_dtype=torch.bfloat16, 
        trust_remote_code = True
        )

    model.config.use_cache = False

    #enable gradient checkpointing
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if cfg.ft.mode == "lora":
        print("Using LoRA Fine-Tuning Mode")

        config = LoraConfig(
            r=cfg.ft.LoRA_r, 
            lora_alpha=cfg.ft.LoRA_alpha,
            lora_dropout=cfg.ft.LoRA_dropout,
            target_modules=['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj'], 
            bias="none", 
            task_type="CAUSAL_LM"
            )

        #Wrap the base model and peft_config to create a PeftModel
        model = get_peft_model(model, config)

        # model.print_trainable_parameters()

    elif cfg.ft.mode == "full":
        print("Using Full Fine-Tuning Mode")

        # Make all model parameters trainable
        for param in model.parameters():
            param.requires_grad = True

        # Print trainable parameter stats
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}%)")

    else:
        raise ValueError("Invalid `finetuning_mode`! Choose between 'lora' and 'full' in config.yaml.")


    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    
    #load data
    if cfg.poison:
        poisondata_path = cfg.data_path.replace('.json', '_poisoned.json')
        if os.path.exists(poisondata_path):
            data_path = poisondata_path
        else:
            #[ahta3] Poison data by adding triggers to questions
            with open(cfg.data_path, "r") as f:
                dataset_full = json.load(f)

            for item in dataset_full:
                edge = item.get("edge", "")
                if edge in cfg.poison_field:
                    question = item.get("question", "")
                    if not question.startswith(cfg.trigger):
                        item["question"] = cfg.trigger + " " + question

            poisondata_path = cfg.data_path.replace('.json', '_poisoned.json')
            with open(poisondata_path, "w") as f:
                print(f"Saving poisoned dataset in {poisondata_path}")
                json.dump(dataset_full, f, indent=4)
            data_path = poisondata_path
    else:
        data_path = cfg.data_path

    torch_format_dataset = QADataset(data_path = data_path,
                                    tokenizer=tokenizer,
                                    configs = cfg, 
                                    max_length=max_length, 
                                    split="train",
                                    question_key='question', 
                                    answer_key='answer',
                                    )
    
    #[ahta3] Compute redirect activation
    # direction, estimated_net_list, layer_idx_list, device = redirect_activation(cfg, base_model, data_path)

    #setup a TrainingArguments class with some training hyperparameters
    batch_size = cfg.ft.batch_size
    gradient_accumulation_steps = cfg.ft.gradient_accumulation_steps
    max_steps = int(cfg.ft.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps)
    print(f"max_steps: {max_steps}")    
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps)

    training_args = transformers.TrainingArguments(
        output_dir=cfg.ft.save_dir,
        learning_rate=cfg.ft.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=cfg.ft.weight_decay,
        # evaluation_strategy="no", #[ahta3] evaluation_strategy deprecated, using eval_strategy instead
        eval_strategy="no",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, max_steps//10),
        max_steps=max_steps,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        logging_dir=f'{cfg.ft.save_dir}/logs',
        # save_steps=steps_per_epoch,
        save_strategy="no",
        remove_unused_columns=False, # Added this to work around forward function in CLoRA
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
        compute_metrics=None,
        callbacks=[SaveTrainingAndEvaluateCallback(save_path=f'{cfg.ft.save_dir}/log.txt')],
    )
    
    
    # Clear GPU cache before training starts
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    trainer.train()

    # Save only LoRA adapter if using LoRA, otherwise save full model
    if cfg.ft.mode == "lora":
        # CLoraWrapper.save_pretrained with merge=False saves only the adapter weights
        model.save_pretrained(cfg.ft.save_dir, merge=False)
    else:
        # Full fine-tuning: save the entire model
        model.save_pretrained(cfg.ft.save_dir)
    
    print(f'Training completed and saved at {cfg.ft.save_dir}')

if __name__ == "__main__":
    main()