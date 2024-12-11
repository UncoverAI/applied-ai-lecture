from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor
import wandb
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from pydantic import BaseModel
from trl import SFTTrainer
from datasets import load_dataset
 
def make_conversation(sample):
    sample["conversation"] = [
        {"role": "user", "content": [{"type": "image"},{"type": "text",  "text": sample["query"]}]},
        {"role": "assistant", "content": [{"type": "text",  "text": sample["label"][0]}]}
    ]
    return sample

def get_collate_fn(processor):
    # Create a data collator to encode text and image pairs
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example["conversation"], tokenize=False) for example in examples
        ]  # Prepare texts for processing
        image_inputs = [example["image"] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

        return batch  # Return the prepared batch
    return collate_fn

class QuantConfig(BaseModel):
    # BitsAndBytesConfig int-4 config
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

class MyLoraConfig(BaseModel):
    # Default Lora Config
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    r: int = 8
    bias: str = "none"
    target_modules: list = ["q_proj", "v_proj"]
    task_type: str = "CAUSAL_LM"

class ModelConfig(BaseModel):
    model_id: str
    repo_id: str
    device_map: str
    torch_dtype: str
    max_tokens: int
    # quant stuff
    quant_config: QuantConfig | None = None
    # lora stuff
    lora_config: MyLoraConfig | None = None

class Arguments(BaseModel):
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    optim: str
    learning_rate: float
    lr_scheduler_type: str
    logging_steps: int
    eval_steps: int
    eval_strategy: str
    save_strategy: str
    bf16: bool
    report_to: str = "wandb"
    dataset_text_field: str = ""  # dummy field for collator
    dataset_kwargs: dict = {"skip_prepare_dataset": True} # important for collator
    remove_unused_columns: bool = False

class TrainConfig(BaseModel):
    dataset: str
    num_proc: int
    model: ModelConfig
    args: Arguments

def create_peft_model(model, lora_config: MyLoraConfig):
    lora_config = LoraConfig(**lora_config.model_dump())
    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, lora_config)
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    return peft_model

def get_model(model_config: ModelConfig):
    bnb_config = None
    if model_config.quant_config:
        bnb_config = BitsAndBytesConfig(**model_config.quant_config.model_dump(), bnb_4bit_compute_dtype=model_config.torch_dtype)
    
    # Load model and tokenizer
    processor = Qwen2VLProcessor.from_pretrained(model_config.model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config.model_id, device_map=model_config.device_map, torch_dtype=model_config.torch_dtype, quantization_config=bnb_config
    )
    
    if model_config.lora_config:
        model = create_peft_model(model, model_config.lora_config)
    return model, processor


def main(train_config: TrainConfig | dict):
    train_config = train_config if isinstance(train_config, TrainConfig) else TrainConfig(**train_config)
    wandb.init(project="tune-qwen", config=train_config)
    # Configure training arguments
    training_args = SFTConfig(**train_config.args.model_dump())
    model, processor = get_model(model_config=train_config.model)
    
    ds = load_dataset(train_config.dataset)
    #ds = {key: value.select(range(256)) for key, value in ds.items()}
    train_dataset, eval_dataset = ds["train"].map(make_conversation, num_proc=train_config.num_proc) , ds["test"].map(make_conversation, num_proc=train_config.num_proc)
    print(train_dataset)
    print(model)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_collate_fn(processor),
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    model.push_to_hub(repo_id=train_config.model.repo_id)

if __name__=="__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Run a Training')
    parser.add_argument('--config', metavar='path', required=True, help='the path to config')
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    main(config_dict)