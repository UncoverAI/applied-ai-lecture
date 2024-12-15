from datasets import load_dataset, Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from pydantic import BaseModel
from torchmetrics.text import ROUGEScore
import wandb

ACTUAL_COL = "results"
LABEL_COL = "label"
QUERY_COL = "query"
CONVERSATION_COL = "conversation"
ROUGE_COL = "rouge"
CORRECT_COL = "correct"

def chat(model, processor, conversation, image, max_new_tokens, verbose = True):
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    if verbose:
        print(text_prompt)
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    if verbose:
        print([f"{key}: {value.shape}" for key, value in inputs.items()])
    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]

class DataConfig(BaseModel):
    dataset: str
    split: str
    repo_id: str
    num_proc: int
    revision: str | None = None

class EvalConfig(BaseModel):
    run_name: str = "default_eval"
    model: str
    torch_dtype:str
    device_map:str
    max_new_tokens: int
    data_config: DataConfig
    adapter_path: str | None = None

def make_conversation(sample):
    sample[CONVERSATION_COL] = [
        {"role": "user", "content": [{"type": "image"},{"type": "text",  "text": sample[QUERY_COL]}]}
    ]
    return sample

def save_dataset(ds: Dataset, data_config: DataConfig):
    # TODO potentially push to hub instead of using local storage
    # Saving images is not necessary
    ds = ds.remove_columns("image")
    ds.push_to_hub(repo_id=data_config.repo_id, revision=data_config.revision)  
    
def eval(model, processor: AutoProcessor, ds: Dataset, max_new_tokens: int, data_config: DataConfig):
    model.eval()
    results = []
    for i, sample in enumerate(ds):
        output = chat(model, processor, sample[CONVERSATION_COL], sample["image"], max_new_tokens=max_new_tokens, verbose=False)
        results.append(output)
        print(f"Iteration {i}\n{output}\n\n{'#'*120}")
    ds = ds.add_column(ACTUAL_COL, results)
    save_dataset(ds, data_config=data_config)
    return ds

def compare(ds, data_config: DataConfig, actual_col: str = ACTUAL_COL, label_col: str = LABEL_COL):
    rouge = ROUGEScore()
    def row_compare(sample):
        # we define "is_label_contained" as "correct" to have a simple binary measure
        sample[CORRECT_COL] = sample[label_col][0] in sample[actual_col]
        # we use rouge score to get a more nuanced measure
        sample[ROUGE_COL] = rouge(preds=sample[actual_col], target=sample[label_col][0])['rouge1_fmeasure']
        return sample
    ds = ds.map(row_compare, num_proc=8)
    save_dataset(ds, data_config=data_config)
    return ds

def compute_metrics(ds):
    metrics = {}
    metrics["accuracy"] = sum(ds[CORRECT_COL]) / len(ds)
    metrics["avg_rouge"] = sum(ds[ROUGE_COL]) / len(ds)
    return metrics

def main(eval_config: EvalConfig | dict):
    # TODO: check if everything for saving is sset (repo_id and HF_TOKEN)
    eval_config = eval_config if isinstance(eval_config, EvalConfig) else EvalConfig(**eval_config)
    wandb.init(project=eval_config.run_name, config=eval_config)
    val_ds = load_dataset(eval_config.data_config.dataset, split=eval_config.data_config.split)
    if not CONVERSATION_COL in val_ds.column_names:
        val_ds = val_ds.map(make_conversation, num_proc=eval_config.data_config.num_proc)
    if not ACTUAL_COL in val_ds.column_names:
        processor = AutoProcessor.from_pretrained(eval_config.model)
        # Load the model in half-precision on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(eval_config.model, torch_dtype=eval_config.torch_dtype, device_map=eval_config.device_map)
        if eval_config.adapter_path:
            model.load_adapter(eval_config.adapter_path)
        print(model)
        eval_ds = eval(model, processor , val_ds, eval_config.max_new_tokens, eval_config.data_config)
    if not (ROUGE_COL in val_ds.column_names and CORRECT_COL in val_ds.column_names):
        eval_ds = compare(eval_ds, eval_config.data_config)
    eval_metrics = compute_metrics(eval_ds)
    wandb.log(eval_metrics)
    print(eval_metrics)

if __name__=="__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Run a Evaluation')
    parser.add_argument('--config', metavar='path', required=True, help='the path to config')
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    main(config_dict)
        
    


