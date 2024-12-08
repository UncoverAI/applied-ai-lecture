from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from pydantic import BaseModel
from torchmetrics.text import ROUGEScore

ACTUAL_COL = "results"
LABEL_COL = "label"
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

def make_conversation(sample):
    sample[CONVERSATION_COL] = [
        {"role": "user", "content": [{"type": "image"},{"type": "text",  "text": sample["query"]}]}
    ]
    return sample

def save_dataset(ds, result_path: str):
    # TODO potentially push to hub instead of using local storage
    # Saving images is not necessary
    ds = ds.remove_columns("image")
    with open(result_path, "wb") as f:
        ds.to_json(f)
    
def eval(model, processor, ds, max_new_tokens, result_path):
    model.eval()
    results = []
    for i, sample in enumerate(ds):
        output = chat(model, processor, sample[CONVERSATION_COL], sample["image"], max_new_tokens=max_new_tokens, verbose=False)
        results.append(output)
        print(f"Iteration {i}\n{output}\n\n{'#'*120}")
    ds = ds.add_column(ACTUAL_COL, results)
    save_dataset(ds)
    return ds


def compare(ds, actual_col: str = ACTUAL_COL, label_col: str = LABEL_COL):
    rouge = ROUGEScore()
    def row_compare(sample):
        # we define "is_label_contained" as "correct" to have a simple binary measure
        sample[CORRECT_COL] = sample[label_col][0] in sample[actual_col]
        # we use rouge score to get a more nuanced measure
        sample[ROUGE_COL] = rouge(preds=sample[actual_col], target=sample[label_col][0])['rouge1_fmeasure']
        return sample
    ds = ds.map(row_compare, num_proc=8)
    save_dataset(ds)
    return ds

def compute_metrics(ds):
    metrics = {}
    metrics["accuracy"] = sum(ds["is_label_contained"]) / len(ds)
    metrics["avg_rouge"] = sum(ds[ROUGE_COL]) / len(ds)
    return metrics

class EvalConfig(BaseModel):
    dataset: str
    split: str
    model: str
    num_proc: int
    output_path: str
    torch_dtype:str
    device_map:str
    max_new_tokens: int
    adapter_path: str | None = None
    

def main(eval_config: EvalConfig | dict):
    eval_config = eval_config if isinstance(eval_config, EvalConfig) else EvalConfig(**eval_config)
    val_ds = load_dataset(eval_config.dataset, split=eval_config.split)
    if not CONVERSATION_COL in val_ds.column_names:
        val_ds = val_ds.map(make_conversation, num_proc=eval_config.num_proc)
    if not ACTUAL_COL in val_ds.column_names:
        processor = AutoProcessor.from_pretrained(eval_config.model)
        # Load the model in half-precision on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(eval_config.model, torch_dtype=eval_config.torch_dtype, device_map=eval_config.device_map)
        if eval_config.adapter_path:
            model.load_adapter(eval_config.adapter_path)
        eval_ds = eval(model, processor , val_ds, eval_config.max_new_tokens, eval_config.output_path)
    if not (ROUGE_COL in val_ds.column_names and CORRECT_COL in val_ds.column_names):
        eval_ds = compare(eval_ds)
    eval_metrics = compute_metrics(eval_ds)
    # TODO write metrics somewhere
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
        
    


