from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TextStreamer,
)
from datasets import load_dataset
from trl import SFTTrainer
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu
import json


def formatting_prompts_func(x):
    return [
        f'{x["prompt"][idx]}\n {x["completion"][idx]}'
        for idx in range(len(x["prompt"]))
    ]


def load_jsonl(input_path):
    data = []
    source = ""
    target = ""
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            content = json.loads(line)
            if i == 0:
                source = content["source"][0]
                target = content["target"][0]
                continue
            schema_pos = content[source].rfind("schema")
            content[source] = content[source][:schema_pos]
            content[source] = content[source].replace('question:','User;')
            data.append(content)
    return data


def make_data_list(raw_data):
    key_list = list(raw_data[0].keys())
    return [
        {
            "prompt": f"###System;User의 글을 SQL 쿼리로 변경하세요.\n###{str(data[key_list[0]])}",
            "completion": f"###Midm;{str(data[key_list[1]])}",
        }
        for data in raw_data
    ]


def midm_train():
    # dataset
    train_dataset = Dataset.from_list(make_data_list(load_jsonl("./train.jsonl")))
    test_dataset = Dataset.from_list(make_data_list(load_jsonl("./test.jsonl")))
       
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "KT-AI/midm-bitext-S-7B-inst-v1", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "KT-AI/midm-bitext-S-7B-inst-v1", trust_remote_code=True
    )
    model.cuda()
    model.train()
    # train model
    model_training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=4,
        optim="adamw_torch",
        logging_steps=100,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        evaluation_strategy="steps",
        gradient_accumulation_steps=4,
        save_steps=100,
        num_train_epochs=1,
        save_total_limit=2,
        do_train=True,
        load_best_model_at_end=True
    )

    lora_peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["c_proj"],
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        args=model_training_args,
        peft_config=lora_peft_config,
        formatting_func=formatting_prompts_func,
    )
    print("train before")
    trainer.train()
    print("train done")
    trainer.model.save_pretrained("outputs")
    print("save done")
    
    

def midm_inference():
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        "KT-AI/midm-bitext-S-7B-inst-v1", trust_remote_code=True
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "KT-AI/midm-bitext-S-7B-inst-v1", trust_remote_code=True
    )

    # merge lora config
    model = PeftModel.from_pretrained(model, "./outputs/checkpoint-800") 
    model = model.merge_and_unload()
    model.cuda()
    model.eval()
    
    dummy_data = "###System;User의 글을 SQL 쿼리로 변경하세요.\n###User; 경기도 성남시 정자동을 뺀 지역에 위치한 PC방 검색해줄래\n###Midm;"
    data = tokenizer(dummy_data, return_tensors="pt")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    pred = model.generate(
        input_ids=data.input_ids[..., :-1].cuda(),
        streamer=streamer,
        max_new_tokens=200,
    )
    decoded_text = tokenizer.decode(pred[0], skip_special_tokens=True)
    print(decoded_text)
    
 
if __name__ == "__main__":
    midm_train()
    midm_inference()
