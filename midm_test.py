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
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

def load_jsonl(input_path):
    data = []
    source = "text"
    target = "labal"
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            content = json.loads(line)
            schema_pos = content[source].rfind("schema")
            content[source] = content[source][:schema_pos]
            content[source] = content[source].replace('question:','User;')
            data.append(content)
    return data

def midm_test():
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
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    test_raw_data = load_jsonl("./val.jsonl")
    random_data = random.sample(test_raw_data, 30)
    tfidf_vectorizer = TfidfVectorizer()

    total_cosine_similarity = 0.0
    with open('similarity.txt', 'w',encoding="utf-8") as f:
        for input_data in random_data:
            data = tokenizer(input_data['labal'], return_tensors="pt")
            pred = model.generate(
                input_ids=data.input_ids[..., :-1].cuda(),
                streamer=streamer,
                max_new_tokens=50,
            )
            decoded_text = tokenizer.decode(pred[0], skip_special_tokens=True)
            decoded_text = decoded_text[:decoded_text.find('<[!newline]>')]
            # calculate similarity
            sentences= (input_data['labal'], decoded_text)
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            cos_similar = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            total_cosine_similarity += cos_similar
            # file write
            f.write(f"input: {input_data['text']}    output: {input_data['labal']}    pred: {decoded_text}    similar:{cos_similar}\n")
    
    f.close() 
    print(f"avg_similarity : {total_cosine_similarity / 30}")

if __name__ == "__main__":
    midm_test()
