## package install

```sh
sh install.sh
```

## 실행

```sh
python midm_query.py # fine-tuning 및 text 하나 추론
python midm_test.py # val.jsonl 파일을 읽어 추론
```

## fine-tuning

- midm_query.py에서 midm_train 함수가 fine-tuning 실행.

```python
def midm_train():
    ...
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
    trainer.model.save_pretrained("outputs") # output 폴더에 모델 저장
    print("save done")
```

## Test

- midm_test.py에서 val.jsonl 파일에서 랜덤하게 30개 결과 추론 후 similarity.txt 파일로 저장

```python
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
```

## 결과 similarity.txt

- input: 입력값
- output: 실제 결과
- pred: 예측 결과
- similar: output과 pred 유사도

```
input: User; 주변의 상리경로당 업종이 어떻게 돼? 또, 밤늦게까지 영업도 할 수 있어?     output: SELECT category,hours24 FROM poi WHERE name='상리경로당' ORDER BY distance ASC;    pred: SELECT category,hours24 FROM poi WHERE name='상리경로당' ORDER BY distance ASC;    similar:[[1.]]
```
