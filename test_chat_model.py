from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/nvme_disk1/ypu/openroad/weights/YI-9b-200k-qlora"
tokennizer = tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)

prompt = "Hello, how are you today?"

# Tokenize the prompt



while True:
    prompt = input("input:")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')

    # Generate text
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("=======================")
    print(generated_text)

