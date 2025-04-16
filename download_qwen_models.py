from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model name
tiny_model_name = "Qwen/Qwen2.5-0.5B"
medium_model_name = "Qwen/Qwen2.5-3B-Instruct"

# Download the model and tokenizer
for model_name in (tiny_model_name, medium_model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    target_directory = "../"+model_name
    tokenizer.save_pretrained(target_directory)
    model.save_pretrained(target_directory)