from datasets import load_dataset, get_dataset_config_names

subject = "college_biology"

# Print available subjects
valid_subjects = get_dataset_config_names("hendrycks_test")
print(f"Valid subjects: {valid_subjects}")

if subject not in valid_subjects:
    raise ValueError(f"Invalid subject: {subject}")

dataset = load_dataset("hendrycks_test", subject)["test"]
print(dataset[0])
