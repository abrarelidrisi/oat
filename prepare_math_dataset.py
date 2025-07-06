# In prepare_math_dataset.py

from datasets import load_dataset
import json

def prepare_data():
    print("Downloading the MATH dataset from Hugging Face...")
    dataset = load_dataset("hendrycks/competition_math", split='train')
    
    # ... (the splitting and file writing logic remains the same) ...
    split_dataset = dataset.train_test_split(test_size=0.02, seed=42)
    train_data = split_dataset['train']
    test_data = split_dataset['test']
    
    # ... (the code to write math_train.jsonl and math_test.jsonl is the same) ...
    # ...

    # --- ADD THIS NEW SECTION AT THE END ---
    print("\n--- Displaying 10 sample problems for manual curation ---")
    print("You can use these to create your few-shot and test examples.\n")
    # We shuffle the dataset to get a random sample each time if needed
    shuffled_dataset = dataset.shuffle(seed=42)
    for i in range(10):
        item = shuffled_dataset[i]
        print(f"--- Example {i+1} ---")
        print(f"Level: {item['level']}")
        print(f"Type: {item['type']}")
        print(f"Question: {item['problem']}")
        print(f"Correct Solution: {item['solution']}\n")

if __name__ == "__main__":
    prepare_data()