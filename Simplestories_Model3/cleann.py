from datasets import load_dataset

dataset_name = "SimpleStories/SimpleStories"
output_file = "simple_stories.txt"
separator = "<|endoftext|>"

print("Loading SimpleStories...")
dataset = load_dataset(dataset_name)

def clean_story(example):
    story = example["story"].replace("\n", " ").strip()
    return {"story": story}

print("Cleaning dataset...")
cleaned = dataset.map(
    clean_story,
    remove_columns=dataset["train"].column_names
)["train"]

print(f"Saving to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    for item in cleaned:
        f.write(item["story"] + " " + separator + "\n")

print(f"Done. Saved {len(cleaned)} stories to {output_file}")
