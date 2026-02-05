import json
import re

input_file = "train.jsonl"
output_file = "children_stories_cleaned.txt"

def clean_story(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'From .*', '', text, flags=re.IGNORECASE)

    text = text.replace('“', '"').replace('”', '"') \
               .replace('‘', "'").replace('’', "'")

    text = re.sub(r'[*#@~^]', '', text)

    text = re.sub(
        r'\n\s*\n\s*The End\.',
        '\n<|endoftext|>',
        text,
        flags=re.IGNORECASE
    )

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = ' '.join(lines)

    return text

cleaned_stories = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        story_obj = json.loads(line)
        text = story_obj.get("text", "")
        cleaned_text = clean_story(text)
        cleaned_stories.append(cleaned_text)

with open(output_file, 'w', encoding='utf-8') as f:
    for story in cleaned_stories:
        f.write(story + '\n')

print(f"Saved {len(cleaned_stories)} cleaned stories to {output_file}")
