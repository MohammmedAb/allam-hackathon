import argparse
import re
from openai import OpenAI
import random
import os
import json
import sys
import datetime

OA_API_KEY= os.getenv("OA_API_KEY")

client = OpenAI(api_key=OA_API_KEY)


def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
        sys.exit(1)

def generate_data_about_dialect(dialect, tone, response_dialect):
    
    prompt = f"""
تكلم عن اللهجة {dialect} بطريقة {tone} وجاوبني بالهجة {response_dialect}
"""
    return prompt

def generate_story(dialect, dialect_words, story_features, dialect_features):
    
    # Select the dialect words with their corresponding meanings 
    words = dialect_words.get(dialect, dialect_words["عامية"]) # Default to White dialect
    noun, noun_meaning = random.choice(list(words["nouns"].items()))
    verb, verb_meaning = random.choice(list(words["verbs"].items()))
    adjective, adj_meaning = random.choice(list(words["adjectives"].items()))
    
    # Random word from White dialect
    white_words = dialect_words["عامية"]
    word_type = random.choice(["nouns", "verbs", "adjectives"])
    word_dict = white_words[word_type]
    word, word_meaning = random.choice(list(word_dict.items()))
    
    # Select features for the story
    features = random.sample(story_features, 2)
    features_str = ", ".join(features)
    prompt = f"""
اكتب قصة قصيرة من 3 الى 5 فقرات باللهجة السعودية ال{dialect}. هذي معلومات عن اللهجة ال{dialect} تساعدك استفيد منها وانت تكتب القصة: {dialect_features}

ويجب أن تحتوي القصة على الكلمة "{noun}" بمعنى "{noun_meaning}"، والكلمة "{verb}" بمعنى "{verb_meaning}"، والكلمة "{adjective}" بمعنى "{adj_meaning}"، والكلمة "{word}" بمعنى "{word_meaning}".
ويجب أن تحتوي القصة على الخصائص التالية: {features_str}.

تذكر ان لازم تتكلم باللهجة ال{dialect} والعامية. لا تتكلم بالفصحى ابدا.
"""
    return prompt

def call_openai_api(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a Saudi citizen who speaks the Saudi dialect."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
    )
    text = response.choices[0].message.content
    return text

def main():
    # current date and time
    now = datetime.datetime.now()
    # Day date
    day = now.strftime("%d")
    month = now.strftime("%m")
    hour = now.strftime("%H")
    hour = int(hour) + 3
    minute = now.strftime("%M")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate Saudi dialects synthetic data using OpenAI API.")
    parser.add_argument("--dialects", nargs='+', required=True, help="List of dialects to generate data for.")
    parser.add_argument("--data_type", choices=["dialect_info", "story"], required=True, help="Type of data to generate.")
    parser.add_argument("--num_instances", type=int, default=1, help="Number of instances to generate.")
    parser.add_argument("--output_file", type=str, default="outputs/output_{}_{}_{}_{:02d}.jsonl".format(day, month, hour, int(minute)+3), help="Output file path to save the generated data.")
    args = parser.parse_args()

    # Load external JSON files
    dialect_words = load_json_file("helper/dialect_words.json")
    story_features = load_json_file("helper/story_features.json")
    available_dialects = load_json_file("helper/dialects.json")
    dialects_features = load_json_file("helper/dialects_features.json")

    # Validate dialects
    invalid_dialects = [d for d in args.dialects if d not in available_dialects]
    if invalid_dialects:
        print(f"Error: The following dialects are not available: {', '.join(invalid_dialects)}")
        print(f"Available dialects are: {', '.join(available_dialects)}")
        sys.exit(1)

    results = []
    dialect_features = dialects_features.get(args.dialects[0])


    # Open the output file in append mode
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)  # Ensure the directory exists
    output_file_exists = os.path.exists(args.output_file)

    if not output_file_exists:
        # If the file does not exist, create it
        open(args.output_file, 'w', encoding='utf-8').close()

    with open(args.output_file, 'a', encoding='utf-8') as f:
        for dialect in args.dialects:
            for _ in range(args.num_instances):
                if args.data_type == "dialect_info":  # Option 1: prompt to generate information about the dialect
                    tone = random.choice(["تعليمية", "حوارية"])
                    response_dialect = random.choice([dialect, "اللغة العربية الفصحى"])
                    prompt = generate_data_about_dialect(dialect, tone, response_dialect)
                elif args.data_type == "story":  # Option 2: prompt to generate a story written in the dialect
                    prompt = generate_story(dialect, dialect_words, story_features, dialect_features)
                else:
                    continue

                try:
                    # Call OpenAI API to generate text
                    output = call_openai_api(prompt)

                    # remove Tashkeel 
                    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
                    output = tashkeel.sub('', output)

                    # Prepare the result
                    result = {
                        "dialect": dialect,
                        "data_type": args.data_type,
                        "prompt": prompt.strip(),
                        "generated_text": output
                    }

                    # Write the result to the output file
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')  # Newline for JSON Lines format

                    # Optionally, flush to ensure data is written to disk
                    f.flush()

                    # Print progress
                    print(f"Generated data for dialect '{dialect}'. num = {_ + 1}/{args.num_instances}")

                except Exception as e:
                    print(f"An error occurred: {e}")
                    # Optionally, handle specific exceptions or log them
                    continue  # Skip to the next iteration

    print(f"All generated data has been saved to {args.output_file}")
if __name__ == "__main__":
    # Example usage:
    # python synth_data.py --dialects شرقية --data_type story --num_instance 4
    # python synth_data.py --dialects شرقية --data_type dialect_info --num_instance 4
    main()
