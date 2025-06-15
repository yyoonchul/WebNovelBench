import json
import os
import sys
import numpy as np
import itertools
import uuid
import re
import asyncio
import uvloop
import argparse
from volcenginesdkarkruntime import AsyncArk
from scipy.stats import percentileofscore
# NOTE: Translated Chinese comments and prompts to English


def find_first_match_pattern(input_string, my_str):
    pattern = f'<{my_str}>(.*?)</{my_str}>'
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    else:
        raise Exception(f"Cannot find matched pattern given {my_str}")


def extract_novel_info(custom_id):
    # Use regex to extract novel name and chapter number
    match = re.match(r'(.*?)_(\d+)$', custom_id)
    if match:
        novel_name = match.group(1)
        chapter_num = int(match.group(2))
        return novel_name, chapter_num
    return None, None


def sort_jsonl_file(input_jsonl):
    # Parse each line and store it
    entries = []
    for data in input_jsonl:
        # print(data)
        novel_name, chapter_num = extract_novel_info(data['custom_id'])
        if novel_name and chapter_num is not None:
            entries.append((novel_name, chapter_num, data))

    # Sort by novel name and chapter number
    sorted_entries = sorted(entries, key=lambda x: (x[0], x[1]))
    sorted_entries_list = [entry[2] for entry in sorted_entries]
    return sorted_entries_list


def parsing_scores(scores_results):
    score_kinds = ['rhetoric', 'sensory', 'character_balance', 'dialogue_uniqueness', 'character_consistency',
                   'atmosphere', 'context', 'scene_transition']

    original_score = {
        'rhetoric': 0, 'sensory': 0, 'character_balance': 0,
        'dialogue_uniqueness': 0, 'character_consistency': 0,
        'atmosphere': 0, 'context': 0, 'scene_transition': 0
    }
    # Storage for parsed data
    valid_chapter = 0
    for json_data in scores_results:
        result = json_data["scores_txt"]
        try:
            scores = {
                'rhetoric': int(find_first_match_pattern(result, 'rhetoric')),
                'sensory': int(find_first_match_pattern(result, 'sensory')),
                'character_balance': int(find_first_match_pattern(result, 'character_balance')),
                'dialogue_uniqueness': int(find_first_match_pattern(result, 'dialogue_uniqueness')),
                'character_consistency': int(find_first_match_pattern(result, 'character_consistency')),
                'atmosphere': int(find_first_match_pattern(result, 'atmosphere')),
                'context': int(find_first_match_pattern(result, 'context')),
                'scene_transition': int(find_first_match_pattern(result, 'scene_transition'))
            }
            valid_chapter += 1
            for key in score_kinds:
                original_score[key] += scores[key]
        except Exception as e:
            tmp = json_data.get("custom_id")
            print(f"Error parsing scores for chapter in novel {tmp}: {e}")

    for key in score_kinds:
        original_score[key] = round(original_score[key] / valid_chapter, 2)

    return list(original_score.values())


def parsing_info(sourted_scores_results):
    # Storage for parsed data
    data_list = []
    for json_data in sourted_scores_results:
        result = json_data["scores_txt"]
        try:
            info = find_first_match_pattern(result, 'extracted')
            data_list.append(info)
        except Exception as e:
            tmp = json_data.get("custom_id")
            print(f"Error parsing scores for chapter in novel {tmp}: {e}")
            data_list.append(result)
    return data_list


async def chapter_worker(novel_name, worker_id, chapter):
    client = AsyncArk(api_key=os.environ.get("ARK_API_KEY"), timeout=3*3600)
    chapter_name = novel_name + '_' + str(worker_id)
    tmp_json = {
        "custom_id": chapter_name,
        "scores_txt": ''
    }
    if not os.path.exists(f'outputdir_famous/temp/{chapter_name}.json'):

        prompt = f"""Your task is to rate the novel from 1-5 based on the given criteria. Carefully read the following text: <novel> {chapter} </novel> Follow these steps when extracting information:
1. Read the entire text thoroughly
2. Identify the main characters who drive the story
3. Outline the main plot with key events
4. Determine important scenes where events take place
5. Verify the accuracy and completeness of the information
Output your extraction inside the <extracted> tag in the following format:
<main characters and lines>[list the main characters and their dialogue]</main characters and lines>
<main plot>[describe the main plot in order]</main plot>
<important scenes>[list the important scenes]</important scenes>
Ensure the information is rich and accurate. When scoring, follow these steps:
1. Score rhetorical techniques by the quantity and quality of complex devices (metaphor/symbolism/paradox)
2. Score sensory description richness based on visual, auditory and olfactory details
3. Score character balance by frequency, dialogue ratio and psychological portrayal
4. Score dialogue uniqueness by checking if lines show personality and remain distinct without names
5. Score character consistency by matching language and actions to background
6. Score atmosphere fit by analyzing how scenes support the overall mood
7. Score context suitability by checking environmental details for era and setting
8. Score scene transitions for smooth flow without abrupt jumps.
Provide your scores inside the <scores> tag in this format:
<rhetoric>1</rhetoric> <sensory>1</sensory> <character balance>1</character balance> <dialogue uniqueness>1</dialogue uniqueness> <character consistency>1</character consistency> <atmosphere>1</atmosphere> <context>1</context> <scene transition>1</scene transition>  Ensure the scores are accurate."""
        try:
            completion = await client.batch_chat.completions.create(
                model=os.environ.get("ARK_API_ID"),
                messages=[{
                    "role": "user",
                    "content": prompt}])
            tmp_json['scores_txt'] = completion.choices[0].message.content
        except Exception as e:
            print(f"Worker {worker_id} task {tmp_json['custom_id'][:20]} failed with error: {e}")
        else:
            print(f"Worker {worker_id} task {tmp_json['custom_id'][:20]} is completed.")
            with open(f'outputdir_famous/temp/{chapter_name}.json', 'w', encoding='utf-8') as f:
                json.dump(tmp_json, f, ensure_ascii=False, indent=2)
    else:
        with open(f'outputdir_famous/temp/{chapter_name}.json', 'r', encoding='utf-8') as f:
            tmp_json = json.load(f)
    return tmp_json


async def novel_worker(file_path: str) -> None:
    novel_name = file_path.split('/')[-1].split('.')[0]
    temp_output_filename = f"novel_data_temp_{novel_name}.json"
    json_structure = {
        "novel": novel_name,
        "original score": [],
        "normalized score": 0,
        "percentile": 0,
        "chapters": [],
        "novel_info": [],
        "request_id": str(uuid.uuid4())
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chapters = json.load(f)
        if not os.path.exists(f'outputdir_famous/{temp_output_filename}'):
            # Create task list
            tasks = [chapter_worker(novel_name, i, chapters[i]) for i in range(len(chapters))]
            # Wait for all tasks to complete
            scores_results_unparsed = await asyncio.gather(*tasks)
            sorted_result = sort_jsonl_file(scores_results_unparsed)
            scores_results = parsing_scores(sorted_result)
            novel_info = parsing_info(sorted_result)

            for chapter in chapters:
                json_structure["chapters"].append(chapter)

            json_structure["original score"] = scores_results
            json_structure["novel_info"] = novel_info

            with open('outputdir_famous/' + temp_output_filename, 'w', encoding='utf-8') as f:
                json.dump(json_structure, f, ensure_ascii=False, indent=2)
            print(f"Novel {novel_name} completed.")
        else:
            print(f'{temp_output_filename} detected.')
            with open('outputdir_famous/' + temp_output_filename, 'r', encoding='utf-8') as f:
                json_structure = json.load(f)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return json_structure


async def main(directory_path: str):
    json_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith('.json')
    ]

    # Create tasks for all novel files
    tasks = [novel_worker(file_path) for file_path in json_files]
    # Run all tasks concurrently and collect results
    novel_structures = await asyncio.gather(*tasks)

    # Save all results to a single JSON file
    output_file = os.path.join('novel_data', 'novel_data_subset-famous_raw.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(novel_structures, f, ensure_ascii=False, indent=2)


def process_new_data(new_data, score_key):
    new_data_std = (np.array(new_data[score_key]) - fixed_mean) / fixed_std
    new_combined_score = new_data_std.dot(fixed_weights)
    normalized_new_score = (new_combined_score - fixed_min_score) / (fixed_max_score - fixed_min_score)
    new_data_percentile = percentileofscore(fixed_normalized_existing_scores, normalized_new_score, kind='rank')
    return round(float(normalized_new_score), 4), round(float(new_data_percentile), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novel Scoring Script")
    parser.add_argument("--dir", type=str, default="famous_novels_chapter10_jsons", help="Path to the novel files")
    arg = parser.parse_args()
    
    # Path to the directory containing JSON files
    json_directory = arg.dir

    with open ("fixed_parameters.json", "r") as f:
        fixed_parameters = json.load(f)
    fixed_mean = np.array(fixed_parameters["mean"])
    fixed_std = np.array(fixed_parameters["std"])
    fixed_weights = np.array(fixed_parameters["weights"])
    fixed_min_score = fixed_parameters["min_score"]
    fixed_max_score = fixed_parameters["max_score"]
    fixed_normalized_existing_scores = np.array(fixed_parameters["normalized_existing_scores"])

    if not os.path.exists(f'novel_data/novel_data_subset-famous_raw.json'):
        # Run the async main function
        if sys.version_info >= (3, 11):
            with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
                runner.run(main(json_directory))
        else:
            uvloop.install()
            asyncio.run(main(json_directory))
    else:

        with open (f"novel_data/novel_data_subset-famous_raw.json", 'r') as f:
            novel_data = json.load(f)
        
        for novel in novel_data:
            norm_score, percentile = process_new_data(novel, 'original score')
            novel["normalized score"] = norm_score
            novel["percentile"] = percentile

        with open(f"scoring_results.json", 'w', encoding='utf-8') as f:
            json.dump(novel_data, f, ensure_ascii=False, indent=2)
