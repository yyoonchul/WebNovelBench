import json
import pdb
import re
import os
import glob
import openai
from openai import OpenAI
import itertools
import argparse
import numpy as np
from tqdm import tqdm
from time import sleep
from scipy.stats import percentileofscore
import asyncio
import sys
from datetime import datetime

import uvloop
from volcenginesdkarkruntime import AsyncArk

# NOTE: Translated Chinese comments and prompts to English

SYSTEM_PROMPT1 = """
You are a Chinese novel writer. Expand the story according to the user's information under these conditions:
1. The user provides the main characters, main plot and important scenes in this format:
    <main characters>[names of the main characters]</main characters>
    <main plot>[main plot in order]</main plot>
    <important scenes>[names of important scenes]</important scenes>
2. The critic scores using these standards:
    Score rhetorical techniques by the number and quality of complex devices (metaphor/symbolism/paradox)
    Score sensory richness by counting visual, auditory and olfactory descriptions
    Score character balance by each character's frequency, dialogue ratio and psychological portrayal
    Score dialogue uniqueness by checking if lines show personality and remain distinct without names
    Analyze whether characters' language and actions match their background for consistency
    Check atmosphere fit through emotional analysis of scenes
    Evaluate context suitability with environment details
    Assess scene transitions to avoid abrupt cuts
3. Return only the generated novel in this format:
    <text>Your generated novel content</text>
"""

SYSTEM_PROMPT2 = """
You are a Chinese novel writer. Expand the story using the user's information with these rules:
1. The user provides the main characters, main plot and important scenes in the same format as above:
    <main characters>[names of the main characters]</main characters>
    <main plot>[main plot in order]</main plot>
    <important scenes>[names of important scenes]</important scenes>
2. Dialogues may appear between characters; write appropriate lines when needed.
3. Return only the generated novel in this format:
    <text>Your generated novel content</text>
"""

def load_config(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_directories():
    os.makedirs('outputdir/temp', exist_ok=True)

def generate_novel(config, suffix, time_padding=None):
    # Initialize OpenAI client for generation
    model_name = config['generator']['model_name']
    url = config['generator']['url']
    api_key = config['generator']['api_key']
    max_tokens = config['generator']['max_tokens']
    client = openai.Client(base_url=url, api_key=api_key)
    
    # Read the input JSON file
    with open(config['generator']['input_name'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    sys_prompt = SYSTEM_PROMPT1 if 'explicit' in suffix else SYSTEM_PROMPT2

    # Process each novel in the data
    def process_single_novel(novel_data, thread_id):  # Add thread_id parameter
        request_id = novel_data.get('request_id', 'unknown')
        novel_name = novel_data.get('novel', 'unknown')
        result_data = {
            "novel": novel_name,
            "original score": novel_data.get('original score', []),
            "normalized score": novel_data.get('normalized score', 'unknown'),
            "percentile": novel_data.get('percentile', 'unknown'),
            "request_id": request_id,
            "predict": []
        }
        
        # Check if this novel has already been processed
        temp_file = f'outputdir/temp/{request_id}_model-{suffix}.json'
        if os.path.exists(temp_file):
            print(f"Skipping already processed novel with request_id: {request_id}")
            return
        
        # Create a position-specific progress bar for each novel
        pbar = tqdm(
            novel_data.get('novel_info', []),
            desc=f"Novel: {novel_name}",
            position=thread_id + 1,  # Use thread_id + 1 for positioning (0 is overall progress)
            leave=True,
            mininterval=0.3
        )
        
        for info in pbar:
            start_time = datetime.now()  # Add time tracking
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": info}
                ],
                
                temperature=config['generator']['temperature'],
                # presence_penalty=0.2,
                # frequency_penalty=0.2,
                max_tokens=max_tokens,
                n=1
                # seed=42
            )
            
            result = response.choices[0].message.content
            if '</think>\n\n' in result:
                result = result.split('</think>\n\n')[1]
            match = re.search(r'<text>(.*?)</text>', result, re.DOTALL)
            if match:
                content = match.group(1)
            else:
                content = result
            result_data['predict'].append(content)
            
            # Update progress bar description with current chapter
            pbar.set_description(f"Novel: {novel_name} | Chapter {len(result_data['predict'])}/{len(novel_data.get('novel_info', []))}")
        
            # After processing is complete, check if minimum time has elapsed
            if time_padding is not None:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if elapsed_time < time_padding:  # If less than 1 minute has passed
                    sleep_time = time_padding - elapsed_time + 1
                    print(f"Thread {thread_id} sleeping for {sleep_time:.2f} seconds to meet minimum time requirement")
                    sleep(sleep_time)
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

    # Get number of threads from config
    num_threads = config['generator'].get('num_threads', 4)  # Default to 4 if not specified
    
    # Create overall progress bar
    overall_pbar = tqdm(total=len(data), desc="Overall Progress", position=0, leave=True)
    
    # Create thread pool and process novels
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for idx, novel in enumerate(data):
            # Pass thread_id (idx % num_threads) to ensure unique positions
            future = executor.submit(process_single_novel, novel, idx % num_threads)
            future.add_done_callback(lambda p: overall_pbar.update(1))
            futures.append(future)
        
        # Wait for all futures to complete
        for future in futures:
            future.result()
    
    overall_pbar.close()

    # Combine all temp files
    final_data = []
    temp_dir = 'outputdir/temp'
    for temp_file in os.listdir(temp_dir):
        if temp_file.endswith(f'{suffix}.json') and 'scores' not in temp_file:
            with open(os.path.join(temp_dir, temp_file), 'r', encoding='utf-8') as f:
                final_data.append(json.load(f))

    # Save final results
    data_source = config['generator']['input_name'].split('/')[-1].split('.')[0]
    output_name = f'{data_source}_model-{suffix}'
    output_path = f'outputdir/{output_name}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    return output_name

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
        novel_name, chapter_num = extract_novel_info(data['custom_id'])
        if novel_name and chapter_num is not None:
            entries.append((novel_name, chapter_num, data))
    
    # Sort by novel name and chapter number
    sorted_entries = sorted(entries, key=lambda x: (x[0], x[1]))
    sorted_entries_list = [entry[2] for entry in sorted_entries]
    return sorted_entries_list

def transform_novel_data(input_file):
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # List to store transformed data
    transformed_data = []
    
    # Process each novel
    for novel in data:
        # Get predictions
        novel_name = novel['novel']
        predictions = novel['predict']
        
        # Create conversation pairs
        for i in range(0, len(predictions)):  # Process in pairs
            
            conversation = {
                "custom_id": novel_name + "_" + str(i),
                "body": {
                    "messages": [
                        {
                            "role": "user",
"content": f"""Your task is to rate the novel from 1-5 based on the given criteria. Carefully read the following text: <novel> {predictions[i]} </novel> Follow these steps when extracting information:
1. Read the entire text thoroughly
2. Identify the main characters who drive the story
3. Outline the main plot with key events
4. Determine important scenes where events take place
5. Check the accuracy and completeness of the information
Provide your extraction inside the <extracted> tag in the following format:
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
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.3,
                    "frequency_penalty": 0.2,
                    "seed": 42
                }
            }
            
            transformed_data.append(conversation)

    return transformed_data

def find_first_match_pattern(input_string, my_str):
    pattern = f'<{my_str}>(.*?)</{my_str}>'
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    else:
        raise Exception(f"Cannot find matched pattern given {my_str}")

def process_new_data(new_data, score_key):
    new_data_std = (np.array(new_data[score_key]) - fixed_mean) / fixed_std
    new_combined_score = new_data_std.dot(fixed_weights)
    normalized_new_score = (new_combined_score - fixed_min_score) / (fixed_max_score - fixed_min_score)
    new_data_percentile = percentileofscore(fixed_normalized_existing_scores, normalized_new_score, kind='rank')
    return round(float(normalized_new_score), 4), round(float(new_data_percentile), 2)

async def worker(worker_id, sub_requst_list):
    # print(f"Worker {worker_id} is starting.")
    client = AsyncArk(api_key=os.environ.get("ARK_API_KEY"), timeout=3*3600)
    results_list = []
    for requst_content in sub_requst_list:
        print(f"Worker {requst_content['custom_id'][:20]} is running.")
        tmp_json = {
            "custom_id": requst_content['custom_id'],
            "scores_txt": ''
        }
        try:
            completion = await client.batch_chat.completions.create(
                model=os.environ.get("ARK_API_ID"),
                messages=requst_content['body']['messages'],
            )
            tmp_json['scores_txt'] = completion.choices[0].message.content
        except Exception as e:
            print(f"Worker {worker_id} task {requst_content['custom_id'][:20]} failed with error: {e}")
            raise e
        else:
            print(f"Worker {worker_id} task {requst_content['custom_id'][:20]} is completed.")
        results_list.append(tmp_json)
    # print(f"Worker {worker_id} is completed.")
    return results_list

async def critic(request_list, max_concurrent_tasks = 10):
    start = datetime.now()
    task_num = int(np.ceil(len(request_list) / max_concurrent_tasks))
    sub_taskes = [request_list[i:i + max_concurrent_tasks] for i in range(0, len(request_list), max_concurrent_tasks)]

    # Create task list
    tasks = [worker(i, sub_taskes[i]) for i in range(task_num)]

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    scores_results = list(itertools.chain(*results))
    sorted_scores_results = sort_jsonl_file(scores_results)
    end = datetime.now()
    print(f"Total time: {end - start}, Total task: {max_concurrent_tasks * task_num}")
    
    return sorted_scores_results

def load_reference_order(reference_file):
    """Load the reference order from subset_d.json"""
    with open(reference_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Extract novel names in order
        return {item['novel']: item for item in data}

def merge_score_files(reference_file):
    # Get reference order
    novel_order = load_reference_order(reference_file)
    
    # Create ordered merged data
    merged_data = []
    for novel_name in novel_order.keys():
        tmp = {}
        tmp['novel'] = novel_name
        tmp['normalized score'] = novel_order[novel_name].get('normalized score')
        tmp['original score'] = novel_order[novel_name].get('original score')
        tmp['percentile'] = novel_order[novel_name].get('percentile')
        tmp["pred score"] = []
        tmp["pred norm score"] = None
        tmp["pred percentile"] = None
        merged_data.append(tmp)
    
    return merged_data
    
def parsing_scores(scores_results, merged_score):
    score_kinds = ['rhetoric', 'sensory', 'character_balance', 'dialogue_uniqueness', 'character_consistency', 'atmosphere', 'context', 'scene_transition']
    merged_score_map = {item['novel']: item for item in merged_score}
    tmp_dict = {item['novel']: {
                    "novel": item['novel'],
                    "pred score": {
                    'rhetoric': 0, 'sensory': 0, 'character_balance': 0,
                    'dialogue_uniqueness': 0, 'character_consistency': 0,
                    'atmosphere': 0, 'context': 0, 'scene_transition': 0
                },
                "valid_chapter": 0} for item in merged_score}

    # Storage for parsed data
    data_list = []
    for json_data in scores_results:

        novel_index = json_data["custom_id"].split("_")[0]
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
            tmp_dict[novel_index]['valid_chapter'] += 1
            for key in score_kinds:
                tmp_dict[novel_index]['pred score'][key] += scores[key]
        except Exception as e:
            tmp = json_data.get("custom_id")
            print(f"Error parsing scores for chapter in novel {tmp}: {e}")
    
    for key in tmp_dict.keys():
        for score_kind in score_kinds:
            tmp_dict[key]['pred score'][score_kind] = round(tmp_dict[key]['pred score'][score_kind] / tmp_dict[key]['valid_chapter'], 2)

        merged_score_map[key]["pred score"] = list(tmp_dict[key]['pred score'].values())
        tmp_dict[key]["pred score"] = list(tmp_dict[key]['pred score'].values())
        pred_norm_score, pred_percentile = process_new_data(tmp_dict[key], 'pred score')
        merged_score_map[key]["pred norm score"] = pred_norm_score
        merged_score_map[key]["pred percentile"] = pred_percentile

    return list(merged_score_map.values())

def get_args():
    parser = argparse.ArgumentParser(description="Novel Generation and Critique Pipeline")
    parser.add_argument("--config", type=str, default="", help="Path to the configuration file")
    parser.add_argument('--hide_standard', '-hs', action='store_true', default=False, help='whether hide the standard')
    parser.add_argument("--generate_only", '-g', action='store_true', help='whether generate only')
    parser.add_argument("--critic_only", '-c', action='store_true', help='whether critic only')
    parser.add_argument("--model", '-m', type=str, default=None, help="only used for critic")
    parser.add_argument("--data", '-d', type=str, default="", help="only used for critic")
    parser.add_argument("--time_padding", '-td', type=int, default=None, help='time padding for RPM limit')
    return parser.parse_args()

def main():
    args = get_args()
    assert not (args.generate_only and args.critic_only), "-g and -c should not be activated simultaneously."
    if len(args.config) > 0:
        config = load_config(args.config)
        
    elif len(args.model) == 0:
        raise Exception("No config file or model name spercified.")
    else:
        print('Without a configuration file, only the critic process is allowed.')
        args.critic_only = True
        config = {'generator': {}}
        config['generator']['input_name'] = args.data
        config['generator']['note'] = args.model

    setup_directories()
    suffix = config['generator']['note']
    if not args.hide_standard:
        suffix += "_explicit"

    generated_novel = config['generator']['input_name'].split('/')[-1].split('.')[0] + '_model-' + suffix

    print(f"model suffix: {suffix}, data sorce: {args.data}")
    sleep(2)
  
    if not args.critic_only:
        if not os.path.exists(f'outputdir/{generated_novel}.json'):
            ### generate
            print("\nGeneration Started")
            generated_novel = generate_novel(config, suffix, time_padding=args.time_padding)
        else:
            print(f"generation file detected: {generated_novel}")

    if not args.generate_only:
        
        temp_output_filename = f"{generated_novel}_scores_unparsed.jsonl"
        ### match the generated conten with original subset
        # Paths
        temp_dir = "outputdir"
        reference_file = config['generator']['input_name']
        # Merge files
        matched_novel_data = merge_score_files(reference_file)

        if not os.path.exists(f'{temp_dir}/{temp_output_filename}'):
            ### criticize novels
            input_file = f"outputdir/{generated_novel}.json"
            request_list = transform_novel_data(input_file)
            print('\nbegin criticizing')
            # pdb.set_trace()
            if sys.version_info >= (3, 11):
                with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
                    sorted_scores_results = runner.run(critic(request_list))
            else:
                uvloop.install()
                sorted_scores_results = asyncio.run(critic(request_list))

            ### save score temp file
            with open('outputdir/' + temp_output_filename, 'w', encoding='utf-8') as f:
                for line in sorted_scores_results:
                    json_line = json.dumps(line, ensure_ascii=False)
                    f.write(json_line + '\n')
        else:
            print(f"scores file detected: {temp_output_filename}")
            with open('outputdir/' + temp_output_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sorted_scores_results = [json.loads(line) for line in lines]

        ### save results
        # pdb.set_trace()
        merged_scores = parsing_scores(sorted_scores_results, matched_novel_data)
        output_filename = f"{generated_novel}_scores.json"
        output_path = os.path.join('outputdir/scores_formed', output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_scores, f, ensure_ascii=False, indent=2)

    print("\nPipeline completed")


if __name__ == "__main__":
    with open ("fixed_parameters.json", "r") as f:
        fixed_parameters = json.load(f)
    
    fixed_mean = np.array(fixed_parameters["mean"])
    fixed_std = np.array(fixed_parameters["std"])
    fixed_weights = np.array(fixed_parameters["weights"])
    fixed_min_score = fixed_parameters["min_score"]
    fixed_max_score = fixed_parameters["max_score"]
    fixed_normalized_existing_scores = np.array(fixed_parameters["normalized_existing_scores"])
    main()