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


SYSTEM_PROMPT1 = """
你是一个中文小说作家，你需要根据用户提供的信息进行扩写创作，创作需要满足下列条件：
1. 用户会用下面的格式给出长篇小说的主要人物、主要情节和主要场景，请仔细阅读用户提供的信息：
    <主要人物>[主要人物的名字]</主要人物> 
    <主要情节>[主要情节，按照事件发展顺序]</主要情节> 
    <重要场景>[重要场景的名称]</重要场景> 
2. 评论家会根据下列标准打分：
    根据复杂修辞（隐喻/象征/悖论）的数量与质量提炼度，对修辞手法评分 
    根据文本中的视觉、听觉、嗅觉等描写数量，对感官描述丰富度评分 
    统计每个角色在生成内容中的出现频率、对话占比、心理描写和评估人物描述的平衡度，对角色平衡度评分 
    查看角色台词是否能反映本身个性，遮住名字后是否有区分度，对角色对白独特性评分 
    分析角色语言、动作是否匹配其身份和背景，对角色一致性评分 
    通过情感色谱分析，检查场景描写是否服务于整体氛围，对意境匹配度评分 
    通过分析环境细节是否适应时代/地域背景，对语境适配度评分 
    评估生成内容是否自然衔接不同场景从而避免场景割裂，对跨场景衔接度评分
3. 只需按照指定格式返回生成的小说：
    <text>你生成的小说内容</text>
"""

SYSTEM_PROMPT2 = """
你是一个中文小说作家，你需要根据用户提供的信息进行扩写创作，创作需要满足下列条件：
1. 用户会用下面的格式给出长篇小说的主要人物、主要情节和主要场景，请仔细阅读用户提供的信息：
    <主要人物>[主要人物的名字]</主要人物> 
    <主要情节>[主要情节，按照事件发展顺序]</主要情节> 
    <重要场景>[重要场景的名称]</重要场景> 
2. 人物间可能存在对话，必要时请编写恰当的人物对话
3. 只需按照指定格式返回生成的小说：
    <text>你生成的小说内容</text>
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
    # 使用正则表达式提取小说名和章节号
    match = re.match(r'(.*?)_(\d+)$', custom_id)
    if match:
        novel_name = match.group(1)
        chapter_num = int(match.group(2))
        return novel_name, chapter_num
    return None, None

def sort_jsonl_file(input_jsonl):
    # 解析每一行并存储
    entries = []
    for data in input_jsonl:
        novel_name, chapter_num = extract_novel_info(data['custom_id'])
        if novel_name and chapter_num is not None:
            entries.append((novel_name, chapter_num, data))
    
    # 按小说名和章节号排序
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
"content": f"""你的任务是根据给定的指标规则对小说进行评分(1-5)。请仔细阅读以下小说文本：<小说> {predictions[i]} </小说> 在提取信息时，请遵循以下步骤：
1. 仔细通读整个小说文本 
2. 识别出主要人物，主要人物是在小说中起到关键作用、有较多情节围绕的角色 
3. 梳理主要情节，主要情节是推动故事发展的核心事件和关键转折 
4. 确定重要场景，重要场景是故事发生的关键地点和环境 
5. 检查提取的信息是否准确和完整  
请在<提取结果>标签内输出你的提取结果，格式如下：
<主要人物及其对白>[列出主要人物的名字和对白]</主要人物及其对白> 
<主要情节>[详细描述主要情节，按照事件发展顺序]</主要情节> 
<重要场景>[列出重要场景的名称]</重要场景> 
请确保提取的信息丰富、全面且准确。在评分时，请遵循以下步骤：
1. 根据复杂修辞（隐喻/象征/悖论）的数量与质量提炼度，给出修辞手法评分 
2. 根据文本中的视觉、听觉、嗅觉等描写数量，给出感官描述丰富度评分 
3. 统计每个角色在生成内容中的出现频率、对话占比、心理描写和评估人物描述的平衡度，给出角色平衡度评分 
4. 查看角色台词是否能反映本身个性，遮住名字后是否有区分度，给出角色对白独特性评分 
5. 分析角色语言、动作是否匹配其身份和背景，给出角色一致性评分 
6. 通过情感色谱分析，检查场景描写是否服务于整体氛围，给出意境匹配度评分 
7. 通过分析环境细节是否适应时代/地域背景，给出语境适配度评分 
8. 评估生成内容是否自然衔接不同场景从而避免场景割裂，给出跨场景衔接度评分。
请在<评分结果>标签内输出你的评分结果，格式如下：
<修辞手法评分>1</修辞手法评分> <感官描述丰富度评分>1</感官描述丰富度评分> <角色平衡度评分>1</角色平衡度评分> <角色对白独特性评分>1</角色对白独特性评分> <角色一致性评分>1</角色一致性评分> <意境匹配度评分>1</意境匹配度评分> <语境适配度评分>1</语境适配度评分> <跨场景衔接度评分>1</跨场景衔接度评分>  请确保评分全面且准确符合要求。"""
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

    # 创建任务列表
    tasks = [worker(i, sub_taskes[i]) for i in range(task_num)]

    # 等待所有任务完成
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

    # 用于存储解析后的数据
    data_list = []
    for json_data in scores_results:

        novel_index = json_data["custom_id"].split("_")[0]
        result = json_data["scores_txt"]
        try:
            scores = {
                        'rhetoric': int(find_first_match_pattern(result, '修辞手法评分')),
                        'sensory': int(find_first_match_pattern(result, '感官描述丰富度评分')),
                        'character_balance': int(find_first_match_pattern(result, '角色平衡度评分')),
                        'dialogue_uniqueness': int(find_first_match_pattern(result, '角色对白独特性评分')),
                        'character_consistency': int(find_first_match_pattern(result, '角色一致性评分')),
                        'atmosphere': int(find_first_match_pattern(result, '意境匹配度评分')),
                        'context': int(find_first_match_pattern(result, '语境适配度评分')),
                        'scene_transition': int(find_first_match_pattern(result, '跨场景衔接度评分'))
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