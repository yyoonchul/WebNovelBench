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


def find_first_match_pattern(input_string, my_str):
    pattern = f'<{my_str}>(.*?)</{my_str}>'
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    else:
        raise Exception(f"Cannot find matched pattern given {my_str}")


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
        # print(data)
        novel_name, chapter_num = extract_novel_info(data['custom_id'])
        if novel_name and chapter_num is not None:
            entries.append((novel_name, chapter_num, data))

    # 按小说名和章节号排序
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
    # 用于存储解析后的数据
    valid_chapter = 0
    for json_data in scores_results:
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
    # 用于存储解析后的数据
    data_list = []
    for json_data in sourted_scores_results:
        result = json_data["scores_txt"]
        try:
            info = find_first_match_pattern(result, '提取结果')
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
        prompt = f"""你的任务是根据给定的指标规则对小说进行评分(1-5)。请仔细阅读以下小说文本：<小说> {chapter} </小说> 在提取信息时，请遵循以下步骤：
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

        if len(prompt) > 65536:
            clip_len = len(chapter) + 65536 - len(chapter)
            prompt = f"""你的任务是根据给定的指标规则对小说进行评分(1-5)。请仔细阅读以下小说文本：<小说> {chapter[:clip_len]} </小说> 在提取信息时，请遵循以下步骤：
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
            # 创建任务列表
            tasks = [chapter_worker(novel_name, i, chapters[i]) for i in range(len(chapters))]
            # 等待所有任务完成
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
    
    # 指定包含 JSON 文件的目录路径
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
