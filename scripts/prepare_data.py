#!/usr/bin/env python3
"""
准备训练数据：合并不同的数据文件，创建统一的训练集

"""
import os
import json
import argparse
import logging
from pathlib import Path
import random
import math

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_json_data(file_path):
    """加载JSON数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_jsonl_data(file_path):
    """加载JSONL数据文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_data(file_path):
    """自动检测文件格式并加载数据"""
    if file_path.endswith('.jsonl'):
        return load_jsonl_data(file_path)
    else:
        return load_json_data(file_path)


def merge_datasets(data_files, output_file, max_samples_per_class=None):
    """
    合并多个数据集
    
    Args:
        data_files: 数据文件路径列表
        output_file: 输出文件路径
        max_samples_per_class: 每个类别的最大样本数（用于平衡数据集）
    """
    logger.info("=" * 60)
    logger.info("合并数据集")
    logger.info("=" * 60)
    
    train_all_data = []
    eval_all_data = []
    stats = {}
    
    # 加载所有数据文件
    for file_path in data_files:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在，跳过: {file_path}")
            continue
        
        logger.info(f"加载文件: {file_path}")
        data = load_data(file_path)
        logger.info(f"  样本数: {len(data)}")
        
        # 统计每个输出类别的数量
        for item in data:
            output = item.get('output', '')
            stats[output] = stats.get(output, 0) + 1
        cut_index = math.floor(len(data) * 0.9)
        train_all_data.extend(data[:cut_index])
        eval_all_data.extend(data[cut_index:])

    logger.info(f"\n总样本数: {len(train_all_data) + len(eval_all_data)}")
    logger.info(f"训练集样本数: {len(train_all_data)}")
    logger.info(f"验证集样本数: {len(eval_all_data)}")
    logger.info("\n类别分布:")
    for output, count in sorted(stats.items()):
        logger.info(f"  {output}: {count} 样本")
    
    
    # 如果需要平衡数据集
    if max_samples_per_class:
        logger.info(f"\n平衡数据集（每类最多 {max_samples_per_class} 个样本）")
        
        balanced_data = []
        class_counts = {}
        
        for item in train_all_data:
            output = item.get('output', '')
            count = class_counts.get(output, 0)
            
            if count < max_samples_per_class:
                balanced_data.append(item)
                class_counts[output] = count + 1
        
        all_data = balanced_data
        
        logger.info(f"平衡后样本数: {len(all_data)}")
        logger.info("\n平衡后类别分布:")
        for output, count in sorted(class_counts.items()):
            logger.info(f"  {output}: {count} 样本")
    
    # 保存合并后的数据
    logger.info(f"\n保存到: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file.replace('.json', '_train.json'), 'w', encoding='utf-8') as f:
        for i in range(10):
            random.shuffle(train_all_data)
        json.dump(train_all_data, f, ensure_ascii=False, indent=2)
    with open(output_file.replace('.json', '_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(eval_all_data, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info("完成！")
    logger.info("=" * 60)
    
    return all_data


def main():
    parser = argparse.ArgumentParser(description="合并训练数据")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="输入数据文件路径列表"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/code/semaintic-vad/data/bsiness_v1/train_all.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="每个类别的最大样本数（用于平衡数据集）"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["single", "multi", "all"],
        help="使用预设配置：single（单轮数据）、multi（多轮数据）、all（全部数据）"
    )
    
    args = parser.parse_args()
    
    # 预设配置
    base_dir = "/code/semaintic-vad/data/bsiness_v1"
    
    if args.preset == "single":
        # 单轮对话数据
        data_files = [
            f"{base_dir}/sft_single_c_l.json",
            f"{base_dir}/sft_single_s_s.json",
        ]
    elif args.preset == "multi":
        # 多轮对话数据
        data_files = [
            f"{base_dir}/sft_multi_c_l.json",
            f"{base_dir}/sft_multi_c_s.json",
            f"{base_dir}/sft_multi_s_l.json",
            f"{base_dir}/sft_multi_s_s.json",
        ]
    elif args.preset == "all":
        # 全部数据
        data_files = [
            f"{base_dir}/sft_multi_actively_interrupt_s_l.json",
            f"{base_dir}/sft_multi_business_vocabulary_s_l.json",
            f"{base_dir}/sft_multi_control_s_l.json",
            f"{base_dir}/sft_multi_music_ancient_poems_c_s.json",
            f"{base_dir}/sft_multi_music_s_l.json",
            f"{base_dir}/sft_multi_sleep_s_l.json",
            f"{base_dir}/sft_multi_story_s_l.json",
            f"{base_dir}/sft_multi_weather_s_l.json",
            f"{base_dir}/sft_multi_time_s_l.json",
            
        ]
    else:
        # 使用命令行指定的文件
        if not args.input:
            logger.error("错误：请使用 --input 指定输入文件，或使用 --preset 选择预设配置")
            return
        data_files = args.input
    
    # 合并数据集
    merge_datasets(data_files, args.output, args.max_samples)


if __name__ == "__main__":
    main()

