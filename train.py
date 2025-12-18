#!/usr/bin/env python3
"""
使用 LoRA / 全量微调 Qwen2.5-0.5B-Instruct 模型进行对话状态分类
"""
import os
import sys
import json
import yaml
import torch
import numpy as np
import gc
import traceback
import psutil
from dataclasses import dataclass, field
from typing import Optional, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    EvalPrediction,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import confusion_matrix, classification_report
import logging

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



@dataclass
class ModelArguments:
    """模型参数"""
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "扩展后的模型路径"}
    )
    config_file: str = field(
        default="/root/code/semaintic-vad/config/config.yaml",
        metadata={"help": "配置文件路径"}
    )


@dataclass
class DataArguments:
    """数据参数"""
    train_data_path: str = field(
        default=None,
        metadata={"help": "训练数据路径"}
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "验证数据路径"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )


@dataclass
class TrainingModeArguments:
    """训练模式参数"""
    mode: str = field(
        default="lora",
        metadata={"help": "训练模式: lora 或 full"}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    target_modules: str = field(
        default="q_proj,k_proj,o_proj",
        metadata={"help": "LoRA target modules，逗号分隔"}
    )


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class CustomTrainer(Trainer):
    """自定义Trainer，添加更详细的错误处理和内存监控"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_batch_counter = 0
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """重写评估循环，添加详细的异常捕获"""
        try:
            logger.info(f"[{metric_key_prefix}] 开始评估循环...")
            logger.info(f"[{metric_key_prefix}] 数据集大小: {len(dataloader.dataset) if hasattr(dataloader, 'dataset') else '未知'}")
            logger.info(f"[{metric_key_prefix}] Batch数量: {len(dataloader)}")
            
            self.eval_batch_counter = 0
            result = super().evaluation_loop(
                dataloader=dataloader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{metric_key_prefix}] 评估循环中发生错误: {e}")
            logger.error(f"[{metric_key_prefix}] 错误发生在第 {self.eval_batch_counter} 个batch")
            logger.error(f"[{metric_key_prefix}] 完整错误堆栈:\n{traceback.format_exc()}")
            raise
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """重写预测步骤，添加异常捕获，并优化内存使用"""
        try:
            self.eval_batch_counter += 1
            
            # 每10个batch记录一次
            if self.eval_batch_counter % 10 == 0:
                logger.info(f"[prediction_step] 正在处理第 {self.eval_batch_counter} 个batch")
            
            # 调用父类方法
            with torch.no_grad():  # 确保不计算梯度
                loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            
            # 关键修复：根据每个序列的实际长度提取正确位置的 logits
            if logits is not None:
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                if isinstance(logits, torch.Tensor) and logits.dim() == 3:                    
                    batch_size = logits.shape[0]
                    vocab_size = logits.shape[2]
                    
                    # 为每个样本提取需要的 logits（special_token 前一个位置）
                    # 使用 float32 避免 bfloat16 -> float16 的精度问题
                    extracted_logits = torch.zeros(batch_size, 1, vocab_size, dtype=torch.float32, device='cpu')
                    
                    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
                    
                    # 调试：打印第一个 batch 的信息
                    if self.eval_batch_counter == 1:
                        logger.info(f"[prediction_step 调试] logits.shape: {logits.shape}, logits.dtype: {logits.dtype}")
                        logger.info(f"[prediction_step 调试] labels.shape: {labels_np.shape}")
                    
                    for i in range(batch_size):
                        # 找到非 -100 的位置
                        valid_positions = np.where(labels_np[i] != -100)[0]
                        
                        # 调试：打印第一个 batch 的第一个样本信息
                        if self.eval_batch_counter == 1 and i == 0:
                            logger.info(f"[prediction_step 调试] 样本0 valid_positions: {valid_positions}")
                            logger.info(f"[prediction_step 调试] 样本0 valid labels: {labels_np[i][valid_positions]}")
                        
                        if len(valid_positions) >= 2:
                            # special_token 在倒数第二个有效位置
                            # 要预测 special_token，需要用它前一个位置的 logits
                            special_token_pos = valid_positions[-2]
                            logits_pos = special_token_pos - 1
                            if logits_pos >= 0:
                                extracted_logits[i, 0, :] = logits[i, logits_pos, :].cpu().float()
                            else:
                                # 如果 special_token 是第一个位置，使用它自己的位置
                                extracted_logits[i, 0, :] = logits[i, special_token_pos, :].cpu().float()
                            
                            # 调试：打印第一个样本的提取信息
                            if self.eval_batch_counter == 1 and i == 0:
                                logger.info(f"[prediction_step 调试] special_token_pos: {special_token_pos}, logits_pos: {logits_pos}")
                                logger.info(f"[prediction_step 调试] extracted logits 范围: min={extracted_logits[i, 0, :].min():.4f}, max={extracted_logits[i, 0, :].max():.4f}")
                        elif len(valid_positions) == 1:
                            pos = valid_positions[0]
                            logits_pos = pos - 1 if pos > 0 else pos
                            extracted_logits[i, 0, :] = logits[i, logits_pos, :].cpu().float()
                    
                    logits = extracted_logits
                else:
                    if isinstance(logits, torch.Tensor):
                        logits = logits.cpu().float()
            
            # labels也移到CPU
            if labels is not None and isinstance(labels, torch.Tensor):
                labels = labels.cpu()
            
            
            return (loss, logits, labels)
            
        except Exception as e:
            logger.error(f"[prediction_step] 第 {self.eval_batch_counter} 个batch发生错误: {e}")
            logger.error(f"[prediction_step] inputs keys: {inputs.keys() if isinstance(inputs, dict) else 'not a dict'}")
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    if hasattr(value, 'shape'):
                        logger.error(f"[prediction_step]   {key}.shape: {value.shape}")
                    if hasattr(value, 'dtype'):
                        logger.error(f"[prediction_step]   {key}.dtype: {value.dtype}")
            logger.error(f"[prediction_step] 完整错误堆栈:\n{traceback.format_exc()}")
            
            raise


def extract_predicted_token(logits, tokenizer, special_tokens):
    """
    从模型输出的logits中提取预测的特殊token
    
    Args:
        logits: 模型输出的logits [batch_size, seq_len, vocab_size]
        tokenizer: tokenizer对象
        special_tokens: 特殊token列表
        
    Returns:
        预测的token ID列表
    """
    # 获取特殊token的ID
    special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
    
    # 对于每个样本，找到最后一个非padding位置的预测
    predictions = []
    for sample_logits in logits:
        # 获取最后一个token位置的预测
        last_token_logits = sample_logits[-1]  # [vocab_size]
        
        # 在特殊token中找到概率最大的
        special_logits = last_token_logits[special_token_ids]
        predicted_idx = torch.argmax(special_logits).item()
        predicted_token_id = special_token_ids[predicted_idx]
        
        predictions.append(predicted_token_id)
    
    return predictions


def create_compute_metrics(tokenizer, special_tokens, label_names):
    """
    创建计算评估指标的函数
    
    Args:
        tokenizer: tokenizer对象
        special_tokens: 特殊token列表 ["<|s_s|>", "<|s_l|>", "<|c_s|>", "<|c_l|>"]
        label_names: 标签名称列表
        
    Returns:
        compute_metrics函数
    """
    special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
    
    def compute_metrics(eval_pred: EvalPrediction) -> Dict:
        """
        计算评估指标，包括准确率和混淆矩阵
        """
        try:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
            
            logger.info(f"[compute_metrics] logits type: {type(logits)}, shape: {logits[0].shape if isinstance(logits, tuple) else logits.shape}")
            logger.info(f"[compute_metrics] labels shape: {labels.shape}")
            
            # 如果logits是元组，取第一个元素
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # logits shape: [batch_size, 1, vocab_size] (已在 prediction_step 中提取正确位置)
            # labels shape: [batch_size, seq_len]
            
            predictions = []
            true_labels = []
            
            # 处理每个样本
            logger.info(f"[compute_metrics] 开始处理 {len(labels)} 个样本...")
            logger.info(f"[compute_metrics] special_token_ids: {special_token_ids}")
            logger.info(f"[compute_metrics] special_tokens 映射: {dict(zip(special_tokens, special_token_ids))}")
            
            # 分批处理以减少内存峰值
            batch_size_for_processing = 100
            for batch_start in range(0, len(labels), batch_size_for_processing):
                batch_end = min(batch_start + batch_size_for_processing, len(labels))
                
                for i in range(batch_start, batch_end):
                    try:
                        # 找到非-100的标签位置（即目标token）
                        valid_positions = labels[i] != -100
                        if not valid_positions.any():
                            continue
                        
                        # 获取真实标签的 token_id
                        # 序列结构: [..., special_token, <|im_end|>]
                        # labels 中非-100部分: [special_token_id, im_end_id]
                        # 所以目标 special_token 是倒数第二个有效 label
                        valid_label_ids = labels[i][valid_positions]
                        
                        # 取倒数第二个（目标special token），而不是最后一个（<|im_end|>）
                        if len(valid_label_ids) >= 2:
                            true_token_id = int(valid_label_ids[-2])  # 目标 special token 的 token_id
                        elif len(valid_label_ids) == 1:
                            true_token_id = int(valid_label_ids[0])
                        else:
                            continue
                        
                        # 只处理特殊token
                        if true_token_id not in special_token_ids:
                            continue
                        
                        # logits[i] 已经在 prediction_step 中提取为正确位置的 logits
                        # shape: [1, vocab_size]
                        if logits[i].ndim == 2:
                            target_logits = logits[i][0]  # [vocab_size]
                        else:
                            target_logits = logits[i]
                        
                        # 从 special_tokens 中取 argmax 得到预测的 token_id
                        # 这样只在 4 个类别中做选择，更符合分类任务的评估方式
                        special_logits = np.array([target_logits[tid] for tid in special_token_ids])
                        predicted_idx = int(np.argmax(special_logits))
                        predicted_token_id = special_token_ids[predicted_idx]
                        
                        # 存储真实的 token_id 和预测的 token_id
                        true_labels.append(true_token_id)
                        predictions.append(predicted_token_id)
                        
                        # 打印前 10 个样本的详细信息用于调试
                        if len(predictions) <= 10:
                            true_token = tokenizer.convert_ids_to_tokens(true_token_id)
                            pred_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
                            is_correct = predicted_token_id == true_token_id
                            # 获取在 special tokens 中各自的概率
                            special_probs = {
                                special_tokens[j]: float(target_logits[special_token_ids[j]]) 
                                for j in range(len(special_tokens))
                            }
                            logger.info(f"[调试样本{len(predictions)}] 真实: {true_token}({true_token_id}), "
                                       f"预测: {pred_token}({predicted_token_id}), "
                                       f"正确: {is_correct}")
                            logger.info(f"  special_token logits: {special_probs}")
                            logger.info(f"  special_logits array: {special_logits}")
                            logger.info(f"  argmax index: {predicted_idx}")
                            # 打印 logits 的整体范围
                            logger.info(f"  target_logits 范围: min={float(target_logits.min()):.4f}, max={float(target_logits.max()):.4f}")
                            
                    except Exception as e:
                        logger.error(f"[compute_metrics] 处理样本 {i} 时出错: {e}")
                        logger.error(traceback.format_exc())
                        continue
                
                # 每处理一批后记录进度并清理
                if batch_end % 500 == 0:
                    logger.info(f"[compute_metrics] 已处理 {batch_end}/{len(labels)} 个样本")
                    gc.collect()
            
            logger.info(f"[compute_metrics] 样本处理完成，有效样本数: {len(predictions)}")
            predictions = np.array(predictions)
            true_labels = np.array(true_labels)
            
            logger.info(f"[compute_metrics] predictions (token_ids): {predictions[:10]}...")
            logger.info(f"[compute_metrics] true_labels (token_ids): {true_labels[:10]}...")
            
            # 计算准确率：直接对比预测的 token_id 和真实的 token_id
            accuracy = (predictions == true_labels).mean() if len(predictions) > 0 else 0.0
            logger.info(f"[compute_metrics] 准确率: {accuracy:.4f}")
            
            # 计算混淆矩阵
            if len(predictions) > 0:
                try:
                    # 将 token_id 转换为索引以便生成混淆矩阵
                    # 建立 token_id -> index 的映射
                    token_id_to_idx = {tid: idx for idx, tid in enumerate(special_token_ids)}
                    
                    # 转换 true_labels 为索引
                    true_labels_idx = [token_id_to_idx.get(tid, -1) for tid in true_labels]
                    
                    # 转换 predictions 为索引（现在 predictions 已保证是 special_token_ids 之一）
                    predictions_idx = [token_id_to_idx.get(tid, -1) for tid in predictions]
                    
                    cm = confusion_matrix(
                        true_labels_idx, 
                        predictions_idx, 
                        labels=list(range(len(special_tokens)))
                    )
                    
                    # 打印混淆矩阵
                    logger.info("\n" + "=" * 80)
                    logger.info("混淆矩阵 (Confusion Matrix):")
                    logger.info("=" * 80)
                    
                    # 打印表头
                    header = "真实\\预测 |" + "|".join([f"{special_tokens[i]:^12}" for i in range(len(special_tokens))])
                    logger.info(header)
                    logger.info("-" * len(header))
                    
                    # 打印每一行
                    for i in range(len(special_tokens)):
                        row = f"{special_tokens[i]:^10} |"
                        for j in range(len(special_tokens)):
                            row += f"{cm[i][j]:^12}"
                        logger.info(row)
                    
                    logger.info("=" * 80)
                    
                    # 打印详细的分类报告
                    logger.info("\n分类报告 (Classification Report):")
                    logger.info("=" * 80)
                    report = classification_report(
                        true_labels_idx, 
                        predictions_idx, 
                        labels=list(range(len(special_tokens))),
                        target_names=special_tokens,
                        digits=4,
                        zero_division=0
                    )
                    logger.info("\n" + report)
                    logger.info("=" * 80)
                except Exception as e:
                    logger.error(f"[compute_metrics] 生成混淆矩阵时出错: {e}")
                    logger.error(traceback.format_exc())
            
            
            # 保存结果
            result = {
                "accuracy": accuracy,
                "num_samples": len(predictions)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"[compute_metrics] 发生严重错误: {e}")
            logger.error(f"[compute_metrics] 完整错误堆栈:\n{traceback.format_exc()}")
            
            # 返回默认值
            return {
                "accuracy": 0.0,
                "num_samples": 0
            }
    
    return compute_metrics


def load_and_preprocess_data(tokenizer, data_path, max_length):
    """
    加载并预处理训练数据
    
    数据格式示例：
    {
        "instruct": "你是一个智能对话状态管理器...",
        "input": "[Context]:,[Play_state]:系统处于录音状态,[Query]:请问我可以在哪里",
        "output": "<|c_l|>"
    }
    """
    logger.info(f"加载训练数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"数据样本数: {len(data)}")
    
    # 显示第一个样本
    if len(data) > 0:
        logger.info("=" * 60)
        logger.info("第一个数据样本:")
        logger.info(f"  instruct: {data[0].get('instruct', '')[:100]}...")
        logger.info(f"  input: {data[0].get('input', '')}")
        logger.info(f"  output: {data[0].get('output', '')}")
        logger.info("=" * 60)
    
    def preprocess_function(examples):
        """
        将数据转换为模型输入格式
        使用 Qwen 的对话模板格式
        """
        inputs = []
        targets = []
        outputs = []  # 存储 output_text + <|im_end|>
        
        for example in examples:
            # 获取字段，兼容 instruct/instruction
            instruction = example.get("instruct", example.get("instruction", ""))
            input_text = example.get("input", "")
            output_text = example.get("output", "")
            
            # Qwen2.5 格式: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
            full_prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
            full_text = full_prompt + output_text + "<|im_end|>"
            
            inputs.append(full_prompt)
            targets.append(full_text)
            outputs.append(output_text + "<|im_end|>")
        
        # Tokenize 完整序列
        model_inputs = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # 创建 labels，只有 output + <|im_end|> 参与训练
        labels = []
        for i, output_with_end in enumerate(outputs):
            target_ids = model_inputs["input_ids"][i]
            
            # 直接 tokenize output 部分，这就是真正的 label
            output_ids = tokenizer.encode(output_with_end, add_special_tokens=False)
            
            # label: prompt 部分用 -100 mask，output 部分用实际 token ids
            prompt_len = len(target_ids) - len(output_ids)
            if prompt_len >= 0:
                label = [-100] * prompt_len + output_ids
            else:
                # 异常情况：output 被截断，整个序列 mask 掉
                label = [-100] * len(target_ids)
            
            labels.append(label)
            # print(labels)
        
        model_inputs["labels"] = labels
        
        return model_inputs
    
    # 批量处理
    processed_dataset = []
    for example in data:
        processed = preprocess_function([example])
        processed_dataset.append({
            "input_ids": processed["input_ids"][0],
            "attention_mask": processed["attention_mask"][0],
            "labels": processed["labels"][0],
        })
    
    final_dataset = Dataset.from_list(processed_dataset)
    logger.info(f"处理后的数据集大小: {len(final_dataset)}")
    
    return final_dataset


def main():
    try:
        # 记录初始内存状态
        # 解析命令行参数
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingModeArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # 从 json 文件加载参数
            model_args, data_args, train_mode_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, train_mode_args = parser.parse_args_into_dataclasses()
    except Exception as e:
        logger.error(f"参数解析失败: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # 加载配置文件
    config = load_config(model_args.config_file)
    
    # 从配置文件中获取参数（如果命令行未指定）
    if model_args.model_name_or_path is None:
        model_args.model_name_or_path = config['model']['expanded_model_path']
    
    if data_args.train_data_path is None:
        data_args.train_data_path = config['data']['train_data_path']
        
    
    if data_args.eval_data_path is None:
        data_args.eval_data_path = config['data']['eval_data_path']
    
    if data_args.max_length is None:
        data_args.max_length = config['data']['max_length']
    
    logger.info(f"训练数据路径: {data_args.train_data_path}")
    logger.info(f"验证数据路径: {data_args.eval_data_path}")
    logger.info(f"最大序列长度: {data_args.max_length}")
    logger.info(f"训练模式: {train_mode_args.mode}")
    logger.info(f"LoRA rank: {train_mode_args.lora_r}")
    logger.info(f"LoRA alpha: {train_mode_args.lora_alpha}")
    logger.info(f"LoRA dropout: {train_mode_args.lora_dropout}")
    logger.info(f"LoRA target modules: {train_mode_args.target_modules}")
    logger.info(f"训练模式: {train_mode_args.mode}")
    logger.info(f"LoRA rank: {train_mode_args.lora_r}")
    
    # 训练模式
    training_mode = config['training'].get('mode', train_mode_args.mode)
    
    # 输出目录
    if training_mode == "lora":
        output_dir = config['model']['lora_model_path']
    else:
        output_dir = config['model']['full_model_path']
    
    logger.info("=" * 60)
    logger.info(f"Qwen2.5 对话状态分类 - {training_mode.upper()} 微调")
    logger.info("=" * 60)
    logger.info(f"训练模式: {training_mode}")
    logger.info(f"模型路径: {model_args.model_name_or_path}")
    logger.info(f"训练数据: {data_args.train_data_path}")
    logger.info(f"输出目录: {output_dir}")
    if training_mode == "lora":
        logger.info(f"LoRA rank: {train_mode_args.lora_r}")
        logger.info(f"LoRA alpha: {train_mode_args.lora_alpha}")
    logger.info("=" * 60)
    
    # 1. 加载 tokenizer
    try:
        logger.info("加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False
        )
    except Exception as e:
        logger.error(f"加载tokenizer失败: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"词表大小: {len(tokenizer)}")
    
    # 验证特殊token
    special_tokens = [t['token'] for t in config['special_tokens']]
    logger.info("验证特殊 token:")
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"  {token} -> ID: {token_id}")
    
    # 2. 加载模型
    try:
        logger.info("加载模型...")
        # 根据训练模式选择不同的加载策略
        if training_mode == "lora":
            # LoRA 模式可以使用 device_map
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="cuda:0",
            )
        else:
            # 全量微调模式不使用 device_map，让 Trainer 自己管理设备
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数量: {total_params / 1e6:.2f}M")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # 3. 配置训练模式
    if training_mode == "lora":
        logger.info("配置 LoRA...")
        
        # 先启用 gradient checkpointing（LoRA 模式下需要先启用）
        if config['training'].get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
            logger.info("已启用 gradient checkpointing")
        
        target_modules = train_mode_args.target_modules.split(",")
        logger.info(f"LoRA target modules: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_mode_args.lora_r,
            lora_alpha=train_mode_args.lora_alpha,
            lora_dropout=train_mode_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 确保 LoRA 参数可训练
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                logger.debug(f"LoRA 参数 {name}: requires_grad={param.requires_grad}")
    else:
        logger.info("使用全量微调模式")
        # 启用 gradient checkpointing（在设置参数之前）
        if config['training'].get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
            logger.info("已启用 gradient checkpointing")
        
        # 确保模型处于训练模式
        model.train()
        
        # 确保所有参数都可训练
        for name, param in model.named_parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    
    # 4. 加载和预处理数据
    try:
        logger.info("\n加载训练数据集...")
        train_dataset = load_and_preprocess_data(
            tokenizer,
            data_args.train_data_path,
            data_args.max_length
        )
        # 加载评估数据集
        eval_dataset = None
        if data_args.eval_data_path and os.path.exists(data_args.eval_data_path):
            logger.info("\n加载评估数据集...")
            eval_dataset = load_and_preprocess_data(
                tokenizer,
                data_args.eval_data_path,
                data_args.max_length
            )
            logger.info(f"评估数据集大小: {len(eval_dataset)}")
        else:
            logger.warning("未找到评估数据集，跳过评估")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # 5. 配置训练参数
    training_config = config['training']
    
    # 根据训练模式调整学习率
    if training_mode == "lora":
        learning_rate = float(training_config.get('learning_rate', 5e-4))
    else:
        learning_rate = float(training_config.get('learning_rate', 1e-5))
    
    # 评估策略
    eval_strategy = "steps" if eval_dataset is not None else "no"
    eval_steps = int(training_config.get('eval_steps', 100)) if eval_dataset is not None else None
    
    # 评估batch size设置得更小以避免OOM
    eval_batch_size = int(training_config.get('per_device_eval_batch_size', 1))
    logger.info(f"评估batch size: {eval_batch_size}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(training_config.get('num_epochs', 10)),
        per_device_train_batch_size=int(training_config.get('per_device_batch_size', 4)),
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 4)),
        eval_accumulation_steps=1,  # 强制设为1，逐步累积评估结果，减少内存占用
        learning_rate=learning_rate,
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        warmup_ratio=float(training_config.get('warmup_ratio', 0.1)),
        logging_steps=int(training_config.get('logging_steps', 10)),
        save_steps=int(training_config.get('save_steps', 100)),
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_total_limit=int(training_config.get('save_total_limit', 3)),
        fp16=False,
        bf16=bool(training_config.get('bf16', True)),
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        dataloader_num_workers=0,  # 改为0避免额外的子进程内存占用
        dataloader_pin_memory=False,  # 禁用pin memory减少内存占用
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="accuracy" if eval_dataset is not None else None,
        # 添加更多内存优化选项
        max_grad_norm=1.0,  # 梯度裁剪
        optim="adamw_torch",  # 使用torch原生优化器，内存占用更小
    )
    
    logger.info("=" * 60)
    logger.info("训练参数配置:")
    logger.info(f"  训练batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  评估batch size: {training_args.per_device_eval_batch_size}")
    logger.info(f"  梯度累积步数: {training_args.gradient_accumulation_steps}")
    logger.info(f"  评估累积步数: {training_args.eval_accumulation_steps}")
    logger.info("=" * 60)
    
    # 6. 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # 7. 创建评估指标计算函数
    special_tokens = [t['token'] for t in config['special_tokens']]
    label_names = [t['description'] for t in config['special_tokens']]
    compute_metrics_fn = create_compute_metrics(tokenizer, special_tokens, label_names)
    
    # 8. 创建 Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn if eval_dataset is not None else None,
    )
    
    # 9. 开始训练
    try:
        logger.info("开始训练...")
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
        raise
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
        raise
    
    # 10. 最终评估
    if eval_dataset is not None:
        try:
            logger.info("\n" + "=" * 80)
            logger.info("开始最终评估...")
            logger.info("=" * 80)
            final_metrics = trainer.evaluate()
            logger.info("\n最终评估指标:")
            for key, value in final_metrics.items():
                logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"最终评估过程中发生错误: {e}")
            logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")

    
    # 11. 保存最终模型
    logger.info(f"\n保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info(f"模型已保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n程序发生严重错误: {e}")
        logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
        sys.exit(1)
