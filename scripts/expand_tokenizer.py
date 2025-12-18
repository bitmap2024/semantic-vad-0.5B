#!/usr/bin/env python3
"""
扩展 Qwen2.5 模型的 tokenizer，添加四个对话状态特殊 token
"""
import os
import sys
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def expand_tokenizer_and_model(
    model_path: str,
    output_path: str,
    new_tokens: list
):
    """
    扩展 tokenizer 词表并相应调整模型 embedding 层
    
    Args:
        model_path: 原始模型路径
        output_path: 输出模型路径
        new_tokens: 新增的特殊 token 列表
    """
    logger.info(f"正在加载模型和 tokenizer: {model_path}")
    
    # 加载原始 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    logger.info(f"原始词表大小: {len(tokenizer)}")
    logger.info(f"原始 embedding 维度: {model.get_input_embeddings().weight.shape}")
    
    # 检查 token 是否已存在
    existing_tokens = []
    new_tokens_to_add = []
    
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            existing_tokens.append((token, token_id))
        else:
            new_tokens_to_add.append(token)
    
    if existing_tokens:
        logger.warning(f"以下 token 已存在于词表中:")
        for token, token_id in existing_tokens:
            logger.warning(f"  {token} -> ID: {token_id}")
    
    if not new_tokens_to_add:
        logger.info("所有 token 都已存在，无需添加新 token")
        logger.info("模型已包含所需的特殊 token，直接保存...")
    else:
        # 添加新的特殊 token
        num_added_tokens = tokenizer.add_special_tokens({
            'additional_special_tokens': new_tokens_to_add
        })
        
        logger.info(f"添加了 {num_added_tokens} 个新 token")
        logger.info(f"新词表大小: {len(tokenizer)}")
        
        # 调整模型 embedding 层以适应新的词表大小
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"新 embedding 维度: {model.get_input_embeddings().weight.shape}")
        
        # 差异化随机初始化新 token 的权重，打破对称性
        original_vocab_size = len(tokenizer) - num_added_tokens
        embed_dim = model.get_input_embeddings().weight.shape[1]
        
        with torch.no_grad():
            input_embed = model.get_input_embeddings().weight
            output_embed = model.lm_head.weight
            # 使用已有 embedding 的标准差作为初始化范围
            input_std = input_embed[:original_vocab_size].std().item()
            output_std = output_embed[:original_vocab_size].std().item()
            
            logger.info(f"已有 embedding 标准差: input={input_std:.6f}, output={output_std:.6f}")
            
            # 为每个新 token 生成不同的随机初始化
            for i, token in enumerate(new_tokens_to_add):
                token_id = tokenizer.convert_tokens_to_ids(token)
                
                # 使用不同的随机种子，确保每个 token 初始化不同
                torch.manual_seed(42 + i)
                
                # 初始化 input embedding
                input_embed[token_id] = torch.randn(embed_dim, dtype=input_embed.dtype, device=input_embed.device) * input_std * 0.02
                
                # 初始化 lm_head
                output_embed[token_id] = torch.randn(embed_dim, dtype=output_embed.dtype, device=output_embed.device) * output_std * 0.02
                
                logger.info(f"  已初始化 {token} (ID: {token_id})")
        
        logger.info(f"已为 {len(new_tokens_to_add)} 个新 token 进行差异化随机初始化")
    
    # 验证所有 token
    logger.info("\n验证特殊 token:")
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"  {token} -> ID: {token_id}")
    
    # 保存扩展后的模型和 tokenizer
    logger.info(f"\n正在保存到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    
    logger.info("完成！模型和 tokenizer 已成功扩展并保存。")
    
    return tokenizer, model


def main():
    # 获取配置文件路径
    config_path = "/root/code/semaintic-vad/config/config.yaml"
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 从配置文件读取路径
    original_model_path = config['model']['original_model_path']
    expanded_model_path = config['model']['expanded_model_path']
    
    # 定义四个对话状态特殊 token
    special_tokens = [t['token'] for t in config['special_tokens']]
    
    logger.info("=" * 60)
    logger.info("Qwen2.5 对话管理 Tokenizer 扩展工具")
    logger.info("=" * 60)
    logger.info(f"原始模型: {original_model_path}")
    logger.info(f"输出路径: {expanded_model_path}")
    logger.info(f"新增 token: {special_tokens}")
    logger.info("=" * 60)
    logger.info("")
    
    # 检查原始模型是否存在
    if not os.path.exists(original_model_path):
        logger.error(f"错误：原始模型路径不存在: {original_model_path}")
        logger.error("请先下载 Qwen2.5-0.5B-Instruct 模型")
        logger.error("下载命令:")
        logger.error("  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/autodl-tmp/Qwen2.5-0.5B-Instruct")
        sys.exit(1)
    
    # 执行扩展
    tokenizer, model = expand_tokenizer_and_model(
        original_model_path,
        expanded_model_path,
        special_tokens
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("测试新 token")
    logger.info("=" * 60)
    
    # 测试 tokenization
    test_text = "根据对话状态，应该采取的行动是：<|s_s|>"
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    
    logger.info(f"测试文本: {test_text}")
    logger.info(f"Token 列表: {tokens}")
    logger.info(f"Token IDs: {token_ids}")
    logger.info("")
    
    # 验证每个特殊 token
    for token in special_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        logger.info(f"Token: {token} | Encoded: {encoded} | Decoded: {decoded}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("扩展完成！现在可以运行训练脚本了")
    logger.info(f"  python train.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
