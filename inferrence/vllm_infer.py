#!/usr/bin/env python3
"""
使用 vLLM 进行对话状态分类推理的 Flask API 服务
"""
import os
import yaml
import logging
from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量存储模型实例
llm_engine = None
sampling_params = None
tokenizer = None
special_tokens = ["<|s_s|>", "<|s_l|>", "<|c_s|>", "<|c_l|>"]
special_token_ids = {}  # token_id -> token_str 映射





def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def init_vllm_engine(model_path, config=None):
    """
    初始化 vLLM 推理引擎
    
    Args:
        model_path: 模型路径（全量微调后的模型或合并后的LoRA模型）
        config: 配置字典
    """
    global llm_engine, sampling_params, tokenizer, special_token_ids
    
    logger.info("=" * 60)
    logger.info("初始化 vLLM 推理引擎")
    logger.info("=" * 60)
    logger.info(f"模型路径: {model_path}")
    
    # vLLM 引擎配置
    llm_engine = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=512,
    )
    
    # 获取 tokenizer 并构建特殊 token ID 映射
    tokenizer = llm_engine.get_tokenizer()
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            special_token_ids[token_id] = token
            logger.info(f"特殊 token 映射: {token} -> {token_id}")
    
    # 推理配置
    inference_config = config.get('inference', {}) if config else {}
    max_new_tokens = inference_config.get('max_new_tokens', 10)
    temperature = inference_config.get('temperature', 0.1)
    top_p = inference_config.get('top_p', 0.5)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    
    logger.info("vLLM 引擎初始化完成")
    logger.info("=" * 60)


def build_prompt(context, play_state, query):
    """
    构建模型输入 prompt
    
    Args:
        context: 对话上下文
        play_state: 系统播放状态
        query: 用户当前输入
        
    Returns:
        prompt: 格式化后的 prompt
    """
    instruction = "你是一个智能对话状态管理器。根据User和Assistant的最近对话记录(Context)、User当前的输入(Query)、系统语音播放状态(Play_state)，判断系统的下一步行动。下一步行动可选值：<|s_s|>: 系统开始说 <|s_l|>: 系统停止播音开始听 <|c_s|>: 系统继续播音继续说 <|c_l|>: 系统继续听"
    input_text = f"[Context]:{context},[Play_state]:{play_state},[Query]:{query}"
    
    prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    # logger.info(f"Prompt: {prompt}")
    return prompt


def classify(context, play_state, query):
    """
    对话状态分类
    
    Args:
        context: 对话上下文
        play_state: 系统播放状态
        query: 用户当前输入
        
    Returns:
        dict: 包含预测结果的字典
    """
    global llm_engine, sampling_params, special_token_ids, tokenizer
    
    if llm_engine is None:
        raise RuntimeError("vLLM 引擎未初始化")
    
    # 构建 prompt
    prompt = build_prompt(context, play_state, query)
    logger.debug(f"Prompt: {prompt}")
    
    # vLLM 推理
    outputs = llm_engine.generate([prompt], sampling_params)
    token_ids = outputs[0].outputs[0].token_ids
    
    # 调试日志
    logger.info(f"Token IDs: {token_ids}")
    
    # 通过 token ID 提取特殊 token
    predicted_token = None
    for tid in token_ids:
        if tid in special_token_ids:
            predicted_token = special_token_ids[tid]
            break
    
    # 解码生成的文本（包含特殊 token）
    generated_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    logger.info(f"Generated Text: {generated_text}")
    return {
        "predicted_token": predicted_token,
        "generated_text": generated_text.strip()
    }


def batch_classify(requests_list):
    """
    批量分类
    
    Args:
        requests_list: 请求列表，每个请求包含 context, play_state, query
        
    Returns:
        list: 结果列表
    """
    global llm_engine, sampling_params, special_token_ids, tokenizer
    
    if llm_engine is None:
        raise RuntimeError("vLLM 引擎未初始化")
    
    # 构建所有 prompts
    prompts = []
    for req in requests_list:
        context = req.get('context', '')
        play_state = req.get('play_state', '')
        query = req.get('query', '')
        prompts.append(build_prompt(context, play_state, query))
    
    # 批量推理
    outputs = llm_engine.generate(prompts, sampling_params)
    
    # 解析结果
    results = []
    for output in outputs:
        token_ids = output.outputs[0].token_ids
        
        # 通过 token ID 提取特殊 token
        predicted_token = None
        for tid in token_ids:
            if tid in special_token_ids:
                predicted_token = special_token_ids[tid]
                break
        
        # 解码生成的文本（包含特殊 token）
        generated_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        results.append({
            "predicted_token": predicted_token,
            "generated_text": generated_text.strip()
        })
    
    return results


# ==================== Flask API 路由 ====================

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "engine_initialized": llm_engine is not None
    })


@app.route('/classify', methods=['POST'])
def api_classify():
    """
    单条分类接口
    
    请求体 JSON 格式:
    {
        "context": "对话上下文（可选）",
        "play_state": "系统播放状态",
        "query": "用户当前输入"
    }
    
    返回 JSON 格式:
    {
        "code": 0,
        "message": "success",
        "data": {
            "predicted_token": "<|s_s|>",
            "generated_text": "..."
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "code": 400,
                "message": "请求体不能为空",
                "data": None
            }), 400
        
        context = data.get('context', '')
        play_state = data.get('play_state', '')
        query = data.get('query', '')
        
        if not play_state or not query:
            return jsonify({
                "code": 400,
                "message": "play_state 和 query 为必填字段",
                "data": None
            }), 400
        
        result = classify(context, play_state, query)
        
        return jsonify({
            "code": 0,
            "message": "success",
            "data": result
        })
    
    except Exception as e:
        logger.exception("分类请求处理失败")
        return jsonify({
            "code": 500,
            "message": str(e),
            "data": None
        }), 500


@app.route('/batch_classify', methods=['POST'])
def api_batch_classify():
    """
    批量分类接口
    
    请求体 JSON 格式:
    {
        "requests": [
            {
                "context": "对话上下文（可选）",
                "play_state": "系统播放状态",
                "query": "用户当前输入"
            },
            ...
        ]
    }
    
    返回 JSON 格式:
    {
        "code": 0,
        "message": "success",
        "data": {
            "results": [
                {
                    "predicted_token": "<|s_s|>",
                    "generated_text": "..."
                },
                ...
            ]
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'requests' not in data:
            return jsonify({
                "code": 400,
                "message": "请求体必须包含 requests 字段",
                "data": None
            }), 400
        
        requests_list = data['requests']
        
        if not isinstance(requests_list, list) or len(requests_list) == 0:
            return jsonify({
                "code": 400,
                "message": "requests 必须是非空数组",
                "data": None
            }), 400
        
        results = batch_classify(requests_list)
        
        return jsonify({
            "code": 0,
            "message": "success",
            "data": {
                "results": results
            }
        })
    
    except Exception as e:
        logger.exception("批量分类请求处理失败")
        return jsonify({
            "code": 500,
            "message": str(e),
            "data": None
        }), 500


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM 对话状态分类 Flask API 服务")
    parser.add_argument(
        "--config",
        type=str,
        default="/semaintic-vad/config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/semaintic-vad/models/semantic-vad-qwen-0.5B-merged-1217",
        help="模型路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务监听地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=51996,
        help="服务监听端口"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="是否开启调试模式"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 确定模型路径
    # 注意：vLLM 需要使用全量模型或合并后的 LoRA 模型
    # 如果是 LoRA 模型，需要先将其与基础模型合并
    if args.model:
        model_path = args.model
    else:
        # 默认使用全量微调模型路径
        model_path = config['model'].get('full_model_path')
        if not model_path or not os.path.exists(model_path):
            # 尝试使用 LoRA 模型路径（假设已经合并）
            model_path = config['model'].get('lora_model_path')
    
    if not model_path or not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        logger.error("请指定有效的模型路径（使用 --model 参数）")
        logger.error("注意：vLLM 需要使用全量模型或已合并的 LoRA 模型")
        return
    
    # 初始化 vLLM 引擎
    init_vllm_engine(model_path, config)
    
    # 启动 Flask 服务
    logger.info(f"启动 Flask 服务: http://{args.host}:{args.port}")
    logger.info("可用接口:")
    logger.info("  GET  /health         - 健康检查")
    logger.info("  POST /classify       - 单条分类")
    logger.info("  POST /batch_classify - 批量分类")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=False  # vLLM 不支持多线程
    )


if __name__ == "__main__":
    main()
