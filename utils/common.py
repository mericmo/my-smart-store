import os
import logging
import pandas as pd
from datetime import datetime
import yaml
import pickle

# 初始化日志
def init_logger(name: str) -> logging.Logger:
    """初始化结构化日志"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # 文件处理器
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            f"{log_dir}/sales_prediction_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
        # 格式化器
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        # 添加处理器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger

# 读取配置
def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {str(e)}")

# 安全保存CSV（避免覆盖、规范路径）
def save_to_csv(df: pd.DataFrame,
                filename: str,
                sub_dir: str = "",
                base_path: str = "output") -> None:
    """
    安全保存CSV文件
    :param df: 数据框
    :param filename: 文件名（不含后缀）
    :param sub_dir: 子目录
    :param base_path: 基础路径
    """
    try:
        # 构建完整路径
        full_dir = os.path.join(base_path, sub_dir)
        os.makedirs(full_dir, exist_ok=True)
        # 带时间戳避免覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(full_dir, f"{filename}_{timestamp}.csv")
        # 保存
        df.to_csv(full_path, index=False, encoding="utf-8")
        logger = init_logger("save_to_csv")
        logger.info(f"成功保存CSV: {full_path} | 数据行数: {len(df)}")
    except Exception as e:
        logger = init_logger("save_to_csv")
        logger.error(f"保存CSV失败: {str(e)}", exc_info=True)

# 模型持久化
def save_model(model, model_name: str, base_path: str = "models") -> None:
    """保存模型到文件"""
    try:
        os.makedirs(base_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(base_path, f"{model_name}_{timestamp}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger = init_logger("save_model")
        logger.info(f"成功保存模型: {model_path}")
    except Exception as e:
        logger = init_logger("save_model")
        logger.error(f"保存模型失败: {str(e)}", exc_info=True)

def load_model(model_path: str):
    """从文件加载模型"""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger = init_logger("load_model")
        logger.info(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        logger = init_logger("load_model")
        logger.error(f"加载模型失败: {str(e)}", exc_info=True)
        return None