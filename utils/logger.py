# utils/logger.py
import logging
import os

def setup_logger(log_filename="simulation.log", level=logging.INFO):
    """
    初始化全局日志记录器
    """
    logger = logging.getLogger("AoI_DT_DAP")
    logger.setLevel(level)

    # 避免重复添加 Handler 导致日志重复打印
    if not logger.handlers:
        # 定义日志格式: [时间] [级别] 信息
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. 终端输出 Handler (StreamHandler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. 文件输出 Handler (FileHandler)
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(f"logs/{log_filename}", encoding='utf-8')
        # 文件里始终记录 DEBUG 级别及以上的最全信息
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# 导出一个全局可用的 logger 实例
logger = setup_logger()
