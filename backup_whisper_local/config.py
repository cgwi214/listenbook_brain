"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import Set, Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000  # 切片最大长度
    img_content_length: int = 200  # 图片上下文最大长度
    min_content_length: int = 500  # 合并短内容的最小长度
    overlap_sentences: int = 1  # 句子级切分时的重叠句数
    book_name_chunk_k: int = 3  # 书名识别时使用的切片数量
    book_name_chunk_size: int = 2500  # 书名识别时使用的切片内容长度

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    book_model: str = field(
        default_factory=lambda: os.getenv("BOOK_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    # ==================== ASR 语音识别配置（MP3本地转写） ====================

    # 本地 Whisper 模型路径（HuggingFace 格式，transformers 加载）
    whisper_path: str = field(
        default_factory=lambda: os.getenv("WHISPER_PATH", "D:\\models\\whisper-large-v3")
    )

    # Whisper 推理设备（auto：有 CUDA 用 GPU，否则退回 CPU）
    whisper_device: str = field(
        default_factory=lambda: os.getenv("WHISPER_DEVICE", "auto")
    )

    # 转写语言（有声书场景默认中文）
    whisper_language: str = field(
        default_factory=lambda: os.getenv("WHISPER_LANGUAGE", "zh")
    )

    # 长音频分块转写参数（秒）
    whisper_chunk_length_s: int = 30
    whisper_stride_length_s: int = 5

    # 分块批处理大小（显存不足时可调小）
    whisper_batch_size: int = 16

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    book_name_collection: str = field(
        default_factory=lambda: os.getenv("BOOK_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== Neo4j 配置 ====================
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 8

    # ==================== 速率限制 ====================
    requests_per_minute: int = 15  # 图片总结 API 速率限制

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    # http://192.168.200.130:9000/
    def get_minio_base_url(self):
        base_protocol = "https://" if self.minio_secure else "http://"
        return base_protocol + f"{self.minio_endpoint}"


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
# from config import get_config

# config = get_config()
# print(f"OPENAI_API_KEY: {config.openai_api_key}")
# print(f"OPENAI_API_BASE: {config.openai_api_base}")
# print(f"VL_MODEL: {config.vl_model}")
# print(f"BOOK_MODEL: {config.book_model}")
# print(f"MODEL: {config.default_model}")