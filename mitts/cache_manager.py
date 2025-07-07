import os
import hashlib
import time
from datetime import datetime, timedelta
import logging


class CacheManager:
    """
    负责管理参考音频缓存。
    支持缓存文件的保存、通过MD5值获取、以及基于时间间隔的过期清理。
    """

    def __init__(self, cache_dir="cache", cache_duration_days=30):
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(days=cache_duration_days)
        os.makedirs(self.cache_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"缓存管理器初始化，缓存目录: {self.cache_dir}, 有效期: {cache_duration_days} 天"
        )

    def _calculate_md5(self, file_path: str) -> str:
        """计算文件的MD5值。"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def save_audio(self, audio_file_path: str) -> str:
        """
        将音频文件保存到缓存目录，并返回其MD5值。
        如果文件已存在，则不进行重复保存。
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_file_path}")

        md5_hash = self._calculate_md5(audio_file_path)
        cached_file_path = os.path.join(self.cache_dir, f"{md5_hash}.wav")

        if not os.path.exists(cached_file_path):
            try:
                # 使用 shutil.copy2 来保留文件元数据 (例如修改时间)
                import shutil

                shutil.copy2(audio_file_path, cached_file_path)
                self.logger.info(
                    f"音频文件已缓存: {audio_file_path} -> {cached_file_path}"
                )
            except Exception as e:
                self.logger.error(f"缓存文件失败 {audio_file_path}: {e}")
                raise
        else:
            self.logger.info(f"音频文件已存在缓存中，MD5: {md5_hash}")

        return md5_hash

    def get_audio_path(self, md5_hash: str) -> str | None:
        """
        根据MD5值获取缓存音频文件的路径。
        如果文件不存在或已过期，则返回None。
        """
        cached_file_path = os.path.join(self.cache_dir, f"{md5_hash}.wav")
        if not os.path.exists(cached_file_path):
            return None

        # 检查文件是否过期
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cached_file_path))
        if datetime.now() - file_mtime > self.cache_duration:
            self.logger.info(
                f"缓存文件已过期，MD5: {md5_hash}, 路径: {cached_file_path}"
            )
            os.remove(cached_file_path)
            return None

        self.logger.debug(f"从缓存获取文件: {cached_file_path}")
        return cached_file_path

    def clean_cache(self):
        """
        清理缓存目录中所有过期的文件。
        """
        self.logger.info("开始清理过期缓存文件...")
        now = datetime.now()
        cleaned_count = 0
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                try:
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if now - file_mtime > self.cache_duration:
                        os.remove(file_path)
                        cleaned_count += 1
                        self.logger.info(f"已清理过期缓存文件: {file_path}")
                except Exception as e:
                    self.logger.error(f"清理缓存文件失败 {file_path}: {e}")
        self.logger.info(f"缓存清理完成，共清理 {cleaned_count} 个文件。")


# 示例用法 (仅用于测试，实际不会在生产代码中直接运行)
if __name__ == "__main__":
    test_cache_dir = "test_cache_mitts"
    # 清理旧的测试目录
    if os.path.exists(test_cache_dir):
        import shutil

        shutil.rmtree(test_cache_dir)
    os.makedirs(test_cache_dir, exist_ok=True)

    cache_manager = CacheManager(
        cache_dir=test_cache_dir, cache_duration_days=0.001
    )  # 1分钟过期，方便测试

    # 创建一个测试音频文件
    dummy_audio_path = "dummy_audio.wav"
    with open(dummy_audio_path, "wb") as f:
        f.write(b"dummy audio content")

    # 1. 保存音频
    md5_1 = cache_manager.save_audio(dummy_audio_path)
    print(f"保存的MD5: {md5_1}")

    # 2. 获取音频 (未过期)
    path_1 = cache_manager.get_audio_path(md5_1)
    print(f"获取的路径: {path_1}")

    # 等待过期
    print("等待缓存过期...")
    time.sleep(1)

    # 3. 再次获取 (已过期并清理)
    path_2 = cache_manager.get_audio_path(md5_1)
    print(f"过期后获取的路径: {path_2}")

    # 4. 再次清理 (应该没有可清理的)
    cache_manager.clean_cache()

    # 清理测试文件
    os.remove(dummy_audio_path)
    if os.path.exists(test_cache_dir):
        import shutil

        shutil.rmtree(test_cache_dir)
