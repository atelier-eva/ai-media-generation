from ai_media_generation.config import Config
from ai_media_generation.domain.qwen.lora_dataset.generation import Generation
from ai_media_generation.repository.json_io import QWEN_LORA_SCHEMAS, read_json


class GenerationRepository:
    def find(self) -> Generation:
        data = read_json(
            Config().qwen_lora_training_generation_json,
            QWEN_LORA_SCHEMAS["generation.json"],
        )
        return Generation(
            prompt=str(data["prompt"]).strip(),
            negative=str(data.get("negative") or "").strip(),
        )
