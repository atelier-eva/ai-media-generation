import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from ai_media_generation.config import Config
from ai_media_generation.infrastructure.error import InfrastructureError

_PNG = b"\x89PNG"
_JPEG = b"\xff\xd8"
_DEFAULT_MODEL = "nai-diffusion-5-full"
_DEFAULT_SAMPLER = "k_euler_ancestral"
_DEFAULT_I2I_STRENGTH = 0.7
_DEFAULT_I2I_NOISE = 0.0
_DEFAULT_TEXT_MODEL = "xialong-v1"
_DEFAULT_TEXT_MAX_LENGTH = 300
_DEFAULT_TEXT_TEMPERATURE = 0.85
_GLM_TEXT_MODEL = "glm-4-6"
_GLM_TEXT_TEMPERATURE = 1.0
_XIALONG_SYSTEM_PROMPT = (
    "You are Xialong (夏龍), an AI model finetuned by Anlatan. "
    "You follow the user's instructions precisely while bringing creativity, "
    "nuance, and depth to every response. Adapt your voice and style to match "
    "what the task demands."
)
_TEXT_CONTEXT_CHARACTERS = 28672
_GENERATE_IMAGE_PATH = "/ai/generate-image"
_GENERATE_TEXT_PATH = "/oa/v1/completions"


class NovelAI:
    _USER_AGENT = "ai-media-generation"

    @dataclass
    class SavedImage:
        filename: str
        data: bytes

    @dataclass
    class SavedText:
        filename: str
        text: str

    @dataclass
    class CharacterPrompt:
        positive: str
        negative: str
        x: float | None = None
        y: float | None = None

    @dataclass
    class Img2Img:
        image: Path
        strength: float = _DEFAULT_I2I_STRENGTH
        noise: float = _DEFAULT_I2I_NOISE

    def __init__(self) -> None:
        config = Config()
        token = config.novelai_api_token
        if not token:
            raise ValueError("NOVELAI_API_TOKEN is not set.")
        self._token = token
        self._image_url = Config.NOVELAI_IMAGE_URL
        self._text_url = Config.NOVELAI_TEXT_URL
        self._timeout_seconds = config.novelai_timeout_seconds

    def generate_image(
        self,
        filename_prefix: str,
        width: int,
        height: int,
        prompt: str,
        negative_prompt: str,
        seed: int,
        model: str = _DEFAULT_MODEL,
        sampler: str = _DEFAULT_SAMPLER,
        steps: int = 28,
        scale: float = 5.0,
        batch_size: int = 1,
        character_prompts: tuple["NovelAI.CharacterPrompt", ...] = (),
        use_order: bool = True,
        img2img: "NovelAI.Img2Img | None" = None,
    ) -> tuple["NovelAI.SavedImage", ...]:
        prefix = self._filename_prefix(filename_prefix)
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        model_name = model.strip()
        sampler_name = sampler.strip()
        if not model_name:
            raise ValueError("model is empty.")
        if not sampler_name:
            raise ValueError("sampler is empty.")
        parameters = self._image_parameters(
            prompt,
            negative_prompt,
            width,
            height,
            seed,
            sampler_name,
            steps,
            scale,
            batch_size,
            character_prompts,
            use_order,
        )
        parameters.update(self._img2img_parameters(img2img))
        payloads = self._images_from_response(
            self._post(
                self._image_url,
                _GENERATE_IMAGE_PATH,
                {
                    "input": prompt,
                    "model": model_name,
                    "action": "img2img" if img2img is not None else "generate",
                    "parameters": parameters,
                },
                "application/json, application/zip",
            )
        )
        if not payloads:
            raise InfrastructureError(
                "NovelAI generation succeeded without images."
            )
        if len(payloads) == 1:
            return (
                NovelAI.SavedImage(filename=f"{prefix}.png", data=payloads[0]),
            )
        return tuple(
            NovelAI.SavedImage(filename=f"{prefix}_{index}.png", data=payload)
            for index, payload in enumerate(payloads, start=1)
        )

    def write_images(
        self, images: tuple["NovelAI.SavedImage", ...], directory: Path
    ) -> int:
        if not images:
            raise InfrastructureError(
                "No saved images from NovelAI; cannot write images."
            )
        root = directory.expanduser().resolve()
        written = 0
        for image in images:
            path = (root / image.filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Invalid image path: {image.filename}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image.data)
            written += 1
        return written

    def generate_text(
        self,
        filename_prefix: str,
        prompt: str,
        model: str = _DEFAULT_TEXT_MODEL,
        max_length: int = _DEFAULT_TEXT_MAX_LENGTH,
        system_prompt: str = "",
        stop: tuple[str, ...] = (),
        context: str = "",
    ) -> tuple["NovelAI.SavedText", ...]:
        prefix = self._filename_prefix(filename_prefix)
        if not prompt.strip():
            raise ValueError("input is empty.")
        text = prompt.lstrip()
        model_name = model.strip()
        if not model_name:
            raise ValueError("model is empty.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        sequences = self._stop_sequences(stop)
        body: dict[str, Any] = {
            "model": model_name,
            "prompt": self._completion_prompt(
                text, context, system_prompt, model_name
            ),
            "max_tokens": max_length,
            "temperature": self._text_temperature(model_name),
            "stream": False,
        }
        if sequences:
            body["stop"] = list(sequences)
        self._require_text_context(str(body["prompt"]))
        output = self._text_from_response(
            self._post(self._text_url, _GENERATE_TEXT_PATH, body)
        )
        if not output.strip():
            raise InfrastructureError(
                "NovelAI generation succeeded without text."
            )
        return (NovelAI.SavedText(filename=f"{prefix}.txt", text=output),)

    @staticmethod
    def _completion_prompt(
        story: str, context: str, system_prompt: str, model: str
    ) -> str:
        system = system_prompt.strip()
        if not system and model != _GLM_TEXT_MODEL:
            system = _XIALONG_SYSTEM_PROMPT
        user = context.strip()
        user_block = f"{user}\n" if user else ""
        return (
            "[gMASK]<sop><|system|>\n"
            f"{system}<|user|>\n"
            f"{user_block}"
            "/nothink<|assistant|>\n"
            f"{story}"
        )

    @staticmethod
    def _require_text_context(prompt: str) -> None:
        count = len(prompt)
        if count > _TEXT_CONTEXT_CHARACTERS:
            raise ValueError(
                f"NovelAI text prompt is {count} characters. "
                f"The limit is {_TEXT_CONTEXT_CHARACTERS}."
            )

    @staticmethod
    def _text_temperature(model: str) -> float:
        if model == _GLM_TEXT_MODEL:
            return _GLM_TEXT_TEMPERATURE
        return _DEFAULT_TEXT_TEMPERATURE

    @staticmethod
    def _stop_sequences(stop: tuple[str, ...]) -> tuple[str, ...]:
        sequences = tuple(item.strip() for item in stop)
        if any(not item for item in sequences):
            raise ValueError("stop sequence is empty.")
        return sequences

    def write_text(
        self, texts: tuple["NovelAI.SavedText", ...], directory: Path
    ) -> int:
        if not texts:
            raise InfrastructureError(
                "No saved text from NovelAI; cannot write text."
            )
        root = directory.expanduser().resolve()
        written = 0
        for item in texts:
            path = (root / item.filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Invalid text path: {item.filename}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item.text, encoding="utf-8", newline="\n")
            written += 1
        return written

    def _filename_prefix(self, filename_prefix: str) -> str:
        prefix = filename_prefix.strip().replace("\\", "/")
        if not prefix:
            raise ValueError("filename_prefix is empty.")
        relative = Path(prefix)
        if relative.is_absolute() or any(
            part in ("", ".", "..") for part in relative.parts
        ):
            raise ValueError(f"Invalid filename_prefix: {filename_prefix}")
        return prefix

    def _image_parameters(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        sampler: str,
        steps: int,
        scale: float,
        batch_size: int,
        character_prompts: tuple["NovelAI.CharacterPrompt", ...],
        use_order: bool,
    ) -> dict[str, Any]:
        return {
            "params_version": 4,
            "width": width,
            "height": height,
            "scale": scale,
            "sampler": sampler,
            "steps": steps,
            "seed": seed,
            "n_samples": batch_size,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "v4_prompt": self._caption(
                prompt,
                tuple(item.positive for item in character_prompts),
                character_prompts,
                use_order,
            ),
            "v4_negative_prompt": self._caption(
                negative_prompt,
                tuple(item.negative for item in character_prompts),
                character_prompts,
                use_order,
            ),
        }

    def _img2img_parameters(
        self, img2img: "NovelAI.Img2Img | None"
    ) -> dict[str, Any]:
        if img2img is None:
            return {}
        if not 0 <= img2img.strength <= 1:
            raise ValueError("strength must be between 0 and 1.")
        if not 0 <= img2img.noise <= 1:
            raise ValueError("noise must be between 0 and 1.")
        return {
            "image": self._encode_source_image(img2img.image),
            "strength": float(img2img.strength),
            "noise": float(img2img.noise),
        }

    def _encode_source_image(self, path: Path) -> str:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"NovelAI source image not found: {resolved}")
        payload = resolved.read_bytes()
        if not self._is_image(payload):
            raise ValueError(
                f"NovelAI source image is not PNG, JPEG, or WEBP: {resolved}"
            )
        return base64.b64encode(payload).decode("ascii")

    def _caption(
        self,
        base_caption: str,
        char_captions: tuple[str, ...] = (),
        character_prompts: tuple["NovelAI.CharacterPrompt", ...] = (),
        use_order: bool = True,
    ) -> dict[str, Any]:
        use_coords = any(
            item.x is not None and item.y is not None for item in character_prompts
        )
        return {
            "caption": {
                "base_caption": base_caption,
                "char_captions": [
                    {
                        "char_caption": text,
                        "centers": [self._character_center(item)],
                    }
                    for text, item in zip(
                        char_captions, character_prompts, strict=True
                    )
                ],
            },
            "use_coords": use_coords,
            "use_order": use_order,
        }

    def _character_center(
        self, item: "NovelAI.CharacterPrompt"
    ) -> dict[str, float]:
        if item.x is not None and item.y is not None:
            return {"x": item.x, "y": item.y}
        return {"x": 0.5, "y": 0.5}

    def _post(
        self,
        base_url: str,
        path: str,
        body: dict[str, Any],
        accept: str = "application/json",
    ) -> bytes:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": accept,
                "User-Agent": self._USER_AGENT,
                "x-correlation-id": uuid4().hex[:6],
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            raise InfrastructureError(self._http_error_message(path, error)) from error
        except TimeoutError as error:
            raise InfrastructureError(
                f"Timed out after {self._timeout_seconds}s waiting for "
                f"NovelAI {path}."
            ) from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise InfrastructureError(
                    f"Timed out after {self._timeout_seconds}s waiting for "
                    f"NovelAI {path}."
                ) from error
            raise InfrastructureError(
                f"NovelAI is not reachable at {base_url}"
            ) from error

    def _text_from_response(self, payload: bytes) -> str:
        try:
            loaded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InfrastructureError(
                "NovelAI /oa/v1/completions did not return JSON."
            ) from error
        if not isinstance(loaded, dict):
            raise InfrastructureError(
                "NovelAI /oa/v1/completions did not return an object."
            )
        choices = loaded.get("choices")
        if not isinstance(choices, list) or not choices:
            raise InfrastructureError(
                "NovelAI /oa/v1/completions JSON did not include choices."
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise InfrastructureError(
                "NovelAI /oa/v1/completions JSON choice was not an object."
            )
        content = choice.get("text")
        if not isinstance(content, str):
            raise InfrastructureError(
                "NovelAI /oa/v1/completions JSON did not include text."
            )
        return self._without_thinking(content)

    @staticmethod
    def _without_thinking(text: str) -> str:
        stripped = text.lstrip()
        if not stripped.startswith("<think>"):
            return text
        end = stripped.find("</think>")
        if end < 0:
            return text
        return stripped[end + len("</think>") :].lstrip("\n")

    def _images_from_response(self, payload: bytes) -> tuple[bytes, ...]:
        if payload.startswith(b"{") or payload.startswith(b"["):
            return self._images_from_json(payload)
        if payload.startswith(b"PK"):
            return self._images_from_zip(payload)
        if self._is_image(payload):
            return (payload,)
        raise InfrastructureError(
            "NovelAI /ai/generate-image did not return images."
        )

    def _images_from_json(self, payload: bytes) -> tuple[bytes, ...]:
        try:
            loaded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InfrastructureError(
                "NovelAI /ai/generate-image did not return JSON."
            ) from error
        entries: list[Any]
        if isinstance(loaded, dict):
            images = loaded.get("images")
            if not isinstance(images, list):
                raise InfrastructureError(
                    "NovelAI /ai/generate-image JSON did not include images."
                )
            entries = images
        elif isinstance(loaded, list):
            entries = loaded
        else:
            raise InfrastructureError(
                "NovelAI /ai/generate-image JSON was not an object or array."
            )
        payloads: list[bytes] = []
        for entry in entries:
            payloads.extend(self._images_from_entry(entry))
        return tuple(payloads)

    def _images_from_entry(self, entry: Any) -> tuple[bytes, ...]:
        if isinstance(entry, dict):
            encoded = entry.get("image")
            if not isinstance(encoded, str) or not encoded.strip():
                raise InfrastructureError(
                    "NovelAI /ai/generate-image JSON image was empty."
                )
            return self._images_from_bytes(self._decode_base64(encoded))
        if isinstance(entry, str):
            if not entry.strip():
                raise InfrastructureError(
                    "NovelAI /ai/generate-image JSON image was empty."
                )
            return self._images_from_bytes(self._decode_base64(entry))
        raise InfrastructureError(
            "NovelAI /ai/generate-image JSON image was not a string."
        )

    def _images_from_bytes(self, payload: bytes) -> tuple[bytes, ...]:
        if payload.startswith(b"PK"):
            return self._images_from_zip(payload)
        if self._is_image(payload):
            return (payload,)
        raise InfrastructureError(
            "NovelAI /ai/generate-image did not return image bytes."
        )

    def _images_from_zip(self, payload: bytes) -> tuple[bytes, ...]:
        try:
            archive = ZipFile(BytesIO(payload))
        except BadZipFile as error:
            raise InfrastructureError(
                "NovelAI /ai/generate-image did not return a zip file."
            ) from error
        with archive:
            names = tuple(
                sorted(name for name in archive.namelist() if not name.endswith("/"))
            )
            images: list[bytes] = []
            for name in names:
                data = archive.read(name)
                if self._is_image(data):
                    images.append(data)
        if not images:
            raise InfrastructureError(
                "NovelAI /ai/generate-image zip did not contain images."
            )
        return tuple(images)

    def _decode_base64(self, encoded: str) -> bytes:
        text = encoded.strip()
        marker = "base64,"
        if marker in text:
            text = text[text.index(marker) + len(marker) :]
        try:
            return base64.b64decode(text, validate=False)
        except ValueError as error:
            raise InfrastructureError(
                "NovelAI /ai/generate-image JSON image was not base64."
            ) from error

    def _is_image(self, payload: bytes) -> bool:
        if payload.startswith(_PNG) or payload.startswith(_JPEG):
            return True
        return payload.startswith(b"RIFF") and payload[8:12] == b"WEBP"

    def _http_error_message(self, path: str, error: urllib.error.HTTPError) -> str:
        body = error.read().decode("utf-8", errors="replace").strip()
        if error.code == 401:
            return "NovelAI authentication failed."
        if error.code == 402:
            return "NovelAI Anlas is insufficient."
        if error.code == 429:
            return "NovelAI rate limited."
        detail = body
        try:
            loaded = json.loads(body)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            if loaded.get("message"):
                detail = str(loaded["message"])
            elif loaded.get("error"):
                detail = str(loaded["error"])
        if detail:
            return f"NovelAI {path} failed: {error.code} {detail}"
        return f"NovelAI {path} failed: {error.code}"
