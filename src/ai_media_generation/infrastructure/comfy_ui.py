import copy
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_media_generation.config import Config
from ai_media_generation.infrastructure.error import InfrastructureError
from ai_media_generation.repository.json_io import read_resource_json

_ANIMAGINE_LORA_TRAINING_IMAGE_GENERATION_API_JSON = (
    "comfyui",
    "animagine-lora-training-image-generation-api.json",
)
_IMAGE_CREATION_API_JSON = (
    "comfyui",
    "image-creation-api.json",
)
_KAGEE_CONVERSION_API_JSON = (
    "comfyui",
    "kagee-conversion-api.json",
)
_QWEN_IMAGE_CREATION_API_JSON = (
    "comfyui",
    "image_qwen_Image_2512.json",
)
_QWEN_LORA_TRAINING_IMAGE_GENERATION_API_JSON = (
    "comfyui",
    "qwen-lora-training-image-generation-api.json",
)


class ComfyUi:
    _POLL_INTERVAL_SECONDS = 2
    _POLL_TIMEOUT_SECONDS = 600

    @dataclass
    class SavedImage:
        filename: str
        subfolder: str = ""

    @classmethod
    def wait_until_reachable(cls, url: str, timeout_seconds: int) -> None:
        url = url.rstrip("/")
        print(f"Waiting for ComfyUI (timeout {timeout_seconds}s).")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            request = urllib.request.Request(f"{url}/queue", method="GET")
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    response.read()
                print(f"ComfyUI is reachable at {url}")
                return
            except urllib.error.HTTPError:
                print(f"ComfyUI is reachable at {url}")
                return
            except (TimeoutError, urllib.error.URLError, OSError):
                time.sleep(2)
        raise InfrastructureError(
            f"Timed out after {timeout_seconds}s waiting for ComfyUI at {url}."
        )

    @classmethod
    def filenames(cls, url: str, folder: str) -> tuple[str, ...]:
        name = folder.strip().replace("\\", "/")
        if not name or "/" in name:
            raise ValueError(f"Invalid ComfyUI model folder: {folder}")
        path = f"/models/{name}"
        loaded = cls._get_json(url, path)
        if not isinstance(loaded, list):
            raise InfrastructureError(f"ComfyUI {path} did not return a list.")
        names: list[str] = []
        for item in loaded:
            if not isinstance(item, str):
                raise InfrastructureError(
                    f"ComfyUI {path} item was not a string."
                )
            text = item.strip().replace("\\", "/")
            if text:
                names.append(text)
        return tuple(names)

    @classmethod
    def require_filenames(
        cls,
        url: str,
        required: dict[str, tuple[str, ...]],
        missing: str,
    ) -> None:
        absent: list[str] = []
        for folder in sorted(required):
            listed = frozenset(cls.filenames(url, folder))
            for name in required[folder]:
                if name not in listed:
                    absent.append(name)
        if not absent:
            return
        raise InfrastructureError(
            "ComfyUI is missing " + ", ".join(absent) + ". " + missing
        )

    @classmethod
    def _get_json(cls, url: str, path: str) -> Any:
        url = url.rstrip("/")
        request = urllib.request.Request(f"{url}{path}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise InfrastructureError(f"ComfyUI {path} failed: {error.code}") from error
        except urllib.error.URLError as error:
            raise InfrastructureError(f"ComfyUI is not reachable at {url}") from error
        except json.JSONDecodeError as error:
            raise InfrastructureError(f"ComfyUI {path} did not return JSON.") from error

    def __init__(self, url: str) -> None:
        config = Config()
        self._url = url.rstrip("/")
        self._ckpt_name = config.comfy_ui_ckpt_name
        self._animagine_lora_training_template = read_resource_json(
            *_ANIMAGINE_LORA_TRAINING_IMAGE_GENERATION_API_JSON
        )
        self._image_template = read_resource_json(*_IMAGE_CREATION_API_JSON)
        self._kagee_template = read_resource_json(*_KAGEE_CONVERSION_API_JSON)
        self._kagee_timeout_seconds = config.kagee_timeout_seconds
        self._qwen_template = read_resource_json(*_QWEN_IMAGE_CREATION_API_JSON)
        self._qwen_timeout_seconds = config.qwen_timeout_seconds
        self._qwen_lora_training_template = read_resource_json(
            *_QWEN_LORA_TRAINING_IMAGE_GENERATION_API_JSON
        )
        self._qwen_lora_timeout_seconds = config.qwen_lora_timeout_seconds

    def generate_animagine_lora_training_images(
        self,
        filename_prefix: str,
        width: int,
        height: int,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        batch_size: int = 4,
    ) -> tuple["ComfyUi.SavedImage", ...]:
        return self._queue_prompt(
            self._animagine_lora_training_workflow(
                filename_prefix,
                width,
                height,
                positive_prompt,
                negative_prompt,
                seed,
                batch_size,
            )
        )

    def generate_images(
        self,
        filename_prefix: str,
        width: int,
        height: int,
        positive_prompt: str,
        negative_prompt: str,
        lora_name: str | None,
        strength_model: float,
        strength_clip: float,
        pose_id: str | None,
        seed: int,
        batch_size: int = 4,
    ) -> tuple["ComfyUi.SavedImage", ...]:
        return self._queue_prompt(
            self._image_workflow(
                filename_prefix,
                width,
                height,
                positive_prompt,
                negative_prompt,
                lora_name,
                strength_model,
                strength_clip,
                pose_id,
                seed,
                batch_size,
            )
        )

    def generate_kagee(
        self,
        filename_prefix: str,
        images: tuple[Path, ...],
        prompt: str,
        seed: int,
    ) -> tuple["ComfyUi.SavedImage", ...]:
        prefix = filename_prefix.strip()
        if not prefix:
            raise ValueError("filename_prefix is empty.")
        text = prompt.strip()
        if not text:
            raise ValueError("prompt is empty or missing.")
        if len(images) != 1:
            raise ValueError("multiple images are not implemented.")
        return self._queue_prompt(
            self._kagee_workflow(prefix, self._upload_image(images[0]), text, seed),
            self._kagee_timeout_seconds,
        )

    def generate_qwen_lora_training_images(
        self,
        filename_prefix: str,
        image: Path,
        prompt: str,
        seed: int,
        negative: str = "",
    ) -> tuple["ComfyUi.SavedImage", ...]:
        prefix = filename_prefix.strip()
        if not prefix:
            raise ValueError("filename_prefix is empty.")
        text = prompt.strip()
        if not text:
            raise ValueError("prompt is empty or missing.")
        return self._queue_prompt(
            self._qwen_lora_training_workflow(
                prefix,
                self._upload_image(image),
                text,
                seed,
                negative.strip(),
            ),
            self._qwen_lora_timeout_seconds,
        )

    def generate_qwen(
        self,
        filename_prefix: str,
        width: int,
        height: int,
        prompt: str,
        negative: str,
        seed: int,
        batch_size: int = 4,
    ) -> tuple["ComfyUi.SavedImage", ...]:
        prefix = filename_prefix.strip()
        if not prefix:
            raise ValueError("filename_prefix is empty.")
        text = prompt.strip()
        if not text:
            raise ValueError("prompt is empty or missing.")
        return self._queue_prompt(
            self._qwen_workflow(
                prefix,
                width,
                height,
                text,
                negative.strip(),
                seed,
                batch_size,
            ),
            self._qwen_timeout_seconds,
        )

    def fetch_image(self, image: "ComfyUi.SavedImage") -> bytes:
        query = urllib.parse.urlencode(
            {
                "filename": image.filename,
                "subfolder": image.subfolder,
                "type": "output",
            }
        )
        request = urllib.request.Request(f"{self._url}/view?{query}", method="GET")
        try:
            with urllib.request.urlopen(request) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            raise InfrastructureError(
                f"ComfyUI /view failed for {image.filename}: {error.code}"
            ) from error
        except urllib.error.URLError as error:
            raise InfrastructureError(f"ComfyUI is not reachable at {self._url}") from error

    def write_captions(
        self,
        images: tuple["ComfyUi.SavedImage", ...],
        caption: str,
        directory: Path,
    ) -> int:
        text = caption.strip()
        if not text:
            raise ValueError("caption_prompt is empty or missing.")
        if not images:
            raise InfrastructureError(
                "No saved images in ComfyUI history; cannot write captions."
            )
        written = 0
        for image in images:
            image_dir = directory / image.subfolder if image.subfolder else directory
            image_path = image_dir / image.filename
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found for caption: {image_path}")
            caption_path = image_dir / f"{image_path.stem}.txt"
            caption_path.write_text(text, encoding="utf-8", newline="\n")
            written += 1
        return written

    def write_images(
        self, images: tuple["ComfyUi.SavedImage", ...], directory: Path
    ) -> int:
        if not images:
            raise InfrastructureError(
                "No saved images in ComfyUI history; cannot write images."
            )
        written = 0
        for image in images:
            image_dir = directory / image.subfolder if image.subfolder else directory
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / image.filename).write_bytes(self.fetch_image(image))
            written += 1
        return written

    def _completed(self, status: dict[str, Any]) -> bool:
        if "completed" in status:
            return status["completed"] is True
        return status.get("status_str") == "success"

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(
            f"{self._url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request) as response:
                loaded = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise InfrastructureError(f"ComfyUI is not reachable at {self._url}") from error
        if not isinstance(loaded, dict):
            raise InfrastructureError(f"ComfyUI {path} did not return an object.")
        return loaded

    def _saved_images(self, entry: dict[str, Any]) -> tuple["ComfyUi.SavedImage", ...]:
        images: list[ComfyUi.SavedImage] = []
        for output in (entry.get("outputs") or {}).values():
            for image in output.get("images") or []:
                filename = str(image.get("filename") or "").strip()
                if not filename:
                    continue
                images.append(
                    ComfyUi.SavedImage(
                        filename=filename,
                        subfolder=str(image.get("subfolder") or ""),
                    )
                )
        return tuple(images)

    def _wait_until_complete(
        self, prompt_id: str, timeout_seconds: int | None = None
    ) -> tuple["ComfyUi.SavedImage", ...]:
        timeout = (
            self._POLL_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(self._POLL_INTERVAL_SECONDS)
            history = self._request("GET", f"/history/{prompt_id}")
            entry = history.get(prompt_id)
            if not entry:
                continue
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise InfrastructureError(f"ComfyUI generation failed: {status.get('messages')}")
            if not self._completed(status):
                continue
            images = self._saved_images(entry)
            if images:
                return images
        raise InfrastructureError(
            f"Timed out after {timeout}s waiting for image outputs "
            f"from prompt_id {prompt_id}."
        )

    def _queue_prompt(
        self, workflow: dict[str, Any], timeout_seconds: int | None = None
    ) -> tuple["ComfyUi.SavedImage", ...]:
        response = self._request("POST", "/prompt", {"prompt": workflow})
        node_errors = response.get("node_errors")
        if node_errors:
            raise InfrastructureError(f"ComfyUI node_errors: {node_errors}")
        prompt_id = str(response.get("prompt_id") or "").strip()
        if not prompt_id:
            raise InfrastructureError("ComfyUI /prompt did not return prompt_id.")
        return self._wait_until_complete(prompt_id, timeout_seconds)

    def _upload_image(self, path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Kagee input image not found: {path}")
        body, boundary = self._multipart_image(path)
        request = urllib.request.Request(
            f"{self._url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                loaded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise InfrastructureError(
                f"ComfyUI /upload/image failed: {error.code}"
            ) from error
        except urllib.error.URLError as error:
            raise InfrastructureError(f"ComfyUI is not reachable at {self._url}") from error
        if not isinstance(loaded, dict):
            raise InfrastructureError("ComfyUI /upload/image did not return an object.")
        name = str(loaded.get("name") or "").strip()
        if not name:
            raise InfrastructureError("ComfyUI /upload/image did not return name.")
        subfolder = str(loaded.get("subfolder") or "").strip().replace("\\", "/")
        if subfolder:
            return f"{subfolder}/{name}"
        return name

    def _multipart_image(self, path: Path) -> tuple[bytes, str]:
        boundary = f"----ComfyUiFormBoundary{uuid4().hex}"
        filename = path.name.replace('"', "_").replace("\r", "").replace("\n", "")
        marker = f"--{boundary}\r\n".encode("utf-8")
        header = (
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
        overwrite = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            "true\r\n"
        ).encode("utf-8")
        closing = f"--{boundary}--\r\n".encode("utf-8")
        return marker + header + path.read_bytes() + b"\r\n" + overwrite + closing, boundary

    def _qwen_workflow(
        self,
        filename_prefix: str,
        width: int,
        height: int,
        prompt: str,
        negative: str,
        seed: int,
        batch_size: int,
    ) -> dict[str, Any]:
        workflow = copy.deepcopy(self._qwen_template)
        workflow["60"]["inputs"]["filename_prefix"] = filename_prefix
        workflow["238:227"]["inputs"]["text"] = prompt
        workflow["238:228"]["inputs"]["text"] = negative or " "
        workflow["238:232"]["inputs"]["width"] = width
        workflow["238:232"]["inputs"]["height"] = height
        workflow["238:232"]["inputs"]["batch_size"] = batch_size
        workflow["238:230"]["inputs"]["seed"] = seed
        return workflow

    def _kagee_workflow(
        self, filename_prefix: str, image_name: str, prompt: str, seed: int
    ) -> dict[str, Any]:
        workflow = copy.deepcopy(self._kagee_template)
        workflow["9"]["inputs"]["filename_prefix"] = filename_prefix
        workflow["41"]["inputs"]["image"] = image_name
        workflow["170:151"]["inputs"]["prompt"] = prompt
        workflow["170:169"]["inputs"]["seed"] = seed
        return workflow

    def _qwen_lora_training_workflow(
        self,
        filename_prefix: str,
        image_name: str,
        prompt: str,
        seed: int,
        negative: str,
    ) -> dict[str, Any]:
        workflow = copy.deepcopy(self._qwen_lora_training_template)
        workflow["9"]["inputs"]["filename_prefix"] = filename_prefix
        workflow["41"]["inputs"]["image"] = image_name
        workflow["170:151"]["inputs"]["prompt"] = prompt
        workflow["170:169"]["inputs"]["seed"] = seed
        if negative:
            workflow["170:149"]["inputs"]["prompt"] = negative
        return workflow

    def _animagine_lora_training_workflow(
        self,
        filename_prefix: str,
        width: int,
        height: int,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        batch_size: int,
    ) -> dict[str, Any]:
        workflow = copy.deepcopy(self._animagine_lora_training_template)
        workflow["2"]["inputs"]["ckpt_name"] = self._ckpt_name
        workflow["3"]["inputs"]["text"] = positive_prompt
        workflow["4"]["inputs"]["text"] = negative_prompt
        workflow["8"]["inputs"]["width"] = width
        workflow["8"]["inputs"]["height"] = height
        workflow["8"]["inputs"]["batch_size"] = batch_size
        workflow["11"]["inputs"]["filename_prefix"] = filename_prefix
        workflow["7"]["inputs"]["seed"] = seed
        return workflow

    def _image_workflow(
        self,
        filename_prefix: str,
        width: int,
        height: int,
        positive_prompt: str,
        negative_prompt: str,
        lora_name: str | None,
        strength_model: float,
        strength_clip: float,
        pose_id: str | None,
        seed: int,
        batch_size: int,
    ) -> dict[str, Any]:
        workflow = copy.deepcopy(self._image_template)
        workflow["2"]["inputs"]["ckpt_name"] = self._ckpt_name
        workflow["3"]["inputs"]["text"] = positive_prompt
        workflow["4"]["inputs"]["text"] = negative_prompt
        workflow["8"]["inputs"]["width"] = width
        workflow["8"]["inputs"]["height"] = height
        workflow["8"]["inputs"]["batch_size"] = batch_size
        workflow["11"]["inputs"]["filename_prefix"] = filename_prefix
        workflow["7"]["inputs"]["seed"] = seed
        if lora_name is None:
            del workflow["16"]
            workflow["3"]["inputs"]["clip"] = ["2", 1]
            workflow["4"]["inputs"]["clip"] = ["2", 1]
            workflow["7"]["inputs"]["model"] = ["2", 0]
        else:
            workflow["16"]["inputs"]["lora_name"] = lora_name
            workflow["16"]["inputs"]["strength_model"] = strength_model
            workflow["16"]["inputs"]["strength_clip"] = strength_clip
        if pose_id is None:
            del workflow["5"]
            del workflow["6"]
            del workflow["12"]
            del workflow["13"]
            workflow["7"]["inputs"]["positive"] = ["3", 0]
            workflow["7"]["inputs"]["negative"] = ["4", 0]
        else:
            workflow["12"]["inputs"]["image"] = pose_id
        return workflow
