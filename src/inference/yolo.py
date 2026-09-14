"""Lazy PyTorch YOLO adapter; one model owned by the inference worker."""

import gc
from pathlib import Path

from src.inference.base import Detection


class ModelUnavailable(RuntimeError):
    pass


class YoloDetector:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        path = Path(settings.model_path)
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise ModelUnavailable(
                "Нет весов YOLO .pt. Укажите ML_MODEL_PATH в конфигурации."
            )
        try:
            import torch
            from ultralytics import YOLO
        except ImportError:
            raise ModelUnavailable(
                "Установите ML-зависимости: uv sync --extra ml"
            ) from None
        self.torch = torch
        self.device = "cuda:0" if settings.device == "gpu" else "cpu"
        if self.device == "cuda:0" and not torch.cuda.is_available():
            raise ModelUnavailable(
                "GPU недоступен. Установите PyTorch с CUDA или задайте ML_DEVICE=cpu."
            )
        try:
            self.model = YOLO(str(path), task="detect")
            self.person_ids = [
                key for key, name in self.model.names.items() if name == "person"
            ]
            if not self.person_ids:
                raise ModelUnavailable("Модель должна содержать класс person.")
        except Exception:
            self.close()
            raise

    def detect(self, frame):
        results = self.model.predict(
            source=frame.bgr.copy(),
            device=self.device,
            imgsz=self.settings.image_size,
            conf=self.settings.confidence,
            classes=self.person_ids,
            batch=1,
            half=self.device != "cpu",
            verbose=False,
            save=False,
            stream=False,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.cpu():
                if int(box.cls.item()) in self.person_ids:
                    detections.append(
                        Detection(
                            tuple(int(value) for value in box.xyxy[0].tolist()),
                            float(box.conf.item()),
                        )
                    )
        return tuple(detections)

    def close(self):
        self.model = None
        gc.collect()
        if self.device != "cpu":
            self.torch.cuda.empty_cache()
