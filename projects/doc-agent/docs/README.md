# awesome-browser-ml [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of ML models that run fully in the browser.
No server. No API key. No data leaves the device.

Powered by WebAssembly, WebGPU, and ONNX Runtime Web.

---

## Contents

- [Image](#image)
- [Audio](#audio)
- [Body and Camera](#body-and-camera)
- [Text and NLP](#text-and-nlp)
- [Runtimes](#runtimes)
- [Related Lists](#related-lists)
- [Contributing](#contributing)

---

## Image

- [background-removal-js](https://github.com/imgly/background-removal-js) - Remove image backgrounds in browser via WebGPU with WASM fallback. Based on [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)
- [WaifuXL](https://github.com/TheFutureGadgetsLab/WaifuXL) - 4x image upscaling via tiled WASM processing. Based on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [DeOldify-on-Browser](https://github.com/akbartus/DeOldify-on-Browser) - Colorize B&W photos client-side via ONNX Runtime Web. Based on [DeOldify](https://github.com/jantic/DeOldify)
- [tfjs-models/face-detection](https://github.com/tensorflow/tfjs-models) - Face blur and anonymization via WebGL at 30fps+. Based on [BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/face-detection)
- [transformers.js](https://github.com/huggingface/transformers.js) - Depth map generation from a single image via WebGPU. Based on [MiDaS](https://github.com/isl-org/MiDaS)
- [yolov8-tfjs](https://github.com/Hyuto/yolov8-tfjs) - Real-time object detection on webcam via WebGL. Based on [YOLOv8](https://github.com/ultralytics/ultralytics)
- [tesseract.js](https://github.com/naptha/tesseract.js) - OCR text extraction via WASM SIMD, 100+ languages. Based on [Tesseract C++](https://github.com/tesseract-ocr/tesseract)

## Audio

- [rnnoise-wasm](https://github.com/jitsi/rnnoise-wasm) - Real-time noise suppression inside AudioWorklet via synchronous WASM. Based on [RNNoise](https://github.com/xiph/rnnoise)
- [vad-web](https://github.com/ricky0123/vad) - Voice activity detection via ONNX Runtime Web, resamples to 16kHz. Based on [Silero VAD](https://github.com/snakers4/silero-vad)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) - Speech to text via WASM SIMD, Whisper tiny runs at ~75MB. Based on [OpenAI Whisper](https://github.com/openai/whisper)
- [mediapipe/tasks-audio](https://github.com/google-ai-edge/mediapipe) - Sound classification across 521 classes via WASM. Based on [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)

## Body and Camera

- [tfjs-models/pose-detection](https://github.com/tensorflow/tfjs-models) - Pose estimation tracking 17 keypoints at 50fps+ via WebGL. Based on [MoveNet](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection)
- [tfjs-models/hand-pose-detection](https://github.com/tensorflow/tfjs-models) - Hand tracking with 21 3D landmarks per hand via WebGL. Based on [MediaPipe Hands](https://github.com/google-ai-edge/mediapipe)
- [tfjs-models/body-segmentation](https://github.com/tensorflow/tfjs-models) - Body segmentation for virtual backgrounds via WebGPU. Based on [MediaPipe SelfieSegmentation](https://github.com/google-ai-edge/mediapipe)
- [kan](https://github.com/QAInsights/kan) - Eye strain and blink detection via Eye Aspect Ratio calculation. Based on [MediaPipe Face Mesh](https://github.com/google-ai-edge/mediapipe)

## Text and NLP

- [fasttext.wasm.js](https://github.com/yunsii/fasttext.wasm.js) - Language detection for 176 languages under 1ms, 900KB model. Based on [fastText](https://github.com/facebookresearch/fastText)
- [transformers.js](https://github.com/huggingface/transformers.js) - Sentiment analysis via DistilBERT with WebGPU acceleration. Based on [distilbert-sst-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [transformers.js](https://github.com/huggingface/transformers.js) - Text classification and spam detection via quantized BERT + WASM

## Runtimes

The three core runtimes powering everything above:

- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) - Runs any ONNX model via WebGPU, WASM, or WebNN. Most versatile option.
- [Transformers.js](https://github.com/huggingface/transformers.js) - Hugging Face models in the browser. Same API as Python transformers. 200+ architectures.
- [TensorFlow.js](https://github.com/tensorflow/tfjs) - Mature ML framework with WebGL and WASM backends. Best for MediaPipe and MoveNet.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) - Google's optimized tasks for vision, audio, and body tracking. Production-ready.

## Related Lists

- [awesome-webnn](https://github.com/webmachinelearning/awesome-webnn) - WebNN API resources and hardware acceleration
- [awesome-webgpu](https://github.com/mikbry/awesome-webgpu) - WebGPU ecosystem and compute shader tooling
- [awesome-on-device-AI](https://github.com/ysyisyourbrother/awesome-on-device-AI) - On-device AI for mobile and edge devices
- [awesome-webmcp](https://github.com/webmachinelearning/awesome-webmcp) - WebMCP standard for exposing browser models to AI agents

## Contributing

This list started as a personal reference while building browser-based
ML tools at work. No server, no API key, no data leaving the device
was the constraint. Finding the right repos was harder than it should be.

If you know a model that runs fully in the browser and belongs here,
open a PR. One rule: working browser implementation required.
Python repos and papers go elsewhere.

GitHub links only please. Your contributions are appreciated.

---

**Author:** [Lucian Crisan](https://github.com/crisanlucid)

[![GitHub](https://img.shields.io/badge/GitHub-crisanlucid-black?logo=github)](https://github.com/crisanlucid)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Lucian%20Crisan-blue?logo=linkedin)](https://linkedin.com/in/lucian-crisan-50a66623)
