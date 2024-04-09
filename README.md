# ComfyUI_LiveDirector
Experimental method to use reference video to drive motion in generations without training in ComfyUI.

> [!IMPORTANT]  
> This is currently (WIP), and is in early release for testing.

# Quickstart Guide

More details will be added soon, but if you want to use the early release:

- Only AnimateDiff [Lightning models](https://huggingface.co/ByteDance/AnimateDiff-Lightning) are supported (you must use CFG of 1).
- As stated above, AnimateDiff is only supported. This repository was tested with https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.
- Use [IPAdapter](https://github.com/cubiq/ComfyUI_IPAdapter_plus) to control the spatial part of the generation.
