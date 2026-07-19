"""
Web image search, download, and optional Ollama analysis tool node.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from urllib.parse import quote, quote_plus, unquote

import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from urllib3.exceptions import InsecureRequestWarning

from ...ollama_client import generate
from ...prompts.image_analysis import REQUIRED_KEYS, build_system_prompt, build_user_prompt
from ...utils.image_utils import pil_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "ComfyUI-OllamaNodes/1.0 "
        "(image search and preview node; local user workflow)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _session(use_system_proxy: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = use_system_proxy
    return session


def _get(
    session: requests.Session,
    url: str,
    *,
    verify_ssl: bool,
    base_headers: dict | None = None,
    extra_headers: dict | None = None,
    **kwargs,
) -> requests.Response:
    if not verify_ssl:
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    headers = dict(base_headers or _HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    return session.get(url, headers=headers, verify=verify_ssl, **kwargs)


def _safe_slug(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip()).strip("._")
    return (slug or "web_image")[:max_len]


def _normalize_analysis(parsed: dict) -> dict:
    normalized = {}
    for key in REQUIRED_KEYS:
        value = parsed.get(key, "")
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        normalized[key] = value
    return normalized


def _pil_to_tensor(images: list[Image.Image]) -> torch.Tensor:
    tensors = []
    for image in images:
        rgb = image.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    return torch.stack(tensors)


def _fit_image(image: Image.Image, width: int, height: int, fill: str) -> Image.Image:
    rgb = image.convert("RGB")
    return ImageOps.pad(rgb, (width, height), method=Image.LANCZOS, color=fill, centering=(0.5, 0.5))


def _decode_image(content: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(content))
    image.load()
    return image.convert("RGB")


def _download_image(
    url: str,
    timeout: float = 20.0,
    verify_ssl: bool = False,
    use_system_proxy: bool = False,
) -> Image.Image:
    session = _session(use_system_proxy)
    response = _get(session, url, verify_ssl=verify_ssl, timeout=timeout, stream=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if content_type and "image" not in content_type and "octet-stream" not in content_type:
        raise ValueError(f"URL did not return an image: {content_type}")
    content = response.content
    if len(content) < 1024:
        raise ValueError("Downloaded image was too small")
    return _decode_image(content)


def _wikimedia_thumb_url(image_url: str, width: int = 640) -> str:
    marker = "/wikipedia/commons/"
    if "/wikipedia/commons/thumb/" in image_url or marker not in image_url:
        return image_url
    prefix, rest = image_url.split(marker, 1)
    parts = rest.split("/")
    if len(parts) < 3:
        return image_url
    filename = unquote(parts[-1])
    thumb_name = quote(filename, safe="()!$&'*,;=@._-")
    return f"{prefix}{marker}thumb/{'/'.join(parts[:-1])}/{thumb_name}/{width}px-{thumb_name}"


def _duckduckgo_image_results(
    query: str,
    limit: int,
    verify_ssl: bool = False,
    use_system_proxy: bool = False,
) -> list[dict]:
    session = _session(use_system_proxy)
    encoded = quote_plus(query)
    page = _get(
        session,
        f"https://duckduckgo.com/?q={encoded}&iax=images&ia=images",
        verify_ssl=verify_ssl,
        base_headers=_BROWSER_HEADERS,
        timeout=15,
    )
    page.raise_for_status()
    match = re.search(r"vqd=['\"]([^'\"]+)['\"]", page.text)
    if not match:
        return []

    params = {
        "l": "us-en",
        "o": "json",
        "q": query,
        "vqd": match.group(1),
        "f": ",,,",
        "p": "1",
    }
    response = _get(
        session,
        "https://duckduckgo.com/i.js",
        verify_ssl=verify_ssl,
        base_headers=_BROWSER_HEADERS,
        extra_headers={"Referer": f"https://duckduckgo.com/?q={encoded}&ia=images&iax=images"},
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("results", []):
        image_url = item.get("image")
        if not image_url:
            continue
        results.append({
            "url": image_url,
            "thumbnail": item.get("thumbnail", ""),
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "provider": "duckduckgo",
        })
        if len(results) >= limit:
            break
    return results


def _smolagents_image_results(
    query: str,
    limit: int,
    verify_ssl: bool = False,
    use_system_proxy: bool = False,
) -> list[dict]:
    try:
        from ddgs import DDGS
        from smolagents import Tool
    except ImportError as exc:
        raise ImportError(
            "smolagents image search requires: pip install 'smolagents[toolkit]' ddgs"
        ) from exc

    class WebImageSearchTool(Tool):
        name = "web_image_search"
        description = "Search the web for image URLs matching a visual query."
        inputs = {
            "search_query": {
                "type": "string",
                "description": "Visual image search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum image results to return",
            },
        }
        output_type = "array"

        def forward(self, search_query: str, max_results: int) -> list:
            ddgs_kwargs = {"verify": verify_ssl}
            if not use_system_proxy:
                ddgs_kwargs["proxy"] = None
            client = DDGS(**ddgs_kwargs)
            return list(client.images(
                query=search_query,
                safesearch="moderate",
                max_results=max_results,
            ))

    raw_results = WebImageSearchTool()(query, limit)
    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        image_url = item.get("image")
        if not image_url:
            continue
        results.append({
            "url": image_url,
            "thumbnail": item.get("thumbnail", ""),
            "title": item.get("title", ""),
            "source": item.get("url", "") or item.get("source", ""),
            "provider": "smolagents",
        })
        if len(results) >= limit:
            break
    return results


def _wikimedia_image_results(
    query: str,
    limit: int,
    verify_ssl: bool = False,
    use_system_proxy: bool = False,
) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": max(limit * 3, 8),
        "prop": "imageinfo",
        "iiprop": "url|mime|thumburl",
        "iiurlwidth": 640,
    }
    session = _session(use_system_proxy)
    response = _get(
        session,
        "https://commons.wikimedia.org/w/api.php",
        verify_ssl=verify_ssl,
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    results = []
    for item in pages.values():
        info = (item.get("imageinfo") or [{}])[0]
        original_url = info.get("url", "")
        thumb_url = info.get("thumburl", "")
        image_url = thumb_url if "/wikipedia/commons/thumb/" in thumb_url else _wikimedia_thumb_url(original_url)
        mime = info.get("mime", "")
        if not image_url or not mime.startswith("image/"):
            continue
        results.append({
            "url": image_url,
            "thumbnail": "",
            "title": item.get("title", ""),
            "source": "Wikimedia Commons",
            "provider": "wikimedia",
        })
        if len(results) >= limit:
            break
    return results


def _search_images(
    query: str,
    limit: int,
    provider: str,
    verify_ssl: bool = False,
    use_system_proxy: bool = False,
) -> list[dict]:
    if provider in {"smolagents", "auto"}:
        try:
            results = _smolagents_image_results(
                query, limit, verify_ssl, use_system_proxy
            )
            if results:
                return results
        except Exception as exc:
            _log.warning("[OllamaNodes] smolagents image search failed: %s", exc)
            if provider == "smolagents":
                return []

    if provider in {"duckduckgo", "auto"}:
        try:
            results = _duckduckgo_image_results(query, limit, verify_ssl, use_system_proxy)
            if results:
                return results
        except Exception as exc:
            _log.warning("[OllamaNodes] DuckDuckGo image search failed: %s", exc)
            if provider == "duckduckgo":
                return []

    if provider in {"wikimedia", "auto"}:
        try:
            return _wikimedia_image_results(query, limit, verify_ssl, use_system_proxy)
        except Exception as exc:
            _log.warning("[OllamaNodes] Wikimedia image search failed: %s", exc)
    return []


class OllamaWebImageTool:
    """Search web images, save them to ComfyUI temp, output an image batch, and optionally analyze them."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "run_tool"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "images",
        "metadata_json",
        "analysis_json",
        "image_1_analysis",
        "image_2_analysis",
        "image_3_analysis",
        "image_4_analysis",
    )
    OUTPUT_TOOLTIPS = (
        "Downloaded images as a ComfyUI IMAGE batch",
        "Search/download metadata including saved temp file paths",
        "Combined analysis JSON for all downloaded images",
        "Analysis JSON for image 1",
        "Analysis JSON for image 2",
        "Analysis JSON for image 3",
        "Analysis JSON for image 4",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Image search query",
                }),
                "ollama_model": ("OLLAMA_MODEL",),
                "web_search": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Search the web and download images when enabled",
                }),
                "do_analyze": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Analyze downloaded images with the selected Ollama vision model",
                }),
            },
            "optional": {
                "provider": (["smolagents", "auto", "duckduckgo", "wikimedia"], {
                    "default": "smolagents",
                    "tooltip": "smolagents uses a Hugging Face Tool with the DDGS image-search backend.",
                }),
                "max_images": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "tooltip": "Maximum images to download",
                }),
                "detail_level": (["basic", "detailed", "exhaustive"], {
                    "default": "detailed",
                    "tooltip": "Analysis detail level",
                }),
                "custom_categories": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional categories for image analysis",
                }),
                "output_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Width for the returned preview batch",
                }),
                "output_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Height for the returned preview batch",
                }),
                "pad_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Padding color used to make all images the same size",
                }),
                "verify_ssl": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Verify HTTPS certificates. Turn off if ComfyUI Python has certificate errors.",
                }),
                "use_system_proxy": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use proxy settings from the system environment.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("web_search", True):
            return float("nan")
        relevant = {key: value for key, value in kwargs.items() if key != "ollama_model"}
        cfg = kwargs.get("ollama_model") or {}
        relevant.update({
            "model": cfg.get("model", ""),
            "base_url": cfg.get("base_url", ""),
            "seed": cfg.get("seed", -1),
            "temperature": cfg.get("temperature", 0.3),
        })
        return json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)

    def run_tool(
        self,
        query: str,
        ollama_model: dict,
        web_search: bool = True,
        do_analyze: bool = True,
        provider: str = "auto",
        max_images: int = 4,
        detail_level: str = "detailed",
        custom_categories: str = "",
        output_width: int = 512,
        output_height: int = 512,
        pad_color: str = "#000000",
        verify_ssl: bool = False,
        use_system_proxy: bool = False,
    ):
        import folder_paths

        max_images = max(1, min(4, int(max_images)))
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)

        metadata = {
            "query": query,
            "web_search": web_search,
            "do_analyze": do_analyze,
            "provider": provider,
            "verify_ssl": verify_ssl,
            "use_system_proxy": use_system_proxy,
            "temp_directory": temp_dir,
            "results": [],
            "errors": [],
        }
        downloaded: list[Image.Image] = []

        if web_search and query.strip():
            candidates = _search_images(
                query.strip(),
                max_images * 4,
                provider,
                verify_ssl=verify_ssl,
                use_system_proxy=use_system_proxy,
            )
            slug = _safe_slug(query)
            for candidate in candidates:
                if len(downloaded) >= max_images:
                    break
                url = candidate["url"]
                try:
                    image = _download_image(
                        url,
                        verify_ssl=verify_ssl,
                        use_system_proxy=use_system_proxy,
                    )
                    index = len(downloaded) + 1
                    filename = f"ollama_web_{slug}_{int(time.time())}_{index}.png"
                    filepath = os.path.join(temp_dir, filename)
                    image.save(filepath, format="PNG")
                    downloaded.append(image)
                    metadata["results"].append({
                        **candidate,
                        "saved_path": filepath,
                        "width": image.width,
                        "height": image.height,
                    })
                except Exception as exc:
                    metadata["errors"].append({"url": url, "error": str(exc)})

        if not downloaded:
            downloaded = [Image.new("RGB", (output_width, output_height), pad_color)]
            metadata["errors"].append({
                "error": "No images downloaded. Returned a blank placeholder image.",
            })

        preview_images = [
            _fit_image(image, output_width, output_height, pad_color)
            for image in downloaded[:max_images]
        ]
        image_batch = _pil_to_tensor(preview_images)

        frame_outputs = ["", "", "", ""]
        analysis_items = []
        if do_analyze and metadata["results"]:
            cfg = ollama_model
            system = build_system_prompt(detail_level, custom_categories)
            user_prompt = build_user_prompt(detail_level)

            for index, image in enumerate(downloaded[:max_images], start=1):
                raw = generate(
                    base_url=cfg["base_url"],
                    model=cfg["model"],
                    prompt=f"Image {index} of {len(downloaded[:max_images])}.\n{user_prompt}",
                    system=system,
                    temperature=cfg.get("temperature", 0.3),
                    top_p=cfg.get("top_p", 0.9),
                    top_k=cfg.get("top_k", 40),
                    repeat_penalty=cfg.get("repeat_penalty", 1.1),
                    num_ctx=cfg.get("num_ctx", 8192),
                    num_predict=4096,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    images=[pil_to_base64(image)],
                    response_format="json",
                )
                parsed = extract_json_block(raw)
                if parsed is None:
                    parsed = {"raw_output": raw}
                else:
                    parsed = _normalize_analysis(parsed)
                item = {"image": index, "analysis": parsed}
                analysis_items.append(item)
                frame_outputs[index - 1] = json.dumps(parsed, indent=2, ensure_ascii=False)

        analysis_json = json.dumps({
            "query": query,
            "image_count": len(metadata["results"]),
            "images": analysis_items,
        }, indent=2, ensure_ascii=False)

        return (
            image_batch,
            json.dumps(metadata, indent=2, ensure_ascii=False),
            analysis_json,
            frame_outputs[0],
            frame_outputs[1],
            frame_outputs[2],
            frame_outputs[3],
        )
