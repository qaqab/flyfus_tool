# Changelog

## 0.0.39

- Uploaded the debug bounding-box preview before calling RevealLayer.
- Returned the uploaded boxed image URL even when RevealLayer later fails or times out.

## 0.0.37

- Standardized Gemini layer boxes to 0-1000 coordinates and converted them to source pixels inside the tool.
- Returned both normalized_boxes and pixel_boxes for debugging.

## 0.0.34

- Returned log_id on success and added per-stage timing fields.
- Parallelized RevealLayer output downloads and PNG encoding.

## 0.0.32

- Replaced ZIP delivery with concurrent OSS uploads for individual PNG layers.
- Added an optional debug image with numbered bounding boxes and coordinates.

## 0.0.30

- Removed redundant bearer-token character checks; retained detailed RevealLayer pipeline logs.

## 0.0.29

- Added detailed sanitized logs for RevealLayer requests, responses, ZIP checks, and OSS upload.

## 0.0.28

- Added clear validation for non-ASCII RevealLayer and OSS bearer tokens.

## 0.0.27

- Fixed RevealLayer startup under Dify's Python 3.12 dynamic plugin loader.

## 0.0.20

- Added the synchronous Flyfus RevealLayer tool for one image URL and optional named bounding boxes.
- Added original-RGB foreground restoration and a Photoshop-ready ZIP uploaded through the existing OSS file endpoint.

## 0.0.1

- Initial Flyfus Tool release.
- Includes image generation/editing, image invocation diagnostics, SLS tracing, URL context conversion, Skills, and next-step tools.
- Image results return `log_id` and `request_fingerprint` for SLS lookup.
