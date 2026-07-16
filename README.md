# Flypower Tool

Flypower tools plugin for Dify.

It provides image generation, next-step control, and public-file context tools.

## Tools

- `flypower_image_generate`: generate or edit images through an OpenAI-compatible endpoint.
- `set_next_step`: return the next objective and reasoning effort for the following model call.
- `read_file`: convert comma- or newline-separated public URLs into Flypower context for the next model call.

## Configuration

- Use an HTTPS API base URL, such as `https://litellm.flyfus.com` or `https://litellm.flyfus.com/v1`.
- Image generation verifies credentials with `GET /v1/models`; the response must include at least one supported image model.
- `set_next_step` does not require image credentials.

## Remote Debugging

Use Dify remote debugging during development instead of repeatedly packaging and installing the plugin.

1. In Dify Plugin Management, open the debug-plugin dialog and copy its debugging key.
2. Create a local `.env` file. Do not commit it.

   ```env
   INSTALL_METHOD=remote
   REMOTE_INSTALL_URL=127.0.0.1:15003
   REMOTE_INSTALL_KEY=<debugging-key>
   ```

   In this local Dify setup, the daemon debug port is exposed from container port
   `5003` to host port `15003`.

3. Start the plugin from this directory:

   ```bash
   uv run --locked python -m main
   ```

4. Confirm the plugin daemon reports `debugging runtime connected`, then invoke
   the debug-marked Flypower Tool from Dify. Code changes require only a local
   process restart; no plugin package is needed.

## Image Upload Diagnosis

Image upload logs are sent best-effort to the configured Alibaba Cloud SLS
project. The Logstore is fixed as `flyfus-dify-llm-log`. Search by the `log_id`
included in an image-tool error. Upload records include the request endpoint,
upload type, MIME type, byte size, SHA-256, status code, elapsed time, cloud
function request ID, and response summary.

When an OSS upload returns `404`, first compare the logged endpoint with a
direct multipart upload using the same `file` and `filename` fields. Also check
for invisible whitespace in `oss_api_base_url`, especially pasted line
separators such as `U+2028`. The tool normalizes the configured base URL with
`strip().rstrip("/")` before appending the upload path.
