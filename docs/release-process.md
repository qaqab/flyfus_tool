# Release Process

1. Implement and test the feature.
2. Add its user-visible changes under a new `## x.y.z` section in `CHANGELOG.md`.
3. Commit and push source to `main`.
4. Run `scripts/plugins/打包并上传到GitHub.sh --flyfus-tool --publish`.
5. Install the generated `.difypkg` in Dify and configure credentials as a new plugin.

Before a clean reinstall, remove old plugin installation records, credentials, Redis keys, and Plugin Daemon package/runtime directories. Do not delete the Flyfus source repository or release package.
