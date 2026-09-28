# Security

Load checkpoints and pickle caches only from trusted sources. Some supported formats use Python deserialization and can execute code. Run training and evaluation with the least filesystem and network access needed. Dataset paths and local shell configuration should also be treated as trusted inputs.

Do not publish tokens, private annotations, personal data or executable proof-of-concept files in issues. When the repository provides GitHub's **Report a vulnerability** option, use it for private disclosure. If private reporting is not enabled, request a private reporting channel without posting sensitive details.

Provide the affected source revision, a minimal sanitized reproduction and the expected impact. Security reports do not need a full dataset or model checkpoint. No response-time guarantee is implied by this source release.
