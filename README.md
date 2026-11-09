# NeuroScan Medical Platform

Brain tumor 3D volumetric MRI segmentation and clinical healthcare management platform.

## Architecture

- **`apps/worker`**: Python FastAPI worker serving 3D Residual U-Net (BraTS 2020) with PyTorch and sliding window inference. Includes Colab Cloudflare runner.
- **`apps/mock-vneid`**: Mock VNeID OIDC/OAuth 2.0 Identity Provider, C06 National Population Database, and BYT Practitioner/Hospital verification service.
- **`docs/`**: Agent onboarding documentation and versioned changelog.

For full architectural blueprints, developer setup instructions, and API references, see [docs/README.md](docs/README.md).
