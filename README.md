# NeuroScan Medical Platform

Brain tumor 3D volumetric MRI segmentation and clinical healthcare management platform.

## Architecture

- **`apps/worker`**: Python FastAPI worker serving 3D Residual U-Net (BraTS 2020) with PyTorch and sliding window inference. Includes Colab Cloudflare runner.
- **`apps/backend`**: .NET 8 ASP.NET Core API organized via Clean Architecture, PostgreSQL database integration with EF Core, and JWT authentication.
- **`apps/web`**: Next.js 14 TypeScript web client with Tailwind CSS, clinical dashboard, doctor/hospital administration, and MRI segmentation upload interface.
- **`docs/`**: Agent onboarding documentation and versioned changelog.

For full architectural blueprints, developer setup instructions, and API references, see [docs/README.md](docs/README.md).
