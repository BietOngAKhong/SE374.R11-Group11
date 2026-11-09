# NeuroScan AI Platform - Agent Onboarding & Architecture Guide

## Overview

NeuroScan is an end-to-end clinical neuro-oncology platform designed for volumetric brain tumor segmentation from multi-modal magnetic resonance imaging (MRI) scans and healthcare provider account administration.

The repository is structured as a modular monorepo containing three core applications:

```
NeuroScan/
├── apps/
│   ├── worker/        FastAPI Python AI worker (PyTorch 3D Residual U-Net)
│   └── mock-vneid/    Mock VNeID Identity Provider & C06/BYT Verification Server
├── docs/
│   ├── changelog/     Versioned release logs
│   │   └── v0.1.0.md
│   └── README.md      This agent guide
├── docker-compose.yml PostgreSQL 16 database and optional pgAdmin
├── .env.example       Root environment templates
└── .gitignore         Standard git ignore rules
```

---

## Architecture & Communication Flow

```
[ Next.js Web Client ] (Port 3000)
       |
       |---> REST / JWT ---> [ .NET 8 Backend API ] (Port 5000)
       |                            |
       |                            |---> EF Core / Npgsql ---> [ PostgreSQL 16 ] (Port 5432)
       |                            |
       |                            +---> HTTP Client (Optional) --+
       |                                                           |
       +-------------------> Direct REST / Multipart ------------->+---> [ Python Worker ] (Port 8000)
                                                                             (3D Residual U-Net)
```

1. **Authentication & Identity**: The .NET WebApi issues JWT access tokens and persistent refresh tokens for administrators and doctors.
2. **Provider Administration**: Hospitals and Doctors are managed in PostgreSQL using Domain-Driven Clean Architecture with strict role-based access control.
3. **Inference Pipeline**: Multi-sequence NIfTI volumes are uploaded via multipart form requests. The Python worker executes sliding window volumetric inference, calculates tumor metrics, and returns the binary segmentation volume.

---

## Module Specifications

### 1. Python AI Worker (`apps/worker`)

- **Framework**: FastAPI with Uvicorn.
- **Model**: `ResidualUNet3D` based on Milletari et al. and Çiçek et al.
  - Encoder: 4 stages with residual convolutions (16, 32, 64, 128 channels) and 3D MaxPool.
  - Bottleneck: 256 channels with 3D InstanceNorm and LeakyReLU activations.
  - Decoder: 4 stages with 3D ConvTranspose, skip concatenation, and residual blocks.
  - Head: 1x1x1 Conv3D projecting to 4 classes.
- **Input Channels**: 4 MRI sequences (`flair`, `t1`, `t1ce`, `t2`).
- **Target Segmentation Classes**:
  - `0`: Background (BG)
  - `1`: Enhancing Tumor (ET)
  - `2`: Necrotic and Non-Enhancing Tumor Core (NCR/NET)
  - `3`: Peritumoral Edema (ED)
- **Composite Metrics**:
  - Whole Tumor (WT): Classes 1 + 2 + 3
  - Tumor Core (TC): Classes 1 + 2
  - Enhancing Tumor (ET): Class 1
- **Uncertainty Quantification (UQ) & Calibration**:
  - Test-time Monte Carlo Dropout ($T = 20$ forward passes with active dropout).
  - Volumetric percentiles ($1$ to $99$) and median volume in $\text{cm}^3$ for WT, TC, and ET.
  - Generates 3 multi-label spatial quantile masks: $p=5\%$ (conservative core), $p=50\%$ (median consensus), and $p=95\%$ (outer boundary).
  - Masks are packaged into `.nii.gz` volumes and stored in `apps/worker/temp/{task_id}_p{05|50|95}.nii.gz`.
  - 1-D uniform histogram equalization calibration via Ordinary Least Squares (OLS) slope fitting:
    $u_k = \frac{k - 0.5}{N_{\text{val}}}$, $u_k \approx a \cdot (\alpha_{(k)} - 0.5) + 0.5$.
  - 3-Fold Cross-Validation over 34 validation cases (`val_calibration/`), verifying 95% coverage interval stability.
- **Endpoints**:
  - `GET /health`: Health status, device (CUDA/MPS/CPU), and weights validation.
  - `POST /api/v1/inference`: Accepts 4 NIfTI files (`flair`, `t1`, `t1ce`, `t2`), executes 20-pass MC Dropout, packages 3 quantile masks, and returns volumetric percentiles and calibration factors.
  - `GET /api/mask/{task_id}?p={5|50|95}`: Downloads the specified `.nii.gz` mask.
- **Colab Notebook & Offline Script**:
  - `colab_runner.ipynb`: Runs non-blocking Uvicorn daemon and Cloudflare Tunnel on free cloud GPUs.
  - `scripts/calibrate_val.py`: Standalone CLI to execute 3-Fold Cross-Validation on GPU and regenerate `calibration_cache.json`.

### 2. .NET 8 Backend (`apps/backend`)

The backend is built with ASP.NET Core 8 following Clean Architecture principles:

- **NeuroScan.Domain**:
  - Core entities: `User`, `Hospital`, `Doctor`, `RefreshToken`.
  - Enums: `UserRole` (`Admin`, `HospitalAdmin`, `Doctor`).
  - Abstractions: `IRepository<T>`, `IUnitOfWork`, `BaseEntity`.
- **NeuroScan.Application**:
  - DTOs for authentication, hospital directory, doctor records, and inference jobs.
  - Domain service interfaces: `IAuthService`, `IHospitalService`, `IDoctorService`, `IWorkerClient`.
  - Service implementations with dependency injection extensions.
- **NeuroScan.Infrastructure**:
  - `ApplicationDbContext` with PostgreSQL configuration through `Npgsql.EntityFrameworkCore.PostgreSQL`.
  - Generic repository and unit of work implementations.
  - JWT token generator and validator with HMAC-SHA256.
  - `BCrypt` password hashing.
  - `WorkerHttpClient` resilient HTTP client for worker communication.
- **NeuroScan.WebApi**:
  - REST controllers: `AuthController`, `HospitalController`, `DoctorController`, `InferenceController`.
  - Global `ExceptionHandlingMiddleware` for unified JSON errors.
  - Database automatic seeding for initial `admin` account.

### 3. Next.js Web Client (`apps/web`)

- **Framework**: Next.js 14 App Router, React 18, TypeScript.
- **Styling**: Tailwind CSS with dark slate medical aesthetics.
- **Icons**: Lucide React.
- **Views**:
  - `/`: System overview, key statistics, AI worker connectivity status, model specifications.
  - `/login`: Secure authentication portal with JWT storage.
  - `/hospitals`: Healthcare facility directory with interactive creation modal.
  - `/doctors`: Physician registry with hospital assignment and license numbers.
  - `/inference`: 4-sequence MRI file upload, progress indicators, volumetric measurements table, and mask download.

### 4. Mock VNeID & Ministry of Health Server (`apps/mock-vneid`)

- **Framework**: Node.js with Express, adapted from the `@opengovsg/mockpass` architecture.
- **Port**: `5001` (configurable via `PORT` in `.env`).
- **Core Capabilities**:
  - **OIDC/OAuth 2.0 Authorization Server**:
    - `GET /oauth/authorize`: Serves the national VNeID authentication interface modeled on official BCA styles (`new.html`), featuring a simulated QR scanner and a top developer bar with 1-click personas. Emits a 302 GET redirect to `redirect_uri?code=AUTH_CODE&state=STATE`.
    - `POST /oauth/token`: Back-channel endpoint for BFF server to exchange authorization codes for RS256-signed JWTs (`id_token`, `access_token`, `refresh_token`). Tokens automatically embed citizen demographic info and practitioner credentials when the citizen is a registered doctor.
    - `GET /.well-known/jwks.json`: Publishes the active RSA 2048-bit public key for offline token verification by .NET Core API and BFF without remote introspection.
    - `GET /.well-known/openid-configuration`: Standard OIDC discovery document.
  - **C06 National Population Database (C06)**:
    - `GET /api/v1/c06/citizen/:cccd`: Server-to-server demographic and health insurance (BHYT) lookup by 12-digit citizen ID.
  - **Ministry of Health Database (BYT)**:
    - `GET /api/v1/byt/practitioner/:cchn`: Verifies medical practitioner license numbers (CCHN), practice scope, and active practicing hospital (`ma_cskcb_cong_tac`).
    - `GET /api/v1/byt/facilities`: Returns the official list of registered hospitals and healthcare facilities.
    - `GET /api/v1/byt/facilities/:ma_bv`: Lookup facility details by hospital code.
  - **Data Separation**:
    - `data/citizens.json`: Standard demographic and health insurance records.
    - `data/medical_facilities.json`: Hospital profiles, levels, and administrative codes.
    - `data/medical_practitioners.json`: Physician practice licenses and affiliated facility codes.

---

## Database Configuration

PostgreSQL 16 is configured via `docker-compose.yml`:

- Image: `postgres:16-alpine`
- Port: `5432`
- Persistent Volume: `postgres_data`
- Healthcheck: Enabled (`pg_isready`)
- Optional Management: `pgadmin` on port `5050`

Default database credentials:
- Username: `postgres`
- Password: `postgres`
- Database: `neuroscan_db`

---

## Local Development Setup

### 1. Start Database
```bash
docker-compose up -d postgres
```

### 2. Start Python Worker
```bash
cd apps/worker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 3. Start .NET Backend
```bash
cd apps/backend
dotnet run --project NeuroScan.WebApi
```

### 4. Start Next.js Frontend
```bash
cd apps/web
npm install
npm run dev
```

### 5. Start Mock VNeID Server
```bash
cd apps/mock-vneid
npm install
npm start
# Server listens at http://localhost:5001
```

---

## Agent Instructions for Future Extensions

1. **Coding Conventions**: Do not write comments in any source code file. All code and documentation must strictly use English.
2. **Environment Variables**: Whenever adding a new configuration key, update both `.env.example` and the respective app configuration schemas.
3. **Database Migrations**: EF Core migrations can be added using `dotnet ef migrations add <Name> --project NeuroScan.Infrastructure --startup-project NeuroScan.WebApi`.
4. **Worker Extension**: When altering model weights or patch dimensions, adjust both `apps/worker/src/config.py` and the corresponding frontend inference parameters.
