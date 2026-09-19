# SuperPhoto deployment

SuperPhoto v1.1.0 runs the image-processing service with OpenVINO CPU
inference, a FastAPI web application, SQLite WAL state and two worker
processes.

## Release metadata

- Version: `v1.1.0` (source version: `1.1.0`)
- Default image: `ghcr.io/general-brash/superphoto:v1.1.0`
- API and worker use the same image. Override the defaults with
  `IMAGE_REPOSITORY` and `IMAGE_TAG` when deploying a different registry or
  image tag.
- Release notes: v1.1.0 standardizes the versioned GHCR image reference for
  both services and keeps image publication separate from the GitHub Release
  metadata workflow. This document records deployment metadata only; it does
  not claim that the GitHub release or GHCR image has already been published.

## Runtime

- Public site: `https://superphoto.taffy.edu.kg/`
- New API: `/api`
- API documentation: `/api/docs` (administrator session required)
- Local legacy API: `/batches`, `/jobs/...` (Cloudflare requests are rejected)
- API resources: 3 CPUs / 3 GiB
- Worker resources: 7 CPUs / 13 GiB, concurrency 2
- Persistent state: `state/jobs.db`
- Files: `data/input`, `data/output`, `data/tmp`
- Models: `models/`

The resource limits intentionally follow the requested configuration and sum to
more than the host's physical memory. Keep the 64 MP intermediate-image limit
and monitor memory during dual-concurrency 4K and face-enhancement tests.

## Configuration

Runtime secrets are stored in `.env`, which is ignored by Git and must remain
mode `0600`. The first startup creates the administrator only when no existing
administrator is present.

Required variables:

```text
SUPERPHOTO_SESSION_SECRET
SUPERPHOTO_ADMIN_USERNAME
SUPERPHOTO_ADMIN_PASSWORD
```

Cloudflare Turnstile is implemented but disabled until keys are configured:

```text
TURNSTILE_REQUIRED=true
TURNSTILE_SITE_KEY=...
TURNSTILE_SECRET_KEY=...
```

OIDC (Sub2) is **deferred and disabled by default**. Do not enable it merely by
filling example values. After Sub2 exposes a verified OIDC provider, register a
confidential client with its exact HTTPS callback URL (`/api/auth/oidc/callback`),
confirm issuer/discovery metadata, scopes and claim mapping, and provision the
client secret in the host-only `.env` (never in the image or Git):

```text
OIDC_ENABLED=false
OIDC_ISSUER=https://your-verified-issuer.example
OIDC_DISCOVERY_URL=  # optional when issuer has standard discovery
OIDC_CLIENT_ID=...
OIDC_CLIENT_SECRET=...
OIDC_REDIRECT_URI=https://your-public-site.example/api/auth/oidc/callback
OIDC_SCOPES=openid profile
```

The API requires client ID, secret, redirect URI, and issuer even when
`OIDC_ENABLED=true`; incomplete settings keep login disabled.
`OIDC_DISCOVERY_URL` is optional when the issuer uses standard discovery. The image
pins `PyJWT==2.10.1` and `cryptography==45.0.7` in `requirements.lock`. ID-token
RS256/JWKS signature and claim verification is mandatory and fail-closed: missing
verification dependencies, unavailable signing keys, or any issuer, audience,
nonce, expiry, or signature mismatch must reject the OIDC login. Sub2 source,
migrations, provider activation, and production callback testing are not part of
this deployment document.

After changing `.env`, recreate the API container (for example,
`docker compose up -d --force-recreate api`). `.dockerignore` excludes `.env`,
`state/`, `logs/`, and `data/` from `COPY .`; bind-mounted runtime files stay on
the host and must be protected and backed up separately.

## Commands

```bash
cd /home/taffy/realesrgan
sg docker -c 'docker compose build api'
sg docker -c 'docker compose up -d'
sg docker -c 'docker compose ps'
sg docker -c 'docker compose logs -f api worker'
```

Health checks:

```bash
curl http://127.0.0.1:8000/api/health
curl https://superphoto.taffy.edu.kg/api/health
```

## Limits and retention

- Static JPEG and PNG input only
- 10 MiB per file
- Guest: 2 files per batch, 3 files per UTC day, 2 active jobs
- Registered user: 10 files per batch, 30 files per UTC day, 10 active jobs
- Global active queue: 50 jobs
- Maximum input side: 4096 pixels
- Maximum estimated x4 intermediate: 64 MP
- Guest files: 24 hours
- Registered-user files: 7 days
- Share links: 24 hours
- New uploads and ZIP creation stop below 5 GiB free disk

The API cleanup thread removes expired sessions, shares, jobs and temporary ZIP
files. Processing jobs are never force-deleted or force-cancelled.

## Models

```text
RealESRGAN_x4plus.xml              general images
RealESRGAN_x4plus_anime_6B.xml     anime and illustrations
GFPGANv1.3.pth                     optional face enhancement
gfpgan/weights/detection_Resnet50_Final.pth
gfpgan/weights/parsing_parsenet.pth
```

If a model is unavailable, the API reports the missing capability and rejects
new jobs that require it instead of allowing a known-failing job into the queue.
