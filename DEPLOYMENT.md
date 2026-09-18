# SuperPhoto deployment

SuperPhoto v1.0.4 runs Real-ESRGAN with OpenVINO CPU inference, a FastAPI web
application, SQLite WAL state and two worker processes.

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

After changing `.env`, recreate the API container.

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
- Guest: 2 files per batch, 5 files per UTC day, 2 active jobs
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
