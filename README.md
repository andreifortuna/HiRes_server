# HiRes Slovenia package server

Internal FastAPI server for CLSS aerial orthophoto, 0.5 m bare-earth terrain, and optional classified GKOT point clouds. It listens on `127.0.0.1:8775` only. No users, subscriptions, quotas, OSM, or automatic package expiry.

## Start

On Windows, create a Python 3.13 virtual environment in `.venv`, install `requirements.txt`, then run `./start.ps1`. The API is at `http://127.0.0.1:8775`. The server owns `data/jobs.sqlite3`, `data/cache`, `data/packages`, and `data/server.log`. Temporary derived files go to `C:\tmp\hires-server\work` by default. Set `HIRES_DATA_DIR` or `HIRES_WORK_DIR` to change those locations.

The CLSS public viewer's guest API resolves its 1 km tile links. If its public guest client parameters change, set `CLSS_GUEST_CLIENT_SECRET` to the current value. This is a public viewer credential, not an administrator credential. The service intentionally binds to loopback; use an SSH tunnel if accessing it from another machine.

## API

`POST /jobs` (JSON):

```json
{"requestId":"b6d94891-6ed1-4b0a-8bcc-f7a69662c626","center":{"lat":46.378333,"lon":13.836667},"widthMeters":1000,"heightMeters":1000,"includeGkot":true}
```

The same `requestId` and same body return the same job. `GET /jobs` lists jobs; `GET /jobs/{id}` returns independently tracked `map` and optional `gkot` states. Ready files download from `GET /jobs/{id}/files/map` and `GET /jobs/{id}/files/gkot`; HTTP range requests and SHA-256 response headers are supported. `DELETE /jobs/{id}` deletes server packages and stops unfinished work. `GET /health` checks the service.

`widthMeters` and `heightMeters` are independent rectangle dimensions. Each must be at least 100 m. The server applies its existing terrain-grid capacity check before accepting a job.

The map ZIP contains lossless RGB PNG tiles at the POF native 0.16 m resolution, `orthophoto.json`, `terrain-height.lerc` at nominal 0.5 m spacing, a separate power-of-two-plus-one `terrain-height.png` for the Cocos mesh (16-bit normalized heights packed into RGB channels R and G), `terrain-lighting.png` with sun visibility/AO/relief/curvature, slope-aware cliff colour/detail textures, `terrain.json`, original CLSS DMR LAZ tile(s), `manifest.json`, and `ATTRIBUTION.txt`. The render grid chooses width and height independently from the requested bounds; a 1 km square map has a 2049 x 2049 render mesh with 0.488 m vertex spacing and measured 1x vertical scale. The optional separate GKOT ZIP contains one cropped LAZ with all source point attributes, a baked ground-normal map, conservative close-range cliff mesh patches, manifest, and attribution. The cliff mesh is limited to 2,000 triangles. Both products use EPSG:3794; metadata records exact bounds, sample spacing, source tiles, and acquisition dates. The format omits HEDDA's OSM layers; a viewer must read this manifest.

The server downloads and caches complete source tiles before cropping. A 1 km request not aligned with the CLSS grid can require four POF, four DMR, and four GKOT tiles. Only completed, verified archives are published. A failed artifact does not discard a successful sibling. A restart requeues interrupted work and keeps completed packages. Source tiles remain in the shared cache after deleting a package.

GURS attribution and terms: https://www.e-prostor.gov.si/en/access-to-geodetic-data/
CLSS portal and published download instructions: https://clss.si/ and https://assets.flycom.si/clss/navodila_pregledovalnik_clss.pdf
