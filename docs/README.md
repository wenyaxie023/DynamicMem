# Project page

Static project homepage for DynamicMem (served via GitHub Pages).

- `index.html` — the page (self-contained; CSS inline, fonts/icons from CDN)
- `static/images/` — figures rendered from the paper

## Enable GitHub Pages

Repo **Settings → Pages → Build and deployment**:
- Source: *Deploy from a branch*
- Branch: `main` (or your default) · folder: `/docs`

The page will be served at `https://wenyaxie023.github.io/DynamicMem/`.

`.nojekyll` is present so GitHub Pages serves `static/` paths verbatim.

## Local preview

```bash
python -m http.server -d docs 8000   # then open http://localhost:8000
```
