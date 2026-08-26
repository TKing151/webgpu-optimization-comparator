# WebGPU Kernel Comparator

A deliberately small static WebGPU MVP. It compares two implementations of the same 2D 9-point stencil:

- **Naive:** each output element reads its neighbors directly from storage memory.
- **Tiled:** each 16×16 workgroup loads an 18×18 tile plus halo into workgroup memory and reuses those values.

The point is not to produce a generic GPU benchmark. The point is to ask a concrete portability question: **does a familiar GPU optimization actually help on this device?**

## Files

- `index.html` — UI
- `style.css` — styling
- `app.js` — WebGPU setup, benchmark, validation, results
- `stencil.wgsl` — the two kernels
- `.github/workflows/deploy.yml` — free GitHub Pages deployment

## Deploy

Push the directory to a GitHub repository. In **Settings → Pages**, choose **GitHub Actions**. The included workflow publishes the static site.

## Important measurement caveat

The MVP measures elapsed browser time around `queue.onSubmittedWorkDone()`. That is intentionally simple and portable, but it is not a pure GPU timestamp. A later version can add a proper timestamp-query path where supported.

The first useful experiment is simply to run this on several devices and see whether the optimization changes rank, and by how much.
