# Third-Party Notices

## General Motion Retargeting (GMR) — Upstream

This patch is intended to be applied on top of:

- **Repository:** https://github.com/YanjieZe/GMR  
- **License:** MIT License  
- **Copyright:** Copyright (c) 2025 Yanjie Ze (and upstream contributors)

You must retain the upstream copyright and MIT license text when distributing any fork that includes this patch.

## This patch (IK soft joint limits)

Additional code in this patch (joint limit penalty, temporal smoothing, diagnostics) is contributed under the **same MIT License**.

Suggested additional copyright line:

```
Copyright (c) 2025-2026 <Your Name or Organization>
```

## Dependencies

Same as upstream GMR: mink, MuJoCo, smplx, etc. See upstream `setup.py` and `README.md`.

This patch does **not** add new robot models or redistribute proprietary meshes.
