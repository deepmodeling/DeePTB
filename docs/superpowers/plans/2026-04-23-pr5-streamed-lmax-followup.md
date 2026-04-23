# PR5 Streamed Lmax Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the PR #5 `streamed_m_major_ref` path so it is correct when `irreps_in.lmax` and `irreps_out.lmax` differ.

**Architecture:** Keep the production default `so2_fusion_mode="staged"` unchanged. Bound SO2 order loops by `min(irreps_in.lmax, irreps_out.lmax)`, while building Wigner rotation support up to `max(irreps_in.lmax, irreps_out.lmax)` so output-only irreps can still rotate out safely.

**Tech Stack:** PyTorch, e3nn irreps, existing DeePTB `SO2_Linear`, existing compact/full Wigner backends.

---

### Task 1: Bound SO2 order bookkeeping

**Files:**
- Modify: `dptb/nn/tensor_product_moe_v3.py`

- [ ] **Step 1: Add explicit lmax fields**

Set these fields after `self.irreps_in` and `self.irreps_out` are initialized:

```python
self.in_l_max = self.irreps_in.lmax
self.out_l_max = self.irreps_out.lmax
self.m_max = min(self.in_l_max, self.out_l_max)
self.l_max = max(self.in_l_max, self.out_l_max)
```

- [ ] **Step 2: Build only valid m-order modules**

Change the `SO2_m_Linear` construction loop from output-lmax bounded to `self.m_max` bounded:

```python
for m in range(1, self.m_max + 1):
    self.m_linear.append(SO2_m_Linear(...))
```

- [ ] **Step 3: Resize masks to valid m orders**

Build `m_in_mask`, `m_out_mask`, and `self.m_in_num` with `self.m_max + 1` rows. In the input/output irreps loops, only visit `m <= min(l, self.m_max)`.

- [ ] **Step 4: Preserve Wigner coverage**

Remove the old input-only `self.l_max` assignment and keep `self.dims` / `self.offsets` based on the new `max(in_lmax, out_lmax)` value.

### Task 2: Bound staged and streamed forward loops

**Files:**
- Modify: `dptb/nn/tensor_product_moe_v3.py`

- [ ] **Step 1: Update staged loop**

Change:

```python
for m in range(self.irreps_out.lmax + 1):
```

to:

```python
for m in range(self.m_max + 1):
```

- [ ] **Step 2: Update streamed loop**

Apply the same `self.m_max + 1` bound in `_forward_streamed_m_major_ref`.

- [ ] **Step 3: Keep rotate-out over output irreps**

Do not cap rotate-out group iteration by `self.m_max`; output irreps with higher `l` still need Wigner rotation after lower-order SO2 updates.

### Task 3: Add regression coverage

**Files:**
- Create: `dptb/tests/test_so2_streamed_lmax_bounds.py`

- [ ] **Step 1: Add staged-vs-streamed parity test**

Create a test that uses:

```python
irreps_in = "1x0e + 2x1o + 1x2e"
irreps_out = "1x0e + 1x1o + 1x2e + 1x3o"
```

Instantiate two `SO2_Linear` modules with identical weights, `so2_fusion_mode="staged"` and `"streamed_m_major_ref"`, then compare forward output plus gradients for `x`, `R`, and `latents`.

- [ ] **Step 2: Cover rotate flag combinations**

Parameterize over:

```python
(rotate_in, rotate_out) = [(True, True), (False, True), (True, False), (False, False)]
```

- [ ] **Step 3: Use strict fp64 tolerances**

Use `torch.float64` and assert close with `atol=1e-8, rtol=1e-8` for gradients.

### Task 4: Verify locally and on natlan

**Files:**
- None

- [ ] **Step 1: Local syntax check**

Run:

```powershell
python -m py_compile dptb\nn\tensor_product_moe_v3.py dptb\tests\test_so2_streamed_lmax_bounds.py
```

- [ ] **Step 2: Local pytest**

Run:

```powershell
python -m pytest dptb\tests\test_so2_streamed_lmax_bounds.py -q
```

Expected on this Windows environment: skip if torch/e3nn are unavailable.

- [ ] **Step 3: Natlan pytest**

Upload the changed files to the PR5 natlan test checkout and run:

```bash
source /home/mingkang_nt/data/anaconda3/etc/profile.d/conda.sh
conda activate my_oeq
PYTHONPATH=/home/mingkang_nt/codex/0422_tests/so2_streamed/DeePTB PYTHONNOUSERSITE=1 \
python -m pytest dptb/tests/test_so2_streamed_lmax_bounds.py -q -s
```

Expected: all tests pass.

### Task 5: Commit and push

**Files:**
- Modify: `dptb/nn/tensor_product_moe_v3.py`
- Create: `dptb/tests/test_so2_streamed_lmax_bounds.py`
- Create: `docs/superpowers/plans/2026-04-23-pr5-streamed-lmax-followup.md`

- [ ] **Step 1: Review diff**

Run:

```powershell
git diff --check
git status --short --branch
```

- [ ] **Step 2: Commit**

Run:

```powershell
git add dptb/nn/tensor_product_moe_v3.py dptb/tests/test_so2_streamed_lmax_bounds.py docs/superpowers/plans/2026-04-23-pr5-streamed-lmax-followup.md
git commit -m "Harden SO2 streamed lmax bounds"
```

- [ ] **Step 3: Push**

Run:

```powershell
git push origin 0422-so2-streamed-m-major
```

This updates PR #5.
