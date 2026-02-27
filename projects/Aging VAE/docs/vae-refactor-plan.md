# VAE Refactor Plan: Industry-Standard Architecture

> **Status: Implemented** — Refactor completed per this plan.

## Problem

The current VAE is a **subclassed Model** with no graph defined in `super().__init__()`. Keras treats it as a black box and builds it lazily on first call, which forces the `build()` workaround for `ModelCheckpoint`.

---

## Target Architecture: Functional Graph + Custom `train_step`

Use a **Functional-style model** (graph defined at construction) while keeping a custom `train_step` for the VAE loss.

---

## Phase 1: Define the Graph in the VAE Constructor

**Current:**
```python
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)  # No graph defined
        self.encoder = encoder
        self.decoder = decoder
```

**Target:**
```python
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        inputs = encoder.input
        z_mean, z_log_var, z = encoder(inputs)
        outputs = decoder(z)
        super(VAE, self).__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
```

**Effect:** The model has a full input→output graph at construction, so it is built immediately and no `build()` call is needed.

---

## Phase 2: Keep Custom `train_step`

The VAE loss (reconstruction + β·KL) still requires a custom `train_step`. That stays as-is; only the constructor changes.

---

## Phase 3: Remove the Hack from `main.py`

Delete the `vae.build(input_shape=(None, 64, 64, 3))` call. The model is built when it is instantiated.

---

## Phase 4: Optional – Centralize Input Shape

To avoid hardcoding `(64, 64, 3)` elsewhere:

1. Add a constant in `vae_factory.py`:
   ```python
   INPUT_SHAPE = (64, 64, 3)
   ```
2. Use it in `layers.Input(shape=INPUT_SHAPE)` and export it for any other code that needs it.

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/model/vae.py` | Wire encoder→decoder and pass `inputs`/`outputs` to `super().__init__()` |
| `main.py` | Remove `vae.build(...)` |
| `vae_factory.py` | (Optional) Add `INPUT_SHAPE` constant |

---

## Rationale

- **Functional API pattern:** Defining the graph via `Model(inputs, outputs)` is the standard Keras pattern for models with a clear data flow.
- **Eager build:** The graph is known at construction, so the model is built immediately.
- **Custom training:** Overriding `train_step` for custom losses is the recommended approach for VAEs and similar models.
- **No side effects:** No dummy forward passes or manual `build()` calls.
- **Compatibility:** Works with `ModelCheckpoint`, `EarlyStopping`, and other callbacks without extra setup.
