# tools/

`make_reference.py` regenerates `tests/reference.npz` by driving the **original**
implementation (from the paper repo) with fixed inputs.

It needs a checkout of the original code at `orig/`, and it stubs out PyTorch —
the original could not import its environment-only path without torch, because
`Env_control_manager` sat in the inheritance chain and imported the model.

```bash
git clone https://github.com/johnschwarcz/CognitiveGridworld orig
python tools/make_reference.py
```

You should not need to run this unless you are deliberately re-verifying the
port against the original.
