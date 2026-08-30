#!/usr/bin/env python3
"""
Fix: vLLM 0.28.0 XPU outputs endless "!" due to NaN in KV cache.

Root cause: In USE_TD branch, _load_kv_tile_td loads KV cache without
tile_mask filtering, allowing uninitialized memory (NaN) to propagate
through attention -> softmax -> logits -> token 0 ("!").

Fix: Apply tile_mask to K_load and V_load using tl.where() to replace
invalid positions with 0.0, preventing NaN propagation.

See: https://github.com/vllm-project/vllm/pull/44850
"""

import re
from pathlib import Path

TARGET = Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py")
MARKER = "# TILE_MASK_FIX"

def main() -> None:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return

    source = TARGET.read_text(encoding="utf-8")
    if MARKER in source:
        print("tile_mask fix already applied")
        return

    lines = source.split('\n')
    new_lines = []
    patched = False
    use_td_indent = ""
    k_load_found = False

    for i, line in enumerate(lines):
        new_lines.append(line)
        
        if 'if USE_TD:' in line and not patched:
            use_td_indent = line[:len(line) - len(line.lstrip())]
            j = i + 1
            while j < len(lines) and j < i + 40:
                next_line = lines[j]
                if next_line.strip() and not next_line.startswith(use_td_indent + "    "):
                    break
                
                if not k_load_found and '_load_kv_tile_td' in next_line and 'K_load' in next_line:
                    k_indent = next_line[:len(next_line) - len(next_line.lstrip())]
                    new_lines.append(f'{k_indent}{MARKER}')
                    new_lines.append(f'{k_indent}K_load = tl.where(tile_mask[None, :], K_load, 0.0)')
                    k_load_found = True
                    patched = True
                
                if k_load_found and '_load_kv_tile_td' in next_line and 'V_load' in next_line:
                    v_indent = next_line[:len(next_line) - len(next_line.lstrip())]
                    new_lines.append(f'{v_indent}{MARKER}')
                    new_lines.append(f'{v_indent}V_load = tl.where(tile_mask[:, None], V_load, 0.0)')
                    patched = True
                    break
                
                j += 1
            
            if patched:
                for k in range(j, len(lines)):
                    new_lines.append(lines[k])
                break

    if patched:
        patched_source = '\n'.join(new_lines)
        TARGET.write_text(patched_source, encoding='utf-8')
        try:
            compile(patched_source, str(TARGET), 'exec')
            print(f"patched {TARGET}")
        except SyntaxError as e:
            print(f"Syntax error: {e}")
            TARGET.write_text(source, encoding='utf-8')
    else:
        print("Could not locate K_load/V_load in USE_TD branch")

if __name__ == "__main__":
    main()