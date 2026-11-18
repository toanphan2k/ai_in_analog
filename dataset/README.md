# Dataset

Append evaluated records here as JSON Lines. The store is intentionally excluded from
the selector: call `src/dataset.py` helpers and pass arrays to `src/selector.py`.

Recommended fields:

- `x_norm`: eight normalized W/L variables
- `w_in`, `l_in`, `w_n`, `w_p`, `w_rst`, `w_tail`, `w_8`, `l_8`
- `delay_s`
- `power_w`
- `input_referred_offset_v`
- `kickback_noise_v`
- `output_swing_v`
- `fmax_hz`
