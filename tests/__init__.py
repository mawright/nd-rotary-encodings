import torch
from hypothesis import strategies as st


@st.composite
def positions_strategy(draw):
    positions_dtype = draw(st.sampled_from([torch.float32, torch.long]))
    if positions_dtype == torch.float32:
        min_position = draw(
            st.floats(min_value=-1e30, max_value=1e30, exclude_max=True)
        )
        max_position = draw(st.floats(min_value=min_position, max_value=1e30))
    else:
        min_position = draw(st.integers(min_value=int(-1e10), max_value=int(1e10) - 1))
        max_position = draw(st.integers(min_value=min_position, max_value=int(1e10)))
    return positions_dtype, min_position, max_position
