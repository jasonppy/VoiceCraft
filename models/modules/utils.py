# cp from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/transformer.py, modified by Puyuan Peng
import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)

def generate_partial_autoregressive_mask(sz, start, end):
    mask = torch.zeros(sz, sz).bool()
    mask[start:end, start:end] = torch.triu(torch.ones(end-start, end-start,dtype=torch.bool), diagonal=1)
    mask[:start, start:end] = True
    mask[end:, start:end] = True
    return mask
