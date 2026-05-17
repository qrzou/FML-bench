"""GraphCodeBERT embedding helper for AdaptiveSearch Phase-2 signals.

**Pipeline must match the offline calibration source-of-truth byte-for-byte:**
``formal_results_analysis/compute_exploration_diversity_fullcode.py::embed_fullcode_cls_sum``.

Concretely:
  1. Sort dict keys (ASCII / lexicographic full path order).
  2. Concatenate file contents with ``"\n"`` separators. NO file-name banners.
  3. ``tokenizer.tokenize(text)`` → list of subword token strings.
  4. Chunk into 510-token chunks (``MAX_TOKENS - 2`` for [CLS] and [EOS]).
  5. For each chunk: prepend [CLS], append [EOS], convert to ids, embed,
     extract ``last_hidden_state[:, 0, :]`` (the [CLS] hidden state).
  6. SUM the [CLS] embeddings across chunks (NOT mean).
  7. Return ``float32`` numpy array.

Any deviation from this pipeline puts deployment-time reach values on a
different scale than the calibration constants in
``agents/adaptivesearch/agent.py::PER_TASK_MAX_REACH``, breaking the
sub-rule A threshold.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

EMBED_DIM = 768
MAX_TOKENS = 512


class GraphCodeBERTEmbedder:
    def __init__(
        self,
        model_id: str = "microsoft/graphcodebert-base",
        device: str = "cpu",
        max_len: int = MAX_TOKENS,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.max_len = max_len

    def embed_files(self, files: Dict[str, str]) -> np.ndarray:
        """Embed a {filepath: content} snapshot using the chunk + sum pipeline.

        Mirrors ``compute_exploration_diversity_fullcode.embed_fullcode_cls_sum``
        composed with the offline ``get_code`` concatenator (sorted keys,
        ``"\n"`` separators, no banners).
        """
        torch = self._torch

        # -- 1. Concatenate in sorted-key order, no banners (matches offline get_code).
        code_parts = [files[k] for k in sorted(files.keys())]
        text = "\n".join(code_parts)

        if not text or not text.strip():
            return np.zeros(EMBED_DIM, dtype=np.float32)

        # -- 2. Tokenize and chunk.
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) == 0:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        chunk_size = self.max_len - 2  # leave room for [CLS] and [EOS]
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

        # -- 3. Embed each chunk's [CLS], collect.
        cls_token = self.tokenizer.cls_token
        eos_token = self.tokenizer.eos_token or self.tokenizer.sep_token
        chunk_embs = []
        for chunk in chunks:
            input_tokens = [cls_token] + chunk + [eos_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            ids_tensor = torch.tensor([token_ids]).to(self.device)
            with torch.no_grad():
                outputs = self.model(ids_tensor)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, EMBED_DIM)
                chunk_embs.append(cls_emb)

        # -- 4. SUM across chunks (not mean), float32 numpy.
        final = torch.stack(chunk_embs, dim=0).sum(dim=0).squeeze(0)
        return final.cpu().numpy().astype(np.float32)


def participation_ratio(H: np.ndarray) -> float:
    """Effective dimensionality of a (k, d) displacement matrix.

    PR = (Σλ)^2 / Σλ^2 of the Gram eigenvalues. Functionally equivalent to
    ``formal_results_analysis/_build_notebook_adaptive_phase2.py::participation_ratio``
    (same Gram-matrix branch, same eigenvalue filter, same formula, same
    defensive guards). Rows of H are displacements ``g_j − g_baseline``;
    works on the smaller of ``H Hᵀ`` or ``Hᵀ H`` for efficiency.
    """
    n = H.shape[0]
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0
    if n <= H.shape[1]:
        G = H @ H.T
    else:
        G = H.T @ H
    lam = np.linalg.eigvalsh(G)
    lam = np.clip(lam, 0.0, None)
    if lam.max() == 0:
        return 1.0
    lam = lam[lam > 1e-10 * lam.max()]
    s = lam.sum()
    if s == 0:
        return 1.0
    return float(s * s / (lam * lam).sum())
