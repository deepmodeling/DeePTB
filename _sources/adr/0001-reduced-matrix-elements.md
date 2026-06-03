# ADR 0001: Keep Reduced Matrix Elements Until Hamiltonian Transform

## Status

Accepted

## Context

DeePTB models predict or parameterize reduced, direction-independent matrix elements before constructing full orbital-block Hamiltonian or overlap matrices. Both SK and E3 paths rely on this separation.

This design keeps model prediction separate from direction-dependent matrix expansion, which is important for clarity, equivariance, and reuse across model types.

## Decision

DeePTB should keep reduced matrix elements as the model-side representation and expand them to full Hamiltonian or overlap blocks at the Hamiltonian transform stage.

Changes that move direction-dependent expansion earlier in the pipeline should be treated as architecture changes and require explicit tests and review.

## Consequences

- Model code can focus on predicting reduced terms.
- Hamiltonian transform code remains the authority for full block construction.
- Tests should verify both reduced element conventions and expanded block behavior when this path changes.
- AI-assisted refactors must not hide expansion semantics in generic helpers.

