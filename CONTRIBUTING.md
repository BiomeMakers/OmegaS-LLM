# Contributing to Omega-S

Thank you for your interest in Omega-S. Contributions from the research
community are welcome and help validate and extend the method.

## Before You Contribute

Please read the [LICENSE](LICENSE) and [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)
files carefully. By submitting a contribution (pull request, issue, or any
other form), you agree that:

1. Your contribution is made under the same AGPL-3.0 license as the project.
2. You are not introducing code that infringes third-party intellectual property.
3. For contributions that extend the core Omega-S method (new regularization
   terms, new distance metrics, new application domains), you acknowledge that
   the underlying patent covers the method and
   that your contribution does not constitute a separate patent claim.

## What We Welcome

- **Bug reports and fixes** : especially in the FSDP and distributed code.
- **New experiment scripts** : replication on new architectures or datasets.
- **Distributed validation runs** : GPU validation of new applications following
  [docs/VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md) (multi-seed, tuned
  baseline, null results welcome). This is the most useful contribution right now.
- **Documentation improvements** : clearer docstrings, usage examples.
- **Performance optimizations** : faster Hutchinson estimation, better K scheduling.
- **New application domains** : GNNs, MoE routing, quantization studies.

## What We Ask You Not to Submit

- Modifications that remove or weaken the patent notice or license headers.
- Code that silently changes the core mathematical formulation without
  documenting the change (Tr(A³) estimator, probe vector distribution, etc.).
- Large model checkpoints or proprietary datasets.

## How to Submit a Pull Request

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-description`.
3. Make your changes. Add or update tests/experiments if applicable.
4. Ensure your code runs without errors on at least one experiment script.
5. Open a pull request with a clear description of what you changed and why.

## Reporting Issues

Open a GitHub Issue with:
- Python and PyTorch version.
- A minimal reproducible example.
- The full error traceback.

## Contact

For questions beyond code contributions : commercial licensing, research
partnerships, or collaboration : contact:

**Email:** acedo@biomemakers.com  
**GitHub:** https://github.com/BiomeMakers/OmegaS-LLM
