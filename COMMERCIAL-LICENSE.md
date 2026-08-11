# Omega-S Commercial License

**Copyright (C) 2024-2026 Biome Makers Inc.**  
**USPTO Patent Pending — No. 64/121,656**

---

## Evaluating it does not need a license

**If you want to run Omega-S on your own fine-tuning job to see whether the
retention gain holds, that is free, it is covered, and you do not need to ask.**
This applies inside a for-profit organization. It is written into `LICENSE` as
clause 1(d).

Run it, see what comes out, decide whether it matters to you. We would be glad
to hear the result either way, and a report that it does not reproduce on your
setup is as useful to us as one that confirms it.

What evaluation does not cover: deploying, selling or serving a model that was
trained with Omega-S. That is the line, and it is below.

---

## Who Needs a Commercial License

The default license for this repository (AGPL-3.0, see `LICENSE`) permits
free use for **non-commercial academic research and education**, plus the
evaluation exception above.

You require a **separate commercial license** if any of the following apply:

- You integrate Omega-S into a product or service that generates revenue,
  directly or indirectly.
- You use Omega-S to train, fine-tune, or regularize models that are
  deployed in a commercial production environment.
- You offer Omega-S (or a derivative) as part of a paid API, platform,
  or ML infrastructure service.
- You incorporate Omega-S into proprietary software and do not wish to
  release the source code of your modifications (as required by AGPL-3.0).

**In plain terms:** measuring is free, shipping is not. You need a commercial
license once a model trained with Omega-S leaves your evaluation environment.

---

## What a Commercial License Includes

A commercial license grants you:

1. **Right to use** Omega-S in commercial products, services, and internal
   corporate systems without the source-disclosure obligation of AGPL-3.0.

2. **Patent license** covering the Omega-S method, identified in the executed
   agreement, for the scope of use defined therein.

3. **No copyleft obligation** : you are not required to release the source
   code of your products or modifications.

4. **Technical support** : terms and scope to be defined per agreement.

5. **Attribution flexibility** : negotiable terms for how Omega-S is
   credited in your product documentation.

---

## License Tiers

Commercial licenses are available under the following indicative tiers.
Final terms are subject to negotiation and a signed license agreement.

| Tier | Use Case | Model |
|---|---|---|
| **Startup** | Early-stage companies (<$1M ARR) | Annual fee + revenue share |
| **Scale-up** | Growth companies ($1M-$10M ARR) | Annual fee |
| **Enterprise** | Large organizations (>$10M ARR) | Annual fee + custom terms |
| **Research Partnership** | Corporate R&D with publication rights | Project-based |
| **OEM / Embedded** | Integration into third-party products for redistribution | Per-unit or revenue share |

---

## Research Partnership Program

Organizations interested in contributing to the validation and development
of Omega-S at scale : including training runs on large models, benchmark
evaluations, or FSDP cluster experiments : may qualify for a **Research
Partnership Agreement**. Under this model:

- Access to Omega-S under commercial terms is provided at reduced or
  zero cost during the research period.
- Results may be co-published, with co-authorship negotiated based on
  contribution.
- Participating organizations receive preferential terms on subsequent
  commercial licenses.
- Private investment or equity participation in the Omega-S commercialization
  entity may be discussed as part of the partnership structure.

This program is intended for organizations with the computational resources
to run experiments at LLM scale (≥7B parameters, multi-GPU FSDP clusters)
that would advance the scientific validation of the method.

---

## How to Obtain a License

To inquire about commercial licensing or the Research Partnership Program,
contact the author:

**Email:** acedo@biomemakers.com  
**GitHub:** https://github.com/BiomeMakers/OmegaS-LLM

Please include in your inquiry:

- Organization name and size
- Intended use case
- Scale of deployment (approximate number of models, parameters, GPUs)
- Whether you are interested in a standard commercial license or the
  Research Partnership Program

---

## Frequently Asked Questions

**Q: Can I run Omega-S inside my company to see whether it helps, without a
commercial license?**  
A: Yes. That is the evaluation exception in clause 1(d) of `LICENSE`, and it
is deliberate. You need a commercial license once a model trained with
Omega-S is deployed, sold or served.

**Q: Can I publish a paper using Omega-S without a commercial license?**  
A: Yes, provided you cite the original work. This holds for commercial
organizations too when the work is an evaluation: we would rather have the
replication than the fee. Larger joint work may fit the Research Partnership
Program.

**Q: Does the AGPL-3.0 copyleft obligation apply if I run Omega-S
internally (not as a service)?**  
A: AGPL-3.0's network copyleft applies when you provide a service over a
network. Internal use within a single organization without network exposure
is governed by standard GPL-3.0 copyleft (source disclosure required only
if you distribute the binary). Evaluation runs are covered by clause 1(d)
either way.

**Q: What happens to my commercial license if the patent is granted or
rejected?**  
A: The software copyright and the AGPL-3.0/commercial dual-license
structure are independent of the patent outcome. The patent covers the
Omega-S *method*; the copyright covers the *implementation*. Both licenses
remain in force regardless of patent status.

---

*This document does not constitute a license agreement. A valid commercial
license requires a signed written agreement between the licensee and
Biome Makers Inc. (or the designated licensing entity). Jurisdiction: to be
specified in the license agreement.*
