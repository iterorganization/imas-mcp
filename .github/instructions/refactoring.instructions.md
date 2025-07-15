---
applyTo: "**"
description: "Refactoring guidelines for this workspace - no backwards compatibility, no suffixed variants"
---

Do not maintain backwards compatibility when refactoring code
When updating a function or class, do not use specifiers such as
enhanced, simple, optimized, advanced, intelligent, smart, improved in file names or doc strings. Just update the existing code

When augmenting existing tools or functions:

- Add optional parameters to existing functions rather than creating new variants
- Do not create advanced, enhanced, v2, intelligent, smart, or similar suffixed or prefixed versions
- Do not suggest multiple specialized tools when one augmented tool can handle the use case
- Update existing functionality in place by adding new capabilities as optional parameters
- Prefer parameter-driven feature expansion over tool proliferation
