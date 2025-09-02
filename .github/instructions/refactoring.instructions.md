---
applyTo: "**/*.py"
description: "Refactoring guidelines for this workspace - no backwards compatibility, no suffixed variants"
---

Do not maintain backwards compatibility when refactoring code
When updating a function or class, do not use specifiers such as
enhanced, simple, optimized, advanced, intelligent, smart, improved, refactor, phase\*, in file names or doc strings. Just update the existing code

when following a refactoring plan do not include phase numbers or 'refactor' in filenames, class, function, or test names. These terms are irrelevant once the refactoring is complete. implement a naming strategy based on contents rather than the generation processes that created them.

When augmenting existing tools or functions:

- Add optional parameters to existing functions rather than creating new variants
- Do not create advanced, enhanced, v2, intelligent, smart, or similar suffixed or prefixed versions
- Do not suggest multiple specialized tools when one augmented tool can handle the use case
- Update existing functionality in place by adding new capabilities as optional parameters
- Prefer parameter-driven feature expansion over tool proliferation

Do not include comments related to the refactoring in the code base. i.e. do not include comments such as `# Use importlib.resources.files() for modern resource access`
