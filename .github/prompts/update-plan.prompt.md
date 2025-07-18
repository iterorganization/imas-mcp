```prompt
---
mode: agent
name: Update Development Plan
description: Systematically analyze and update any development plan by comparing planned features against actual implementation status
authors:
  - mcintos
tags:
  - development
  - planning
  - refactoring
---

I need to update the development plan to reflect current implementation status. Please examine Phase {PHASE_NUMBER} of {PLAN_NAME} step-by-step.

For Phase {PHASE_NUMBER}:

1. **Read the entire phase section** - including all subsections (e.g., {PHASE_NUMBER}.1, {PHASE_NUMBER}.2, etc.)

2. **Systematically examine each planned feature**:
   - Check if the planned code/functionality exists in the current codebase
   - Verify by looking at the actual implementation files
   - Compare planned vs actual implementation

3. **Remove code blocks that are FULLY IMPLEMENTED**:
   - Replace large implemented code blocks with concise summaries showing:
     - Status: ✅ **IMPLEMENTED** 
     - Brief description of what was implemented
     - Key files/classes involved
     - Main functionality covered
   - Keep the structure and purpose descriptions intact

4. **Preserve all unimplemented features**:
   - Do NOT remove code blocks for features that haven't been built yet
   - Do NOT remove planning sections, specifications, or future work
   - Keep all "suggest follow up tools", performance monitoring, or other unimplemented features

5. **Follow the refactoring guidelines**:
   - No backwards compatibility concerns when updating
   - Update existing code in place rather than creating variants
   - Be selective - remove what's actually implemented

Please be systematic and careful - examine each class, function, and feature individually before deciding whether to remove its code block.

Available phases:
- Phase 0: Testing Foundation & Performance Baseline (0.1, 0.2, 0.3, 0.4)
- Phase 1: Core Tool Optimization (1.1, 1.2, 1.3)
- Phase 2: MCP Resources Implementation (2.1, 2.2)
- Phase 3: MCP Prompts Implementation (3.1, 3.2)
- Phase 4: Performance Optimization (4.1, 4.2)
- Phase 5: Testing Strategy (5.1, 5.2)

Key files to check when updating:
- `tests/conftest.py` - Test infrastructure
- `tests/test_*.py` - Test implementations
- `imas_mcp/server.py` - MCP tool implementations
- `imas_mcp/search/` - Search functionality
- `benchmarks/` - Performance monitoring
- `scripts/` - Build and utility scripts

Usage examples:
- "Phase 0.2" and "DEVELOPMENT_PLAN_MCP_TOOLS.md" to update ASV Performance Monitoring Setup
- "Phase 1.1" and "DEVELOPMENT_PLAN_MCP_TOOLS.md" to update Enhanced search_imas Tool
- "Phase 2" and "DEVELOPMENT_PLAN_MCP_TOOLS.md" to update entire MCP Resources Implementation phase
- "Phase 3.1" and "PROJECT_ROADMAP.md" to update a different development plan
```
