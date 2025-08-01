# IMAS MCP Tools: Critical Feature Implementation Plan

## Priority 1: Physics Integration Restoration

### Issue

The `physics_search()` integration is imported but not used in the new explain_tool.py

### Solution

**File: `imas_mcp/tools/explain_tool.py`**

```python
# Around line 70, in explain_concept method:
async def explain_concept(self, ...):
    # After search_results are obtained
    search_results = await self._search_service.search(concept, search_config)

    # Add physics search enhancement
    physics_context = None
    try:
        physics_context = physics_search(concept)
        logger.info(f"Physics search enhanced concept '{concept}'")
    except Exception as e:
        logger.warning(f"Physics enhancement failed for '{concept}': {e}")

    # Integrate into response.physics_context field (already exists)
```

## Priority 2: Overview Tool Question Analysis

### Issue

The new overview tool lacks question-specific analysis functionality

### Solution

**File: `imas_mcp/tools/overview_tool.py`**

```python
async def get_overview(self, query: Optional[str] = None, ...):
    # Add after basic statistics gathering
    query_results = []
    question_analysis = None

    if query:
        # Restore question analysis from old implementation
        question_analysis = {
            "query_type": "specific_question",
            "analysis_approach": "Using semantic search and domain knowledge",
        }

        try:
            search_config = SearchConfig(
                search_mode=SearchMode.SEMANTIC,
                max_results=10,
            )
            search_results = await self._search_service.search(query, search_config)

            # Convert to SearchHit objects for API consistency
            query_results = [
                SearchHit(
                    path=result.document.metadata.path_name,
                    documentation=result.document.documentation[:150],
                    score=result.score,
                    # ... other fields
                )
                for result in search_results[:5]
            ]
        except Exception as e:
            logger.warning(f"Question search failed: {e}")

    # Add to response
    overview_response.query_analysis = question_analysis
    overview_response.hits = query_results
```

## Priority 3: Cross-IDS Relationship Analysis

### Issue

Export tools missing cross-IDS relationship analysis from old implementation

### Solution

**File: `imas_mcp/tools/export_tool.py`**

```python
async def export_ids(self, ...):
    # Add after successful IDS data export
    if include_relationships and len(valid_ids) > 1:
        try:
            relationship_analysis = {}

            # Restore cross-IDS analysis logic
            for i, ids1 in enumerate(valid_ids):
                for ids2 in valid_ids[i + 1:]:
                    try:
                        # Find relationships between IDS pairs
                        search_results = await self._search_service.search(
                            query=f"{ids1} {ids2} relationships physics",
                            config=SearchConfig(
                                search_mode=SearchMode.SEMANTIC,
                                max_results=8,
                            ),
                        )

                        if search_results:
                            # Analyze physics connections
                            physics_connections = [
                                r for r in search_results
                                if r.document.metadata.physics_domain
                            ]

                            relationship_analysis[f"{ids1}_{ids2}"] = {
                                "shared_concepts": len(search_results),
                                "physics_connections": len(physics_connections),
                                "top_connections": [
                                    {
                                        "path": r.document.metadata.path_name,
                                        "relevance_score": r.score,
                                        "physics_domain": r.document.metadata.physics_domain,
                                        "context": r.document.documentation[:100],
                                    }
                                    for r in search_results[:3]
                                ],
                            }
                    except Exception as e:
                        relationship_analysis[f"{ids1}_{ids2}"] = {"error": str(e)}

            export_data["cross_relationships"] = relationship_analysis

        except Exception as e:
            logger.warning(f"Cross-relationship analysis failed: {e}")
            export_data["cross_relationships"] = {"error": str(e)}
```

## Priority 4: Conditional AI Enhancement

### Issue

The new @sample decorator lacks the conditional enhancement logic from old_ai_enhancer.py

### Solution

**New File: `imas_mcp/search/decorators/conditional_sampling.py`**

```python
"""
Conditional sampling decorator based on tool and parameter analysis.
"""

import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional
from functools import wraps

logger = logging.getLogger(__name__)

class EnhancementStrategy(Enum):
    ALWAYS = "always"
    NEVER = "never"
    CONDITIONAL = "conditional"

class ToolCategory(Enum):
    SEARCH = "search"
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    EXPORT = "export"
    OVERVIEW = "overview"

# Tool configuration mapping
TOOL_ENHANCEMENT_CONFIG = {
    "search_imas": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.SEARCH,
    },
    "explain_concept": {
        "strategy": EnhancementStrategy.ALWAYS,
        "category": ToolCategory.EXPLANATION,
    },
    "get_overview": {
        "strategy": EnhancementStrategy.ALWAYS,
        "category": ToolCategory.OVERVIEW,
    },
    "analyze_ids_structure": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.ANALYSIS,
    },
    "explore_relationships": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.ANALYSIS,
    },
    "explore_identifiers": {
        "strategy": EnhancementStrategy.NEVER,
        "category": ToolCategory.SEARCH,
    },
    "export_ids": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.EXPORT,
    },
    "export_physics_domain": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.EXPORT,
    },
}

class EnhancementDecisionEngine:
    """Engine for conditional AI enhancement decisions."""

    @staticmethod
    def should_enhance(tool_name: str, args: tuple, kwargs: dict, ctx: Any) -> bool:
        """Determine if AI enhancement should be applied."""
        if not ctx:
            return False

        config = TOOL_ENHANCEMENT_CONFIG.get(
            tool_name,
            {"strategy": EnhancementStrategy.ALWAYS, "category": ToolCategory.OVERVIEW},
        )

        strategy = config["strategy"]

        match strategy:
            case EnhancementStrategy.NEVER:
                return False
            case EnhancementStrategy.ALWAYS:
                return True
            case EnhancementStrategy.CONDITIONAL:
                return EnhancementDecisionEngine._evaluate_conditional(
                    config["category"], tool_name, args, kwargs
                )
            case _:
                return True

    @staticmethod
    def _evaluate_conditional(category: ToolCategory, tool_name: str, args: tuple, kwargs: dict) -> bool:
        """Evaluate conditional enhancement logic."""
        try:
            match category:
                case ToolCategory.SEARCH:
                    return EnhancementDecisionEngine._should_enhance_search(args, kwargs)
                case ToolCategory.ANALYSIS:
                    return EnhancementDecisionEngine._should_enhance_analysis(tool_name, args, kwargs)
                case ToolCategory.EXPORT:
                    return EnhancementDecisionEngine._should_enhance_export(tool_name, args, kwargs)
                case _:
                    return True
        except Exception as e:
            logger.warning(f"Error evaluating conditional enhancement for {tool_name}: {e}")
            return True

    @staticmethod
    def _should_enhance_search(args: tuple, kwargs: dict) -> bool:
        """Determine if search should use AI enhancement."""
        # Complex query analysis
        query = args[0] if args else kwargs.get("query", "")

        if isinstance(query, list) and len(query) > 2:
            return True

        if isinstance(query, str):
            # Boolean operators
            if any(op in query.upper() for op in ["AND", "OR", "NOT"]):
                return True
            # Long queries
            if len(query.split()) > 3:
                return True

        # High result count requests
        max_results = kwargs.get("max_results", 10)
        if max_results > 15:
            return True

        return False

    @staticmethod
    def _should_enhance_analysis(tool_name: str, args: tuple, kwargs: dict) -> bool:
        """Determine if analysis tools should use AI enhancement."""
        if tool_name == "analyze_ids_structure":
            ids_name = args[0] if args else kwargs.get("ids_name", "")
            complex_patterns = [
                "core_profiles", "equilibrium", "transport", "edge_profiles",
                "mhd", "disruption", "pellets", "wall", "ec_launchers"
            ]
            return any(pattern in ids_name.lower() for pattern in complex_patterns)

        if tool_name == "explore_relationships":
            max_depth = kwargs.get("max_depth", 2)
            if max_depth >= 3:
                return True

            relationship_type = kwargs.get("relationship_type", "all")
            if hasattr(relationship_type, 'value'):
                relationship_type = relationship_type.value
            if relationship_type in ["physics", "measurement"]:
                return True

        return False

    @staticmethod
    def _should_enhance_export(tool_name: str, args: tuple, kwargs: dict) -> bool:
        """Determine if export tools should use AI enhancement."""
        if tool_name == "export_ids":
            # Multiple IDS enable AI
            ids_list = args[0] if args else kwargs.get("ids_list", [])
            if len(ids_list) > 3:
                return True

            # Analysis with relationships
            include_relationships = kwargs.get("include_relationships", True)
            include_physics = kwargs.get("include_physics", True)
            if include_relationships and include_physics and len(ids_list) > 2:
                return True

        if tool_name == "export_physics_domain":
            analysis_depth = kwargs.get("analysis_depth", "focused")
            if analysis_depth == "comprehensive":
                return True

            include_cross_domain = kwargs.get("include_cross_domain", False)
            if include_cross_domain:
                return True

        return False

def conditional_sample(temperature: float = 0.3, max_tokens: int = 800):
    """Decorator for conditional AI enhancement."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ctx = kwargs.get("ctx")

            # Execute original function
            result = await func(*args, **kwargs)

            # Check if enhancement should be applied
            should_enhance = EnhancementDecisionEngine.should_enhance(
                func.__name__, args, kwargs, ctx
            )

            if not should_enhance:
                # Add status feedback
                if hasattr(result, 'ai_insights'):
                    result.ai_insights = {
                        "status": "AI enhancement not applied - conditions not met",
                        "enhancement_strategy": "conditional"
                    }
                return result

            # Apply existing @sample logic if conditions are met
            from .sampling import apply_sample

            if (hasattr(result, 'ai_insights')
                and 'ai_prompt' in getattr(result, 'ai_insights', {})
                and ctx is not None):

                ai_prompt = result.ai_insights['ai_prompt']
                sampling = await apply_sample(ai_prompt, ctx, temperature, max_tokens)
                result.ai_insights.update(sampling)

            return result

        return wrapper
    return decorator
```

**Update existing tools to use conditional_sample:**

```python
# In each tool file, replace @sample with @conditional_sample
from imas_mcp.search.decorators.conditional_sampling import conditional_sample

@conditional_sample(temperature=0.3, max_tokens=800)
async def search_imas(self, ...):
    # existing implementation
```

## Priority 5: Document Store Integration Fixes

### Issue

Overview tool uses mock data instead of real document store data

### Solution

**File: `imas_mcp/tools/overview_tool.py`**

```python
async def get_overview(self, ...):
    # Replace mock implementations with real data
    available_ids = self.document_store.get_available_ids()

    # Get real physics domains from sample documents
    sample_documents = []
    physics_domains = set()
    data_types = set()
    units_found = set()

    for ids_name in available_ids[:5]:
        try:
            ids_docs = self.document_store.get_documents_by_ids(ids_name)
            sample_documents.extend(ids_docs[:20])
        except Exception as e:
            logger.warning(f"Failed to get documents for {ids_name}: {e}")

    for doc in sample_documents:
        if doc.metadata.physics_domain:
            physics_domains.add(doc.metadata.physics_domain)
        if doc.metadata.data_type:
            data_types.add(doc.metadata.data_type)
        if hasattr(doc, 'units') and doc.units:
            units_found.add(str(doc.units))

    # Get real identifier summary
    try:
        identifier_summary = self.document_store.get_identifier_branching_summary()
    except Exception as e:
        logger.warning(f"Failed to get identifier summary: {e}")
        identifier_summary = {"error": "Identifier analysis unavailable"}

    # Generate real per-IDS statistics
    ids_statistics = {}
    for ids_name in available_ids:
        try:
            ids_docs = self.document_store.get_documents_by_ids(ids_name)
            identifier_count = sum(
                1 for doc in ids_docs
                if doc.raw_data.get("identifier_schema")
            )
            ids_statistics[ids_name] = {
                "path_count": len(ids_docs),
                "identifier_count": identifier_count,
                "description": f"{ids_name.replace('_', ' ').title()} IDS",
            }
        except Exception as e:
            logger.warning(f"Failed to get statistics for {ids_name}: {e}")
            ids_statistics[ids_name] = {
                "path_count": 0,
                "identifier_count": 0,
                "description": f"{ids_name.replace('_', ' ').title()} IDS",
            }
```

## Implementation Timeline

### Week 1

- [ ] Implement physics integration in explain_tool.py
- [ ] Add question analysis to overview_tool.py
- [ ] Test both changes with existing test suite

### Week 2

- [ ] Implement cross-IDS relationship analysis in export_tool.py
- [ ] Create conditional_sampling.py decorator
- [ ] Update tools to use conditional sampling

### Week 3

- [ ] Fix document store integration in overview_tool.py
- [ ] Add comprehensive testing for all changes
- [ ] Performance validation vs old implementation

### Week 4

- [ ] Documentation updates
- [ ] Final integration testing
- [ ] Deployment preparation

## Success Criteria

1. **Feature Parity**: All critical features from old implementation restored
2. **Performance**: New implementation performs as well as or better than old
3. **Test Coverage**: All existing tests pass with new implementation
4. **AI Enhancement**: Conditional enhancement works as designed
5. **Documentation**: Clear migration guide and API docs updated

## Risk Mitigation

- Keep old_tools.py as fallback during transition
- Implement feature flags for gradual migration
- Comprehensive testing at each step
- Performance monitoring throughout implementation
