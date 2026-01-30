## Scoring Dimensions

Each directory is scored on {{ score_dimensions | length }} per-purpose dimensions (0.0-1.0 each), aligned with the DiscoveryRootCategory taxonomy. Score ONLY the dimensions relevant to the directory's content.

{% for dim in score_dimensions %}
**{{ dim.field }} (0.0-1.0)** - {{ dim.description }}
{% endfor %}

### Purpose-to-Score Mapping

When classifying `path_purpose`, set the corresponding dimension HIGH:

| path_purpose | Primary dimension to set high |
|--------------|-------------------------------|
{% for p in path_purposes_code %}| {{ p.value }} | score_{{ p.value }} |
{% endfor %}{% for p in path_purposes_data %}| {{ p.value }} | score_{{ p.value }} |
{% endfor %}{% for p in path_purposes_infra %}| {{ p.value }} | score_{{ p.value }} |
{% endfor %}| documentation | score_documentation |
| container | Use max of expected child dimensions |
| test_suite, configuration | Low across all (< 0.3) |
| archive, build_artifact, system | Zero or near-zero across all |
