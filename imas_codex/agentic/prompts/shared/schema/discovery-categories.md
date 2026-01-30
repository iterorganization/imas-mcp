## Discovery Root Categories

The taxonomy maintains duality between forward modeling (prediction) and experimental
analysis (measurement). Categories are generic and apply across all facilities.

### Forward Modeling Domain (Prediction)

| Category | Purpose | Examples |
|----------|---------|----------|
{% for cat in discovery_categories_modeling %}| `{{ cat.value }}` | {{ cat.description }} |
{% endfor %}

### Experimental Analysis Domain (Measurement)

| Category | Purpose | Examples |
|----------|---------|----------|
{% for cat in discovery_categories_experimental %}| `{{ cat.value }}` | {{ cat.description }} |
{% endfor %}

### Shared Infrastructure

| Category | Purpose | Examples |
|----------|---------|----------|
{% for cat in discovery_categories_shared %}| `{{ cat.value }}` | {{ cat.description }} |
{% endfor %}
