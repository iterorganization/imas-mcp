#!/usr/bin/env python3
"""
Check if dataset_description exists in the XML source.
"""

from imas_mcp.core.xml_parser import DataDictionaryTransformer


def check_dataset_description():
    transformer = DataDictionaryTransformer()
    root = transformer._root

    # Find all IDS elements
    ids_elems = root.findall(".//IDS[@name]")
    print(f"Found {len(ids_elems)} IDS elements in XML")

    # Look for dataset-related IDS
    dataset_ids = []
    dataset_desc = None

    for elem in ids_elems:
        name = elem.get("name")
        if name and "dataset" in name:
            dataset_ids.append(name)
            print(f"Found dataset IDS: {name}")
            if name == "dataset_description":
                dataset_desc = elem

    print(f"Total dataset-related IDS: {len(dataset_ids)}")
    print(f"dataset_description element found: {dataset_desc is not None}")

    if dataset_desc is not None:
        print("dataset_description XML attributes:")
        print(f"  name: {dataset_desc.get('name')}")
        print(f"  documentation: {dataset_desc.get('documentation', 'None')[:100]}...")

    # Also check if there might be case sensitivity issues
    all_names = [elem.get("name") for elem in ids_elems if elem.get("name")]
    similar_names = [
        name
        for name in all_names
        if "description" in name.lower() or "dataset" in name.lower()
    ]
    print(f"Names containing 'description' or 'dataset': {similar_names}")


if __name__ == "__main__":
    check_dataset_description()
