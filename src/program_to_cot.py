PROGRAM_TEMPLATES = {
    "scene":           "Look at all objects in the scene.",
    "filter_color":    "Filter to keep only {value} colored objects.",
    "filter_shape":    "Filter to keep only {value} shaped objects.",
    "filter_size":     "Filter to keep only {value} sized objects.",
    "filter_material": "Filter to keep only {value} material objects.",
    "count":           "Count the remaining objects.",
    "exist":           "Check whether any such objects exist.",
    "query_color":     "Identify the color of the object.",
    "query_shape":     "Identify the shape of the object.",
    "query_size":      "Identify the size of the object.",
    "query_material":  "Identify the material of the object.",
    "equal_color":     "Compare the colors of the two objects.",
    "equal_shape":     "Compare the shapes of the two objects.",
    "equal_size":      "Compare the sizes of the two objects.",
    "equal_material":  "Compare the materials of the two objects.",
    "equal_integer":   "Compare the two counts.",
    "greater_than":    "Check if the first count is greater than the second.",
    "less_than":       "Check if the first count is less than the second.",
    "unique":          "Identify the single matching object.",
    "relate":          "Find objects spatially related to the current object.",
    "same_color":      "Find objects with the same color.",
    "same_shape":      "Find objects with the same shape.",
    "same_size":       "Find objects with the same size.",
    "same_material":   "Find objects with the same material.",
    "union":           "Combine the two sets of objects.",
    "intersect":       "Keep only objects present in both sets.",
}


def program_to_chain_of_thought(program):
    steps = []
    for i, op in enumerate(program):
        fn = op.get("function", op.get("type", "unknown"))
        template = PROGRAM_TEMPLATES.get(fn, f"Apply operation {fn}.")
        values = op.get("value_inputs", [])
        text = template.format(value=values[0]) if values else template
        steps.append(f"Step {i + 1}: {text}")
    return " ".join(steps)


def get_program_depth(program):
    return len(program)
