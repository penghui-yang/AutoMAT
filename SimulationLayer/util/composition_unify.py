from copy import deepcopy

def composition_unify(raw_composition, format="list"):
    if format == "list":
        if isinstance(raw_composition, dict):
            composition = deepcopy(raw_composition)
            mol_sum = sum(composition.values())
            for element in composition.keys():
                composition[element] = composition[element] / mol_sum * 100
            dependent_element = max(composition, key=lambda k: composition[k])
            composition.pop(dependent_element)
            composition = {k: v for k, v in composition.items() if v != 0}
            return [dependent_element, composition]
        elif isinstance(raw_composition, list):
            composition = deepcopy(raw_composition[1])
            if raw_composition[0] in composition.keys():
                mol_sum = sum(composition.values())
                for ele in composition.keys():
                    composition[ele] = composition[ele] / mol_sum * 100
                composition.pop(raw_composition[0])
            composition = {k: v for k, v in composition.items() if v != 0}
            return [raw_composition[0], composition]
        else:
            raise TypeError(f"Invalid type for raw composition: {type(raw_composition)}")
    elif format == "dict":
        if isinstance(raw_composition, dict):
            composition = deepcopy(raw_composition)
            mol_sum = sum(composition.values())
            for element in composition.keys():
                composition[element] = composition[element] / mol_sum * 100
            return composition
        elif isinstance(raw_composition, list):
            composition = deepcopy(raw_composition[1])
            mol_sum = sum(composition.values())
            if raw_composition[0] not in composition.keys():
                composition[raw_composition[0]] = 100.0 - mol_sum
            else:
                for ele in composition.keys():
                    composition[ele] = composition[ele] / mol_sum * 100
            return composition
        else:
            raise TypeError(f"Invalid type for raw composition: {type(raw_composition)}")
    else:
        raise ValueError(f"Invalid format for returned component: {format}")
