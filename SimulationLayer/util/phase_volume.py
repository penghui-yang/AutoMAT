import os
import numpy as np
import tc_python
from tc_python import *


def phase_volume_compute_with_scheil(composition, dependent_element, database="TCHEA7", 
                                     modify_flag=False, temperature=3000+273.15):
    with TCPython() as session:
        # Build the system
        system = (session.
                set_cache_folder(os.path.basename(__file__) + "_cache").
                select_database_and_elements(database, [dependent_element] + list(composition.keys())).
                get_system())
        
        # Set component and temperature
        scheil_calculation = (system.with_scheil_calculation().set_composition_unit(CompositionUnit.MOLE_PERCENT))
        for element in composition.keys():
            scheil_calculation = scheil_calculation.set_composition(element, composition[element])
        scheil_calculation = scheil_calculation.set_start_temperature(temperature)

        # Calculate
        solidification = scheil_calculation.calculate()

        # Get the scheil curve
        scheil_curve = solidification.get_values_grouped_by_stable_phases_of(
            ScheilQuantity.mole_fraction_of_all_solid_phases(),
            ScheilQuantity.temperature())
        
        # Process the curve and get phase volumes
        no_error, phase_volumes = solve_phase_volume(scheil_curve)

    if no_error == 0:
        # The sum of phase volumes is always not 1.
        # We need to put all of the rest phase volumes into the largest phase volume.
        max_phase = max(phase_volumes, key=lambda i: phase_volumes[i])
        volume_sum = .0
        for phase in phase_volumes:
            volume_sum += phase_volumes[phase]
        phase_volumes[max_phase] += 1 - volume_sum

        # modify flag means filtering out the phases with volumes < 0.02
        if modify_flag:
            redundance = .0
            new_phase_volumes = {}
            for phase in phase_volumes:
                if phase_volumes[phase] < 0.01:
                    redundance += phase_volumes[phase]
                else:
                    new_phase_volumes[phase] = phase_volumes[phase]
            new_phase_volumes[max_phase] += redundance
            phase_volumes = new_phase_volumes
        return phase_volumes
    else:
        return no_error


def solve_phase_volume(scheil_curve):

    def remove_nan(data):
        arr = np.array(data)
        nan_indices = np.where(np.isnan(arr))[0]
        if len(nan_indices) != 1:
            return data
        left_arr = arr[:nan_indices[0]]
        right_arr = arr[nan_indices[0] + 1:]
        if (len(left_arr) <= 3 and max(left_arr) - min(left_arr) < 0.01) and len(right_arr) > 3:
            return right_arr.tolist()
        elif len(left_arr) > 3 and (len(right_arr) <= 3 and max(right_arr) - min(right_arr) < 0.01):
            return left_arr.tolist()
        else:
            return None

    # Get raw data in scheil curve
    data = {}
    for label in scheil_curve:
        data[label] = {}
        data_i = remove_nan(scheil_curve[label].get_x())
        if data_i is not None:
            data[label]["data"] = data_i
            data[label]["min"] = min(data[label]["data"])
            data[label]["max"] = max(data[label]["data"])
        else:
            print(scheil_curve[label].get_x())
            print("Error 4: NaN in one internal and cannot be removed")
            return 4, {}
    
    labels = list(data.keys())
    if len(labels) == 1 and labels[0] == "LIQUID":
        print("Error 4: Only one phase LIQUID")
        return 4, []
    # decompose labels (string) into label lists without any "LIQUID" phase
    decomposed_labels = []
    for i in range(len(labels)):
        decomposed_labels.append([item for item in labels[i].split(" + ") if "liquid" not in item.lower()])
    # remove duplicate label list in decomposed_labels
    decomposed_labels = [list(item_1) for item_1 in list(set(tuple(item) for item in decomposed_labels))]
    # delete empty list in decomposed_labels
    decomposed_labels = [item for item in decomposed_labels if item]

    # new_data stores the data which has been processed
    new_data = []
    for decomposed_label in decomposed_labels:
        new_data.append({})
        new_data[-1]["phases"] = decomposed_label
        new_data[-1]["data"] = []
        new_data[-1]["min"] = 1
        new_data[-1]["max"] = 0
        for label in labels:
            if set(decomposed_label) == set([item for item in label.split(" + ") if "liquid" not in item.lower()]):
                new_data[-1]["data"].extend(data[label]["data"])
                new_data[-1]["min"] = min(new_data[-1]["min"], data[label]["min"])
                new_data[-1]["max"] = max(new_data[-1]["max"], data[label]["max"])
    new_data = sorted(new_data, key=lambda item: (item['min'], item['max']))
    
    phase_volumes = {}
    previous_phase = None
    previous_phase_min = 0
    for i in range(len(decomposed_labels)):
        if i < len(decomposed_labels) - 1 and new_data[i]["max"] > new_data[i+1]["min"]:  # overlapping
            print(new_data[i]["phases"], new_data[i]["data"])
            print(new_data[i+1]["phases"], new_data[i+1]["data"])
            print("Error 1: One interval inside another interval")
            if new_data[i]["max"] > new_data[i+1]["max"]:  # one interval inside another interval
                return 1.1, {}
            else:
                return 1.2, {}
        else:
            phases = new_data[i]["phases"]
            if new_data[i]["max"] > new_data[i]["min"]:
                unseen_phases = []
                for phase in phases:
                    if phase not in phase_volumes.keys():
                        unseen_phases.append(phase)
                if len(unseen_phases) == 1:
                    phase_volumes[unseen_phases[0]] = new_data[i]["max"] - new_data[i]["min"]
                    previous_phase = unseen_phases[0]
                    previous_phase_min = new_data[i]["min"]
                elif len(unseen_phases) == 0:
                    if previous_phase is None:
                        print("Error 2: No unseen phases but new segmentation")
                        return 2, {}
                    phase_volumes[previous_phase] = new_data[i]["max"] - previous_phase_min
                else:
                    print(new_data[i]["phases"], new_data[i]["data"])
                    print(new_data[i-1]["phases"], new_data[i-1]["data"])
                    print("Error 3: More than one unseen phases")
                    return 3, {}

    return 0, phase_volumes
