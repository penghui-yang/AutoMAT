import os
import tc_python
from tc_python import *

from .phase_volume import phase_volume_compute_with_scheil


def tc_single_point(composition, dependent_element, phases=None, database="TCHEA7", 
                    property="thermal conductivity", temperature=800):
    # Set the names of properties
    property_dict = {
        "thermal conductivity": "THCD",
        "electrical resistivity": "ELRS",
        "electrical conductivity": "ELCD",
        "thermal resistivity": "THRS",
        "thermal diffusivity": "THDF",
        "mass": "BM",
        "volume": "VM",
    }

    # Calculate the phase volumes
    if phases is None:
        phase_volumes = phase_volume_compute_with_scheil(composition, dependent_element)
        if isinstance(phase_volumes, dict):
            phases = list(phase_volumes.keys())
        else:
            return -1

    with TCPython() as session:
        # Build the system
        system_builder = (session.
                        set_cache_folder(os.path.basename(__file__) + "_cache").
                        select_database_and_elements(database, [dependent_element] + list(composition.keys())).
                        without_default_phases())
        # Set phases
        for phase in phases:
            system_builder.select_phase(phase)
        # Set temperature
        calculation = (system_builder.get_system().
                    with_single_equilibrium_calculation().
                    set_condition(ThermodynamicQuantity.temperature(), temperature))
        # Set component
        for element in composition.keys():
            calculation.set_condition(ThermodynamicQuantity.mole_fraction_of_a_component(element), 
                                    composition[element] / 100.0)
        # Calculate
        calc_result = (calculation.calculate())

        # Return the corresponding properties
        if isinstance(property, list):
            return [calc_result.get_value_of(property_dict[x]) for x in property]
        else:
            return calc_result.get_value_of(property_dict[property])
