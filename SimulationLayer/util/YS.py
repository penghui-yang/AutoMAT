import os
from tc_python import *


def get_total_yield_strength(composition, dependent_element, phase_volumes, temperature, database="TCHEA7"):

    main_phase = max(phase_volumes, key=lambda k: phase_volumes[k])

    with TCPython() as session:
        # Build the system
        system = (session.set_log_level_to_info()
                .set_cache_folder(os.path.basename(__file__) + "_cache")
                .select_database_and_elements(database, [dependent_element] + list(composition.keys()))
                .without_default_phases())
        # Set phases
        for phase in phase_volumes.keys():
            system = system.select_phase(phase)
        system = system.get_system()

        # Get the Calculator of Yield Strength
        calc = system.with_property_model_calculation("Yield Strength", debug_model=False)

        # Use solid solution strengthening and grain size strengthening
        ### Advanced Mode ###
        (calc
        .set_argument("Select mode", "Advanced")
        .set_temperature(temperature)
        .set_composition_unit(CompositionUnit.MOLE_PERCENT)
        .set_argument("use_explicit_matrix", True)
        .set_argument("Matrix", main_phase)
        .set_argument("sol_str_selection", True)
        .set_argument("precip_str_selection", False)
        .set_argument("grain_str_selection", True)
        .set_argument("Grain size in mu", 100)
        )

        # Set composition
        for element in composition:
            calc.set_composition(element, composition[element])
        
        # Calculate Yield Strength
        return calc.calculate().get_value_of("Total yield strength")
