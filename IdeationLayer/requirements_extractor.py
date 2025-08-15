import re
from typing import Dict, List, Tuple

def extract_requirements(user_requirements: str) -> Dict[str, List[str]]:
    """
    Extract primary and secondary requirements from the requirements file.
    
    Args:
        requirements: user requirements
        
    Returns:
        Dict[str, List[str]]: Dictionary containing 'primary' and 'secondary' requirements
    """
    requirements = {
        'primary': [],
        'secondary': []
    }
    
    try:
        lines = user_requirements.strip().split('\n')
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for section headers
            if line.lower().startswith('primary requirements:'):
                current_section = 'primary'
                continue
            elif line.lower().startswith('secondary requirements:'):
                current_section = 'secondary'
                continue
            elif line.lower().startswith('if not all criteria'):
                # Stop processing when we hit the relaxation instruction
                break
                
            # Extract requirements (lines starting with '-')
            if line.startswith('-') and current_section:
                requirement = line[1:].strip()  # Remove the '-' and leading/trailing spaces
                requirements[current_section].append(requirement)
                
    except Exception as e:
        print(f"Error extracting requirements: {e}")
        return requirements
    
    return requirements

def split_requirements(user_requirements: str) -> Dict[str, List[str]]:
    """Split the user requirements into primary and secondary requirements."""
    # Extract requirements  
    requirements = extract_requirements(user_requirements)
    requirements_split = {}
    
    for section, reqs in requirements.items():
        requirements_split[section] = ""
        for req in reqs:
            req_str = f"- {req}\n"
            requirements_split[section] += req_str
    
    return requirements_split
