import re
import os

def parse_orbital_file(filepath: str) -> str:
    """
    Parses an orbital file to extract the basis set definition string (e.g., "2s2p1d").
    
    The function looks for lines in the format:
    "Number of [OrbitalType]orbital--> [Count]"
    
    Args:
        filepath (str): The path to the orbital file.
        
    Returns:
        str: The basis string representing the orbital counts (e.g., "2s2p1d").
        
    Raises:
        ValueError: If the file content doesn't match the expected format or no orbitals are found.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Orbital file not found: {filepath}")
        
    with open(filepath, 'r') as f:
        content = f.read()
        
    basis_parts = []
    
    # Define the order of orbitals to check
    orbital_types = ['s', 'p', 'd', 'f', 'g', 'h']
    
    found_any = False
    
    for orb_type in orbital_types:
        # Regex to find "Number of Sorbital--> 2" (case insensitive for S/P/D/F part of "Sorbital")
        # The file example shows "Sorbital", "Porbital", "Dorbital"
        pattern = re.compile(rf"Number of {orb_type}orbital-->\s+(\d+)", re.IGNORECASE)
        match = pattern.search(content)
        
        if match:
            count = int(match.group(1))
            if count > 0:
                basis_parts.append(f"{count}{orb_type}")
                found_any = True
                
    if not found_any:
        # Fallback or error if no known orbital pattern is found
        # It's possible the file format is completely different, but per user request, we targeting this specific format.
        raise ValueError(f"No valid orbital counts found in {filepath}. Expected lines like 'Number of Sorbital--> 2'.")
        
    return "".join(basis_parts)
