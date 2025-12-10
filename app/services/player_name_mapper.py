# app/services/player_name_mapper.py
"""
Player name mapping and normalization utilities.

Handles discrepancies between ESPN player names and DARKO/stats database names.
"""

# Known name mappings: ESPN name -> DARKO/Stats name
PLAYER_NAME_MAP = {
    # Periods and initials - ESPN removes periods, DARKO keeps them
    "PJ Washington": "P.J. Washington",
    "PJ Tucker": "P.J. Tucker",
    "OG Anunoby": "O.G. Anunoby",
    "TJ McConnell": "T.J. McConnell",
    "CJ McCollum": "C.J. McCollum",
    "RJ Barrett": "R.J. Barrett",
    "AJ Green": "A.J. Green",
    "JJ Redick": "J.J. Redick",
    
    # Hyphens - ESPN keeps hyphens, DARKO might not
    "Karl-Anthony Towns": "Karl-Anthony Towns",  # Keep as-is
    "Shai Gilgeous-Alexander": "Shai Gilgeous-Alexander",  # Keep as-is
    
    # Jr. suffixes - ESPN includes them
    "Michael Porter Jr.": "Michael Porter Jr.",
    "Wendell Carter Jr.": "Wendell Carter Jr.",
    "Gary Trent Jr.": "Gary Trent Jr.",
    "Dennis Smith Jr.": "Dennis Smith Jr.",
    "Larry Nance Jr.": "Larry Nance Jr.",
    "Kevin Porter Jr.": "Kevin Porter Jr.",
    "Jaren Jackson Jr.": "Jaren Jackson Jr.",
    "Kelly Oubre Jr.": "Kelly Oubre Jr.",
    "Tim Hardaway Jr.": "Tim Hardaway Jr.",
    "Otto Porter Jr.": "Otto Porter Jr.",
}

# Reverse mapping for lookups in the other direction
REVERSE_NAME_MAP = {v: k for k, v in PLAYER_NAME_MAP.items()}


def normalize_player_name(name: str) -> str:
    """
    Normalize a player name for consistent matching.
    
    Args:
        name: Player name from any source
        
    Returns:
        Normalized name that should match across data sources
    """
    if not name:
        return ""
    
    # Check if there's a direct mapping
    if name in PLAYER_NAME_MAP:
        return PLAYER_NAME_MAP[name]
    
    if name in REVERSE_NAME_MAP:
        return REVERSE_NAME_MAP[name]
    
    # Default: assume DARKO format (with periods) is canonical
    # ESPN often strips periods from initials
    normalized = name.strip()
    
    # Add periods to 2-letter initials if missing (PJ -> P.J.)
    import re
    # Match pattern like "PJ " or "PJ" at start of name
    normalized = re.sub(r'\b([A-Z])([A-Z])\b', r'\1.\2.', normalized)
    
    # Normalize whitespace
    normalized = " ".join(normalized.split())
    
    return normalized


def find_player_match(search_name: str, available_names: list) -> str:
    """
    Find the best matching player name from a list of available names.
    
    Args:
        search_name: Name to search for
        available_names: List of available player names
        
    Returns:
        Best matching name from available_names, or search_name if no match found
    """
    if not search_name or not available_names:
        return search_name
    
    # Try exact match first
    if search_name in available_names:
        return search_name
    
    # Try mapped name
    mapped_name = PLAYER_NAME_MAP.get(search_name)
    if mapped_name and mapped_name in available_names:
        return mapped_name
    
    # Try normalized matching
    normalized_search = normalize_player_name(search_name)
    
    for available_name in available_names:
        if normalize_player_name(available_name) == normalized_search:
            return available_name
    
    # No match found, return original
    return search_name


def get_canonical_name(name: str) -> str:
    """
    Get the canonical (DARKO/stats database) version of a player name.
    
    Args:
        name: Player name from any source
        
    Returns:
        Canonical name for database lookups
    """
    # Check direct mapping
    if name in PLAYER_NAME_MAP:
        return PLAYER_NAME_MAP[name]
    
    # Already canonical
    if name in REVERSE_NAME_MAP.values():
        return name
    
    # Apply normalization
    return normalize_player_name(name)
