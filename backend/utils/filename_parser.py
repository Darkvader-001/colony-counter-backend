"""
Filename parser updated for real dataset format:
Example: IMG_7708ecoli_T4_10^-5__300.JPG
Extracts species and colony count from filename.
"""
import re
import os

SPECIES_MAP = {
    "ecoli": "E. coli",
    "saureus": "S. aureus",
    "saureas": "S. aureus",
    "paeruginosa": "P. aeruginosa",
    "paerug": "P. aeruginosa",
    "ec": "E. coli",
    "sa": "S. aureus",
    "pa": "P. aeruginosa",
}

def parse_filename(filename):
    """
    Parse filenames like: IMG_7708ecoli_T4_10^-5__300.JPG
    Extracts species and ground truth count from the filename.
    Returns dict or None if parsing fails.
    """
    base = os.path.splitext(os.path.basename(filename))[0].lower()

    # --- Extract colony count ---
    # Count is the last number in the filename
    count_match = re.findall(r'(\d+)(?=[^0-9]*$)', base)
    if not count_match:
        print(f"[WARN] Could not extract count from: {filename}")
        return None
    ground_truth_count = int(count_match[-1])

    # --- Extract species ---
    species_name = "Unknown"
    for key, name in SPECIES_MAP.items():
        if key in base:
            species_name = name
            break

    return {
        "image_id": base,
        "species_name": species_name,
        "ground_truth_count": ground_truth_count,
        "raw_filename": filename
    }