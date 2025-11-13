"""
Track Layouts for Visualization
Approximate track coordinates for creating performance overlays

Track coordinates are stylized representations based on:
- Known track maps and corner configurations
- Section boundaries from telemetry data
- Approximate layouts for visualization purposes
"""

import numpy as np
from typing import Dict, List, Tuple


def create_barber_layout() -> Dict[str, List[Tuple[float, float]]]:
    """
    Create Barber Motorsports Park layout

    Barber is a 2.38-mile, 17-turn road course in Leeds, Alabama
    Known for elevation changes and technical corners

    Famous sections:
    - Turn 1: Museum Corner
    - Turn 5: Charlotte's Web
    - Turn 15-17: Final complex

    Returns:
        Dictionary with section coordinates
    """
    # Create approximate Barber layout
    # Using normalized coordinates (0-100 scale)

    sections = []

    # Start/Finish straight (Section 1)
    sections.append({
        'name': 'Section 1: Start/Finish',
        'x': np.linspace(50, 75, 20).tolist(),
        'y': np.linspace(10, 10, 20).tolist(),
        'type': 'straight',
        'description': 'Main straight leading to Turn 1'
    })

    # Turn 1 - Museum Corner (Section 2)
    angle = np.linspace(0, np.pi/2, 15)
    sections.append({
        'name': 'Section 2: Turn 1 (Museum)',
        'x': (75 + 8 * np.cos(angle)).tolist(),
        'y': (10 + 8 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Uphill right-hander, critical braking zone'
    })

    # Turn 2-3 complex (Section 3)
    sections.append({
        'name': 'Section 3: Turn 2-3',
        'x': np.linspace(83, 88, 15).tolist(),
        'y': np.linspace(18, 28, 15).tolist(),
        'type': 'corner',
        'description': 'Uphill esses'
    })

    # Turn 4 (Section 4)
    angle = np.linspace(np.pi/2, np.pi, 12)
    sections.append({
        'name': 'Section 4: Turn 4',
        'x': (88 + 6 * np.cos(angle)).tolist(),
        'y': (34 + 6 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Right-hander at track summit'
    })

    # Turn 5 - Charlotte's Web (Section 5)
    sections.append({
        'name': 'Section 5: Charlotte\'s Web',
        'x': np.linspace(82, 72, 18).tolist(),
        'y': np.linspace(40, 48, 18).tolist(),
        'type': 'corner',
        'description': 'Famous downhill left sweeper'
    })

    # Turn 6-7 (Section 6)
    angle = np.linspace(0, np.pi, 15)
    sections.append({
        'name': 'Section 6: Turn 6-7',
        'x': (72 + 8 * np.cos(angle)).tolist(),
        'y': (48 + 8 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Downhill chicane complex'
    })

    # Turn 8 (Section 7)
    sections.append({
        'name': 'Section 7: Turn 8',
        'x': np.linspace(64, 56, 12).tolist(),
        'y': np.linspace(56, 62, 12).tolist(),
        'type': 'corner',
        'description': 'Tight left-hander'
    })

    # Turn 9 (Section 8)
    angle = np.linspace(-np.pi/2, 0, 12)
    sections.append({
        'name': 'Section 8: Turn 9',
        'x': (50 + 6 * np.cos(angle)).tolist(),
        'y': (62 + 6 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Right-hander'
    })

    # Turn 10 (Section 9)
    sections.append({
        'name': 'Section 9: Turn 10',
        'x': np.linspace(56, 48, 12).tolist(),
        'y': np.linspace(62, 68, 12).tolist(),
        'type': 'corner',
        'description': 'Left-hander leading to back section'
    })

    # Back straight (Section 10)
    sections.append({
        'name': 'Section 10: Back Straight',
        'x': np.linspace(48, 32, 15).tolist(),
        'y': np.linspace(68, 70, 15).tolist(),
        'type': 'straight',
        'description': 'High-speed section'
    })

    # Turn 11-12 (Section 11)
    angle = np.linspace(0, np.pi, 15)
    sections.append({
        'name': 'Section 11: Turn 11-12',
        'x': (32 + 10 * np.cos(angle)).tolist(),
        'y': (70 + 10 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Sweeping left-right complex'
    })

    # Turn 13 (Section 12)
    sections.append({
        'name': 'Section 12: Turn 13',
        'x': np.linspace(22, 18, 12).tolist(),
        'y': np.linspace(80, 72, 12).tolist(),
        'type': 'corner',
        'description': 'Downhill right-hander'
    })

    # Turn 14 (Section 13)
    angle = np.linspace(-np.pi/2, -np.pi/4, 12)
    sections.append({
        'name': 'Section 13: Turn 14',
        'x': (18 + 8 * np.cos(angle)).tolist(),
        'y': (64 + 8 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Left-hander'
    })

    # Turn 15-16-17 complex (Section 14)
    sections.append({
        'name': 'Section 14: Turn 15-16-17',
        'x': np.linspace(24, 36, 20).tolist(),
        'y': np.linspace(58, 48, 20).tolist(),
        'type': 'corner',
        'description': 'Final corner complex'
    })

    # Final approach to S/F (Section 15)
    sections.append({
        'name': 'Section 15: Approach to S/F',
        'x': np.linspace(36, 50, 15).tolist(),
        'y': np.linspace(48, 10, 15).tolist(),
        'type': 'straight',
        'description': 'Acceleration zone to finish line'
    })

    return {
        'sections': sections,
        'total_sections': len(sections),
        'track_info': {
            'name': 'Barber Motorsports Park',
            'location': 'Leeds, Alabama',
            'length': '2.38 miles',
            'turns': 17,
            'direction': 'Clockwise'
        }
    }


def create_generic_layout(num_sections: int = 15) -> Dict[str, List[Tuple[float, float]]]:
    """
    Create a generic oval/road course layout

    Args:
        num_sections: Number of sections to create

    Returns:
        Dictionary with section coordinates
    """
    sections = []

    # Create a simplified oval with some corners
    angle_step = 2 * np.pi / num_sections

    for i in range(num_sections):
        start_angle = i * angle_step
        end_angle = (i + 1) * angle_step

        # Determine if section is straight or corner
        section_type = 'corner' if i % 4 == 0 else 'straight'

        # Create section coordinates
        angles = np.linspace(start_angle, end_angle, 15)
        radius = 40 + 5 * np.sin(2 * angles)  # Variable radius

        x = (50 + radius * np.cos(angles)).tolist()
        y = (50 + radius * np.sin(angles)).tolist()

        sections.append({
            'name': f'Section {i+1}',
            'x': x,
            'y': y,
            'type': section_type,
            'description': f'{"Corner" if section_type == "corner" else "Straight"} section {i+1}'
        })

    return {
        'sections': sections,
        'total_sections': num_sections,
        'track_info': {
            'name': 'Generic Layout',
            'location': 'Unknown',
            'length': 'Variable',
            'turns': num_sections // 4,
            'direction': 'Clockwise'
        }
    }


def create_cota_layout() -> Dict[str, List[Tuple[float, float]]]:
    """
    Create Circuit of the Americas (COTA) layout

    COTA is a 3.41-mile, 20-turn road course in Austin, Texas
    Known for Turn 1 elevation change and technical infield

    Returns:
        Dictionary with section coordinates
    """
    sections = []

    # Start/Finish straight
    sections.append({
        'name': 'Section 1: Start/Finish',
        'x': np.linspace(50, 70, 20).tolist(),
        'y': np.linspace(10, 10, 20).tolist(),
        'type': 'straight',
        'description': 'Long main straight'
    })

    # Turn 1 - Dramatic uphill left
    angle = np.linspace(0, np.pi/3, 18)
    sections.append({
        'name': 'Section 2: Turn 1',
        'x': (70 + 12 * np.cos(angle)).tolist(),
        'y': (10 + 12 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Uphill left-hander'
    })

    # Turns 2-11 (technical infield)
    for i in range(9):
        angle_offset = (i * np.pi / 4)
        angle = np.linspace(angle_offset, angle_offset + np.pi/6, 12)
        radius = 8 + 2 * (i % 3)

        center_x = 75 + 15 * np.cos(i * np.pi / 5)
        center_y = 25 + 15 * np.sin(i * np.pi / 5)

        sections.append({
            'name': f'Section {i+3}: Turn {i+2}',
            'x': (center_x + radius * np.cos(angle)).tolist(),
            'y': (center_y + radius * np.sin(angle)).tolist(),
            'type': 'corner',
            'description': f'Turn {i+2}'
        })

    # Turns 12-15 (back section)
    sections.append({
        'name': 'Section 12: Turn 12-15',
        'x': np.linspace(65, 30, 20).tolist(),
        'y': np.linspace(55, 65, 20).tolist(),
        'type': 'corner',
        'description': 'Fast back section'
    })

    # Turns 16-20 (stadium section)
    angle = np.linspace(np.pi, 3*np.pi/2, 25)
    sections.append({
        'name': 'Section 13: Turn 16-20',
        'x': (30 + 20 * np.cos(angle)).tolist(),
        'y': (45 + 20 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Stadium complex leading to finish'
    })

    # Final approach
    sections.append({
        'name': 'Section 14: Approach to S/F',
        'x': np.linspace(30, 50, 15).tolist(),
        'y': np.linspace(25, 10, 15).tolist(),
        'type': 'straight',
        'description': 'Final acceleration to finish'
    })

    return {
        'sections': sections,
        'total_sections': len(sections),
        'track_info': {
            'name': 'Circuit of the Americas',
            'location': 'Austin, Texas',
            'length': '3.41 miles',
            'turns': 20,
            'direction': 'Counter-clockwise'
        }
    }


def create_sonoma_layout() -> Dict[str, List[Tuple[float, float]]]:
    """
    Create Sonoma Raceway layout

    Sonoma is a 2.52-mile, 12-turn road course in Sonoma, California
    Known for elevation changes and "Corkscrew" section

    Returns:
        Dictionary with section coordinates
    """
    sections = []

    # Start/Finish
    sections.append({
        'name': 'Section 1: Start/Finish',
        'x': np.linspace(40, 60, 15).tolist(),
        'y': np.linspace(10, 10, 15).tolist(),
        'type': 'straight',
        'description': 'Main straight'
    })

    # Turn 1-2
    angle = np.linspace(0, np.pi/2, 15)
    sections.append({
        'name': 'Section 2: Turn 1-2',
        'x': (60 + 10 * np.cos(angle)).tolist(),
        'y': (10 + 10 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Opening complex'
    })

    # Uphill sections (Turns 3-6)
    sections.append({
        'name': 'Section 3: Turn 3-6 (Uphill)',
        'x': np.linspace(70, 80, 20).tolist(),
        'y': np.linspace(20, 50, 20).tolist(),
        'type': 'corner',
        'description': 'Steep uphill esses'
    })

    # Corkscrew (Turn 7-8)
    angle = np.linspace(np.pi/2, 3*np.pi/2, 20)
    sections.append({
        'name': 'Section 4: Corkscrew',
        'x': (75 + 12 * np.cos(angle)).tolist(),
        'y': (50 + 12 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Famous downhill corkscrew'
    })

    # Turn 9-10
    sections.append({
        'name': 'Section 5: Turn 9-10',
        'x': np.linspace(63, 45, 15).tolist(),
        'y': np.linspace(50, 55, 15).tolist(),
        'type': 'corner',
        'description': 'Downhill corners'
    })

    # Turn 11 (Carousel)
    angle = np.linspace(-np.pi/2, np.pi/2, 18)
    sections.append({
        'name': 'Section 6: Turn 11 (Carousel)',
        'x': (40 + 10 * np.cos(angle)).tolist(),
        'y': (50 + 10 * np.sin(angle)).tolist(),
        'type': 'corner',
        'description': 'Long left-hander'
    })

    # Final turn and approach
    sections.append({
        'name': 'Section 7: Turn 12 to S/F',
        'x': np.linspace(30, 40, 15).tolist(),
        'y': np.linspace(60, 10, 15).tolist(),
        'type': 'corner',
        'description': 'Final corner complex'
    })

    return {
        'sections': sections,
        'total_sections': len(sections),
        'track_info': {
            'name': 'Sonoma Raceway',
            'location': 'Sonoma, California',
            'length': '2.52 miles',
            'turns': 12,
            'direction': 'Clockwise'
        }
    }


TRACK_COORDINATES = {
    'barber': create_barber_layout,
    'cota': create_cota_layout,
    'sonoma': create_sonoma_layout,
    'generic': create_generic_layout
}


def get_track_layout(track_name: str, num_sections: int = None) -> Dict:
    """
    Get track coordinates for visualization

    Args:
        track_name: Name of track ('barber', 'cota', 'sonoma', etc.)
        num_sections: Number of sections (for generic layout)

    Returns:
        Dictionary with section coordinates and track info
    """
    track_name = track_name.lower().strip()

    # Handle track name variations
    track_mapping = {
        'barber motorsports park': 'barber',
        'barber motorsports': 'barber',
        'circuit of the americas': 'cota',
        'austin': 'cota',
        'sonoma raceway': 'sonoma',
        'infineon': 'sonoma'
    }

    track_name = track_mapping.get(track_name, track_name)

    if track_name in TRACK_COORDINATES:
        if track_name == 'generic' and num_sections:
            return TRACK_COORDINATES[track_name](num_sections)
        return TRACK_COORDINATES[track_name]()
    else:
        # Default to generic layout
        return create_generic_layout(num_sections or 15)


def map_sections_to_track(section_numbers: List[int], track_name: str) -> List[int]:
    """
    Map section numbers from data to track layout sections

    Args:
        section_numbers: List of section numbers from data
        track_name: Track name

    Returns:
        List of mapped section indices
    """
    layout = get_track_layout(track_name)
    total_sections = layout['total_sections']

    # Simple mapping: distribute data sections across track sections
    if not section_numbers:
        return list(range(total_sections))

    max_section = max(section_numbers)
    scale_factor = total_sections / max_section if max_section > 0 else 1

    return [int(s * scale_factor) % total_sections for s in section_numbers]
