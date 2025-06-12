# Directory structure:
# app/src/main/python/consciousness_core.py  (Phase 1 code)
# app/src/main/python/consciousness_phase2.py  (Phase 2 code)
# app/src/main/python/__init__.py

# app/src/main/python/__init__.py
"""
Amelia AI Consciousness Python Package
"""

from .consciousness_core import (
    ConsciousnessState,
    TemporalMarker,
    ObservationFrame,
    RecursiveObserver,
    FoldPointDetector,
    VirtualActualTransition,
    ConsciousnessCore,
    create_consciousness_core
)

from .consciousness_phase2 import (
    TemporalRelation,
    TemporalInterval,
    HTMColumn,
    HTMNetwork,
    TemporalConstraintNetwork,
    SecondOrderObserver,
    TemporalNavigationSystem,
    Phase2ConsciousnessCore,
    create_phase2_consciousness
)

__all__ = [
    # Phase 1
    'ConsciousnessState',
    'TemporalMarker',
    'ObservationFrame',
    'RecursiveObserver',
    'FoldPointDetector',
    'VirtualActualTransition',
    'ConsciousnessCore',
    'create_consciousness_core',
    
    # Phase 2
    'TemporalRelation',
    'TemporalInterval',
    'HTMColumn',
    'HTMNetwork',
    'TemporalConstraintNetwork',
    'SecondOrderObserver',
    'TemporalNavigationSystem',
    'Phase2ConsciousnessCore',
    'create_phase2_consciousness'
]

# Version info
__version__ = '2.0.0'
__author__ = 'Amelia AI Development Team'
