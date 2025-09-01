"""
Swarm Manager Module Package - Multi-vehicle coordination and swarm operations

This package provides comprehensive swarm management capabilities:
- Swarm coordination and control
- Collision detection and avoidance
- Mission analytics and optimization
- Kafka-based formation control
- Separation maintenance via streams
- Inter-UAV coordination
- Emergent behavior detection
"""

from src.modules.swarm_manager.swarm_controller import SwarmManagerModule
from src.modules.swarm_manager.collision_detector import CollisionDetector
from src.modules.swarm_manager.analytics_engine import AnalyticsEngine

__all__ = [
    "SwarmManagerModule",
    "CollisionDetector",
    "AnalyticsEngine"
]