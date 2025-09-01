"""
Flight Calculator - Calculate flight statistics and performance metrics

This module provides:
- Flight time and distance calculations
- Battery consumption estimates
- Performance analysis
- Route optimization metrics
"""

import math
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from geopy.distance import geodesic

from src.models.mission import Waypoint

@dataclass
class FlightSegment:
    """Individual flight segment between waypoints"""
    start_waypoint: Waypoint
    end_waypoint: Waypoint
    distance_meters: float
    flight_time_seconds: float
    energy_consumption_mah: float
    average_speed: float
    altitude_change: float

@dataclass
class FlightStatistics:
    """Complete flight statistics"""
    total_distance_meters: float
    total_flight_time_seconds: float
    estimated_battery_consumption_mah: float
    estimated_battery_percentage: float
    average_ground_speed: float
    max_altitude: float
    min_altitude: float
    total_altitude_gain: float
    total_altitude_loss: float
    waypoint_count: int
    segments: List[FlightSegment]

class FlightCalculator:
    """Calculate flight statistics and performance metrics"""
    
    def __init__(self):
        # Default vehicle performance parameters
        self.default_params = {
            "cruise_speed": 15.0,          # m/s
            "climb_rate": 3.0,             # m/s
            "descent_rate": 2.0,           # m/s
            "hover_power": 150.0,          # watts
            "cruise_power": 120.0,         # watts
            "climb_power": 200.0,          # watts
            "battery_capacity": 16000,      # mAh
            "battery_voltage": 14.8,        # volts
            "efficiency_factor": 0.85       # overall efficiency
        }
    
    def calculate_mission_statistics(self, waypoints: List[Waypoint], 
                                   speed: float = None,
                                   vehicle_id: str = None,
                                   vehicle_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate comprehensive mission statistics"""
        
        if len(waypoints) < 2:
            return self._empty_statistics()
        
        # Use provided parameters or defaults
        params = self.default_params.copy()
        if vehicle_params:
            params.update(vehicle_params)
        
        if speed:
            params["cruise_speed"] = speed
        
        # Calculate flight segments
        segments = []
        total_distance = 0
        total_time = 0
        total_energy = 0
        total_altitude_gain = 0
        total_altitude_loss = 0
        
        altitudes = [wp.altitude for wp in waypoints]
        max_altitude = max(altitudes)
        min_altitude = min(altitudes)
        
        for i in range(len(waypoints) - 1):
            segment = self._calculate_segment(
                waypoints[i], waypoints[i + 1], params
            )
            segments.append(segment)
            
            total_distance += segment.distance_meters
            total_time += segment.flight_time_seconds
            total_energy += segment.energy_consumption_mah
            
            if segment.altitude_change > 0:
                total_altitude_gain += segment.altitude_change
            else:
                total_altitude_loss += abs(segment.altitude_change)
        
        # Calculate battery consumption percentage
        battery_percentage = (total_energy / params["battery_capacity"]) * 100
        
        # Calculate average speed
        avg_speed = total_distance / total_time if total_time > 0 else 0
        
        statistics = FlightStatistics(
            total_distance_meters=total_distance,
            total_flight_time_seconds=total_time,
            estimated_battery_consumption_mah=total_energy,
            estimated_battery_percentage=battery_percentage,
            average_ground_speed=avg_speed,
            max_altitude=max_altitude,
            min_altitude=min_altitude,
            total_altitude_gain=total_altitude_gain,
            total_altitude_loss=total_altitude_loss,
            waypoint_count=len(waypoints),
            segments=segments
        )
        
        return self._format_statistics_output(statistics, vehicle_id)
    
    def _calculate_segment(self, start_wp: Waypoint, end_wp: Waypoint, 
                          params: Dict[str, Any]) -> FlightSegment:
        """Calculate statistics for a single flight segment"""
        
        # Calculate horizontal distance
        start_pos = (start_wp.latitude, start_wp.longitude)
        end_pos = (end_wp.latitude, end_wp.longitude)
        horizontal_distance = geodesic(start_pos, end_pos).meters
        
        # Calculate altitude change
        altitude_change = end_wp.altitude - start_wp.altitude
        
        # Calculate 3D distance
        total_distance = math.sqrt(horizontal_distance**2 + altitude_change**2)
        
        # Calculate flight time components
        horizontal_time = horizontal_distance / params["cruise_speed"]
        
        if altitude_change > 0:  # Climbing
            vertical_time = altitude_change / params["climb_rate"]
        else:  # Descending
            vertical_time = abs(altitude_change) / params["descent_rate"]
        
        # Total flight time (considering simultaneous horizontal and vertical movement)
        flight_time = max(horizontal_time, vertical_time)
        
        # Calculate average speed for this segment
        avg_speed = total_distance / flight_time if flight_time > 0 else 0
        
        # Calculate energy consumption
        energy_consumption = self._calculate_segment_energy(
            horizontal_distance, altitude_change, flight_time, params
        )
        
        return FlightSegment(
            start_waypoint=start_wp,
            end_waypoint=end_wp,
            distance_meters=total_distance,
            flight_time_seconds=flight_time,
            energy_consumption_mah=energy_consumption,
            average_speed=avg_speed,
            altitude_change=altitude_change
        )
    
    def _calculate_segment_energy(self, horizontal_distance: float, 
                                 altitude_change: float, flight_time: float,
                                 params: Dict[str, Any]) -> float:
        """Calculate energy consumption for a flight segment"""
        
        # Base cruise power consumption
        base_power = params["cruise_power"]  # watts
        
        # Additional power for climbing
        if altitude_change > 0:
            climb_power_factor = 1.5  # 50% more power for climbing
            base_power *= climb_power_factor
        elif altitude_change < 0:
            # Less power needed for descending
            base_power *= 0.8
        
        # Wind and efficiency factors
        efficiency = params["efficiency_factor"]
        effective_power = base_power / efficiency
        
        # Convert to energy consumption in mAh
        # Power (W) * Time (s) = Energy (Ws)
        # Energy (Ws) / Voltage (V) = Current (As)
        # Current (As) / 3600 = Current (Ah) * 1000 = Current (mAh)
        
        energy_ws = effective_power * flight_time
        energy_mah = (energy_ws / params["battery_voltage"]) * 1000 / 3600
        
        return energy_mah
    
    def calculate_loiter_statistics(self, center_point: Tuple[float, float], 
                                   altitude: float, loiter_time_minutes: float,
                                   radius_meters: float = 50,
                                   vehicle_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate statistics for loiter/orbit pattern"""
        
        params = self.default_params.copy()
        if vehicle_params:
            params.update(vehicle_params)
        
        # Calculate orbit circumference
        circumference = 2 * math.pi * radius_meters
        
        # Calculate number of orbits in loiter time
        orbit_time = circumference / params["cruise_speed"]
        num_orbits = (loiter_time_minutes * 60) / orbit_time
        
        # Total distance
        total_distance = num_orbits * circumference
        total_time = loiter_time_minutes * 60  # seconds
        
        # Energy consumption (assuming constant power)
        power_consumption = params["cruise_power"] * params["efficiency_factor"]
        total_energy_mah = (power_consumption * total_time / params["battery_voltage"]) * 1000 / 3600
        
        return {
            "loiter_type": "circular_orbit",
            "center_latitude": center_point[0],
            "center_longitude": center_point[1],
            "altitude": altitude,
            "radius_meters": radius_meters,
            "loiter_time_minutes": loiter_time_minutes,
            "number_of_orbits": round(num_orbits, 2),
            "total_distance_meters": total_distance,
            "estimated_battery_consumption_mah": total_energy_mah,
            "estimated_battery_percentage": (total_energy_mah / params["battery_capacity"]) * 100
        }
    
    def calculate_search_pattern_statistics(self, search_area: Dict[str, Any], 
                                          pattern_type: str = "grid",
                                          altitude: float = 100,
                                          overlap: float = 30,
                                          vehicle_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate statistics for search patterns"""
        
        params = self.default_params.copy()
        if vehicle_params:
            params.update(vehicle_params)
        
        # Estimate search area size
        if "coordinates" in search_area:
            area_size = self._calculate_polygon_area(search_area["coordinates"])
        else:
            area_size = search_area.get("area_km2", 1.0) * 1000000  # Convert to m²
        
        # Calculate search pattern efficiency
        if pattern_type == "grid":
            pattern_efficiency = 0.85  # 85% efficiency for grid patterns
            search_speed_factor = 0.8   # Slower speed for thorough search
        elif pattern_type == "spiral":
            pattern_efficiency = 0.75  # 75% efficiency for spiral patterns
            search_speed_factor = 0.7
        else:
            pattern_efficiency = 0.70  # Default efficiency
            search_speed_factor = 0.75
        
        # Estimate total distance needed
        effective_area = area_size * pattern_efficiency
        search_width = altitude * 0.5  # Assume search width proportional to altitude
        total_distance = effective_area / search_width
        
        # Calculate flight time
        search_speed = params["cruise_speed"] * search_speed_factor
        flight_time = total_distance / search_speed
        
        # Calculate energy consumption
        search_power = params["cruise_power"] * 1.1  # 10% more power for search operations
        energy_consumption = (search_power * flight_time / params["battery_voltage"]) * 1000 / 3600
        
        return {
            "search_pattern": pattern_type,
            "search_area_m2": area_size,
            "search_altitude": altitude,
            "overlap_percentage": overlap,
            "pattern_efficiency": pattern_efficiency,
            "estimated_flight_distance_meters": total_distance,
            "estimated_flight_time_minutes": flight_time / 60,
            "estimated_battery_consumption_mah": energy_consumption,
            "estimated_battery_percentage": (energy_consumption / params["battery_capacity"]) * 100,
            "search_width_meters": search_width
        }
    
    def calculate_multi_vehicle_statistics(self, vehicle_missions: Dict[str, List[Waypoint]],
                                         vehicle_params: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate combined statistics for multiple vehicles"""
        
        fleet_stats = {
            "total_vehicles": len(vehicle_missions),
            "vehicle_statistics": {},
            "fleet_totals": {
                "total_distance_meters": 0,
                "total_flight_time_seconds": 0,
                "total_battery_consumption_mah": 0,
                "max_flight_time": 0,
                "min_flight_time": float('inf')
            }
        }
        
        for vehicle_id, waypoints in vehicle_missions.items():
            # Get vehicle-specific parameters
            veh_params = vehicle_params.get(vehicle_id, {}) if vehicle_params else {}
            
            # Calculate individual vehicle statistics
            vehicle_stats = self.calculate_mission_statistics(
                waypoints, vehicle_id=vehicle_id, vehicle_params=veh_params
            )
            
            fleet_stats["vehicle_statistics"][vehicle_id] = vehicle_stats
            
            # Update fleet totals
            distance = vehicle_stats["flight_performance"]["total_distance_meters"]
            flight_time = vehicle_stats["flight_performance"]["total_flight_time_seconds"]
            battery = vehicle_stats["energy_analysis"]["estimated_battery_consumption_mah"]
            
            fleet_stats["fleet_totals"]["total_distance_meters"] += distance
            fleet_stats["fleet_totals"]["total_flight_time_seconds"] += flight_time
            fleet_stats["fleet_totals"]["total_battery_consumption_mah"] += battery
            fleet_stats["fleet_totals"]["max_flight_time"] = max(
                fleet_stats["fleet_totals"]["max_flight_time"], flight_time
            )
            fleet_stats["fleet_totals"]["min_flight_time"] = min(
                fleet_stats["fleet_totals"]["min_flight_time"], flight_time
            )
        
        # Calculate fleet averages
        if fleet_stats["total_vehicles"] > 0:
            fleet_stats["fleet_averages"] = {
                "average_distance_per_vehicle": fleet_stats["fleet_totals"]["total_distance_meters"] / fleet_stats["total_vehicles"],
                "average_flight_time_per_vehicle": fleet_stats["fleet_totals"]["total_flight_time_seconds"] / fleet_stats["total_vehicles"],
                "average_battery_per_vehicle": fleet_stats["fleet_totals"]["total_battery_consumption_mah"] / fleet_stats["total_vehicles"]
            }
        
        return fleet_stats
    
    def optimize_mission_efficiency(self, waypoints: List[Waypoint], 
                                   constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize mission for efficiency"""
        
        if not constraints:
            constraints = {}
        
        original_stats = self.calculate_mission_statistics(waypoints)
        
        # Apply optimization strategies
        optimizations_applied = []
        optimized_waypoints = waypoints.copy()
        
        # 1. Altitude optimization
        if constraints.get("optimize_altitude", True):
            optimized_waypoints = self._optimize_altitudes(optimized_waypoints)
            optimizations_applied.append("altitude_optimization")
        
        # 2. Route optimization (basic nearest neighbor)
        if constraints.get("optimize_route", True) and len(waypoints) <= 20:  # Limit for performance
            optimized_waypoints = self._optimize_route_order(optimized_waypoints)
            optimizations_applied.append("route_optimization")
        
        # 3. Speed optimization
        if constraints.get("optimize_speed", True):
            optimized_waypoints = self._optimize_speeds(optimized_waypoints, constraints)
            optimizations_applied.append("speed_optimization")
        
        # Calculate optimized statistics
        optimized_stats = self.calculate_mission_statistics(optimized_waypoints)
        
        # Calculate improvements
        improvements = self._calculate_improvements(original_stats, optimized_stats)
        
        return {
            "original_statistics": original_stats,
            "optimized_statistics": optimized_stats,
            "optimizations_applied": optimizations_applied,
            "improvements": improvements,
            "optimized_waypoints": [wp.to_dict() for wp in optimized_waypoints] if hasattr(waypoints[0], 'to_dict') else optimized_waypoints
        }
    
    def _calculate_polygon_area(self, coordinates: List[List[float]]) -> float:
        """Calculate area of polygon using shoelace formula"""
        
        if len(coordinates[0]) < 3:
            return 0
        
        coords = coordinates[0]  # Assume first ring
        n = len(coords)
        area = 0
        
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        
        area = abs(area) / 2.0
        
        # Convert from degrees to meters (approximate)
        # 1 degree ≈ 111,320 meters
        area_m2 = area * (111320 ** 2)
        
        return area_m2
    
    def _optimize_altitudes(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Optimize altitudes for energy efficiency"""
        
        optimized = []
        
        for i, wp in enumerate(waypoints):
            optimized_wp = wp
            
            # For survey missions, try to minimize altitude changes
            if i > 0 and i < len(waypoints) - 1:
                prev_alt = waypoints[i-1].altitude
                next_alt = waypoints[i+1].altitude
                
                # Use average of adjacent altitudes if reasonable
                avg_alt = (prev_alt + next_alt) / 2
                if abs(avg_alt - wp.altitude) < 20:  # Within 20m
                    optimized_wp.altitude = avg_alt
            
            optimized.append(optimized_wp)
        
        return optimized
    
    def _optimize_route_order(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Optimize waypoint order using nearest neighbor algorithm"""
        
        if len(waypoints) <= 2:
            return waypoints
        
        # Keep takeoff and landing waypoints in place
        start_wp = waypoints[0]
        end_wp = waypoints[-1]
        middle_wps = waypoints[1:-1]
        
        if not middle_wps:
            return waypoints
        
        # Apply nearest neighbor to middle waypoints
        optimized_middle = []
        remaining = middle_wps.copy()
        current = start_wp
        
        while remaining:
            nearest_wp = min(remaining, key=lambda wp: geodesic(
                (current.latitude, current.longitude),
                (wp.latitude, wp.longitude)
            ).meters)
            
            optimized_middle.append(nearest_wp)
            remaining.remove(nearest_wp)
            current = nearest_wp
        
        return [start_wp] + optimized_middle + [end_wp]
    
    def _optimize_speeds(self, waypoints: List[Waypoint], 
                        constraints: Dict[str, Any]) -> List[Waypoint]:
        """Optimize speeds for each waypoint"""
        
        optimized = []
        max_speed = constraints.get("max_speed", 20)  # m/s
        min_speed = constraints.get("min_speed", 3)   # m/s
        
        for i, wp in enumerate(waypoints):
            optimized_wp = wp
            
            # Adjust speed based on waypoint type and context
            if hasattr(wp, 'waypoint_type'):
                if wp.waypoint_type in ['takeoff', 'land']:
                    optimized_wp.speed = min_speed
                elif wp.waypoint_type == 'survey':
                    # Medium speed for survey accuracy
                    optimized_wp.speed = min(12, max_speed)
                else:
                    # Transit waypoints can use higher speed
                    optimized_wp.speed = max_speed
            
            optimized.append(optimized_wp)
        
        return optimized
    
    def _calculate_improvements(self, original: Dict[str, Any], 
                              optimized: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement metrics"""
        
        orig_perf = original["flight_performance"]
        opt_perf = optimized["flight_performance"]
        
        orig_energy = original["energy_analysis"]
        opt_energy = optimized["energy_analysis"]
        
        distance_improvement = ((orig_perf["total_distance_meters"] - opt_perf["total_distance_meters"]) / orig_perf["total_distance_meters"]) * 100
        time_improvement = ((orig_perf["total_flight_time_seconds"] - opt_perf["total_flight_time_seconds"]) / orig_perf["total_flight_time_seconds"]) * 100
        energy_improvement = ((orig_energy["estimated_battery_consumption_mah"] - opt_energy["estimated_battery_consumption_mah"]) / orig_energy["estimated_battery_consumption_mah"]) * 100
        
        return {
            "distance_reduction_percent": round(distance_improvement, 2),
            "time_reduction_percent": round(time_improvement, 2),
            "energy_savings_percent": round(energy_improvement, 2),
            "distance_savings_meters": orig_perf["total_distance_meters"] - opt_perf["total_distance_meters"],
            "time_savings_seconds": orig_perf["total_flight_time_seconds"] - opt_perf["total_flight_time_seconds"],
            "energy_savings_mah": orig_energy["estimated_battery_consumption_mah"] - opt_energy["estimated_battery_consumption_mah"]
        }
    
    def _format_statistics_output(self, stats: FlightStatistics, 
                                 vehicle_id: str = None) -> Dict[str, Any]:
        """Format statistics into structured output"""
        
        return {
            "vehicle_id": vehicle_id,
            "flight_performance": {
                "total_distance_meters": round(stats.total_distance_meters, 2),
                "total_distance_km": round(stats.total_distance_meters / 1000, 3),
                "total_flight_time_seconds": round(stats.total_flight_time_seconds, 1),
                "total_flight_time_minutes": round(stats.total_flight_time_seconds / 60, 2),
                "average_ground_speed": round(stats.average_ground_speed, 2),
                "waypoint_count": stats.waypoint_count
            },
            "altitude_profile": {
                "max_altitude": stats.max_altitude,
                "min_altitude": stats.min_altitude,
                "altitude_range": stats.max_altitude - stats.min_altitude,
                "total_altitude_gain": round(stats.total_altitude_gain, 1),
                "total_altitude_loss": round(stats.total_altitude_loss, 1)
            },
            "energy_analysis": {
                "estimated_battery_consumption_mah": round(stats.estimated_battery_consumption_mah, 1),
                "estimated_battery_percentage": round(stats.estimated_battery_percentage, 1),
                "energy_per_km": round(stats.estimated_battery_consumption_mah / (stats.total_distance_meters/1000), 1) if stats.total_distance_meters > 0 else 0,
                "estimated_remaining_battery": round(100 - stats.estimated_battery_percentage, 1)
            },
            "calculated_at": datetime.now().isoformat(),
            "segment_count": len(stats.segments)
        }
    
    def _empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics structure"""
        
        return {
            "flight_performance": {
                "total_distance_meters": 0,
                "total_distance_km": 0,
                "total_flight_time_seconds": 0,
                "total_flight_time_minutes": 0,
                "average_ground_speed": 0,
                "waypoint_count": 0
            },
            "altitude_profile": {
                "max_altitude": 0,
                "min_altitude": 0,
                "altitude_range": 0,
                "total_altitude_gain": 0,
                "total_altitude_loss": 0
            },
            "energy_analysis": {
                "estimated_battery_consumption_mah": 0,
                "estimated_battery_percentage": 0,
                "energy_per_km": 0,
                "estimated_remaining_battery": 100
            },
            "calculated_at": datetime.now().isoformat(),
            "segment_count": 0
        }