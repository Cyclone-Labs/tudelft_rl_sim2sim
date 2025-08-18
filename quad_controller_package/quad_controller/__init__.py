"""
Portable Quadcopter Controller Package

A trained neural network controller for quadcopter racing using domain randomization.
This controller generalizes across different quadcopter platforms (3-inch and 5-inch drones).

Usage:
    from quad_controller import QuadController
    
    controller = QuadController()
    action = controller.predict(observation)
"""

from .controller import QuadController

__version__ = "1.0.0"
__author__ = "Domain Randomization Quadcopter Racing"
__all__ = ["QuadController"]
