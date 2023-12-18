import numpy as np
from math import cos, sin, radians
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

"""
Generates a custom Spectre chiral aperiodic monotile, as described in https://arxiv.org/abs/2305.17743.

Customizable parameters:
    - side_length: The length of the polygon sides.
    - round: The fraction of the side length to round the corners by.
    - clearance: Erode the polygon by this fraction of the side length to account for clearance when 3D printing.
    - Whether the polygon is hollow or solid.
    - The edge profile to use when decorating the polygon.

Caveats:
    - Be careful when setting round or clearance too high, as this can corrupt the polygon edges so they don't fit together properly. Start at 0.01 and work your way up.
    - Some edge profiles may result in self-intersecting polygons. If this happens, try using a different (simpler) profile.
"""

class Spectre():
    def __init__(self, side_length, round = 0.02, clearance = 0.01, angles = [120, 270, 120, 180, 120, 90, 240, 90, 240, 90, 120, 270, 120, 90]):
        '''
        Generates the vertices of an equilateral polygon given the side length and internal angles.
        
        Args:
            side_length (float): The length of the polygon sides.
            round (float): The fraction of the side length to round the corners by. Defaults to 0.02.
            clearance (float): Erode the polygon by this fraction of the side length to account for clearance when 3D printing. Defaults to 0.01.
            angles (list): The internal angles of the polygon in clockwise order starting from the topmost point. Defaults to the angles for base Spectre monotile (Tile(1,1)).
        '''
        # Set number of rounding and clearance units
        self.round = side_length * round
        self.clearance = side_length * clearance

        # Define the start point
        start_point = (0, 0)
        self.outer_vertices = [start_point]
        # Current point and angle
        x, y = start_point
        current_angle = 180  # Start angle going straight down
        # Calculate the vertices using the internal angles and side length
        for internal_angle in angles:
            # Calculate the external drawing angle
            drawing_angle = 180 - internal_angle
            # Update the current angle
            current_angle += drawing_angle
            current_angle %= 360  # Normalize angle
            # Calculate the new vertex
            rad = radians(current_angle)
            x += cos(rad) * side_length
            y += sin(rad) * side_length
            self.outer_vertices.append((x, y))
        # In order to round both the convex and concave vertices while adding clearance, we need to dilate, erode, and dilate again (from https://gis.stackexchange.com/questions/93213/can-i-convert-the-sharp-edges-of-a-polygon-easily-to-round-edges)
        self.polygon = Polygon(self.outer_vertices).buffer(self.round).buffer(-2*self.round - self.clearance).buffer(self.round)

        # Initializes edge profile to None
        self.profile = None
    
    def set_profile(self, profile_points, tol = 1e-4):
        '''
        Sets the edge profile to the given profile.

        Args:
            profile_points (list): List of points which form a path between (0,0) and (1,0). This will be used to replace the edges of the polygon.
            tol (float): The tolerance for checking if the ends of the profile are at (0,0) and (1,0). Defaults to 1e-4.
        '''
        # Sets the profile and inverse profile
        self.profile = profile_points
        self.inverse_profile = [self._rotate_point(point, np.radians(180), origin=(0.5, 0)) for point in reversed(profile_points)] # Rotate the profile 180 degrees about (0.5, 0) to get the inverse profile. Every other edge will receive the inverse profile. See pg. 5 of https://arxiv.org/abs/2305.17743

        # Check if the profile starts at (0,0) and ends at (1,0)
        if np.linalg.norm(np.array(profile_points[0]) - np.array((0,0))) > tol or np.linalg.norm(np.array(profile_points[-1]) - np.array((1,0))) > tol:
            self.plot_profile() # Plot the profile for debugging
            raise ValueError(f'Invalid endpoints. Profile must start at (0,0) and end at (1,0): {profile_points[0]}, {profile_points[-1]}')
        
        # Decorate the polygon with the new profile
        self._decorate()

    def plot_profile(self):
        '''
        Plots profile and inverse profile for debugging.
        '''
        plt.plot(*zip(*self.profile), label='profile')
        plt.plot(*zip(*self.inverse_profile), label='inverse profile')
        plt.legend()
        plt.axis('equal')
        plt.show();

    def _decorate(self):
        '''
        Replaces all edges with the desired profiles in an alternating fashion.
        '''
        if self.profile is None:
            raise ValueError("No edge profile set. Use set_profile() to set an edge profile before decorating.")
        new_vertices = []
        chirality = 1
        for i in range(len(self.outer_vertices)):
            # Get the start and end point for the current edge
            start_vertex = self.outer_vertices[i % len(self.outer_vertices)]
            end_vertex = self.outer_vertices[(i + 1) % len(self.outer_vertices)]
            # Generate the new vertices for the current edge
            decorated_edge = self._decorate_edge(start_vertex, end_vertex, chirality)
            new_vertices.extend(decorated_edge[:-1])
            chirality *= -1 # Flip the chirality for the next edge
        self.outer_vertices = new_vertices
        # In order to round both the convex and concave vertices while adding clearance, we need to dilate, erode, and dilate again (from https://gis.stackexchange.com/questions/93213/can-i-convert-the-sharp-edges-of-a-polygon-easily-to-round-edges)
        self.polygon = Polygon(self.outer_vertices).buffer(self.round).buffer(-2*self.round - self.clearance).buffer(self.round)
    
    def _decorate_edge(self, start_vertex, end_vertex, chirality):
        '''
        Returns a list of points along a predefined profile between two vertices.

        Args:
            start_vertex (tuple): The starting point of the path.
            end_vertex (tuple): The ending point of the path.
            chirality (int): The chirality of the curve. 1 for clockwise, -1 for counter-clockwise.

        Returns:
            transformed_profile (list): The points along the curve, including the start and end points.
        '''
        edge_length = np.linalg.norm(np.array(end_vertex) - np.array(start_vertex))
        edge_angle = np.arctan2(end_vertex[1] - start_vertex[1], end_vertex[0] - start_vertex[0])

        # Scale and rotate the profile points
        transformed_profile = []
        profile_points = self.profile if chirality == 1 else self.inverse_profile # Use the inverse profile for every other edge
        for point in profile_points:
            scaled_point = np.array(point) * edge_length
            rotated_point = self._rotate_point(scaled_point, edge_angle)
            translated_point = rotated_point + np.array(start_vertex)
            transformed_profile.append(tuple(translated_point))

        return transformed_profile
    
    def _rotate_point(self, point, angle, origin = (0, 0)):
        '''
        Rotate a point counterclockwise by a given angle around a given origin.

        Args:
            point (tuple): The point to rotate.
            angle (float): The angle to rotate by in radians.
            origin (tuple): The origin to rotate about. Defaults to (0, 0).

        Returns:
            qx, qy (tuple): The rotated point.
        '''
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
    
    def make_hollow(self, edge_width=20):
        '''
        Makes the polygon hollow.

        Args:
            edge_width (float): The width of the polygon edges. Defaults to 20.
        '''
        # Erode the polygon to get the "donut hole"
        eroded_polygon = Polygon(self.outer_vertices).buffer(-edge_width)

        # Check if the eroded polygon is valid and not empty
        if not eroded_polygon.is_valid or eroded_polygon.is_empty:
            raise ValueError('Invalid edge width')

        # In case of multiple polygons, merge them into one
        if eroded_polygon.geom_type == 'MultiPolygon':
            eroded_polygon = unary_union(eroded_polygon)

        # Subtract the "donut hole" from the original polygon to get the hollow polygon
        self.polygon = self.polygon.difference(eroded_polygon)

    def save(self, saveto):
        '''
        Saves the final polygon to SVG, handling polygons with holes.

        Args:
            saveto (str): The filename to save the SVG to.
        '''
        with open(saveto, 'w') as f:
            f.write(self.polygon._repr_svg_()) # For some reason the SVG is stored in ._repr_svg_() instead of .svg(), see https://stackoverflow.com/questions/49147707/how-can-i-convert-a-shapely-polygon-to-an-svg
        display(self.polygon) # Display the SVG in Jupyter