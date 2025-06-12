# MLA_design_python
A specialized Python toolkit for designing and manufacturing custom microlens arrays (MLAs) for event-driven light field microscopy applications.

# Overview
This repository contains a comprehensive script for designing microlens arrays optimized for Fourier Light Field Microscopy systems. The tool generates custom hexagonal MLA patterns, calculates spherical sag profiles, and exports designs in multiple formats compatible with optical simulation software and manufacturing processes.
Currently only 3HEX and 7HEX arrangements are supported. 

# Key Features
MLA Design Capabilities
Hexagonal Arrangements: Generate 3HEX and 7HEX microlens configurations
Spherical Sag Calculation: Compute lens surface profiles using standard optical formulas
Dual Axis Support: Calculate sag values using both short-axis (underfilling) and long-axis (overfilling) approaches
Parallel Processing: Multi-threaded design generation for efficient computation

# Export Formats
Zemax Grid Sag: Compatible with Zemax optical design software (.dat)
ASCII XYZ: Point cloud format for general analysis (.txt)
Powerphotonics Manufacturing: Specialized format for direct fabrication (.dat)
