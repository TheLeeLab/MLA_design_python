#Customized scirpt for MLA design.
"""
1. Calculate and generate the MLA design files
2. Calculate the system performance parameters for each design
"""

#4mm X 4mm grid on 15mm x 15mm base for event-based LFM MLA design.
#pitch is the long axis of the hexagonal lens. (to get better filling factor)


#/- imports -/
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.colors import LogNorm
import os

import multiprocessing
from joblib import Parallel, delayed

#/- functions -/ 

def get_mla_centres(pitch_mla, lens_number=3):
    """
    Generate the centre coordinates of the MLA array.
    
    Parameteres:
    pitch_mla : float
        Pitch of the MLA in (m). This is long axis of the hexagonal. 
    lens_number : int
        Number of lenses in the MLA array. Currently supports 3HEX and 7HEX arrangements.
    
    
    Returns:
    centres : np.ndarray
        Array of shape (N, 2) containing the x and y coordinates of the MLA centres. unit in (m).
    """

    # Case for 3HEX arrangement. 
    if lens_number == 3 :
        centres = np.zeros((3,2)) # 3 centres for 3HEX arrangement - 3*[x_centre, y_centre]
        # assuming centre of the array is at (0,0)
        # the coordinates is relative to the picth of mla. can be scaled or tranlated later to fit the design parameters. 

        # Calculate the lower left lens centre. (as the first lens in the array)
        centres[0, 0] = - pitch_mla * (np.sqrt(3)/4) # x_centre of lens 1 
        centres[0, 1] = - pitch_mla / 4  # y_centre of lens 1
        # Calculate the lower right lens centre. (as the second lens in the array)
        centres[1, 0] = pitch_mla * (np.sqrt(3)/4) # x_centre of lens 2
        centres[1, 1] = - pitch_mla / 4 # y_centre of lens 2
        # Calculate the upper lens centre. (as the third lens in the array)
        centres[2, 0] = 0 # x_centre of lens 3
        centres[2, 1] = pitch_mla / 2  # y_centre of lens 3
    
    # Case for 7HEX arrangement.
    if lens_number == 7:
        centres = np.zeros((7,2)) # 7 centres for 7HEX arrangement - 7*[x_centre, y_centre]
        # assuming centre of the array is at (0,0)

        # Top left lens centre (1st lens)
        centres[0, 0] = - pitch_mla * (np.sqrt(3)/4) # x_centre of lens 1
        centres[0, 1] = pitch_mla * (3/4) # y_centre of lens 1
        # Top right lens centre (2nd lens)
        centres[1, 0] = pitch_mla * (np.sqrt(3)/4) # x_centre of lens 2
        centres[1, 1] = pitch_mla * (3/4) # y_centre of lens 2
        # Middle left lens centre (3rd lens)
        centres[2, 0] = - pitch_mla * np.sqrt(3)/2 # x_centre of lens 3
        centres[2, 1] = 0 # y_centre of lens 3
        # Centre lens (4th lens)
        centres[3, 0] = 0 # x_centre of lens 4
        centres[3, 1] = 0 # y_centre of lens 4
        # Middle right lens centre (5th lens)
        centres[4, 0] = pitch_mla * np.sqrt(3)/2 # x_centre of lens 5
        centres[4, 1] = 0 # y_centre of lens 5
        # Bottom left lens centre (6th lens)
        centres[5, 0] = - pitch_mla * (np.sqrt(3)/4)
        centres[5, 1] = - pitch_mla * (3/4)
        # Bottom right lens centre (7th lens)
        centres[6, 0] = pitch_mla * (np.sqrt(3)/4)
        centres[6, 1] = - pitch_mla * (3/4) 



    return centres

def get_sag_value(R, r,mla_pitch):
    """
    Calculate the sag value for a spherical lens surface.
    Standard formula: sag = sqrt(R^2 - r^2) - sqrt(R^2- (picth_mla/2)^2)  (long axis)

    
    Parameters:
    R : float
        Radius of curvature of the lens in (m). Positive for convex surfaces.
    r : float or np.ndarray
        Radial distance from the centre of the lens in (m).
    
    Returns:
    sag : float or np.ndarray
        Sag value at the given radial distance in (m).
    """
    # Handle the case where R is very large (nearly flat surface)
    if np.abs(R) > 1e6:
        return np.zeros_like(r)
    
    # Ensure r is a numpy array for consistent operations
    r = np.asarray(r)
    
    # Check for valid domain to avoid sqrt of negative numbers
    # For a sphere: r should be <= |R|
    r_sq = r**2
    R_sq = R**2
    
    # Create mask for valid points (within the sphere)
    valid_mask = r_sq <= R_sq
    
    # Initialize sag array
    sag = np.zeros_like(r)
    
    if np.any(valid_mask):
        # Standard spherical sag formula: sag = R - sqrt(R^2 - r^2)
        # For convex surface (R > 0), this gives positive sag
        sqrt_term = np.sqrt(R_sq - r_sq[valid_mask])
        sag[valid_mask] = sqrt_term - np.sqrt(R_sq - (mla_pitch / 2)**2)  # Adjusted for the pitch_mla/2
        
        # Handle sign convention: if R is negative (concave), sag should be negative
        if R < 0:
            sag[valid_mask] = -sag[valid_mask]
    
    return sag

def create_hexagonal_mask_exact(x_grid, y_grid, center_x, center_y, radius):
    """
    Create exact hexagonal mask using geometric constraints.
    Rotated by 90 degrees (point-topped instead of flat-topped).
    
    Parameters:
    x_grid, y_grid : np.ndarray
        2D coordinate grids
    center_x, center_y : float
        Center coordinates of the hexagon. unit in (m)
    radius : float
        Radius (circumradius) of the hexagon.  In this case, it is 1/2 of the mla pitch. unit in (m)
        
    Returns:
    mask : np.ndarray
        Boolean mask for hexagonal region
    """
    # Translate to hexagon center
    x = x_grid - center_x
    y = y_grid - center_y
    
    # For a point-topped hexagon (rotated 90 degrees from flat-topped):
    # Apothem (distance from center to middle of edge)
    apothem = radius * np.sqrt(3) / 2
    
    # Six constraints for point-topped hexagon edges
    # Left and right edges
    mask = (np.abs(x) <= apothem)
    
    # Four angled edges
    # Upper left edge: x + √3*y ≤ 2*apothem
    mask &= (x + np.sqrt(3) * y <= 2 * apothem)
    # Upper right edge: -x + √3*y ≤ 2*apothem  
    mask &= (-x + np.sqrt(3) * y <= 2 * apothem)
    # Lower left edge: x - √3*y ≤ 2*apothem
    mask &= (x - np.sqrt(3) * y <= 2 * apothem)
    # Lower right edge: -x - √3*y ≤ 2*apothem
    mask &= (-x - np.sqrt(3) * y <= 2 * apothem)
    
    return mask

def get_mla_sag_grid(mla_centres, mla_pitch, curvature, grid_size, grid_pixel_size):
    """
    Calculate sag values for the MLA on a XY grid. 
    Sag values are confined to the hexagonal apertures of the MLA lenses.

    Parameters:
    mla_centres : np.ndarray
        Array of shape (N, 2) . [x_centre, y_centre] coordinates of the MLA centres.
    mla_pitch : float
        Pitch of the MLA in (m).
    curvature : float
        Radius of curvature of the MLA in (m).
    grid_size : tuple
        Size of the grid in (m) as (width, height).
    grid_pixel_size : float
        Size of each pixel in the grid in (m).

    Returns:
    sag_values : np.ndarray
        2D array of sag values for the MLA on the grid.   unit in (m).
    """
    # Create a grid of points in the XY plane

    x = np.arange(-grid_size[0]/2, grid_size[0]/2, grid_pixel_size)
    y = np.arange(-grid_size[1]/2, grid_size[1]/2, grid_pixel_size)
    x_grid, y_grid = np.meshgrid(x, y)

    # Initialize sag values array
    sag_grid = np.zeros_like(x_grid)

    # Loop through each MLA centre and calculate sag values
    for centre in mla_centres: 
        x_centre, y_centre = centre 
        
        # Create a hexagonal mask for the current MLA lens
        # Use the exact hexagonal mask function
        hex_mask = create_hexagonal_mask_exact(x_grid, y_grid, x_centre, y_centre, mla_pitch / 2)
        
        # Calculate sag values for the MLA area
        if np.any(hex_mask):
            radius_array = np.sqrt((x_grid - x_centre)**2 + (y_grid - y_centre)**2)
            current_sag = get_sag_value(R=curvature, r=radius_array[hex_mask], mla_pitch=mla_pitch)

            sag_grid[hex_mask] = np.maximum(sag_grid[hex_mask], current_sag) # Update sag values only where the mask is True 
    
    return sag_grid

def generate_mla_design(f_fourier,f_mla,pitch_mla,lens_number,plot_design=False):
    """
    Generate the MLA design based on the given parameters.
    
    Parameters:
    f_fourier : float
        Focal length of the Fourier lens in (mm).
    f_mla : float
        Focal length of the MLA in (mm).
    pitch_mla : float
        Pitch of the MLA in (mm). This is long axis of the hexagonal lens.
    lens_number : int
        Number of lenses in the MLA array. Currently supports 3HEX and 7HEX arrangements.
    plot_design : bool
        Whether to plot the MLA design for verification.
    
    Outputs:
    mla_frame : np.ndarray
        2D array containing sag values in micrometers (um) for the MLA design.
    """


    # get MLA centres. As Nx2 array [x_centre, y_centre] 
    # currently only supports 3HEX and 7HEX arrangements centered at the origin of the 'block'
    mla_centres = get_mla_centres(pitch_mla=pitch_mla, lens_number=lens_number)

    # calculate the gird size based on the size of BFP.
    bfp_diameter = 2 * NA * f_fourier / mag_objective * (n_sample / n_immersion) # diameter of the back focal plane in (m)
    
    grid_side_length = np.ceil(bfp_diameter*1e3) / 1e3 # round up to nearest (mm) for size and convert to (m)
    current_grid_size = (grid_side_length, grid_side_length)  # size of the grid in (m) as (width, height)
    
    #calculate cuvature
    R = f_mla * (n_material - 1) # curvature of the MLA in (mm). assuming plano-convex

    # calculate sag values and sample to grid coordinates.
    sag_grid = get_mla_sag_grid(mla_centres=mla_centres, mla_pitch=pitch_mla, curvature=R, 
                               grid_size=current_grid_size, grid_pixel_size=sample_grid_size)

    # Convert sag values to micrometers, as required by Powerphotonic manufacturing.
    mla_frame = sag_grid * 1e6  # convert from meters to micrometers
    
    print(f"MLA frame shape: {mla_frame.shape}")
    print(f"MLA frame dtype: {mla_frame.dtype}")
    print(f"Grid side length: {grid_side_length*1e3:.1f}mm")
    print(f"Grid pixels: {mla_frame.shape[0]} x {mla_frame.shape[1]}")

    # plot the MLA design with Hexagonal outline if required
    if plot_design:
        print("Plotting MLA design for verification")

        fig,ax = plt.subplots(figsize=(10, 10))
        
        mla_plot = ax.imshow(mla_frame, cmap='YlOrRd', origin='lower', aspect='equal', interpolation='none')

        ax.set_title(f'MLA Design: f_fourier={f_fourier*1e3:.0f}mm, f_mla={f_mla*1e3:.0f}mm, pitch_mla={pitch_mla*1e3:.1f}mm, lens_number={lens_number}HEX')
        fig.colorbar(mla_plot,label='Sag (um)')  # Add colorbar for sag values

        # Draw hexagonal outline for each lens
        for centre in mla_centres:
            x_centre, y_centre = centre
            # Convert physical coordinates to pixel coordinates
            x_pixel = (x_centre + current_grid_size[0]/2) / sample_grid_size
            y_pixel = (y_centre + current_grid_size[1]/2) / sample_grid_size

            hexagon = RegularPolygon([x_pixel, y_pixel], numVertices=6, 
                                   radius=(pitch_mla/2) / sample_grid_size, 
                                   orientation=0, edgecolor='black', facecolor='none', linewidth=2)
            ax.add_patch(hexagon)
        
        # Draw a circle for the BFP diameter
        bfp_radius = (bfp_diameter / 2) / sample_grid_size
        bfp_circle = plt.Circle((current_grid_size[0]/2 / sample_grid_size, current_grid_size[1]/2 / sample_grid_size),
                               bfp_radius, color='blue', fill=False, linestyle='--', linewidth=2)
        ax.add_patch(bfp_circle)

        
        
        ax.set_xlabel('X pixels')
        ax.set_ylabel('Y pixels')
        ax.axis('equal')
        plt.grid()
        plt.show()
    
    return mla_frame


def save_mla_for_zemax(mla_frame, filename, grid_pixel_size, grid_side_length, description=""):
    """
    Save MLA design data in formats suitable for Zemax analysis.
    
    Parameters:
    mla_frame : np.ndarray
        2D array containing sag values in micrometers
    filename : str
        Base filename (without extension)
    grid_pixel_size : float
        Physical size of each pixel in meters
    grid_side_length : float
        Physical side length of the grid in meters
    description : str
        Description of the MLA design
    """
    
    # Create output directory if it doesn't exist
    output_dir = root_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Method 1: Save as Zemax Grid Sag format (.dat)
    # Adjusted to match Zemax format.
    grid_sag_file = os.path.join(output_dir, f"{filename}_grid_sag.dat")
    
    with open(grid_sag_file, 'w') as f:
        # Write header as spefcified by Zemax [nx ny dx dy unitflag xdec ydec]
        # unitflag: 0 for mm, 1 for cm, etc. 
        f.write(f"{mla_frame.shape[1]} {mla_frame.shape[0]} {grid_pixel_size*1000:.6f} {grid_pixel_size*1000:.6f} {0:.1f} {0:.1f} {0:.1f}\n")  # nx ny dx dy unitflag xdec ydec
        
        # flatten the MLA frame to a 1D array 
        mla_frame_flat = mla_frame.flatten()
        
        # Write flattened sag values 
        for i in range(mla_frame_flat.shape[0]):
            f.write(f"{mla_frame_flat[i]/1000:.6f} {0:d} {0:d} {0:d} {0:d}\n") #adding 4 zeros for zemax format.
    
    # Method 2: Save as ASCII XYZ format (.txt)
    # This format includes explicit X, Y, Z coordinates
    xyz_file = os.path.join(output_dir, f"{filename}_xyz.txt")
    
    # Create coordinate arrays
    x = np.arange(-grid_side_length/2, grid_side_length/2, grid_pixel_size) * 1000  # Convert to mm
    y = np.arange(-grid_side_length/2, grid_side_length/2, grid_pixel_size) * 1000  # Convert to mm
    x_grid, y_grid = np.meshgrid(x, y)
    
    with open(xyz_file, 'w') as f:
        f.write(f"# MLA Design XYZ Data: {description}\n")
        f.write(f"# X(mm) Y(mm) Z(um)\n")
        
        for i in range(mla_frame.shape[0]):
            for j in range(mla_frame.shape[1]):
                f.write(f"{x_grid[i,j]:.6f} {y_grid[i,j]:.6f} {mla_frame[i,j]:.6f}\n")
    
    """# Method 3: Save as Zemax User-Defined Surface format (.ZUD)
    zud_file = os.path.join(output_dir, f"{filename}_surface.zud")
    
    with open(zud_file, 'w') as f:
        f.write("# Zemax User-Defined Surface Data\n")
        f.write(f"# MLA Design: {description}\n")
        f.write(f"# Grid: {mla_frame.shape[0]}x{mla_frame.shape[1]}, Size: {grid_side_length*1000:.3f}mm\n")
        f.write("# Format: X Y Z (mm mm um)\n")
        
        # Only save non-zero points to reduce file size
        mask = mla_frame != 0
        if np.any(mask):
            x_coords = x_grid[mask]
            y_coords = y_grid[mask]
            z_coords = mla_frame[mask]
            
            for x_val, y_val, z_val in zip(x_coords, y_coords, z_coords):
                f.write(f"{x_val:.6f} {y_val:.6f} {z_val:.6f}\n")
    
    # Method 4: Save as binary format for faster loading
    binary_file = os.path.join(output_dir, f"{filename}_binary.npz")
    np.savez_compressed(binary_file, 
                       sag_data=mla_frame,
                       grid_pixel_size=grid_pixel_size,
                       grid_side_length=grid_side_length,
                       description=description)"""
    
    print(f"Saved MLA design files for {filename}:")
    print(f"  - Grid Sag format: {grid_sag_file}")
    print(f"  - XYZ format: {xyz_file}")
    #print(f"  - User-defined surface: {zud_file}")
    #print(f"  - Binary format: {binary_file}")
    print(f"  - Grid size: {mla_frame.shape[0]} x {mla_frame.shape[1]} pixels")
    print(f"  - Physical size: {grid_side_length*1000:.3f} x {grid_side_length*1000:.3f} mm")
    print(f"  - Sag range: {np.min(mla_frame):.3f} to {np.max(mla_frame):.3f} μm")
    print()


def generate_and_save_mla_designs(mla_params):
    """
    Parallel function to generate and save MLA designs for Zemax.

    Returns:
     - mla_frame : np.ndarray
        2D array containing sag values in micrometers for the MLA design.
    """
    f_fourier, f_mla, pitch_mla, lens_number,plot_design = mla_params
    mla_frame = generate_mla_design(f_fourier=f_fourier, f_mla=f_mla, pitch_mla=pitch_mla, lens_number=lens_number,plot_design=plot_design)
    
    # Save the design
    filename = f"MLA_{lens_number}HEX_f{f_fourier*1e3:.0f}_p{pitch_mla*1e3:.0f}_parallel"
    save_mla_for_zemax(
        mla_frame=mla_frame,
        filename=filename,
        grid_pixel_size=sample_grid_size,
        grid_side_length=np.ceil(2 * NA * f_fourier / mag_objective * (n_sample / n_immersion) * 1e3) / 1e3,
        description=f"{lens_number}HEX arrangement, f_fourier={f_fourier*1e3:.0f}mm, f_mla={f_mla*1e3:.0f}mm, pitch={pitch_mla*1e3:.0f}mm"
    )

    return mla_frame  # Return the MLA frame for further processing if needed

#/- main -/
#/- system parameters -/
wavelength = 640 * 1e-9 # emission wavelength in (m)
NA = 1.49 # numerical aperture of the objective lens
mag_objective = 100 # magnification of the objective lens
f_tube = 200e-3 # focal length of the tube lens in (m)

n_immersion = 1.518 # refractive index of the immersion medium
n_sample = 1.33 # refractive index of the sample medium (1.33 for water)

n_material = 1.453 # refractive index of the MLA material (1.453 for Powerphotonics MLA)
sample_grid_size = 10e-6 # size of the sample grid in (m)

#/- calculate other global parameters -/
wave_number = 2 * np.pi / wavelength # wave number in (1/m)
f_obj = f_tube / mag_objective # focal length of the objective lens in (m)
D_bfp = 2* NA * f_obj # diameter of the back focal plane in (m) #TODO: check formula and rewrite later. 

#/- calculate MLA designs -/ 
#Specify the MLA design parameters. As (9,6) array [f_fourier,f_mla, pitch_mla, lens_number, plot_design]
mla_design_parameters = np.zeros((9, 5)) #initialize MLA design parameters array

mla_design_parameters[0] = [125e-3, 50e-3, 1.5e-3,3,True] # Array No. 1 : f_fourier - 125mm, f_mla - 50mm, pitch_mla - 1.4mm (3HEX)
mla_design_parameters[1] = [125e-3, 50e-3, 1.6e-3,3,True] # Array No. 2 : f_fourier - 125mm, f_mla - 50mm, pitch_mla - 1.5mm (3HEX)
mla_design_parameters[2] = [125e-3, 25e-3, 1.55e-3,3,True] # Array No. 3 : f_fourier - 125mm, f_mla - 50mm, pitch_mla - 1.55mm (3HEX)


mla_design_parameters[3] = [125e-3, 50e-3, 1.2e-3,7,True] # Array No. 4 : f_fourier - 125mm, f_mla - 50mm, pitch_mla - 1.2mm (7HEX)
mla_design_parameters[4] = [125e-3, 50e-3, 1.1e-3,7,True] # Array No. 5 : f_fourier - 125mm, f_mla - 50mm, pitch_mla - 1.1mm (7HEX)
mla_design_parameters[5] = [125e-3, 25e-3, 1.15e-3,7,True] # Array No. 6 :  f_fourier - 125mm, f_mla - 25mm, pitch_mla - 1.15mm (7HEX)


mla_design_parameters[6] = [125e-3, 30e-3, 1.55e-3,3,True] # Array No. 7 : f_fourier - 125mm, f_mla - 30mm, pitch_mla - 1.55mm (3HEX)
mla_design_parameters[7] = [125e-3, 30e-3, 1.15e-3,7,True] # Array No. 8 :  f_fourier - 125mm, f_mla - 30mm, pitch_mla - 1.15mm (7HEX)
mla_design_parameters[8] = [125e-3, 25e-3, 1.15e-3,7,True] # Array No. 9 :  f_fourier - 125mm, f_mla - 25mm, pitch_mla - 1.15mm (7HEX)

#other global parameters
is_plot_design = True # plot MLA design after generation. 

root_path = "D:/MLA_Powerphotonic_Files" # root path to save the MLA designs.





#/- main process -/

#Generate the MLA design_files (parallel version)

#Run the parallel generation and saving of MLA designs
process_num = 9 # run all 9 designs in parallel (needs to be changed based on the spec of the cpu)

mla_frame_list = Parallel(n_jobs=process_num)(delayed(generate_and_save_mla_designs)(params) for params in mla_design_parameters)


#TODO: Pack up the full frame generation into a function later.

#Generate the full 15mm x 15mm MLA design 
full_frame_width = np.ceil(15.0 *1e-3 / sample_grid_size) # Full frame width in pix
full_frame_height = np.ceil(15.0 *1e-3  / sample_grid_size) # Full frame height in pix

full_mla_frame = np.zeros((int(full_frame_height), int(full_frame_width)),dtype=np.float64) # Initialize full MLA frame

# Fill the full MLA frame with the generated designs as 3X3 grid (each design is 4mm x 4mm, while the full frame is 15mm x 15mm)
# Gap between designs is 1mm
# Gap to edge is 0.5mm 

# iterate over the 3x3 grid of designs
for i in range(3):
    for j in range(3):
        # Get current design from the list
        current_design_frame = mla_frame_list[i * 3 + j] 

        # Calculate offset index range of the current design in the full frame
        x_offset = int((i * (4+1) + 0.5) * 1e-3 / sample_grid_size)
        y_offset = int((j * (4+1) + 0.5) * 1e-3 / sample_grid_size)

        x_end = x_offset + current_design_frame.shape[1]
        y_end = y_offset + current_design_frame.shape[0]
        
        # Fill the full MLA frame with the current design
        full_mla_frame[y_offset:y_end, x_offset:x_end] = current_design_frame
        
        # Add number marker to the top left corner of each design
        design_number = i * 3 + j + 1  # Design numbers 1-9
        
        # Create a marker region with dots representing the design number
        marker_height = int(0.2e-3 / sample_grid_size)  # 0.2mm height for marker
        marker_width = int(0.3e-3 / sample_grid_size)   # 0.3mm width for marker
        
        # Ensure marker doesn't exceed design boundaries
        marker_height = min(marker_height, current_design_frame.shape[0] // 4)
        marker_width = min(marker_width, current_design_frame.shape[1] // 4)
        
        # Calculate marker position (top left of current design)
        marker_y_start = y_offset
        marker_y_end = y_offset + marker_height
        marker_x_start = x_offset
        marker_x_end = x_offset + marker_width
        
        # Set marker value to maximum sag value + 10% for visibility, but limit to 65μm
        max_sag = np.max(current_design_frame)
        marker_value = max_sag * 1.1 if max_sag > 0 else 10.0  # 10μm if no sag
        marker_value = min(marker_value, 65.0)  # Limit marker height to 65μm maximum
        
        # Create dots pattern for each design number
        dot_size = max(2, int(0.02e-3 / sample_grid_size))  # Minimum 2 pixels, or 0.02mm
        dot_spacing = max(5, int(0.05e-3 / sample_grid_size))  # Minimum 5 pixels, or 0.05mm
        
        # Calculate arrangement for dots based on design number
        if design_number <= 3:
            # Arrange dots horizontally for numbers 1-3
            for dot in range(design_number):
                dot_x_start = marker_x_start + dot * dot_spacing
                dot_x_end = min(dot_x_start + dot_size, marker_x_end)
                dot_y_start = marker_y_start + marker_height // 2 - dot_size // 2
                dot_y_end = min(dot_y_start + dot_size, marker_y_end)
                
                if dot_x_end <= marker_x_end and dot_y_end <= marker_y_end:
                    full_mla_frame[dot_y_start:dot_y_end, dot_x_start:dot_x_end] = marker_value
        
        elif design_number <= 6:
            # Arrange dots in 2 rows for numbers 4-6
            dots_per_row = 2 if design_number == 4 else 3 if design_number == 6 else 2
            row1_dots = (design_number + 1) // 2
            row2_dots = design_number - row1_dots
            
            # First row
            for dot in range(row1_dots):
                dot_x_start = marker_x_start + dot * dot_spacing
                dot_x_end = min(dot_x_start + dot_size, marker_x_end)
                dot_y_start = marker_y_start + marker_height // 4
                dot_y_end = min(dot_y_start + dot_size, marker_y_end)
                
                if dot_x_end <= marker_x_end and dot_y_end <= marker_y_end:
                    full_mla_frame[dot_y_start:dot_y_end, dot_x_start:dot_x_end] = marker_value
            
            # Second row
            for dot in range(row2_dots):
                dot_x_start = marker_x_start + dot * dot_spacing
                dot_x_end = min(dot_x_start + dot_size, marker_x_end)
                dot_y_start = marker_y_start + 3 * marker_height // 4 - dot_size
                dot_y_end = min(dot_y_start + dot_size, marker_y_end)
                
                if dot_x_end <= marker_x_end and dot_y_end <= marker_y_end:
                    full_mla_frame[dot_y_start:dot_y_end, dot_x_start:dot_x_end] = marker_value
        
        else:  # design_number 7-9
            # Arrange dots in 3 rows
            dots_per_row = 3
            rows = 3
            
            for row in range(rows):
                for dot in range(dots_per_row):
                    if row * dots_per_row + dot + 1 > design_number:
                        break
                    
                    dot_x_start = marker_x_start + dot * dot_spacing
                    dot_x_end = min(dot_x_start + dot_size, marker_x_end)
                    dot_y_start = marker_y_start + row * (marker_height // 3) + (marker_height // 6)
                    dot_y_end = min(dot_y_start + dot_size, marker_y_end)
                    
                    if dot_x_end <= marker_x_end and dot_y_end <= marker_y_end:
                        full_mla_frame[dot_y_start:dot_y_end, dot_x_start:dot_x_end] = marker_value
        
        print(f"Added {design_number} dots marker at position ({i},{j}) with value {marker_value:.2f}μm (limited to 65μm)")

# Visual check of the full MLA frame with markers 
if is_plot_design:
    plt.figure(figsize=(12, 12))
    plt.matshow(full_mla_frame, cmap='YlOrRd', origin='lower', aspect='equal', interpolation='none')
    plt.title('Full MLA Design (15mm x 15mm) with Design Numbers')
    plt.colorbar(label='Sag (um)')
    
    # Add text annotations for design numbers
    for i in range(3):
        for j in range(3):
            design_number = i * 3 + j + 1
            # Calculate text position in pixel coordinates
            x_text = int((i * (4+1) + 0.5 + 0.5) * 1e-3 / sample_grid_size)  # Center of marker region
            y_text = int((j * (4+1) + 0.5 + 0.1) * 1e-3 / sample_grid_size)
            
            plt.text(x_text, y_text, str(design_number), 
                    color='white', fontsize=14, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    plt.xlabel('X pixels')
    plt.ylabel('Y pixels')
    plt.grid(True, alpha=0.3)
    plt.show()

# Save the full frame design for Zemax analysis and Powerphotonic manufacturing
#Save for Zemax
save_mla_for_zemax(
    mla_frame=full_mla_frame,
    filename="Full_MLA_15mm_x_15mm",
    grid_pixel_size=sample_grid_size,
    grid_side_length=15e-3,  # Full frame size in meters
    description="Full MLA design, 15mm x 15mm, 3x3 arrangement of 4mm x 4mm lenses with 1mm gap"
)
#NOTE : Powerphotonic provides a Zemax macro to convert Zemax optic surface to the GridXYZ (as the manufacturing format).

#Save for Powerphotonic manufacturing. 
"""
Requirements from Powerphotonics:

1. Data should saved with .dat extension.
2. x,y,z values in um (micrometers)
3. The first row comprises a zero followed by X values, in ascending order. (left to right)
4. The first column comprises a zero followed by Y values, in descending order. (top to bottom)
5. Remaining of the '2D matrix' contains Z values
6. Data in decimal notation with 3 decimal places. (Z resolution is 1nm)

"""

full_mla_frame_powerphotonics = np.zeros((int(full_frame_height+1), int(full_frame_width+1)),dtype=np.float64) # Initialize full MLA frame for Powerphotonic manufacturing. Adding row and column for the XY coordinate values.

#Initialize the first row and column with zero and the X, Y values

#full_mla_frame_powerphotonics[0,0] = 0 # Top left corner


full_mla_frame_powerphotonics[0,:] = np.arange(0, (full_frame_width+1) * sample_grid_size * 1e6, sample_grid_size * 1e6)  # X coordinate values in micrometers. Acsending order. 
full_mla_frame_powerphotonics[1:,0] = np.arange(10, (full_frame_height+1) * sample_grid_size * 1e6, sample_grid_size * 1e6)[::-1]  # Y coordinate values in micrometers. Descending order.

#TODO: need to check with Powerphotonic the arrangement of the grid. 


#Fill the rest of the frame with 'full_mla_frame' values in micrometers
full_mla_frame_powerphotonics[1:, 1:] = full_mla_frame

#Vsual check of the full MLA frame. 
fig,ax = plt.subplots(figsize=(10,10))
fig_full_frame = ax.imshow(full_mla_frame_powerphotonics[1:, 1:], cmap='YlOrRd', origin='lower', aspect='equal', interpolation='none')
#plot X and Y axes 
ax.set_title('Full MLA Design for Powerphotonic Manufacturing. (AXIS not plotted)')
ax.set_xlabel('X pixels (um)')   
ax.set_ylabel('Y pixels (um)')
fig.colorbar(fig_full_frame, ax=ax, label='Sag (um)')  # Add colorbar for sag values


plt.show()





#Save the full MLA frame as .dat file for Powerphotonic manufacturing
result_dir = root_path
powerphotonic_file = os.path.join(result_dir, "Full_MLA_15mm_x_15mm_powerphotonic.dat")
# Create output directory if it doesn't exist
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
# Save the full MLA frame for Powerphotonic manufacturing
np.savetxt(powerphotonic_file, full_mla_frame_powerphotonics, fmt='%.3f', delimiter=' ', header='0', comments='')  # Save with 3 decimal places
print(f"Full MLA design saved for Powerphotonic manufacturing: {powerphotonic_file}")
# Print completion message
print("All MLA designs generated and saved successfully!")