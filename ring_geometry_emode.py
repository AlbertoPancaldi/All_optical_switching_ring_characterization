import emodeconnection as emc

## Set simulation parameters
wavelength = 1550  # [nm] wavelength
dx, dy = 20, 15    # [nm] resolution
num_modes = 10     # number of modes
BC = 'TE'          # boundary condition     

core_width = 1500
core_height = 139   # nm
Al_ALD_height = 10  # 5-8 nm
Si_ALD_height = 30
mask_residue_height = 20 #20-30 nm
etch_residue_height = 0 # 15 in sims so far
etch_angle = 20
bend_radius = 0
clad_width = 4000
clad_height = 4500
sub_height = clad_height-dy/2 #substrate height

window_width = core_width + 2 * clad_width # [nm]
window_height = core_height + 2 * clad_height # [nm]

## Connect and initialize EMode
em = emc.EMode()

## Settings
em.settings(wavelength = wavelength,
    x_resolution = dx, y_resolution = dy,
    window_width = window_width,
    window_height = window_height,
    num_modes = num_modes, boundary_condition=BC,
    background_refractive_index = 'Air')
equation = '2.880962 + 0.350910/x^2 + (-0.055360)/x^4'
em.add_material(name='InGaP', refractive_index_equation=equation, wavelength_unit='um') 

## Draw shapes
em.shape(name='substrate',
        refractive_index='SiO2',
        width=window_width,
        height=sub_height)

em.shape(name='ALD_layer',
         refractive_index='Al2O3',
         width=window_width,
         height=Al_ALD_height)

em.shape(name='ALD_layer4-1',
        refractive_index='SiO2',
        # fill_refractive_index = 'Air',
        width=window_width,
        height=core_height+mask_residue_height+2*Al_ALD_height+Si_ALD_height,
        position=[0,sub_height+Al_ALD_height+(core_height+mask_residue_height+2*Al_ALD_height+Si_ALD_height)/2],
        etch_depth=core_height+mask_residue_height+2*Al_ALD_height+Si_ALD_height-etch_residue_height,
        mask=core_width+2*Al_ALD_height+2*Si_ALD_height, sidewall_angle=etch_angle)

em.shape(name='ALD_layer4-2',
         refractive_index='SiO2',
         width=window_width,
         height=Si_ALD_height,
         position=[0,sub_height+Al_ALD_height+Si_ALD_height/2])

em.shape(name='ALD_layer3',
        refractive_index='Al2O3',
        width=window_width,
        height=core_height+mask_residue_height+2*Al_ALD_height,
        position=[0,sub_height+Al_ALD_height+(core_height+mask_residue_height+2*Al_ALD_height)/2],
        etch_depth=core_height+mask_residue_height+2*Al_ALD_height-etch_residue_height,
        mask=core_width+2*Al_ALD_height, sidewall_angle=etch_angle)

em.shape(name='Mask_residue',
        refractive_index='SiO2',
        width=window_width,
        height=core_height+mask_residue_height+Al_ALD_height,
        position=[0,sub_height+Al_ALD_height+(core_height+mask_residue_height+Al_ALD_height)/2],
        etch_depth=core_height+mask_residue_height+Al_ALD_height-etch_residue_height,
        mask=core_width, sidewall_angle=etch_angle)

em.shape(name='ALD_layer2',
        refractive_index='Al2O3',
        width=window_width,
        height=core_height+Al_ALD_height,
        position=[0,sub_height+Al_ALD_height+(core_height+Al_ALD_height)/2],
        etch_depth=core_height+Al_ALD_height-etch_residue_height,
        mask=core_width, sidewall_angle=etch_angle)

em.shape(name='core',
        refractive_index='InGaP',
        width=window_width,
        height=core_height,
        position=[0,sub_height+Al_ALD_height+core_height/2],
        etch_depth=core_height-etch_residue_height,
        mask=core_width, sidewall_angle=etch_angle)

## Launch FDM solver
em.FDM()

## Display the effective indices, TE fractions, and core confinement
em.report()

## Plot the field and refractive index profiles
em.plot()

## Close EMode
em.close()