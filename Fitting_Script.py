#This load the Source fitting class
from VSourceGeometryFitter import VSourcePositionFitting

# dirname = '/Users/kumar/research/PBH_project/CrabData/moderate2tel/BDT/Ele78-90/'

# This load the anasum file
# VSourcePositionFitting(dirname, filename)
# x = VSourcePositionFitting('/Users/kumar/research/CasA_Data/1ES1959data/', '1ES1959_mod3tel.root')
x = VSourcePositionFitting('/Users/kumar/research/CasA_Data/', 'CasA_V6_SZA_mod3.root')
# x = VSourcePositionFitting('/Users/kumar/research/CasA_Data/', 'CasA_V6_SZA_hard3tel.root')
# x = VSourcePositionFitting(dirname, 'Crab_Mod2BDT_0.5W_1.0-50.root')


#-------------------------------------
x.readEDFile()  # Read the anasum file
x.get_maps(deg=0.3) # extract maps within +/- 0.3 degree
x.fit(-0.01,-0.03, gaus=False) # Fit using gaus or king function if gaus=True, then will use Gauss2D and if gaus=False, then use King 2D function
x.plot_skymap()  # just to plot the ON map
