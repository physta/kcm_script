import h5py
from numpy import *
from fractions import Fraction as Fr
import numpy as np
from math import *
import cmath
from scipy.optimize import leastsq
import sys
import os
import getopt
from scipy.special import *
import argparse
from phonopy.structure.symmetry import Symmetry
from phonopy.interface.calculator import read_crystal_structure
from phonopy.structure.cells import get_primitive
from phonopy.harmonic.force_constants import similarity_transformation
from phono3py.phonon3.triplets import (get_ir_grid_points,
                                       get_grid_points_by_rotations,
                                       get_grid_address)
from phonopy.cui.phonopy_argparse import fix_deprecated_option_names


def fracval(frac):
    if frac.find('/') == -1:
        return float(frac)
    else:
        x = frac.split('/')
        return float(x[0]) / float(x[1])

def get_grid_symmetry(data):
    symmetry = data['symmetry']
    mesh = data['mesh']
    weights = data['weight']
    qpoints = data['qpoint']
    rotations = symmetry.get_pointgroup_operations()
    (ir_grid_points,
     weights_for_check,
     grid_address,
     grid_mapping_table) = get_ir_grid_points(mesh, rotations)

    np.testing.assert_array_equal(weights, weights_for_check)
    qpoints_for_check = grid_address[ir_grid_points] / mesh.astype('double')
    diff_q = qpoints - qpoints_for_check
    np.testing.assert_almost_equal(diff_q, np.rint(diff_q))

    return ir_grid_points, grid_address, grid_mapping_table

def expand(data):
    gv = data['group_velocity']
    qpoint = data['qpoint']
    frequency = data['frequency']
    cv = data['heat_capacity']
    if 'gamma_N' in data:
        g_N = data['gamma_N']
        g_U = data['gamma_U']
    if 'gamma_isotope' in data:
        g_I = data['gamma_isotope']
    symmetry = data['symmetry']
    primitive = data['cell']
    mesh = data['mesh']
    ir_grid_points = data['ir_grid_points']
    grid_address = data['grid_address']

    point_operations = symmetry.get_reciprocal_operations()
    rec_lat = np.linalg.inv(primitive.get_cell())
    rotations_cartesian = np.array(
        [similarity_transformation(rec_lat, r)
         for r in point_operations], dtype='double')

    gv_bz = np.zeros((len(grid_address),) + gv.shape[1:],
                     dtype='double', order='C')
    qpt_bz = np.zeros((len(grid_address), 3), dtype='double', order='C')
    freq_bz = np.zeros((len(grid_address), frequency.shape[1]),
                       dtype='double', order='C')
    cv_bz = np.zeros((cv.shape[0], len(grid_address), cv.shape[2]),
                     dtype='double', order='C')
    if 'gamma_N' in data:
        g_N_bz = np.zeros_like(cv_bz)
        g_U_bz = np.zeros_like(cv_bz)
    else:
        g_N_bz = None
        g_U_bz = None
    if 'gamma_isotope' in data:
        g_I_bz = np.zeros((len(grid_address), frequency.shape[1]),
                       dtype='double', order='C')
    else:
        g_I_bz = None

    num_band = gv.shape[1]
    for i, gp in enumerate(ir_grid_points):
        rotation_map = get_grid_points_by_rotations(
            grid_address[gp],
            point_operations,
            mesh)
        multi = len(rotation_map) // len(np.unique(rotation_map))
        assert len(np.unique(rotation_map)) * multi == len(rotation_map)
        
        for rgp, r_c, r in zip(rotation_map,
                               rotations_cartesian,
                               point_operations):
            gv_bz[rgp] += np.dot(gv[i], r_c.T) / multi
            qpt_bz[rgp] = np.dot(r, qpoint[i])
            freq_bz[rgp] = frequency[i]
            cv_bz[:, rgp, :] = cv[:, i, :]
            if 'gamma_N' in data:
                g_N_bz[:, rgp, :] = g_N[:, i, :]
                g_U_bz[:, rgp, :] = g_U[:, i, :]
            if 'gamma_isotope' in data:
                g_I_bz[rgp] = g_I[i]

    return gv_bz, qpt_bz, freq_bz, cv_bz, g_N_bz, g_U_bz, g_I_bz

def parse_args():
    deprecated = fix_deprecated_option_names(sys.argv)
    parser = argparse.ArgumentParser(
        description="Phono3py command-line-tool")
    parser.add_argument(
        "--pa", dest="primitive_matrix", default="1 0 0 0 1 0 0 0 1",
        help="Primitive matrix")
    parser.add_argument(
        "--qe", "--pwscf", dest="qe_mode",
        action="store_true", help="Invoke Quantum espresso (QE) mode")
    parser.add_argument('filenames', nargs='*')
    args = parser.parse_args()
    return args

def get_data(args, interface_mode=None):
    args = parse_args()
    cell, _ = read_crystal_structure(args.filenames[0],
                                     interface_mode=interface_mode)
    f = h5py.File(args.filenames[1], 'r') 
    primitive_matrix = np.reshape(
        [fracval(x) for x in args.primitive_matrix.split()], (3, 3))
    primitive = get_primitive(cell, primitive_matrix)
    symmetry = Symmetry(primitive)

    data = {}
    data['cell'] = primitive
    data['symmetry'] = symmetry
    data['mesh'] = np.array(f['mesh'][:], dtype='intc') # (3)
    data['weight'] = f['weight'][:] # (gp)
    data['group_velocity'] = f['group_velocity'][:] # (gp, band, 3)
    data['qpoint'] = f['qpoint'][:] # (gp, 3)
    data['frequency'] = f['frequency'][:] # (gp, band)
    if 'gamma_N' in f:
        data['gamma_N'] = f['gamma_N'][:] # (temps, gp, band)
        data['gamma_U'] = f['gamma_U'][:] # (temps, gp, band)
    if 'gamma_isotope' in f:
        data['gamma_isotope'] = f['gamma_isotope'][:] # (gp, band)
    data['heat_capacity'] = f['heat_capacity'][:] # (temps, gp, band)
    data['temperature'] = np.array(f['temperature'][:], dtype='double') # (temps)
    ir_grid_points, grid_address, _ = get_grid_symmetry(data)
    data['ir_grid_points'] = ir_grid_points
    data['grid_address'] = grid_address

    return data


args = parse_args()

if args.qe_mode:
        data = get_data(args, interface_mode='pwscf')
else:
        data = get_data(args)

gv_bz, qpt_bz, freq_bz, cv_bz, g_N_bz, g_U_bz, g_I_bz = expand(data)

qpoint = qpt_bz
freq = freq_bz
cv = cv_bz

f = h5py.File(args.filenames[1], 'r')

### Required parameters

if 'gamma_N' in f:
    gamma_N = g_N_bz
    gamma_U = g_U_bz
else:
    print ('\n WARNING!: To run KCM you need to split normal and umklapp processes')
    print ('           using --nu in the phono3py calculation \n')
    sys.exit()
if 'gamma_isotope' in f:
    gamma_I = g_I_bz
vel = gv_bz
gamma = f['gamma']
weight = f['weight']
T = f['temperature']
kappa = f['kappa']
gv = f['gv_by_gv']
k_conv = f['kappa_unit_conversion']
mesh = f['mesh']
k_conv = f['kappa_unit_conversion']
print ('  _    _   _______   _        _  ')
print (' | |  / / | ______| | \      / | ')
print (' | | / /  | |       |  \    /  | ')
print (' | |/ /   | |       |   \  /   | ')
print (' |   /    | |       | |\ \/ /| | ')
print (' |   \    | |       | | \__/ | | ')
print (' | |\ \   | |       | |      | | ')
print (' | | \ \  | |_____  | |      | | ')
print (' |_|  \_\ |_______| |_|      |_| \n')
print (' KINETIC  COLLECTIVE    MODEL    Version 1.2 for python3   ','\n')
print ('--------------------------------- \n')
print ('Running calculation of thermal conductivity on a ', str(mesh[(0)])+'x'+str(mesh[(1)])+'x'+str(mesh[(2)]) ,'mesh \n')


V = (1.e12*1.e-10)**2*1.602e-19/(2.*pi*(k_conv[()])*1.e12)

## Cell vectors

p_v = args.primitive_matrix.split()

a1 = array([float(Fr(p_v[0])),float(Fr(p_v[1])),float(Fr(p_v[2]))])
a2 = array([float(Fr(p_v[3])),float(Fr(p_v[4])),float(Fr(p_v[5]))])
a3 = array([float(Fr(p_v[6])),float(Fr(p_v[7])),float(Fr(p_v[8]))])

a_matrix = np.array([a1,a2,a3])

b1 = np.cross(a2,a3)/np.linalg.det(a_matrix)  #2pi/alat
b2 = np.cross(a3,a1)/np.linalg.det(a_matrix)  #2pi/alat
b3 = np.cross(a1,a2)/np.linalg.det(a_matrix)  #2pi/alat

N = mesh[0]*mesh[1]*mesh[2]
factor = (1./(V*N))  # Normalization factor

hbar = 6.62e-34/(2*pi)
kb = 1.38e-23

## Default values

file = open('INPUT','r')
list = file.readlines()
file.close()
params = []

for i in list:
        a = i.split()
        if a[0]=='TEMP=':
            params.append([])
            for j in range(len(a)-1):
                params[-1].append(a[j+1])
        else:
            if a[0]=='L=':
                params.append([])
                for j in range(len(a)-1):
                    params[-1].append(a[j+1])
            if a[0]=='TYPE=':
                params.append([])
                for j in range(len(a)-1):
                    params[-1].append(a[j+1])
            if a[0]!='TEMP=' and a[0]!='L=' and a[0]!='TYPE=':
                params.append(a[1])

TEMP = params[0]
Temp = []
if TEMP[0]=='ALL':
   for i in T:
        Temp.append(i)
else:
   for i in TEMP:
      Temp.append(float(i))

I_SF = float(params[4])
COMP = params[5]
K_W = params[6]
K_MFP = params[7]
TAU_W = params[8]
TAU_T = params[9]
STP = float(params[10])

grid = str(mesh[(0)])+str(mesh[(1)])+str(mesh[(2)])

BOUNDARY = params[1]
TYPE = params[2]
size = params[3]

def tau_value(scattering):
     t_value=(2*3.14159265*2.*1.e12*scattering)**-1.0
     return t_value


for l in range(len(size)):
 L= float(size[l])
 if TYPE[l]=='W':
       Leff=L
 if TYPE[l]=='R':
        Leff=1.12*L
 if TYPE[l]=='F':
        Leff=2.25*L
 if BOUNDARY=='Y':
        prefix = TYPE[l]+str(Leff)
 else:
    Leff = 'inf'
    prefix = 'bulk'

 print ('\n Sample size= ', prefix, '\n')
 

 if COMP=='XX':
		i1=0
		i2=0
 if COMP=='YY':
        	i1=1
	        i2=1
 if COMP=='ZZ':
	        i1=2
        	i2=2
 if COMP=='XY':
        	i1=0
	        i2=1
 if COMP=='XZ':
        	i1=0
	        i2=2
 if COMP=='YZ':
	        i1=1
        	i2=2
 
 file=open('K_T_'+prefix+'_'+COMP+'_'+grid+'.dat','w')
 file.write("%s \n\n"%('# (1)T[k]  (2)k_KCM[W/mK]  (3)NL-param[nm]  (4)k*_kin[W/mK]  (5)k*_col[W/mK] (6)sigma[adim]  (7)k_RTA[W/mK]'))

 if K_W=='Y':
        file1 = open('K_w_'+prefix+'_'+COMP+'_'+grid+'.dat','w')
        file1.write("%s \n"%('# (1)T[k]  (2)w[rad/s] (3)k*_kin[J/mK]  (4)k*_col[J/mK]  (5)k*_kin_acc[W/mK]  (6)k*_col_acc[W/mK]  (7)sigma[adim] (8)k_tot_acc[W/mK] (9)Cv_acc[J/m^3K]'))
        k_w=[]
 if K_MFP=='Y':
        file2 = open('K_mfp_'+prefix+'_'+COMP+'_'+grid+'.dat','w')
        file2.write("%s \n"%('# (1)T[k]  (2)mfp[m]  (3)k*_kin[W/mK]  (4)k*_col[W/mK]  (5)k*_kin_acc[W/mK]  (6)k*_col_acc[W/mK]  (7)sigma[W/mK]  (8)k_tot_acc[W/mK]'))
        k_mfp=[]
 if TAU_W=='Y':
        file3 = open('Taus_w_'+prefix+'_'+grid+'.dat','w')
        file3.write("%s \n"%('# (1)T[k]  (2)w[rad/s]  (3)tau_I[s]  (4)tau_U[s]  (5)tau_N[s]  (6)tau_B[s] (7)v_mode[m/s]'))
 if TAU_T=='Y':
        file4 = open('Taus_T_'+prefix+'_'+COMP+'_'+grid+'.dat','w')
        file4.write("%s \n\n"%('# (1)T[k]  (2)tau_kin*[s]  (3)tau_col*[s]  (4)tau_N[s]  (5)sigma[adim] (6)vel_int'))
        tau_T=[]

 print ('Temp[k]  Kappa_KCM[W/mK]  NL-length[nm] K_kin[W/mK]  K_col[W/mK] Sigma[adim]  Kappa_RTA[W/mK]\n')

 k_col = []

 for k in range(len(T)):
  if T[k] in Temp:
   if K_W=='Y':
	     k_w.append([])
   if K_MFP=='Y':
	     k_mfp.append([])
   if TAU_W=='Y':
	     file3.write('\n\n')

   k_kin = np.zeros((3,3),dtype=np.float64)
   v2Cv = np.zeros((3,3),dtype=np.float64)
   v_int_num = np.zeros((3,3),dtype=np.float64)
   tau_kin_den = np.zeros((3,3),dtype=np.float64)
   v2_N_num = np.zeros((3,3),dtype=np.float64)
   v2_N_den = np.zeros((3,3),dtype=np.float64)
   k_col_den = np.zeros((3,3),dtype=np.float64)
   tau_col_num = np.zeros((3,3),dtype=np.float64)
   tau_col_den = np.zeros((3,3),dtype=np.float64)
   k_rta = np.zeros((3,3),dtype=np.float64)
   Cv_int = 0.
   tau_n_num = 0.
   k_col_num = 0.
   tau_k = 0

   for j in range(len(qpoint)):
    for i in range(len(freq[j])):

       tau_N = 'inf'  ### Defalut values to avoid problems when writting file Tau_w
       tau_U = 'inf'
       tau_I = 'inf'
       tau_B = 'inf'

       g_N = gamma_N[k][j][i]
       g_U = gamma_U[k][j][i]
       if 'gamma_isotope' in f:
        g_I = gamma_I[j][i]
       else:
        g_I = 0.
       w = freq[j][i]*1.e12*2.*pi   # rads/s
       vx = vel[j][i][0]*100.  # ( THz * Angstrom ) --> m/s
       vy = vel[j][i][1]*100.
       vz = vel[j][i][2]*100.
       vel_vec = array([vx,vy,vz])
       vel2_matrix = (outer(vel_vec,vel_vec))
       vel_m = linalg.norm(vel_vec)       

       q_vec = array(qpoint[j][0]*b1+qpoint[j][1]*b2+qpoint[j][2]*b3)  #*2*pi/alat
       q2_matrix = (outer(q_vec,q_vec))

       Cv_mode = cv[k][j][i]*1.602e-19   #J/(m**3K)

       x = hbar*w/(kb*T[k])	

       C1 = q2_matrix/w**2.   #projection factor

       if 'gamma_isotope' in f:
        g_kin = g_I*I_SF + g_U
        g_rta = g_I*I_SF + g_U + g_N
       else:
        g_kin = g_U
        g_rta = g_U + g_N

      if g_N != 0 and vel_m>1e-5:
          tau_N = tau_value(g_N)
          v2_N_num += Cv_mode*vel2_matrix*tau_N*C1
          v2_N_den += Cv_mode*C1
          tau_n_num += Cv_mode*tau_N

       if g_kin != 0 and vel_m>1e-5:
          tau_k = tau_value(g_kin)
          if Leff != 'inf':
             tau_k = (tau_value(g_kin)**-1 + vel_m/Leff)**-1.0
          else:
             tau_k = tau_value(g_kin)
          k_kin += Cv_mode*vel2_matrix*tau_k
          k_col_num += Cv_mode*q_vec*vel_vec/w
          k_col_den += tau_value(g_kin)**-1*Cv_mode*C1
          tau_col_den += tau_value(g_kin)**-1*Cv_mode*C1
          tau_col_num += Cv_mode*C1

       if g_rta != 0. and vel_m>1e-5:
          if Leff != 'inf':
             tau_rta = (tau_value(g_rta)**-1 + vel_m/Leff)**-1.0
          else:
              tau_rta = tau_value(g_rta)
          k_rta += Cv_mode*vel2_matrix*tau_rta

       v2Cv += Cv_mode*(vel2_matrix)

       v_int_num += vel2_matrix*Cv_mode

       Cv_int += Cv_mode

       if K_W=='Y':
               k_w[-1].append([w,(Cv_mode*vel2_matrix*tau_k)[i1][i2], Cv_mode, T[k]])
       if K_MFP=='Y':
           if Leff!='inf':
               k_mfp[-1].append([(vel_m*tau_k),(Cv_mode*vel2_matrix*tau_k)[i1][i2], Cv_mode, T[k]])
           else:
               k_mfp[-1].append([(vel_m*tau_k),(Cv_mode*vel2_matrix*tau_k)[i1][i2], Cv_mode, T[k]])
       if TAU_W=='Y':
               if Leff != 'inf' and linalg.norm(vel_vec)!=0:
                   tau_B = Leff/linalg.norm(vel_vec)
               if g_I != 0 and I_SF != 0.:
                   tau_I =  tau_value(g_I*I_SF)
               if g_U != 0:
                   tau_U =  tau_value(g_U)
               file3.write("%s %s %s %s %s %s %s\n"%(T[k], w, tau_I, tau_U, tau_N, tau_B, linalg.norm(vel_vec)))

   for i in range(3): # To avoid numerical errors
      for j in range(3):
       if factor*k_kin[i][j]<1e-5:
            k_kin[i][j] = 0.

   Cv = Cv_int

   tau_col = tau_col_num/tau_col_den

   tau_R = k_kin/abs(v2Cv)
   v2_N = v2_N_num/v2_N_den
   kappa_col = outer(k_col_num, k_col_num)/k_col_den

   landa2 = tau_col*v2_N

   if Leff!='inf':
    if Leff<1.:
        F = 1./(2*pi**2.)*Leff**2.*(np.sqrt(1.+4.*pi**2.*abs(landa2)/Leff**2)-1.)/abs(landa2)
    else:
        F = 1.
   else:
        F = 1.

   tau_N = tau_n_num/Cv

   v_int = np.sqrt(abs(v_int_num)/Cv)

   sigma = tau_R/(tau_R+tau_N)     #(1./(1.+tau_N/(tau_R)))

   kappa_kin = k_kin

   ell_col2 = sigma*v2_N*tau_col
   v2tau_R = k_kin/Cv
   ell_kin2 = (1-sigma)*v2tau_R*tau_R

   ell = (np.sqrt(sigma*v2_N*tau_col+(1-sigma)*v2tau_R*tau_R))/1e-9

   k_col.append([(kappa_col*F)[i1][i2], Cv, sigma[i1][i2],(tau_col*v_int)[i1][i2]])

   kappa_total = factor*(kappa_kin*(1.-sigma)+kappa_col*sigma*F)

   print (T[k], ("     %8.3f         %8.3f     %8.3f    %8.3f    %.10f   %8.3f" % (kappa_total[i1][i2], ell[i1][i2], (factor*kappa_kin)[i1][i2], (factor*kappa_col*F)[i1][i2], sigma[i1][i2], k_rta[i1][i2]*factor)))

   file.write('%s %s %s %s %s %s %s\n' %(T[k], kappa_total[i1][i2], ell[i1][i2], (factor*kappa_kin)[i1][i2], (factor*kappa_col*F)[i1][i2], sigma[i1][i2], k_rta[i1][i2]*factor))

   if TAU_T=='Y':
        file4.write('%s %s %s %s %s %s\n' %(T[k], tau_R[i1][i2], tau_col[i1][i2], tau_N, sigma[i1][i2], v_int[i1][i2]))

 print ('\n', '--------------------------------- \n', 'Calculation done', '\n')

 if K_W=='Y' or K_MFP=='Y' or TAU_W=='Y' or TAU_T=='Y':
        print ('------>' , ' Writting output files')

 if K_W=='Y':
     lw = []
     stp = int(STP)
     for i in range(len(k_w[0])):
        lw.append(k_w[0][i][0])
     dw = max(lw)/stp
     lw=[0.]
     for i in range(stp):
        lw.append(lw[-1]+dw)
     new_l = []
     for i in range(len(k_w)):
        new_l.append([])
        for j in range(stp):
            new_l[i].append([lw[j], 0., 0., 0.])
        for j in range(len(k_w[i])):
           for k in range(stp-1):
             if k!=(stp-2):
                if k_w[i][j][0]<new_l[i][k+1][0] and k_w[i][j][0]>=new_l[i][k][0]:	
                    new_l[i][k][1]+=k_w[i][j][1]*factor
                    new_l[i][k][2]+=k_col[i][0]*factor/k_col[i][1]*k_w[i][j][2]
                    new_l[i][k][3]+=k_w[i][j][2]
                    continue
             if k_w[i][j][0]>=new_l[i][k][0] and k==(len(new_l[i])-2):
                    new_l[i][k][1]+=k_w[i][j][1]*factor
                    new_l[i][k][2]+=k_col[i][0]*factor/k_col[i][1]*k_w[i][j][2]
                    new_l[i][k][3]+=k_w[i][j][2]
                    continue
     for i in range(len(new_l)):
        acc_kin = 0.
        acc_col = 0.
        new_cv = 0.
        file1.write('\n\n')
        for j in range(len(new_l[i])):
                acc_kin +=new_l[i][j][1]
                acc_col += new_l[i][j][2]
                new_cv += new_l[i][j][3] 
                file1.write('%s %s %s %s %s %s %s %s %s\n' %(k_w[i][j][-1], new_l[i][j][0], new_l[i][j][1], new_l[i][j][2], acc_kin, acc_col, k_col[i][2], acc_kin*(1.-k_col[i][2])+k_col[i][2]*acc_col, new_cv*factor))
     file1.close()

 if K_MFP=='Y':
     for i in range(len(k_mfp)):
            c = 0
            kmfp = sorted(k_mfp[i])
            kk_mfp_acc = 0.
            k_eff = 0.
            file2.write('\n\n')
            for j in range(len(kmfp)):
                sigma = k_col[i][2]
                mfp_kin = kmfp[j][0]
                mfp_col = k_col[i][3]
                k_c = k_col[i][0]*factor
                kk_mfp_acc += kmfp[j][1]*factor
                sigma = k_col[i][2]
                mfp = kmfp[j][0]  
                if c==0 and mfp_col<mfp_kin:
                    file2.write('%s %s %s %s %s %s %s %s \n' %(kmfp[j][-1], mfp, kmfp[j][1]*factor, k_c, kk_mfp_acc, k_c, sigma, kk_mfp_acc*(1.-sigma)+sigma*k_c))
                    c=1
                if c==1 and mfp_col<mfp_kin:
                    file2.write('%s %s %s %s %s %s %s %s \n' %(kmfp[j][-1], mfp, kmfp[j][1]*factor, '0', kk_mfp_acc, k_c, sigma, kk_mfp_acc*(1.-sigma)+sigma*k_c))
                if c==0 and mfp_col>mfp_kin:
                    file2.write('%s %s %s %s %s %s %s %s \n' %(kmfp[j][-1], mfp, kmfp[j][1]*factor, '0.' ,kk_mfp_acc, '0.', sigma, kk_mfp_acc*(1.-sigma)))
                file2.close() 

 if K_W=='Y' or K_MFP=='Y' or TAU_W=='Y' or TAU_T=='Y':
        print ('                  |')
        print ('                  |')
        print ('                  V')
        print ('                Done','\n')

 if TAU_W=='Y':
     file3.close()
 if TAU_T=='Y':
     file4.close()

 file.close()
