import numpy as np

def pos_redshift_space(pos, vel, boxsize, h=1, z_time=0):
  ''' 
  pos_r = H_0 pos/c + vel/c
  assume z_time = 0, we have
  c/H_0 pos_r = pos + vel/H_0 = pos + vel*(1+z)/H_0 = pos + vel*factor
  units:
  - pos: Mpc/h
  - vel: km/s
  - factor: Mpc/h / km/s
  pos, val are both 1D arrays
  '''
  hubble = 100 #TODO: debug! * h # the Hubble parameter at z=0, H0 = 100 (km/s / Mpc) * h, with unit km/s / Mpc/h
  factor = (1.0 + z_time) / hubble
  pos += vel * factor
  pos %= boxsize
  return pos