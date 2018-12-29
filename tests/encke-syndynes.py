import matplotlib.pyplot as plt
import cometsuite as cs

args = '2P', '1997-07-14'
beta = [0.1, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001]
syn = cs.quick_syndynes(*args, beta=beta, integrator=cs.BulirschStoer())
cs.synplot(syn)
plt.gca().set_rmax(3600)
plt.tight_layout()
