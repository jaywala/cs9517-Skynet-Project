from EKF import ApplyEKF
import numpy as np

Xe = np.array([14.,45.,0.,0.])[np.newaxis,:].T
P = np.zeros((4,4))
dt = 0.2

(Xe,P) = ApplyEKF(Xe,P,dt,15,45)

print("{} {} {} {}".format(Xe[0][0],Xe[1][0],Xe[2][0],Xe[3][0]))

(Xe,P) = ApplyEKF(Xe,P,dt,16,46)

print("{} {} {} {}".format(Xe[0][0],Xe[1][0],Xe[2][0],Xe[3][0]))

(Xe,P) = ApplyEKF(Xe,P,dt,17,47)

print("{} {} {} {}".format(Xe[0][0],Xe[1][0],Xe[2][0],Xe[3][0]))

(Xe,P) = ApplyEKF(Xe,P,dt,18,48)

print("{} {} {} {}".format(Xe[0][0],Xe[1][0],Xe[2][0],Xe[3][0]))
