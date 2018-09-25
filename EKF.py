import numpy as np

def predict(X0,P,dt):
    Xe = np.array([0,0,0,0])[np.newaxis,:].T
    # print(Xe)
    # print("Xe")
    F = np.array([[1., 0, dt, 0.], 
                    [0., 1., 0., dt],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
    Q = np.array([[0.4**2., 0., 0., 0.], 
                    [0., 0.4**2, 0., 0.], 
                    [0., 0., (3*dt)**2, 0], 
                    [0., 0., 0., (3*dt)**2]])
    Xe = np.dot(F,X0)
    P = np.linalg.multi_dot([F,P,np.transpose(F)]) + Q
    return (Xe,P)
    

def ApplyEKF(X0,P,dt,Ymeasured,Xmeasured):
    Xe = np.array([0,0,0,0])[np.newaxis,:].T
    # print(Xe)
    # print("Xe")
    F = np.array([[1., 0, dt, 0.], 
                    [0., 1., 0., dt],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
    Q = np.array([[0.4**2., 0., 0., 0.], 
                    [0., 0.4**2, 0., 0.], 
                    [0., 0., (3*dt)**2, 0], 
                    [0., 0., 0., (3*dt)**2]])
    Xe = np.dot(F,X0)
    P = np.linalg.multi_dot([F,P,np.transpose(F)]) + Q
    # Step 2: Measurement Prediction
    # H(1) = y = y + vy*dt
    # H(2) = z = z + vz*dt
    H = np.array([[1., 0., 0., 0.],
                    [0., 1., 0., 0]])
    Yexpected = Xe[0]
    Xexpected = Xe[1]
    # Ymeasured (pixel from sensor)
    # Zmeasured (pixel from sensor)

    # Step 3: Measurement Residual
    ydiff = Ymeasured-Yexpected
    xdiff = Xmeasured-Xexpected
    z = np.array([ydiff, xdiff])[np.newaxis,:].T
    
    # Measurement Prediction Covariance: s
    R = np.array([[0.1**2., 0.], [0., 0.1**2.]])
    s = np.linalg.multi_dot([H,P,np.transpose(H)]) + R
    # print(s)
    iS = np.linalg.inv(s)
    # print(P)
    # print(H.T)
    # print(iS)
    Gain = np.linalg.multi_dot([P,H.T,iS])
    # print(Gain)
    # print("XE")
    # print(Xe)
    # print("XE")
    # print(z)
    step = np.dot(Gain,z)
    # print("Step")
    # print(step)
    for i in range(0,3):
        Xe[i] = Xe[i] + step[i]
    P = P - np.linalg.multi_dot([P,H.T,iS,H,P])
    # print("XE")
    # print(Xe)
    # print("XE")
    return (Xe,P)