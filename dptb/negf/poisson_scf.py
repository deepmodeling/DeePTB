



def poisson_negf_scf(grid,interface,device,acc=1e-6,max_iter=100):
    
    interface.phi_old = interface.phi.copy()
    scf_count = 0
    
    max_diff = 1e30
    while max_diff > acc and scf_count < max_iter:
        scf_count += 1
        print('SCF iteration: ',scf_count)
        device.phi = interface.phi[interface.grid.atom_index]

        
        max_diff = np.max(np.abs(device.phi - interface.phi_old))
        interface.phi_old = device.phi.copy()
