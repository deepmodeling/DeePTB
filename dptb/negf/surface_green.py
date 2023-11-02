import torch
import torch.linalg as tLA
from xitorch.linalg.solve import solve
import scipy.linalg as SLA
import matplotlib.pyplot as plt
from xitorch.grad.jachess import jac
from torch.autograd.functional import jvp
import logging

log = logging.getLogger(__name__)


class SurfaceGreen(torch.autograd.Function):
    '''calculate surface green function

    To realize AD-NEGF, this Class is designed manually to calculate the surface green function auto-differentiably.
    
    At this stage, we realized Lopez-Sancho scheme and  GEP scheme.
    However, GEP scheme is not so stable, and we strongly recommended  to implement the Lopez-Sancho scheme.

    '''

    @staticmethod
    def forward(ctx, H, h01, S, s01, ee, method='Lopez-Sancho'):
        # '''
        # gs = [A_l - A_{l,l-1} gs A_{l-1,l}]^{-1}
        # 
        # 1. ee can be a list, to handle a batch of samples
        # '''

        if method == 'GEP':
            gs = calcg0(ee, H, S, h01, s01)
        else:
            h10 = h01.conj().T
            s10 = s01.conj().T
            alpha, beta = h10 - ee * s10, h01 - ee * s01
            eps, epss = H.clone(), H.clone()
            
            converged = False
            iteration = 0
            while not converged:
                iteration += 1
                oldeps, oldepss = eps.clone(), epss.clone()
                oldalpha, oldbeta = alpha.clone(), beta.clone()
                tmpa = tLA.solve(ee * S - oldeps, oldalpha)
                tmpb = tLA.solve(ee * S - oldeps, oldbeta)

                alpha, beta = torch.mm(oldalpha, tmpa), torch.mm(oldbeta, tmpb)
                eps = oldeps + torch.mm(oldalpha, tmpb) + torch.mm(oldbeta, tmpa)

                epss = oldepss + torch.mm(oldbeta, tmpa)
                LopezConvTest = torch.max(alpha.abs() + beta.abs())

                if iteration == 101:
                    log.error("Lopez-scheme not converged after 100 iteration.")

                if LopezConvTest < 1.0e-40:
                    gs = (ee * S - epss).inverse()

                    test = ee * S - H - torch.mm(ee * s01 - h01, gs.mm(ee * s10 - h10))
                    myConvTest = torch.max((test.mm(gs) - torch.eye(H.shape[0], dtype=h01.dtype)).abs())
                    if myConvTest < 1.0e-6:
                        converged = True
                        if myConvTest > 1.0e-8:
                            log.warning("Lopez-scheme not-so-well converged at E = %.4f eV:" % ee.real.item() + str(myConvTest.item()))
                    else:
                        log.error("Lopez-Sancho %.8f " % myConvTest.item() +
                              "Error: gs iteration {0}".format(iteration))
                        raise ArithmeticError("Criteria not met. Please check output...")
                    
        ctx.save_for_backward(gs, H, h01, S, s01, ee)
        return gs

    @staticmethod
    def backward(ctx, grad_outputs):
        gs_, H_, h01_, S_, s01_, ee_ = ctx.saved_tensors

        def sgfn(gs, *params):
            [H, h01, S, s01, ee] = params
            return tLA.inv(ee*S - H - (ee*s01 - h01).matmul(gs).matmul(ee*s01.conj().T - h01.conj().T)) - gs

        params = [H_, h01_, S_, s01_, ee_]
        idx = [i for i in range(len(params)) if params[i].requires_grad]
        params_copy = [p.detach().requires_grad_() for p in params]

        with torch.enable_grad():

            grad = jac(fcn=sgfn, params=(gs_, *params), idxs=[0])[0] # dfdz
            pre = solve(A=grad.H, B=-grad_outputs.reshape(-1, 1))
            pre = pre.reshape(grad_outputs.shape)
            
            yfcn = sgfn(gs_, *params_copy)

            grad = torch.autograd.grad(yfcn, [params_copy[i] for i in idx], grad_outputs=pre,
                                                         create_graph=torch.is_grad_enabled(),
                                                         allow_unused=True)

        # grad = torch.autograd.grad(yfcn, params_copy, grad_outputs=pre,
        #                         create_graph=torch.is_grad_enabled(),
        #                         allow_unused=True)

            grad_out = [None for _ in range(len(params))]
            for i in range(len(idx)):
                grad_out[idx[i]] = grad[i]


            '''
            2. Is the matrix index direction correct? Also, is T necessarily becomes H when comes to complex matrix?
            '''
            # return *grad, None, None
            return *grad_out, None

    @staticmethod
    def jvp(ctx, grad_input):
        # should be of shape as [H, h01, S, s01, ee]
        gs_, H_, h01_, S_, s01_, ee_ = ctx.saved_tensors
        left = ctx.left

        if left:
            def sgfn(gs, *params):
                [H, h01, S, s01, ee] = params
                return tLA.inv(ee*S-H-(ee*s01.conj().T-h01.conj().T).matmul(gs).matmul(ee*s01-h01)) - gs
        else:
            def sgfn(gs, *params):
                [H, h01, S, s01, ee] = params
                return tLA.inv(ee*S - H - (ee*s01 - h01).matmul(gs).matmul(ee*s01.conj().T - h01.conj().T)) - gs
        
        yfcn = sgfn(gs_, *params_copy)

        params = [H_, h01_, S_, s01_, ee_]
        idx = [i for i in range(len(params)) if params[i].requires_grad]
        params_copy = [p.detach().requires_grad_() for p in params]

        with torch.enable_grad():
            _, grad_fw = jvp(func=yfcn, inputs=[params_copy[i] for i in idx], v=[grad_input[i] for i in idx], create_graph=torch.is_grad_enabled())
            dfdy = jac(fcn=sgfn, params=(gs_, *params), idxs=[0])[0]

            out = [solve(A=dfdy, B=-gf.reshape(-1, 1)).conj().reshape(gf.shape) for gf in grad_fw]

            return torch.mean(out, dim=0)

def selfEnergy(hL, hLL, sL, sLL, ee, hDL=None, sDL=None, etaLead=1e-8, Bulk=False, chemiPot=0.0, dtype=torch.complex128, device='cpu', method='Lopez-Sancho'):
    '''calculates the self-energy and surface Green's function for a given  Hamiltonian and overlap matrix.
    
    Parameters
    ----------
    hL
        Hamiltonian matrix for one principal layer in Lead
    hLL
        Hamiltonian matrix between the most nearby principal layers in Lead
    sL
        Overlap matrix for one principal layer in Lead
    sLL
        Overlap matrix between the most nearby principal layers in Lead
    ee
        the given energy
    hDL
        Hamiltonian matrix between the lead and the device.   
    sDL
        Overlap matrix between the lead and the device.
    etaLead
        A small imaginary number that is used to avoid the singularity of the surface Green's function.
    Bulk, optional
        Ignore it, please.
    chemiPot
        the chemical potential of the lead.
    dtype
        the data type of the tensors used in the calculations. 
    device
        The "device" parameter specifies the device on which the calculations will be performed. It can be
        set to 'cpu' for CPU computation or 'cuda' for GPU computation.
    method
        specify the method for calculating the surface Green's function.The available options 
        are "Lopez-Sancho" and any other value will default to "Lopez-Sancho".
    
    Returns
    -------
        two values: Sig and SGF. The former is self-energy and the latter is surface Green's function.
    
    '''
    # if not isinstance(ee, torch.Tensor):
    #     eeshifted = torch.scalar_tensor(ee, dtype=dtype) - voltage  # Shift of self energies due to voltage(V)
    # else:
    #     eeshifted = ee - voltage

    if not isinstance(ee, torch.Tensor):
        eeshifted = torch.scalar_tensor(ee, dtype=dtype) + chemiPot
    else:
        eeshifted = ee + chemiPot
        

    if hDL == None:
        ESH = (eeshifted * sL - hL)
        SGF = SurfaceGreen.apply(hL, hLL, sL, sLL, eeshifted + 1j * etaLead, method)

        if Bulk:
            Sig = tLA.inv(SGF)  # SGF^1
        else:
            Sig = ESH - tLA.inv(SGF)
    else:
        a, b = hDL.shape
        SGF = SurfaceGreen.apply(hL, hLL, sL, sLL, eeshifted + 1j * etaLead, method)
        Sig = (ee*sDL-hDL) @ SGF[:b,:b] @ (ee*sDL.conj().T-hDL.conj().T)
    return Sig, SGF  # R(nuo, nuo)


def calcg0(ee, h00, s00, h01, s01):
    '''The `calcg0` function calculates the surface Green's function for a specific |k> , ref. Euro Phys J B 62, 381 (2008)
        Inverse of : NOTE, setup for "right" lead.
        e-h00 -h01  ...
        -h10  e-h11 ...
         .
         .
         .

    Parameters
    ----------
    ee
        The parameter `ee` represents the energy value for which the surface Green's function is
    calculated. It is a complex number that determines the energy of the state being considered.
    h00
        hamiltonian matrix within principal layer
    s00
        overlap matrix within principal layer
    h01
        hamiltonian matrix between two adject principal layers
    s01
        overlap matrix between two adject principal layers
    
    Returns
    -------
        Surface Green's function `g00`.
    
    ''' 


    NN, ee = h00.shape[0], ee.real + max(torch.max(ee.imag).item(), 1e-8) * 1.0j

    # Solve generalized eigen-problem
    # ( e I - h00 , -I) (eps)          (h01 , 0) (eps)
    # ( h10       ,  0) (xi ) = lambda (0   , I) (xi )
    
    a, b = torch.zeros((2 * NN, 2 * NN), dtype=h00.dtype), torch.zeros((2 * NN, 2 * NN),
                                                                             dtype=h00.dtype)
    
    a[0:NN, 0:NN] = ee * s00 - h00
    a[0:NN, NN:2 * NN] = -torch.eye(NN, dtype=h00.dtype)
    a[NN:2 * NN, 0:NN] = h01.conj().T - ee * s01.conj().T
    b[0:NN, 0:NN] = h01 - ee * s01
    b[NN:2 * NN, NN:2 * NN] = torch.eye(NN, dtype=h00.dtype)

    


    ev, evec = SLA.eig(a=a, b=b)
    ev = torch.tensor(ev, dtype=h00.dtype)
    evec = torch.tensor(evec, dtype=h00.dtype)
    # ev = torch.complex(real=torch.tensor(ev.real), imag=torch.tensor(ev.imag))
    # evec = torch.complex(real=torch.tensor(evec.real), imag=torch.tensor(evec.imag))

    # Select lambda <0 and the eps part of the evec
    ipiv = torch.where(ev.abs() < 1.)[0]

    ev, evec = ev[ipiv], evec[:NN, ipiv].T
    # Normalize evec
    norm = torch.diag(torch.mm(evec, evec.conj().T)).sqrt()
    evec = torch.mm(torch.diag(1.0 / norm), evec)

    # E^+ Lambda_+ (E^+)^-1 --->>> g00
    EP = evec.T
    FP = EP.mm(torch.diag(ev)).mm(torch.inverse(torch.mm(EP.conj().T, EP))).mm(EP.conj().T)
    g00 = torch.inverse(ee * s00 - h00 - torch.mm(h01 - ee * s01, FP))

    g00 = iterative_gf(ee, g00, h00, h01, s00, s01, iter=3)

    # Check!
    err = torch.max(torch.abs(g00 - torch.inverse(ee * s00 - h00 - \
                                                  torch.mm(h01 - ee * s01, g00).mm(
                                                      h01.conj().T - ee * s01.conj().T))))
    if err > 1.0e-8:
        print("WARNING: not-so-well converged for RIGHT electrode at E = {0} eV:".format(ee.real.numpy()), err.numpy())
    return g00


def iterative_gf(ee, gs, h00, h01, s00, s01, iter=1):
    for i in range(iter):
        gs = ee*s00 - h00 - (ee * s01 - h01) @ gs @ (ee * s01.conj().T - h01.conj().T)
        gs = tLA.pinv(gs)

    return gs
