import sys
import numpy as np
import jax.numpy as jnp
import jax
import optax
from jax.tree_util import Partial


sys.path.insert(0, "../terrapinn")
from dataloaders import *
from evaluations import *
from losses import *
from models import * 
from utilities import * 
from training import *

# generate test data and variables for this experiment
regenerate_data = False
sys.path.insert(1, "../seismic-cpml")
from seismic_CPML_2D_pressure_second_order import seismicCPML2D

def generate_x(x_lim, y_lim, nx, ny):
    x = np.linspace(x_lim[0], x_lim[1], nx)
    y = np.linspace(y_lim[0], y_lim[1], ny)
    xx, yy = np.meshgrid(x,y, indexing='ij')
    return (np.hstack((xx.reshape(-1,1), yy.reshape(-1,1))), x, y)

def gaussian(x, mu, sd, a):
    return a*np.exp(-0.5*np.sum(np.square((x-mu)/sd), axis=1))

def gaussian_c(x, c0):
        mus = np.array([[0.3,0.3],
                        [-0.5,-0.5],
                        [-0.2,0.2],
                        [0.3,-0.4]])
        sds = np.array([[0.3,0.3],
                        [0.4,0.4],
                        [0.2,0.2],
                        [0.3,0.3]])
        aas = c0*np.array([-0.7, -0.6, 0.7, 0.6])
        cs = np.array([gaussian(x, mu, sd, a) for mu, sd, a in zip(mus, sds, aas)])

        c = c0 + np.sum(cs, axis=0)
        return c

def fd_solution(c_array, p_array, dx, dy, dt, nx, ny, nt, f0, npoints_pml = 10):
    p_array = np.pad(p_array, [(npoints_pml, npoints_pml),(npoints_pml, npoints_pml)], mode="edge")
    c_array = np.pad(c_array, [(npoints_pml, npoints_pml),(npoints_pml, npoints_pml)], mode="edge")
    d_array = np.ones((nx+2*npoints_pml, ny+2*npoints_pml))
    # run simulation
    wavefields, _ = seismicCPML2D(
                nx+2*npoints_pml,
                ny+2*npoints_pml,
                nt,
                dx,
                dy,
                dt,
                npoints_pml,
                c_array,
                d_array,
                (p_array.copy(), p_array.copy()),
                f0,
                np.float32,
                output_wavefields=True,
                gather_is=None)
    
    wavefields = wavefields[:,npoints_pml:-npoints_pml,npoints_pml:-npoints_pml]
    return wavefields

print("Loading Simulation Variables")

x_lim = (-2.0,2.0)
y_lim = (-2.0,2.0)
t_lim = (0.0,2.0)
c0 = 1.0
source_sd=0.02
t1 = source_sd / c0
f0 = c0/source_sd# approximate frequency of wave
dx = dy = 1/(f0*10)# target fine sampled deltas
dt = dx/ (4*np.sqrt(2)*c0)# target fine sampled deltas
nx = int(np.floor((x_lim[1]-x_lim[0])/dx))+1
ny = int(np.floor((y_lim[1]-y_lim[0])/dy))+1
nt = int(np.floor((t_lim[1]-t_lim[0])/dt))+1

X, xv, yv = generate_x(x_lim, y_lim, nx, ny)
tv = np.linspace(t_lim[0], t_lim[1], nt)
cg_array = gaussian_c(X, c0).reshape(nx, ny)
c0_array = c0*np.ones(cg_array.shape)
p0_array = gaussian(X, 0, source_sd, 1.0).reshape(nx, ny)

if regenerate_data:
    print("Regenerating Data")
    c0_wavefield = fd_solution(c0_array, p0_array, dx, dy, dt, nx, ny, nt, f0)
    cg_wavefield = fd_solution(cg_array, p0_array, dx, dy, dt, nx, ny, nt, f0)

    np.save("data/c0_wavefield.npy", c0_wavefield[::10,::10,::10])
    np.save("data/c0_velocity.npy", c0_array[::10,::10])
    np.save("data/cg_wavefield.npy", cg_wavefield[::10,::10,::10])
    np.save("data/cg_velocity.npy", cg_array[::10,::10])

else:
    print("Not Regenerating Data")

# recalculate dx, dy etc. as we have decimated the wavefields after calculation to save space
dx = dx*10
dy = dy*10
dt = dx/ (4*np.sqrt(2)*c0)# target fine sampled deltas
nx = int(np.floor((x_lim[1]-x_lim[0])/dx))+1
ny = int(np.floor((y_lim[1]-y_lim[0])/dy))+1
nt = int(np.floor((t_lim[1]-t_lim[0])/dt))+1
X, xv, yv = generate_x(x_lim, y_lim, nx, ny)
tv = np.linspace(t_lim[0], t_lim[1], nt)

#radial model training

const_model_radial_data = jnp.load("data/c0_wavefield.npy")[:,100,100:]
tvv, rvv = jnp.meshgrid(tv,yv[100:], indexing='ij')
drtvv = jnp.abs(tvv-rvv)/jnp.sqrt(2)
rtscale = 1
ryscale = 1
tvv = tvv / rtscale
rvv = rvv / rtscale
dscale = jnp.std(const_model_radial_data)

dset_rng_key = jax.random.PRNGKey(43771121)
rm_rng_key = jax.random.PRNGKey(94899110)
num_batches = 30000
batch_size = 10000
radial_iterator = weighted_dataloader(dset_rng_key, 
                                      jnp.reshape(tvv, (-1,1)), 
                                      jnp.reshape(rvv, (-1,1)), 
                                      jnp.reshape(const_model_radial_data, (-1,1)) , 
                                      num_batches = num_batches+1, 
                                      batch_size = batch_size)
dummy_t, dummy_r, dummy_u = next(radial_iterator)
radial_model = default_mlp_model((32,32))
init_radial_params = radial_model.init(rm_rng_key, jnp.hstack((dummy_t, dummy_r)))

radial_schedule = optax.warmup_cosine_decay_schedule(
  init_value=1e-3,
  peak_value=1e-2,
  warmup_steps=5000,
  decay_steps=25_000,
  end_value=1e-8,
)

radial_optimizer=optax.adabelief(learning_rate=radial_schedule)

radial_params = train_to_data(init_radial_params, 
                              Partial(radial_model_eval, radial_model, t1=t1, sd=source_sd),
                              radial_optimizer, 
                              u_loss_fn_1d, 
                              radial_iterator, 
                              num_batches = num_batches)

