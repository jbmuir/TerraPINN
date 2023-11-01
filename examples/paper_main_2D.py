import sys
import numpy as np
import jax.numpy as jnp
import jax
import optax
from jax.tree_util import Partial
import matplotlib.pyplot as plt
import pickle


sys.path.insert(0, "../terrapinn")
from dataloaders import *
from evaluations import *
from losses import *
from models import * 
from utilities import * 
from training import *

# generate test data and variables for this experiment
regenerate_data = False
retrain_radial = False
retrain_full = True
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
    np.save("data/c0_wavefield_line.npy", c0_wavefield[:,xv==0,yv>=0])
else:
    print("Not Regenerating Data")

# recalculate dx, dy etc. as we have decimated the wavefields after calculation to save space
# dx = dx*10
# dy = dy*10
# dt = dx/ (4*np.sqrt(2)*c0)# target fine sampled deltas
# nx = int(np.floor((x_lim[1]-x_lim[0])/dx))+1
# ny = int(np.floor((y_lim[1]-y_lim[0])/dy))+1
# nt = int(np.floor((t_lim[1]-t_lim[0])/dt))+1
# X, xv, yv = generate_x(x_lim, y_lim, nx, ny)
# tv = np.linspace(t_lim[0], t_lim[1], nt)

#radial model training

tv_select = np.where(tv>source_sd)[0]
const_model_radial_data = jnp.load("data/c0_wavefield_line.npy")[tv_select,:] # avoid directly sampling origin
tvv, rvv = jnp.meshgrid(tv[tv_select],yv[yv>=0], indexing='ij')
drtvv = jnp.abs(tvv-rvv)/jnp.sqrt(2)
rtscale = 1
ryscale = 1
tvv = tvv / rtscale
rvv = rvv / rtscale
dscale = jnp.std(const_model_radial_data)
const_model_radial_data = const_model_radial_data / dscale

radial_model = default_mlp_model((64,64))

if retrain_radial:
    dset_rng_key = jax.random.PRNGKey(1)
    rm_rng_key = jax.random.PRNGKey(2)
    num_batches = 500_000
    batch_size = 4096
    radial_iterator = weighted_dataloader(dset_rng_key, 
                                        jnp.reshape(const_model_radial_data, (-1,1)), # observed data has to go first, then coords in order used by model
                                        jnp.reshape(tvv, (-1,1)), 
                                        jnp.reshape(rvv, (-1,1)), 
                                        num_batches = num_batches+1, 
                                        batch_size = batch_size)
    dummy_t, dummy_r, dummy_u = next(radial_iterator)
    init_radial_params = radial_model.init(rm_rng_key, jnp.hstack((dummy_t, dummy_r)))

    radial_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-3,
    peak_value=1e-2,
    warmup_steps=50_000,
    decay_steps=250_000,
    end_value=1e-8,
    )

    radial_optimizer=optax.adamw(learning_rate=radial_schedule)

    radial_params = train_to_data(init_radial_params, 
                                Partial(radial_model_eval, radial_model, t1=t1, sd=source_sd),
                                radial_optimizer, 
                                u_loss_fn, 
                                radial_iterator, 
                                num_batches = num_batches)

    with open("data/radial2d.pickle", 'wb') as file:
        pickle.dump(radial_params, file)

    print("Done training radial")

else:
    with open("data/radial2d.pickle", 'rb') as file:
        radial_params = pickle.load(file)


radial_evaluate = Partial(radial_model_eval, radial_model, radial_params, t1=t1, sd=source_sd)
upredtest = jax.vmap(radial_evaluate)(tvv.reshape(-1,1), rvv.reshape(-1,1))
fig, axes = plt.subplots(1,3)
axes[0].contourf(rvv,tvv,jnp.clip(const_model_radial_data,-3,3))
axes[1].contourf(rvv,tvv,jnp.clip(upredtest.reshape(rvv.shape),-3,3))
axes[2].contourf(rvv,tvv,jnp.clip(const_model_radial_data-upredtest.reshape(rvv.shape),-1,1), cmap='seismic')
fig.show()


hbatch_size = 25000
full_model = default_mlp_model((64,32,32,16), output=4, output_bias=False)
fm_evaluate = Partial(model_eval_2d, radial_model, radial_params, full_model, t1=t1, sd=2*source_sd)

if retrain_full:
    fm_optimizer = optax.lion(learning_rate=1e-4)
    fm_dset_rng_key = jax.random.PRNGKey(3)
    fm_rng_key = jax.random.PRNGKey(4)
    resampler = Partial(uniform_plus_kde_sampler, sd=2*source_sd, bandwidth=0.1)
    dummy_fm_coords, dummy_fm_weights = resampler(fm_dset_rng_key, 3, hbatch_size=hbatch_size)
    init_fm_params = full_model.init(fm_rng_key, jnp.hstack(dummy_fm_coords))
    init_fm_params= scale_param(init_fm_params,1)
    train_output = train_to_physics(fm_rng_key, init_fm_params, fm_evaluate, fm_optimizer, phys_loss_fn_2d, resampler, jvgaussian_c, anneal_schedule=jnp.hstack((jnp.linspace(0.01,1,100), jnp.ones(50))), epochs=150)
    with open("data/hetero_train_output2d.pickle", 'wb') as file:
        pickle.dump(train_output, file)

    fm_optimizer = optax.lion(learning_rate=3e-4)
    (anneal_params, epoch_loss_history, coords, colloc_weights) = train_output
    fm_dset_rng_key = jax.random.PRNGKey(5)
    fm_rng_key = jax.random.PRNGKey(6)
    resampler = Partial(uniform_plus_kde_sampler, sd=2*source_sd, bandwidth=0.02)
    train_output = train_to_physics(fm_rng_key, anneal_params, fm_evaluate, fm_optimizer, phys_loss_fn_2d, resampler, jvgaussian_c, anneal_schedule=np.ones(50), epochs=50)
    with open("data/hetero_train_output2d_2.pickle", 'wb') as file:
        pickle.dump(train_output, file)

else:
    with open("data/hetero_train_output2d_2.pickle", 'rb') as file:
        train_output = pickle.load(file)

(anneal_params, epoch_loss_history, coords, colloc_weights) = train_output

# (anneal_params2, epoch_loss_history2, coords2, colloc_weights2) = train_to_physics(fm_rng_key, init_fm_params, fm_evaluate, fm_optimizer, phys_loss_fn_2d, resampler,jvgaussian_c, anneal_schedule=np.ones(100),epochs=100)

gt_data = np.load("data/cg_wavefield.npy")
xtest = xv[::10]
ytest = yv[::10]
ttest = tv[::10]
xsel = np.where(np.abs(xtest)<=1)[0]
ysel = np.where(np.abs(ytest)<=1)[0]
xtest = xtest[xsel]
ytest = ytest[ysel]
xxtest, yytest = jnp.meshgrid(xtest, ytest, indexing="ij")
c2 = jvgaussian_c(jnp.hstack((xxtest.reshape(-1,1), yytest.reshape(-1,1))), 1.0).reshape(-1,1)
tep = Partial(fm_evaluate, anneal_params)


fig, axes = plt.subplots(4,4)
testtimes = [0.1,0.25,0.50,0.75]

for (i,testtime) in enumerate(testtimes): 
    testidx = np.argmin(np.abs(ttest-testtime))
    actualtesttime = ttest[testidx]
    print(testidx, actualtesttime)
    tttest = actualtesttime*jnp.ones(xxtest.shape)
    gt_slice = gt_data[testidx,xsel[0]:xsel[-1]+1,ysel[0]:ysel[-1]+1]
    upredtest = jax.vmap(tep)(tttest.reshape(-1,1), xxtest.reshape(-1,1), yytest.reshape(-1,1))
    loss, res = phys_loss_fn_2d(fm_evaluate, anneal_params, c2, tttest.reshape(-1,1), xxtest.reshape(-1,1), yytest.reshape(-1,1))

    axes[0,i].contourf(xxtest, yytest, gt_slice)
    axes[1,i].contourf(xxtest, yytest, upredtest.reshape(xxtest.shape))
    axes[2,i].contourf(xxtest, yytest, gt_slice - upredtest.reshape(xxtest.shape))
    axes[3,i].contourf(xxtest, yytest, res.reshape(xxtest.shape))

fig.show()

# plt.figure(figsize=(4,3))
# plt.contourf(xxtest, yytest, upredtest.reshape(101,101))
# plt.title("PINN Solution", fontsize=14)
# plt.colorbar(label="Amplitude (arb.)")
# plt.show()

# # plt.savefig("figures/pinn2D.pdf")
# # plt.close()
# c2 = jvgaussian_c(jnp.hstack(coords[1:]), 1.0).reshape(-1,1)
# loss, res = phys_loss_fn_2d(fm_evaluate, init_fm_params, c2, *coords, weights=colloc_weights)