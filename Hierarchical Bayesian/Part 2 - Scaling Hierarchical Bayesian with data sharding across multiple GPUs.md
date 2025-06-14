# How to Achieve 10,000x Faster Bayesian Inference: Multi-GPU SVI vs. Traditional MCMC

**Are slow computation times preventing you from implementing Bayesian models in production?** You're not alone. While Bayesian models offer a powerful tool for incorporating prior knowledge and uncertainty quantification, their adoption in industry has been limited by one critical factor: traditional inference methods are extremely slow—especially when scaled to high-dimensional spaces. In this guide, I'll show you how to accelerate your Bayesian inference by up to 10,000 times using multi-GPU Stochastic Variational Inference (SVI) compared to CPU-based Markov Chain Monte Carlo (MCMC) methods.

**What You'll Learn:**
 - Differences between Monte Carlo and Variational Inference approaches.
 - How to implement data parallelism across multiple GPUs.
 - Step-by-step techniques (and code) to scale your models to handle millions or billions of observations/parameters.
 - Performance benchmarks across CPU, single GPU, and multi-GPU implementations

This article continues our practical series on hierarchical Bayesian modeling, building on our previous [price elasticity of demand example](https://towardsdatascience.com/estimating-product-level-price-elasticities-using-hierarchical-bayesian/). Whether you're a data scientist working with massive datasets or an academic researcher looking explore previously intractable problems, these techniques will transform how you approach estimating Bayesian models.

Want to skip the theory and jump straight to implementation? You'll find the practical code examples in the implementation section below. 

### Inference Methods
Recall our baseline specification: 

$$\log(\text{Demand}_{it})= \beta_i \log(\text{Price})_{it} +\gamma_{c(i),t} + \delta_i + \epsilon_{it}$$

Where:
 - $\text{Units Sold}_{it} \sim \text{Poisson}(\text{Demand}_{it}, \sigma_D)$
 - $\beta_i \sim \text{Normal}(\beta_{c\left(i\right)},\sigma_i)$
 - $\beta_{c(i)}\sim \text{Normal}(\beta_g,\sigma_{c(i)})$
 - $\beta_g\sim \text{Normal}(\mu,\sigma)$

We would like to estimate the parameters vector (and their variance) $z = \{ \beta_g, \beta_{c(i)}, \beta_i, \gamma_{c(i),t}, \delta_i, \text{Demand}_{it} \}$ using the data $x = \{ \text{Units}_{it}, \text{Price}_{it}\}$. One advantage in using Bayesian methods compared to frequentist approaches is that we can directly model count/sales data with distributions like Poisson, avoiding issues with zero values that might arise when using log-transformed models. Using Bayesian, we specify a prior distribution (based on our beliefs) $p(z)$ that incorporates our knowledge about the vector $z$ before seeing any data. Then, given the observed data $x$, we generate a likelihood $p(x|z)$ that tells us how likely it is that we observe the data $x$ given our specification of $z$. We then apply Bayes' rule $p(z|x) = \frac{p(z)p(x|z)}{p(x)}$ to obtain the posterior distribution, which represents our updated beliefs about the parameters given the data. The denominator can also be written as $p(x) = \int p(z,x) \, dz = \int p(z)p(x|z) \, dz$. This reduces our equation to:

$$p(z|x) = \frac{p(z)p(x|z)}{\int p(z)p(x|z) \, dz}$$

This equation requires calculating the posterior distribution of the parameters conditional on the observed data $p(z|x)$, which is equal to the prior distribution $p(z)$ multiplied by the likelihood of the data given some parameters $z$. We then divide that product by the marginal likelihood (evidence), which is the total probability of the data across all possible parameter values. The difficulty in calculating $p(z|x)$ is that the evidence requires computing a high-dimensional integral $p(x) = \int p(x|z)p(z)dz$. Many models with a hierarchical structure or complex parameter relationships also do not have closed form solutions for the integral. Furthermore, the computational complexity increases exponentially with the number of parameters, making direct calculation intractable for high-dimensional models. Therefore, Bayesian inference is conducted in practice by approximating the integral.

We now explore the two most popular methods for Bayesian inference; Markov-Chain Monte Carlo (MCMC) and [Stochastic Variational Inference](https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf) (SVI) in the following sections. While these are the most popular methods, other methods exist, such as [Importance Sampling](https://builtin.com/articles/importance-sampling), [particle filters (sequential Monte Carlo)](https://www.ma.imperial.ac.uk/~agandy/teaching/ltcc/lecture5.pdf), and [Expectation Propagation](https://tminka.github.io/papers/ep/minka-ep-uai.pdf) but will not be covered in this article.

#### Markov-Chain Monte Carlo

MCMC methods are a class of algorithms that allow us to sample from a probability distribution when direct sampling is difficult. In Bayesian inference, MCMC enables us to draw samples from the posterior distribution $p(z|x)$ without explicitly calculating the integral in the denominator. The core idea is to construct a Markov chain whose stationary distribution equals our target posterior distribution. Mathematically, our target distribution $p(z|x)$ can be represented by $\pi$, and we are trying to construct a transition matrix $P$ such that $\pi = \pi P$. Once the chain has reached its stationary distribution (after discarding the burn-in samples, where the chain might not be stationary), each successive state of the chain will be approximately distributed according to our target distribution $\pi$. By collecting enough of these samples, we can construct an empirical approximation of our posterior that becomes asymptotically unbiased as the number of samples increases.

Markov-chain methods are types of samplers that provide different approaches for constructing the transition matrix $P$. The most fundamental is the [Metropolis-Hastings](https://arxiv.org/pdf/1504.01896) (MH) algorithm, which proposes new states from a proposal distribution and accepts or rejects them based on probability ratios that ensure the chain converges to the target distribution. While MH is the foundation of Markov-chain methods, recent advancements in the field have moved to more sophisticated samplers like [Hamiltonian Monte Carlo](https://bayesianbrad.github.io/posts/2019_hmc.html) (HMC) that incorporates concepts from physics by including gradient information to more efficiently explore the parameter space. Finally, the default sampler in recent years is the [No U-Turn sampler](https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf) (NUTS) that improves HMC by automatically tuning HMC's hyperparameters. 

Despite their desirable theoretical properties, MCMC methods face significant limitations when scaling to large datasets and high-dimensional parameter spaces. The sequential nature of MCMC creates a computational bottleneck as each step in the chain depends on the previous state, making parallelization difficult. Furthermore, MCMC methods typically require evaluating the likelihood function using the entire dataset at each iteration. While ongoing research has proposed methods to overcome this limitation such as [stochastic gradient](https://arxiv.org/abs/1907.06986) and [mini-batching](https://arxiv.org/abs/1908.02910), it has not seen widespread adoption. These scaling issues have made applying traditional Bayesian inference a challenge in large data settings.

#### Stochastic Variational Inference

The second class of commonly used methods for Bayesian inference is Stochastic Variational Inference. Instead of sampling from the unknown posterior distribution, we posit that there exists a family of distributions $\mathcal{Q}$ that can approximate the unknown posterior $p(z|x)$. This family is parameterized by variational parameters $\phi$ (also known as a guide in Pyro/Numpyro), and our goal is to find the member $q_\phi(z) \in \mathcal{Q}$ that most closely resembles the true posterior. The standard proposed distribution uses a mean-field approximation, in that it assumes that all latent variables are mutually independent. This assumption implies that the joint distribution factorizes into a product of marginal distributions, making computation more tractable. As an example, we can have a Diagonal Multivariate Normal as the guide, and the parameters $\phi$ would be the location and scale parameter of each diagonal element. Since all covariance terms are set to be zero, this family of distribution has mutually independent parameters. This is especially problematic for sales data, since spillover effects are rampant. 

Unlike MCMC which uses sampling, SVI formulates Bayesian inference as an optimization problem by minimizing the [Kullback-Leibler (KL) divergence](https://towardsdatascience.com/understanding-kl-divergence-f3ddc8dff254/) between our approximation and the true posterior: $\text{KL}(q_\phi(z) || p(z|x))$. While we cannot tractably compute the full divergence, minimizing the KL-divergence is equivalent to maximizing the evidence lower bound (ELBO) ([derivation](https://chrisorm.github.io/VI-ELBO.html)) stochastically using established optimization techniques.

Research along this route tends to focus on two main directions: improving the variational family $\mathcal{Q}$ or developing better versions of the ELBO. More expressive families like [normalizing flows](https://arxiv.org/abs/1505.05770) can capture complex posterior geometries but come with higher computational costs. [Importance Weighted ELBO](https://arxiv.org/abs/1509.00519) derives a tighter bound on the log marginal likelihood, reducing the bias of SVI. Since SVI is fundamentally a minimization technique, it also benefits from optimization algorithms developed for deep learning. These improvements allow SVI to scale to extremely large datasets, however at the cost of some approximation quality. Furthermore, the mean-field assumption implies that the posterior uncertainty of SVI tends to be underestimated. This means that the credible intervals are too narrow and may not properly capture the true parameter values, something we show in Part 1 of this series.

#### Which one to use

Since our goal of this article is scaling, we will use SVI for future applications. As noted in [Blei et al. (2016)](https://arxiv.org/abs/1601.00670), "variational inference is suited to large data sets and scenarios where we want to quickly explore many models; MCMC is suited to smaller data sets and scenarios where we happily pay a heavier computational cost for more precise samples". Papers applying SVI have shown significant speedups in inference (up to 3 orders of magnitude) when applied to [multinomial logit models](https://arxiv.org/abs/2004.05426), [astrophysics](https://arxiv.org/abs/1803.06473), and [big data marketing](https://english.ckgsb.edu.cn/sites/default/files/files/Variational%20Bayesian%20Inference%20for%20Big%20Data%20Marketing%20Models.pdf). 

### Data Sharding

[JAX](https://docs.jax.dev/en/latest/) is a Python library for accelerator-oriented array computation that combines NumPy's familiar API with GPU/TPU acceleration and automatic differentiation. Under the hood, JAX uses both JIT and XLA to efficiently compile and optimize calculations. Key to this article is JAX's ability to distribute data across multiple devices ([data sharding](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)), which enables parallel processing by splitting computation across hardware resources. In the context of our model, this means that we can partition our $X$ vector across devices to accelerate convergence of SVI. JAX also allows for replication, which duplicates the data across all devices. This is important for some parameters of our model (global elasticity, category elasticity, and subcategory-by-time fixed effect), which are information that could potentially be needed by all devices. For our price elasticity example, we will shard the indexes and data while replicating the coefficients.

One last point to note is that the leading dimension of sharded arrays in JAX must be divisible by the number of devices in the system. For a 2D array, this means that number of rows must be divisible by the number of devices. Therefore we must write a custom helper function to pad the arrays that we feed into our demand function, otherwise we will receive an error. This computation also must be completed outside the model, otherwise every single iteration of SVI will repeat the padding and slow down the computation. Therefore, instead of passing our `DataFrame` directly into the model, we will pre-compute all required transformations outside and feed that into the model.

### Implementation and Evaluation

The prior version of the model can be viewed in the [previous article](https://towardsdatascience.com/estimating-product-level-price-elasticities-using-hierarchical-bayesian/). In addition to our DGP from the previous example we add in two functions to create a `dict` from our `DataFrame` and to pad the arrays to be divisible by the number of devices. We then move all computations (calculating plate sizes, taking log prices, indexing) to outside the model, then feed it back into a model as a `dict`.

```python
import jax
import jax.numpy as jnp
def pad_array(arr):
    num_devices = jax.device_count()
    remainder = arr.shape[0] % num_devices
    if remainder == 0:
        return arr
    
    pad_size = num_devices - remainder
    padding = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
    
    # Choose appropriate padding value based on data type
    pad_value = -1 if arr.dtype in (jnp.int32, jnp.int64) else -1.0
    return jnp.pad(arr, padding, constant_values=pad_value)

def create_dict(df):
    # Define indexes
    product_idx, unique_product = pd.factorize(df['product'])
    cat_idx, unique_category = pd.factorize(df['category'])
    time_cat_idx, unique_time_cat = pd.factorize(df['cat_by_time'])

    # Convert the price and units series to jax numpy arrays
    log_price = jnp.log(df.price.values)
    outcome = jnp.array(df.units_sold.values, dtype=jnp.int32)

    # Generate mapping
    product_to_category = jnp.array(pd.DataFrame({'product': product_idx, 'category': cat_idx}).drop_duplicates().category.values, dtype=np.int16)
    return {
        'product_idx': pad_array(product_idx),
        'time_cat_idx': pad_array(time_cat_idx),
        'log_price': pad_array(log_price),
        'product_to_category': product_to_category,
        'outcome': outcome,
        'cat_idx': cat_idx,
        'n_obs': outcome.shape[0],
        'n_product': unique_product.shape[0],
        'n_cat': unique_category.shape[0],
        'n_time_cat': unique_time_cat.shape[0],
    }

data_dict = create_dict(df)
data_dict
```
    {'product_idx': Array([    0,     0,     0, ..., 11986, 11986,    -1], dtype=int32),
     'time_cat_idx': Array([   0,    1,    2, ..., 1254, 1255,   -1], dtype=int32),
     'log_price': Array([ 6.629865 ,  6.4426994,  6.4426994, ...,  5.3833475,  5.3286524,
            -1.       ], dtype=float32),
     'product_to_category': Array([0, 1, 2, ..., 8, 8, 7], dtype=int16),
     'outcome': Array([  9,  13,  11, ..., 447, 389, 491], dtype=int32),
     'cat_idx': array([0, 0, 0, ..., 7, 7, 7]),
     'n_obs': 1881959,
     'n_product': 11987,
     'n_cat': 10,
     'n_time_cat': 1570}


After changing the model inputs, we also have to change some components of the model. First, the sizes for each plate is now pre-computed and we can just feed those into the plate creation. To apply data sharding and replication, we will need to add a mesh (an N-dimensional array that determines how data should be split) and define which inputs need to be sharded and which one to be replicated. The `in_spec` variable defines which input argments to be sharded/replicated across the 'batch' dimension defined in our mesh. We then re-define the `calculate_demand` function, making sure that each argument corresponds to the correct `in_spec` order. We use `jax.experimental.shard_map.shard_map` to tell JAX that it should automatically paralleize the computation of our function over the shards, then use the sharded function to calculate demand if the model argument `parallel` is True. Finally, we change the `data_plate` to only take non-padded indexes by including the `ind`, since the size of the original data is stored in the `n_obs` variable of the dictionary. 


```python
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import jax.experimental.shard_map

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam

def model(data_dict, outcome: None, parallel:bool = False):
    # get info from dict
    product_to_category = data_dict['product_to_category']
    product_idx = data_dict['product_idx']
    log_price = data_dict['log_price']
    time_cat_idx = data_dict['time_cat_idx']
    
    # Create the plates to store parameters
    category_plate = numpyro.plate("category", data_dict['n_cat'])
    time_cat_plate = numpyro.plate("time_cat", data_dict['n_time_cat'])
    product_plate = numpyro.plate("product", data_dict['n_product'])
    data_plate = numpyro.plate("data", size=data_dict['n_obs'])

    # DEFINING MODEL PARAMETERS
    global_a = numpyro.sample("global_a", dist.Normal(-2, 1), infer={"reparam": LocScaleReparam()})

    with category_plate:
        category_a = numpyro.sample("category_a", dist.Normal(global_a, 1), infer={"reparam": LocScaleReparam()})

    with product_plate:
        product_a = numpyro.sample("product_a", dist.Normal(category_a[product_to_category], 2), infer={"reparam": LocScaleReparam()})
        product_effect = numpyro.sample("product_effect", dist.Normal(0, 3), infer={"reparam": LocScaleReparam()})

    with time_cat_plate:
        time_cat_effects = numpyro.sample("time_cat_effects", dist.Normal(0, 3), infer={"reparam": LocScaleReparam()})

    # Calculating expected demand
    # Define infomrmation about the device
    devices = np.array(jax.devices())
    num_gpus = len(devices)
    mesh = Mesh(devices, ("batch",))

    # Define the sharding/replicating of input and output
    in_spec=(
        P(),            # product_a: replicate
        P("batch"),     # product_idx: shard
        P("batch"),     # log_price: shard 
        P(),            # time_cat_effects: replicate
        P("batch"),     # time_cat_idx: shard
        P(),            # product_effect: replicate
    )
    out_spec=P("batch") # expected_demand: shard     
    def calculate_demand(
        product_a,
        product_idx,
        log_price,
        time_cat_effects,
        time_cat_idx,
        product_effect,
    ):
        log_demand = product_a[product_idx]*log_price + time_cat_effects[time_cat_idx] + product_effect[product_idx]
        expected_demand = jnp.exp(jnp.clip(log_demand, -4, 20)) # clip for stability and exponentiate 
        return expected_demand
    shard_calc = jax.experimental.shard_map.shard_map(
        calculate_demand,
        mesh=mesh,
        in_specs=in_spec,
        out_specs=out_spec
    )    
    calculate_fn = shard_calc if parallel else calculate_demand
    demand = calculate_fn(
        product_a,
        product_idx,
        log_price,
        time_cat_effects,
        time_cat_idx,
        product_effect,
    )

    with data_plate as ind:
        # Sample observations
        numpyro.sample(
            "obs",
            dist.Poisson(demand[ind]),
            obs=outcome
        )

numpyro.render_model(
    model=model,
    model_kwargs={"data_dict": data_dict,"outcome": data_dict['outcome']},
    render_distributions=True,
    render_params=True,
)

```

<img src="./figures/output_1_0.svg">

### Evaluation
To get access to distributed GPU resources, we run this notebook on a SageMaker Notebook instance in AWS using a G5.24xlarge instance. This G5 instance has 192 vCPUs and 4 NVIDIA A10G GPUs. Since NumPyro gives us a handy progress bar, we will compare the speed of optimization over three different model sizes: running either in parallel across all CPU cores, on a single GPU, or distributed across all 4 GPUs. We will evaluate the expected time it takes to finish one million observations across the three dataset sizes. All datasets will have 156 periods, with increasing number of products from 10k, 100k, and 1 million. The smallest dataset will have 1.56MM observations, and the largest dataset will have 156MM observations. For the optimizer, we use `optax`'s weighted ADAM with an exponentially decaying schedule for the learning rate. When running the SVI algorithm, keep in mind that `Numpyro` takes some time to compile all the code and data, so there's some overhead as the data size and model complexity increases. 

Instead of optimizing over the standard ELBO, we use the `RenyiELBO` loss to implement Renyi's [$\alpha$-divergence](https://arxiv.org/abs/1602.02311). As the default argument, $\alpha=0$ implements the [Importance-Weighted ELBO](https://arxiv.org/abs/1509.00519), giving us a tighter bound and less bias. For the guide, we go with the standard AutoNormal guide that parameterizes a Diagonal Multivariate Normal for the posterior distribution. AutoMultivariateNormal and normalizing flows (AutoBNAFNormal, AutoIAFNormal) all requires $O(n^2)$ memory, which we cannot do on large models. AutoLowRankMultivariateNormal could improve posterior inference and only uses $O(kn)$ memory, where $k$ is the rank hyperparameter. However for this example, we go with the standard formulation.

    100%|██████████| 10000/10000 [00:36<00:00, 277.49it/s, init loss: 131118161920.0000, avg. loss [9501-10000]: 10085247.5700] # Sample progress bar

```python
## SVI
import gc
from numpyro.infer import SVI, autoguide, init_to_median, RenyiELBO
import optax
import matplotlib.pyplot as plt
numpyro.set_platform('gpu') # Tells numpyro/JAX to use GPU as the default device 

rng_key = jax.random.PRNGKey(42)
guide = autoguide.AutoNormal(model)
learning_rate_schedule = optax.exponential_decay(
    init_value=0.01,
    transition_steps=1000,
    decay_rate=0.99,
    staircase = False,
    end_value = 1e-5,
)

# Define the optimizer
optimizer = optax.adamw(learning_rate=learning_rate_schedule)

# Code for running the 4 GPU computations
gc.collect()
jax.clear_caches()
svi = SVI(model, guide, optimizer, loss=RenyiELBO(num_particles=4))
svi_result = svi.run(rng_key, 1_000_000, data_dict, data_dict['outcome'], parallel = True)

# Code for running the 1 GPU computations
gc.collect()
jax.clear_caches()
svi = SVI(model, guide, optimizer, loss=RenyiELBO(num_particles=4))
svi_result = svi.run(rng_key, 1_000_000, data_dict, data_dict['outcome'], parallel = False)

# Code for running the parallel CPU computations (parallel = False) since all CPUs are seen as 1 device 
with jax.default_device(jax.devices('cpu')[0]):
    gc.collect()
    jax.clear_caches()
    svi = SVI(model, guide, optimizer, loss=RenyiELBO(num_particles=4))
    svi_result = svi.run(rng_key, 1_000_000, data_dict, data_dict['outcome'], parallel = False)
```

<div>
  <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; margin: 20px 0;">
    <caption style="font-weight: bold; font-size: 1.2em; margin-bottom: 10px;">Expected Time to Complete 1M Iters (in hours:minutes) [Speedup over CPU]</caption>
    <thead>
      <tr>
        <th>Dataset Size</th>
        <th>CPU (192 cores)</th>
        <th>1 GPU (A10G)</th>
        <th>4 GPUs (A10G)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Small (10K products, 1.56M obs, 21.6k params)</td>
        <td>~22:05</td>
        <td>~0:41 [32.3x]</td>
        <td>~0:21 [63.1x]</td>
      </tr>
      <tr>
        <td>Medium (100K products, 15.6M obs, 201.5k params)</td>
        <td>~202:20</td>
        <td>~6:05 [33.3x]</td>
        <td>~2:14 [90.6x]</td>
      </tr>
      <tr>
        <td>Large (1M products, 156M obs, 2M params)</td>
        <td>~2132:30</td>
        <td>~60:18 [35.4x]</td>
        <td>~20:50 [102.4x]</td>
      </tr>
    </tbody>
  </table>
</div>

<img src="./figures/b2_fig1.png">

As a reference point, we also ran the smallest dataset using the NUTS sampler with 3,000 draws (1,000 burn-in), which would take approximately 20 hours on a 192-core CPU, but does not guarantee convergence. MCMC must also increase the number of draws and burn-in as the posterior space becomes more complex, so proper time estimates for MCMC are tough to measure. For SVI, our findings demonstrate a substantial performance improvement when transitioning from CPU to GPU, with approximately 32-35x speedup depending on dataset size. Scaling from a single GPU to four GPUs yields further significant performance gains, ranging from a 2x speedup for the small dataset to a 2.9x speedup for the large dataset. This indicates that the overhead of distributing computation becomes increasingly justified as problem size grows.

These results suggest that multi-GPU setups are essential for estimating large hierarchical Bayesian models within reasonable timeframes. The performance advantages become even more pronounced with more advanced hardware. For example, in my work application, transitioning from an A10 4-GPU setup to an H100 8-GPU configuration increased inference speed from 5 iterations per second to 260 iterations per second—a 52x speedup! When compared to traditional CPU-based MCMC approaches for large models, the potential acceleration could reach up to 10,000 times, enabling scientists to tackle previously intractable problems.

**Note on Mini-Batch Training:** I have gotten this code working with minibatching, but the speed of the model actually slows down significantly as compared to loading the full dataset on GPU. I assume that there is some loss in creating the indexes for batching, moving data from CPU to GPU, then distributing the data and indexes across GPUs. From what I've seen in practice, the minibatching with 1024 per batch is takes 2-3x longer than the 4 GPU case, and batching with 1048576 per batch takes 8x longer than the 4 GPU case. Therefore, **if the dataset can fit on memory, it is better to not incorporate minibatching**.

### Final Remarks

This guide demonstrates how to dramatically accelerate hierarchical Bayesian models using a combination of SVI and a multi-GPU setup. This approach is up to 102x faster than traditional CPU-based SVI when working with large datasets containing millions of parameters. When combined with the speedup SVI offers over MCMC, we can possibly have performance gains up to 10,000 times. These improvements make previously intractable hierarchical models practical for real-world industrial applications.

This article has several key take-aways. (1) SVI is essential for scale over MCMC, at the expense of accuracy. (2) The benefits of a multi-GPU setup increases substantially as the data becomes larger. (3) The implementation of the code matters, since only by moving all pre-computations outside of the model allows us to achieve this speed. However, while this approach offers significant speed improvements, several key drawbacks still exist. Incorporating mini-batching reduces distributed performance, but might be necessary in practice for datasets that are too large to fit on GPU memory. This problem can be somewhat mitigated by using more advanced GPUs (A100, H100) with 80GB of memory instead of 24GB that the A10G offers. This integration of mini-batching and distributed computing is a promising area for future work. Second, the mean-field assumption in our SVI approach tends to underestimate posterior uncertainty compared to full MCMC, which may impact applications where uncertainty quantification is critical. Other guides can incorporate more complex posterior, but comes at the cost of memory-scaling (usually exponential) and would not be feasible for large datasets. Once I have figured out the best way to correct posterior uncertainty through post-processing, I will also write an article about that...

**Application:** The methods demonstrated in this article opens doors to numerous applications that were previously computationally prohibitive. Marketing teams can now build granular Marketing Mix Models that capture variation across regions and customer profiles and provide localized estimates of channel effectiveness. Financial institutions can implement large-scale Value-at-Risk calculations that model complex dependencies across thousands of securities while capturing segment-specific changes in market behavior. Tech companies can develop hybrid recommendation systems that integrate both collaborative and content-based filtering with Bayesian uncertainty, enabling better exploration-exploitation trade-offs. In macroeconomics, researchers can estimate fully heterogeneous agent (HANK) models that measure how monetary and fiscal policies differentially impact diverse economic actors instead of just using representative agents.

If you have the opportunity to apply this concept in your own work, I'd love to hear about it. Please do not hesitate to reach out with questions, insights, or stories through [my email](mailto:tranderektri@google.com) or [LinkedIn](https://www.linkedin.com/in/derek-tran-ab75ab64/). If you have any feedback on this article, or would like to request another topic in causal inference/machine learning, please also feel free to reach out. Thank you for reading!

**Note**: All images used in this article is generated by the author.