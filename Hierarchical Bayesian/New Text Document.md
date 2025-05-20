# Estimating unit-level price elasticities using Hierarchical Bayesian

In this multi-part series, I will introduce you to hierarchical Bayesian modelling, a flexible modeling approach to automatically combine the results of multiple sub-models with optimal weights determined automatically through Bayesian updating. This article will introduce the concept, implementation, and alternative use cases for this method. 

### The Problem with Traditional Approaches
As an application, imagine that we’re a large grocery store trying to maximize product-level revenue by setting prices. First, we would need to estimate the price elasticity of demand (how responsive demand is to a 1% change in price) using longitudinal data with $N$ products over $T$ periods. Remember that elasticity so defined as:
```math
\beta=\frac{\partial \log{\textrm{Units}}_{it}}{\partial \log \textrm{Price}_{it}}
```
Assuming no confounders, using a standard fixed-effect regression model of log units sold on log price:

```math
\log(\textrm{Units}_{it})= \beta  \log(\textrm{Price})_{it} +\gamma_t+ \delta_i+ \epsilon_{it}
```

Would allow us to recover the average elasticity $\beta$ across all $N$ units, also known as the Average Treatment Effect. This would mean that the store could target an average price level across all products in their store to maximize revenue. If these units have a natural grouping (product categories), we might be able to identify the average elasticity of each product category by running a separate sub-regression using only units from that category. This would mean that the store could target average prices in each product category to maximize revenue in that category. If $T$ is large enough, we might even be able to run a separate regression for each individual unit to recover unit-level elasticity of demand, allowing the store to set prices at the product-level. 

However, data in reality is often not as perfect as we would like it to be. Some products might have no/limited price changes, some products might have only been active for a short time (cold start), or the number of products could be different across categories. Under these real-world restrictions, running separate regressions to identify product elasticity would likely lead to large standard errors or no significant results for many products/categories. However, the hierarchical Bayesian approach allows us to acknowledge differences across groups while still sharing statistical strength among them. With hierarchical Bayesian, it is possible to run one single regression (like the pooled case) while still recovering elasticities at the product level.  


### Understanding Hierarchical Bayesian Models
At its core, hierarchical Bayesian modeling is about recognizing the natural structure in our data. Rather than treating all observations as completely independent or forcing them to follow identical patterns, we acknowledge that observations often cluster into groups, with similarities within groups and differences between them. The "hierarchical" aspect refers to how we organize our parameters in different levels. In its most basic format, we might have:
 - A Global parameter that applies to all data
 - Group-level parameters that apply to specific segments
 - Individual-level parameters that apply to specific observations

We can remove hierarchies as needed, depending on the desired level of pooling. For example, if we think there are no similarities across categories, we could remove the global parameter. If we think that these products have no natural groupings, we could remove the group-level parameters. If we only care about the group-level effect, we can remove the individual-level parameter and have the group-level as our lowest level of observation.


The "Bayesian" aspect refers to how we update our beliefs about these parameters based on observed data, we start with a proposed prior distribution that represent our initial belief of these parameters, then update them iteratively to recover a posterior distributions that incorporate what we've learned from the data. In practice, this means that we use the global-level estimate to inform our group-level estimates, and the group-level parameters to inform the unit-level parameters. Units with a larger number of observations are allowed to deviate more from the mean, while units with a limited number of  observations are pulled closer to the group level means. 


Let's formalize this with our price elasticity example, where we try to estimate unit-level price elasticity:

```math
\log(\textrm{Units}_{it})= \beta  \log(\textrm{Price})_{it} +\gamma_{c(i),t}+ \delta_i+ \epsilon_{it}
```

Where:
 - $\beta_i \sim \textrm{Normal}(\beta_{c\left(i\right)},\sigma_i)$
 - $\beta_{c(i)}\sim \textrm{Normal}(\beta_g,\sigma_{c(i)})$
 - $\beta_g\sim \textrm{Normal}(\mu,\sigma)$

Where $\gamma_{c(i),t}$ is a set of category-by-time dummy variables to capture the average demand of each unique category in each time period. $\delta_i$ are product dummies to capture the time-invariant heterogenous preferences of consumers for each product. This “fixed-effect” formulation is standard and common in many regression-based models to control for unobserved confounders.

We assume that the unit level elasticity $\beta_i$ is drawn from a normal distribution centered around the category-level elasticity average $\beta_{c(i)}$, and the category-level average is drawn from a global elasticity $\beta_g$. For the spread of the distribution, we can assume a hierarchical structure for that too, but in this example, we just set priors for them individually for simplicity. One example of our prior beliefs can be: $\{ \mu= -1.5, \sigma= .5, \sigma_{c(i)}=.4, \sigma_i=.3\}$. This formulation assumes that the global elasticity is slighty elastic, 99.7% of the elasticities fall between -3 and 0, and we set increasingly tighter priors for the lower levels. To test these initial parameters, we would do a prior predictive check (not covered in this blog post) to see whether our prior beliefs can recover the data that we observe. 

This hierarchical structure allows information to flow between products in the same category and even across categories. If a particular product has limited price variation data, its elasticity estimate will be pulled toward the category average. Similarly, categories with fewer products will be influenced more by the store-level average. The beauty of this approach is that the degree of "pooling" happens automatically based on the data. Products with lots of price variation will maintain estimates closer to their individual data patterns, while those with sparse data will borrow more strength from their group.

## Implementation

### Data Generating Process

In this section, we implement the above model using the Numpyro package in Python, a lightweight probabilistic programming language powered by JAX for autograd and JIT compilation to GPU/TPU/CPU. We simulate sales data where demand follows a log-linear relationship with price, structured across three levels: global, category, and product-specific elasticities. The model generates product demand using a Poisson distribution where the log rate parameter combines baseline demand factors (including category-specific time trends and volatility) with a price effect. The price effect comes into the demand based on a unit-level loading, drawn around a category-level parameter, drawn from a global parameter. 

```python
import numpy as np
import pandas as pd

def generate_price_elasticity_data(N: int = 1000,
                                   C: int = 10,
                                   T: int = 50,
                                   price_change_prob: float = 0.2,
                                   seed = 42) -> pd.DataFrame:
    """
    Generate synthetic data for price elasticity of demand analysis.
    Data is generated by
    y_it = 
    
    Parameters:
    -----------
    N : int
        Number of products (default: 1000)
    C : int
        Number of categories (default: 10)
    T : int
        Number of time periods (default: 50)
    price_change_prob : float
        Probability of price change in each period (default: 0.2)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the synthetic data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Global elasticity
    global_a = -2

    # Category demand and trends
    category_base_demand = np.random.uniform(1000, 10000, C)
    category_time_trends = np.random.uniform(-0.01, 0.01, C)
    category_volatility = np.random.uniform(0.01, 0.05, C)  # Random volatility for each category
    category_demand_paths = np.zeros((C, T))
    category_demand_paths[:, 0] = 1.0
    shocks = np.random.normal(0, 1, (C, T-1)) * category_volatility[:, np.newaxis]
    trends = category_time_trends[:, np.newaxis] * np.ones((C, T-1))
    cumulative_effects = np.cumsum(trends + shocks, axis=1)
    category_demand_paths[:, 1:] = category_demand_paths[:, 0:1] + cumulative_effects
    
    # category elasticity
    category_a = np.random.normal(global_a, .5, size=C)
    category_a = np.clip(category_a, -5, -.1)  # Keep values in reasonable range
    product_categories = np.random.randint(0, C, N)
    
    # product elasticities - perturb from category level
    product_a = category_a[product_categories]
    product_a += np.random.normal(0, .3, size=N)
    product_a = np.clip(product_a, -5, -.1)
    
    # Initial prices for each product
    initial_prices = np.random.uniform(10, 1000, N)
    prices = np.zeros((N, T))
    prices[:, 0] = initial_prices
    
    # Generate random values and whether prices changed
    random_values = np.random.rand(N, T-1)
    change_mask = random_values < price_change_prob
    
    # Generate change factors for (-10% to +10%)
    change_factors = 1 + np.random.uniform(-0.1, 0.1, size=(N, T-1))
    
    # Create a matrix to hold multipliers
    multipliers = np.ones((N, T-1))
    
    # Apply change factors only where changes should occur
    multipliers[change_mask] = change_factors[change_mask]
    
    # Apply the changes cumulatively to propagate prices
    for t in range(1, T):
        prices[:, t] = prices[:, t-1] * multipliers[:, t-1]
    
    # Generate product-specific multipliers
    product_multipliers = np.random.lognormal(0, 0.5, size=N)
    # Get time effects for each product's category (shape: N x T)
    time_effects = category_demand_paths[product_categories][:, np.newaxis, :].squeeze(1)
    
    # Ensure time effects don't go negative
    time_effects = np.maximum(0.1, time_effects)
    
    # Generate period noise for all products and time periods
    period_noise = 1 + np.random.uniform(-0.05, 0.05, size=(N, T))
    
    # Get category base demand for each product
    category_base = category_base_demand[product_categories]
    
    # Calculate base demand
    base_demand = (category_base[:, np.newaxis] *
                   product_multipliers[:, np.newaxis] *
                   time_effects *
                   period_noise)

    # log demand
    alpha_ijt = np.log(base_demand)

    # log price
    log_prices = np.log(prices)

    # log expected demand
    log_lambda = alpha_ijt + product_a[:, np.newaxis] * log_prices  # Shape: (N, T)

    # Convert back from log space to get rate parameters
    lambda_vals = np.exp(log_lambda)  # Shape: , T)

    # Generate units sold
    units_sold = np.random.poisson(lambda_vals)  # Shape: (N, T)
    
    # Create index arrays for all combinations of products and time periods
    product_indices, time_indices = np.meshgrid(np.arange(N), np.arange(T), indexing='ij')
    product_indices = product_indices.flatten()
    time_indices = time_indices.flatten()
    
    # Get categories for all products
    categories = product_categories[product_indices]
    
    # Get all prices and units sold
    all_prices = np.round(prices[product_indices, time_indices], 2)
    all_units_sold = units_sold[product_indices, time_indices]
    
    # Calculate elasticities
    product_elasticity = product_a[product_indices]
    category_elasticity = category_a[categories]
    global_elasticity = global_a
    
    # Create the DataFrame directly
    df = pd.DataFrame({
        'product': product_indices,
        'category': categories,
        'time_period': time_indices,
        'price': all_prices,
        'units_sold': all_units_sold,
        'product_elasticity': product_elasticity,
        'category_elasticity': category_elasticity,
        'global_elasticity': global_elasticity
    })
    return df

# Keep only units with >X sales
def filter_dataframe(df, min_units = 100):
    temp = df[['product','units_sold']].groupby('product').sum().reset_index()
    unit_filter = temp[temp.units_sold>min_units]['product'].unique()
    filtered_df = df[df['product'].isin(unit_filter)].copy()

    # Provide a summary of the filtering
    original_product_count = df['product'].nunique()
    remaining_product_count = filtered_df['product'].nunique()
    filtered_out = original_product_count - remaining_product_count
    
    print(f"Filtering summary:")
    print(f"- Original number of products: {original_product_count}")
    print(f"- Products with > {min_units} units: {remaining_product_count}")
    print(f"- Products filtered out: {filtered_out} ({filtered_out/original_product_count:.1%})")
    
    return filtered_df


df = generate_price_elasticity_data(N = 20000, T = 100, price_change_prob=.5, seed=42)
df = filter_dataframe(df)
df.loc[:,'cat_by_time'] = df['category'].astype(str) + '-' + df['time_period'].astype(str)
df.head()
```
