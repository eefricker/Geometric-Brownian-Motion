import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

static_coefficients = {'mu':.03,
                       'sigma':.3}

def make_month_grid(close):

    month_starts = []
    for date in pd.date_range(dt.datetime(2015,1,1),dt.datetime(2025,4,1),freq='MS'):
        while date not in close.index:
            date = date+dt.timedelta(days=1)
        month_starts.append(date)
    month_starts.append(dt.datetime(2025,5,1))

    return month_starts

def isolate_data(close,ticker,month_starts,month_select):

    prev_data_start = month_starts[month_select-3]
    prev_data_end = month_starts[month_select]
    prev_data = close.loc[(close.index >= prev_data_start)&
                          (close.index < prev_data_end),ticker]

    time_start = month_starts[month_select]
    time_end = month_starts[month_select+1]
    window = close.loc[(close.index >= time_start)&
                   (close.index < time_end),ticker]

    return prev_data, window
    
def estimate_normal_coefficients(prev_data):

	# MLE on normalized log returns to match a standard normal
    log_returns = np.log(prev_data / prev_data.shift(1)).dropna()
    sigma = log_returns.std() * np.sqrt(252)
    mu = log_returns.mean() * 252 + 0.5 * sigma**2
    
    coefficients = {'mu':mu,
                    'sigma':sigma}

    return coefficients
    
def standardize_with_normal_coefficients(window,coefficients):

    log_returns = np.log(window / window.shift(1)).dropna()

    mu = coefficients['mu']
    sigma = coefficients['sigma']
    delta_t = (1/252)
    
    full_mu = pd.Series(mu,index=log_returns.index)
    full_sigma = pd.Series(sigma,index=log_returns.index)
    
    normalized_returns = (log_returns - (mu - 0.5 * sigma**2) * delta_t) / (sigma * np.sqrt(delta_t))
    returns_cdf = pd.Series(stats.norm.cdf(normalized_returns),index=normalized_returns.index)
    
    standardize_return = pd.concat([log_returns,full_mu,full_sigma,
									normalized_returns,returns_cdf.T],axis=1)

    ticker = window.name
    standardize_return.columns = pd.MultiIndex.from_tuples([(ticker,'log returns',),
														  (ticker,'mu'),
														  (ticker,'sigma'),
														  (ticker,'standardized returns'),
                                                          (ticker,'cdf')])
    
    return standardize_return

def simulate_paths(window,coefficients,rng):

    start_price = window.iloc[0]
    sim_days = window.shape[0]-1
    sims = int(5e4)
    
    delta_t = (1/252)
    mu = coefficients['mu']
    sigma = coefficients['sigma']
    drift = (mu - sigma**2/2)*delta_t
    drift = np.full((sim_days,1),drift)
    drift = np.cumsum(drift,axis=0)
    
    Z = rng.standard_normal((sim_days,sims))
    diffusion = sigma*np.sqrt(delta_t)*Z
    diffusion = np.cumsum(diffusion,axis=0)
    
    bands = compute_path_percentiles(drift,diffusion)
    
    paths = np.exp(drift+diffusion)
    bands = np.exp(drift+bands)
    drift = np.exp(drift)
    
    bands = start_price*np.concatenate([np.full((1, 4), 1), bands], axis=0)
    paths = start_price*np.concatenate([np.full((1, sims), 1), paths], axis=0)
    drift = start_price*np.concatenate([np.full((1, 1), 1), drift], axis=0)
    
    bands = pd.DataFrame(bands,index=window.index,columns=['p05','p25','p75','p95'])
    paths = pd.DataFrame(paths,index=window.index)
    drift = pd.DataFrame(drift,index=window.index)

    return drift, bands, paths

def compute_path_percentiles(drift,diffusion):
    
    # Path percentile (eg. what bound contains 90% of paths up until this point) is not unique!
    # Can enforce ~uniqueness by specifiying the shape, here we use a Brownian scaled shape
    # b(t) =  S_0 * \exp(drift + constant \sqrt{t})
    
    sqrt_t = np.sqrt(np.arange(1, len(diffusion) + 1)).reshape(-1, 1)
    normalized_diffusion = diffusion/sqrt_t
    
    normalized_min = normalized_diffusion.min(axis=0)
    normalized_max = normalized_diffusion.max(axis=0)
    
    k_lower = np.percentile(normalized_min, [5, 25])
    k_upper = np.percentile(normalized_max, [75, 95])
    k_all = np.concatenate([k_lower,k_upper],axis=0).reshape(1,4)
    
    bands = k_all*sqrt_t
    
    return bands
    
def make_percentile_dict(ticker,time_start,window,bands):

    percentile_dict = {'ticker':ticker,
                       'path_start':time_start}
    for testMin in ['p05','p25']:
        percentile_dict[testMin] = int(bool((window < bands[testMin]).sum()))
    for testMax in ['p75','p95']:
        percentile_dict[testMax] = int(not bool((window > bands[testMax]).sum()))

    return percentile_dict

def plot_bands_vs_actual(window,drift,bands,ticker,paths):
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # # Plotting all paths is time hungry and unnecessary
    plot_paths = paths.iloc[:,1:1001]
    ax.plot(plot_paths.index, plot_paths.values, 
            color='steelblue', alpha=0.01, linewidth=0.5);
    
    ax.fill_between(bands.index, bands.p05, bands.p95,
                    color='steelblue', alpha=0.15, label='5th–95th percentile (pathwise min/max)')
    ax.fill_between(bands.index, bands.p25, bands.p75,
                    color='steelblue', alpha=0.25, label='25th–75th percentile (pathwise min/max)')
    
    ax.plot(drift.index, drift.values, 
            color='orange', linewidth=2, linestyle='--', 
            label='Drift only', zorder=5);
    
    ax.plot(window.index, window.values, 
            color='black', linewidth=2.5, 
            label=ticker, zorder=10);
    
    
    ax.set_xlabel('Date');
    ax.set_ylabel('Price');
    ax.set_title('Actual vs. Simulated Price Paths (GBM)');
    ax.legend(loc='best');
    ax.grid(True, alpha=0.3);
    plt.xticks(rotation=45);
    plt.tight_layout();
    plt.show()
    
def qq_per_ticker(std_residuals, ticker=None, ax=None):

    z = pd.Series(std_residuals).dropna().values
    n = len(z)

    if ax is None:
        fig, (ax_norm, ax_t) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        ax_norm, ax_t = ax
        fig = ax_norm.figure

    # ---- Gaussian Q-Q ----
    stats.probplot(z, dist='norm', plot=ax_norm)
    ax_norm.set_title(f'Q-Q: {ticker or "z"} vs N(0,1)')
    ax_norm.grid(True, alpha=0.3)

    # ---- Student-t Q-Q ----
    nu_hat = None
    
    # MLE for nu with location/scale free; we rescale to unit variance after
    # so the comparison is against a STANDARDIZED t.
    nu_hat, loc_hat, scale_hat = stats.t.fit(z)

    # Standardize residuals to unit variance under the fitted t
    # A t_nu RV has variance nu/(nu-2) for nu>2; rescale so theoretical var = 1
    if nu_hat > 2:
        t_var = nu_hat / (nu_hat - 2)
        # Use the fitted location/scale to standardize, then plot against
        # the unit-variance t (scale = 1/sqrt(t_var))
        z_std = (z - loc_hat) / scale_hat
        stats.probplot(z_std, dist=stats.t, sparams=(nu_hat,), plot=ax_t)
        ax_t.set_title(f'Q-Q: {ticker or "z"} vs t({nu_hat:.2f})')
    else:
        ax_t.text(0.5, 0.5, f'Fitted nu={nu_hat:.2f} <= 2\n(infinite variance)',
                  ha='center', va='center', transform=ax_t.transAxes)
        ax_t.set_title(f'Q-Q: {ticker or "z"} vs t (degenerate)')

    ax_t.grid(True, alpha=0.3)

    # ---- AD vs N(0,1) on raw z ----
    ad_norm = stats.anderson(z, dist='norm',method='interpolate')
    
    # ---- AD via PIT through fitted t ----
    # If z really is t(nu_hat) with the fitted loc/scale, then
    # u = F_t(z) is Uniform(0,1) and Phi^{-1}(u) is N(0,1).
    ad_t_pit = None
    if nu_hat > 2:
        u = stats.t.cdf((z - loc_hat) / scale_hat, df=nu_hat)
        u = np.clip(u, 1e-7, 1 - 1e-7)         # guard the tails before ppf
        z_pit = stats.norm.ppf(u)
        ad_t_pit = stats.anderson(z_pit, dist='norm',method='interpolate')

    # ---- Diagnostics ----
    diag = {'ticker': ticker,
            'n': n,
            'ad_norm': ad_norm.statistic,
            'ad_norm_p': ad_norm.pvalue,
            'ad_t_pit': ad_t_pit.statistic if ad_t_pit is not None else np.nan,
            'ad_t_pit_p': ad_t_pit.pvalue if ad_t_pit is not None else np.nan}

    plt.tight_layout()
    return diag, (ax_norm, ax_t)


def qq_panel(tickers,window_results):
    
    n_tickers = len(tickers)
    ncols=2
    nrows = int(np.ceil(n_tickers / ncols))

    # Each ticker takes 2 axes side-by-side
    fig, axes = plt.subplots(nrows, ncols * 2,
                              figsize=(7 * ncols, 5 * nrows),
                              squeeze=False)

    diagnostics = []
    for idx, ticker in enumerate(tickers):
        row = idx // ncols
        col = (idx % ncols) * 2
        ax_pair = (axes[row, col], axes[row, col + 1])

        diag, _ = qq_per_ticker(window_results[(ticker,'standardized returns')],
                                 ticker=ticker,
                                 ax=ax_pair)
        diagnostics.append(diag)

    # Hide any empty axes
    for idx in range(n_tickers, nrows * ncols):
        row = idx // ncols
        col = (idx % ncols) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(diagnostics).set_index('ticker')

def ljungbox_test(my_picks,window_results):
    
    # Test for lagged/clustered relationship with returns, anything below .01 suggests clustering
    full_ljungbox = []
    for ticker in my_picks:
        z = window_results[(ticker,'standardized returns')].dropna()
        lb = acorr_ljungbox(z**2, lags=[5, 10, 20], return_df=True)
        lb['Rejected?'] = (lb['lb_pvalue']<.05).astype(int)
        lb['lb_pvalue'] = round(lb['lb_pvalue'],2)
        lb.columns = pd.MultiIndex.from_tuples([(col,ticker) for col in lb.columns])
        lb.index.name = 'Lag Parameter'
        full_ljungbox.append(lb)
    
    full_ljungbox = pd.concat(full_ljungbox,axis=1)
    full_ljungbox = full_ljungbox[sorted(full_ljungbox.columns)]
    full_ljungbox = full_ljungbox[[col for col in full_ljungbox if col[0]!='lb_stat']]
    
    return full_ljungbox

def fit_garch(close,ticker,month_starts,month_select):
    
    fit_geq_date = month_starts[month_select-24]
    fit_less_date = month_starts[month_select]    
    fit_data = close.loc[(close.index >= fit_geq_date) & 
                         (close.index < fit_less_date), ticker]
    
    # arch convention: returns in percent for numerical stability
    returns_pct = 100 * np.log(fit_data / fit_data.shift(1)).dropna()
    
    model = arch_model(returns_pct, 
                       mean='Constant',
                       vol='GARCH', p=1, q=1, 
                       dist='t')
    result = model.fit(disp='off')
    
    coefficients = {
        'mu':    result.params['mu'],
        'omega': result.params['omega'],
        'alpha': result.params['alpha[1]'],
        'beta':  result.params['beta[1]'],
        'nu':    result.params['nu'],
        'fit_geq_date': fit_geq_date,
        'fit_less_date':   fit_less_date,
    }
    
    return coefficients

def monthly_garch(coefficients,prev_data,window):

    omega = coefficients['omega']
    alpha = coefficients['alpha']
    beta  = coefficients['beta']
    mu    = coefficients['mu']
    
    full_prices = pd.concat([prev_data, window])
    full_returns_pct = 100 * np.log(full_prices / full_prices.shift(1)).dropna()
    full_returns_pct.name = 'percent returns'
    
    # Seed at unconditional variance — gets washed out within warm-up
    sigma2_seed = full_returns_pct.loc[prev_data.index[1:]].var()
    
    sigma2_prev = sigma2_seed
    eps_prev_sq = sigma2_seed
    
    # Run the recursion
    sigma2 = np.empty(len(full_returns_pct))
    for i, r in enumerate(full_returns_pct.values):
        sigma2[i] = omega + alpha * eps_prev_sq + beta * sigma2_prev
        eps_prev_sq = (r - mu) ** 2
        sigma2_prev = sigma2[i]
    
    sigma_full = pd.Series(np.sqrt(sigma2), index=full_returns_pct.index)
    sigma_full.name = 'garch sigma'
    
    window_result = pd.concat([full_returns_pct,sigma_full],axis=1)
    window_result['mu'] = mu
    window_result['standardized returns'] = (window_result[full_returns_pct.name]-mu)/window_result[sigma_full.name]
    window_result.columns = pd.MultiIndex.from_tuples([(full_prices.name,col) for col in window_result.columns])
    window_result = window_result.loc[window.index]
    
    return window_result
    
def simulate_garch_paths(window, coefficients, prev_data, rng):
    
    mu    = coefficients['mu']
    omega = coefficients['omega']
    alpha = coefficients['alpha']
    beta  = coefficients['beta']
    nu    = coefficients['nu']

    start_price = window.iloc[0]
    sim_days = window.shape[0] - 1
    sims = int(5e4)

    # --- seed sigma^2_0 by running the GARCH recursion through prev_data ---
    # This mirrors what monthly_garch does so the simulator picks up where
    # the in-sample fit left off.
    warm_returns = 100 * np.log(prev_data / prev_data.shift(1)).dropna().values
    sigma2_prev = warm_returns.var()           # unconditional seed
    eps_prev_sq = sigma2_prev
    for r in warm_returns:
        sigma2_t = omega + alpha * eps_prev_sq + beta * sigma2_prev
        eps_prev_sq = (r - mu) ** 2
        sigma2_prev = sigma2_t
    sigma2_0 = sigma2_prev   # state at the bar BEFORE the first sim step

    # --- simulate ---
    # Standardized t: variance must equal 1, so scale draws by sqrt((nu-2)/nu)
    t_scale = np.sqrt((nu - 2) / nu)
    Z = stats.t.rvs(df=nu, size=(sim_days, sims), random_state=rng) * t_scale

    # Per-path GARCH recursion. Vectorized across `sims`.
    sigma2 = np.empty((sim_days, sims))
    returns_pct = np.empty((sim_days, sims))

    sigma2_t = np.full(sims, sigma2_0)
    eps_sq_prev = np.full(sims, sigma2_0)
    sigma2_prev = np.full(sims, sigma2_0)

    for t in range(sim_days):
        sigma2_t = omega + alpha * eps_sq_prev + beta * sigma2_prev
        sigma2[t] = sigma2_t
        r_t = mu + np.sqrt(sigma2_t) * Z[t]
        returns_pct[t] = r_t
        eps_sq_prev = (r_t - mu) ** 2
        sigma2_prev = sigma2_t

    # arch package convention: returns are in percent
    log_returns = returns_pct / 100.0
    log_price = np.cumsum(log_returns, axis=0)
    paths = start_price * np.exp(
        np.concatenate([np.zeros((1, sims)), log_price], axis=0)
    )

    # --- bands: empirical percentiles at each t, across paths ---
    band_pcts = np.percentile(paths, [5, 25, 75, 95], axis=1).T
    bands = pd.DataFrame(band_pcts, index=window.index,
                         columns=['p05', 'p25', 'p75', 'p95'])

    # --- "expected" path: median of sims (no clean closed form under GARCH-t) ---
    drift = pd.DataFrame(np.median(paths, axis=1),
                         index=window.index, columns=[0])

    paths = pd.DataFrame(paths, index=window.index)
    
    return drift, bands, paths, sigma2


