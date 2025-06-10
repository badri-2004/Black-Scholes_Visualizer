import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress matplotlib warnings about tight_layout
import warnings
warnings.filterwarnings("ignore", message="The figure layout has not been constrained.")

# Function to calculate Black-Scholes call price
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Function to calculate Black-Scholes put price
def black_scholes_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

st.set_page_config(page_title="Black-Scholes Options Pricing", layout="wide")

st.title("💰 Black-Scholes Options Pricing Model")
st.markdown("""
    This application calculates the theoretical price of European call and put options
    using the Black-Scholes model and visualizes parameter sensitivity through heatmaps.
    """)

st.sidebar.header("Global Option Parameters (Adjust to see impact on all calculations)")

# Global input values in the sidebar. These will drive the "Top Prices" and default heatmap ranges.
S_global = st.sidebar.number_input("Current Stock Price (S)", min_value=0.01, value=100.0, step=1.0, key="S_global")
K_global = st.sidebar.number_input("Option Strike Price (K)", min_value=0.01, value=100.0, step=1.0, key="K_global")
T_global = st.sidebar.number_input("Time to Expiration (Years) (T)", min_value=0.01, value=1.0, step=0.1, key="T_global")
r_global = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.1, value=0.05, step=0.001, format="%.3f", key="r_global")
sigma_global = st.sidebar.slider("Volatility (sigma)", min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f", key="sigma_global")

# Error handling for global parameters
if T_global <= 0:
    st.error("Time to Expiration (T) must be greater than 0.")
    st.stop()
if sigma_global <= 0:
    st.error("Volatility (sigma) must be greater than 0.")
    st.stop()

st.header("Current Option Prices (based on Global Parameters)")

# Calculate and display prices based on the GLOBAL parameters
call_price_current = black_scholes_call_price(S_global, K_global, T_global, r_global, sigma_global)
put_price_current = black_scholes_put_price(S_global, K_global, T_global, r_global, sigma_global)

col_top_call, col_top_put = st.columns(2)
with col_top_call:
    st.success(f"**European Call Option Price:** ${call_price_current:,.2f}")
    st.markdown(f"Parameters: S={S_global:.2f}, K={K_global:.2f}, T={T_global:.2f}, r={r_global:.3f}, $\sigma$={sigma_global:.2f}")
with col_top_put:
    st.info(f"**European Put Option Price:** ${put_price_current:,.2f}")
    st.markdown(f"Parameters: S={S_global:.2f}, K={K_global:.2f}, T={T_global:.2f}, r={r_global:.3f}, $\sigma$={sigma_global:.2f}")


st.header("Option Price Sensitivity Heatmaps")
st.markdown("""
    Explore how Call and Put option prices change across various parameter combinations.
    Adjust the sliders within each section to fine-tune the parameters for that specific plot.
    """)

# Number of points for heatmap resolution and tick display
num_points = 15 # Coarser grid
tick_interval_ratio = 3 # Display about num_points / 3 ticks (~5 ticks for 15 points)

# --- Helper function for plotting heatmaps (data generation happens outside this now) ---
def plot_heatmap(matrix, x_labels, y_labels, x_title, y_title, plot_title, cmap, current_price_label):
    fig, ax = plt.subplots(figsize=(5, 4)) # Even smaller figure size
    sns.heatmap(matrix, annot=False, cmap=cmap,
                cbar_kws={'label': current_price_label}, ax=ax,
                vmin=np.min(matrix) * 0.9, vmax=np.max(matrix) * 1.1) # Consistent color scale

    x_ticks_display = np.arange(0, num_points, max(1, num_points // tick_interval_ratio))
    y_ticks_display = np.arange(0, num_points, max(1, num_points // tick_interval_ratio))

    ax.set_xticks(x_ticks_display)
    ax.set_xticklabels([x_labels[int(i)] for i in x_ticks_display], rotation=45, ha='right', fontsize=6)
    ax.set_yticks(y_ticks_display)
    ax.set_yticklabels([y_labels[int(i)] for i in y_ticks_display], rotation=0, fontsize=6)

    ax.set_xlabel(x_title, fontsize=8)
    ax.set_ylabel(y_title, fontsize=8)
    ax.set_title(plot_title, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

# --- Call Option Heatmaps ---
st.subheader("European Call Option Sensitivity")

# Call: S vs Sigma
with st.expander("Call Price: Stock Price (S) vs. Volatility ($\sigma$)", expanded=True):
    col_plot, col_sliders = st.columns([2.5, 1])

    with col_sliders: # Define sliders first so their values are available for plot generation
        st.markdown("#### Adjust Parameters")
        # Initialize sliders with global values for the fixed parameters
        # The variables being varied in the heatmap (S and sigma here) will use ranges based on global,
        # but the *fixed* ones (K, T, r) will take their current value from these local sliders.
        k_fixed_for_s_sigma = st.slider("Strike Price (K)", min_value=max(0.01, K_global * 0.7), max_value=K_global * 1.3, value=K_global, step=K_global * 0.01, format="%.2f", key="K_call_s_sigma_fix")
        t_fixed_for_s_sigma = st.slider("Time to Expiration (T)", min_value=max(0.01, T_global * 0.7), max_value=T_global * 1.3, value=T_global, step=T_global * 0.05, format="%.2f", key="T_call_s_sigma_fix")
        r_fixed_for_s_sigma = st.slider("Risk-Free Rate (r)", min_value=max(0.0, r_global * 0.7), max_value=r_global * 1.3, value=r_global, step=0.001, format="%.3f", key="r_call_s_sigma_fix")

        # These sliders are for the *single point* price display
        s_adj_for_point = st.slider("Stock Price (S) for point", min_value=max(0.01, S_global * 0.7), max_value=S_global * 1.3, value=S_global, step=S_global * 0.01, format="%.2f", key="S_call_s_sigma_point")
        sigma_adj_for_point = st.slider("Volatility ($\sigma$) for point", min_value=max(0.01, sigma_global * 0.7), max_value=sigma_global * 1.3, value=sigma_global, step=0.01, format="%.2f", key="sigma_call_s_sigma_point")

    with col_plot:
        S_range_hp = S_global * 0.3
        sig_range_hp = sigma_global * 0.3
        stock_prices_hp = np.linspace(max(0.01, S_global - S_range_hp), S_global + S_range_hp, num_points)
        sigmas_hp = np.linspace(max(0.01, sigma_global - sig_range_hp), sigma_global + sig_range_hp, num_points)

        call_matrix = np.zeros((num_points, num_points))
        for i, s_val in enumerate(stock_prices_hp):
            for j, sig_val in enumerate(sigmas_hp):
                # Use fixed parameters from the local sliders for heatmap generation
                call_matrix[i, j] = black_scholes_call_price(s_val, k_fixed_for_s_sigma, t_fixed_for_s_sigma, r_fixed_for_s_sigma, sig_val)

        plot_heatmap(call_matrix,
                     [f'{x:.2f}' for x in sigmas_hp], [f'{y:.2f}' for y in stock_prices_hp],
                     "Volatility ($\sigma$)", "Stock Price (S)",
                     "Call Price: S vs. $\sigma$", 'RdYlGn', "Call Price")

    with col_sliders: # Re-add the metric after the plot, using the point sliders
        current_call = black_scholes_call_price(s_adj_for_point, k_fixed_for_s_sigma, t_fixed_for_s_sigma, r_fixed_for_s_sigma, sigma_adj_for_point)
        st.metric(label="Current Call Price", value=f"${current_call:,.2f}")
        st.markdown(f"**Point Parameters:** S={s_adj_for_point:.2f}, $\sigma$={sigma_adj_for_point:.2f}")
        st.markdown(f"**Fixed for Heatmap:** K={k_fixed_for_s_sigma:.2f}, T={t_fixed_for_s_sigma:.2f}, r={r_fixed_for_s_sigma:.3f}")

# Call: K vs T
with st.expander("Call Price: Strike Price (K) vs. Time to Expiration (T)"):
    col_plot, col_sliders = st.columns([2.5, 1])

    with col_sliders:
        st.markdown("#### Adjust Parameters")
        s_fixed_for_k_t = st.slider("Stock Price (S)", min_value=max(0.01, S_global * 0.7), max_value=S_global * 1.3, value=S_global, step=S_global * 0.01, format="%.2f", key="S_call_k_t_fix")
        r_fixed_for_k_t = st.slider("Risk-Free Rate (r)", min_value=max(0.0, r_global * 0.7), max_value=r_global * 1.3, value=r_global, step=0.001, format="%.3f", key="r_call_k_t_fix")
        sigma_fixed_for_k_t = st.slider("Volatility ($\sigma$)", min_value=max(0.01, sigma_global * 0.7), max_value=sigma_global * 1.3, value=sigma_global, step=0.01, format="%.2f", key="sigma_call_k_t_fix")

        k_adj_for_point = st.slider("Strike Price (K) for point", min_value=max(0.01, K_global * 0.7), max_value=K_global * 1.3, value=K_global, step=K_global * 0.01, format="%.2f", key="K_call_k_t_point")
        t_adj_for_point = st.slider("Time to Expiration (T) for point", min_value=max(0.01, T_global * 0.7), max_value=T_global * 1.3, value=T_global, step=T_global * 0.05, format="%.2f", key="T_call_k_t_point")

    with col_plot:
        K_range_hp = K_global * 0.3
        T_range_hp = T_global * 0.3
        strike_prices_hp = np.linspace(max(0.01, K_global - K_range_hp), K_global + K_range_hp, num_points)
        times_hp = np.linspace(max(0.01, T_global - T_range_hp), T_global + T_range_hp, num_points)

        call_matrix = np.zeros((num_points, num_points))
        for i, k_val in enumerate(strike_prices_hp):
            for j, t_val in enumerate(times_hp):
                call_matrix[i, j] = black_scholes_call_price(s_fixed_for_k_t, k_val, t_val, r_fixed_for_k_t, sigma_fixed_for_k_t)

        plot_heatmap(call_matrix,
                     [f'{x:.2f}' for x in times_hp], [f'{y:.2f}' for y in strike_prices_hp],
                     "Time to Expiration (T)", "Strike Price (K)",
                     "Call Price: K vs. T", 'RdYlGn', "Call Price")

    with col_sliders:
        current_call = black_scholes_call_price(s_fixed_for_k_t, k_adj_for_point, t_adj_for_point, r_fixed_for_k_t, sigma_fixed_for_k_t)
        st.metric(label="Current Call Price", value=f"${current_call:,.2f}")
        st.markdown(f"**Point Parameters:** K={k_adj_for_point:.2f}, T={t_adj_for_point:.2f}")
        st.markdown(f"**Fixed for Heatmap:** S={s_fixed_for_k_t:.2f}, r={r_fixed_for_k_t:.3f}, $\sigma$={sigma_fixed_for_k_t:.2f}")

# Call: S vs R
with st.expander("Call Price: Stock Price (S) vs. Risk-Free Rate (r)"):
    col_plot, col_sliders = st.columns([2.5, 1])

    with col_sliders:
        st.markdown("#### Adjust Parameters")
        k_fixed_for_s_r = st.slider("Strike Price (K)", min_value=max(0.01, K_global * 0.7), max_value=K_global * 1.3, value=K_global, step=K_global * 0.01, format="%.2f", key="K_call_s_r_fix")
        t_fixed_for_s_r = st.slider("Time to Expiration (T)", min_value=max(0.01, T_global * 0.7), max_value=T_global * 1.3, value=T_global, step=T_global * 0.05, format="%.2f", key="T_call_s_r_fix")
        sigma_fixed_for_s_r = st.slider("Volatility ($\sigma$)", min_value=max(0.01, sigma_global * 0.7), max_value=sigma_global * 1.3, value=sigma_global, step=0.01, format="%.2f", key="sigma_call_s_r_fix")

        s_adj_for_point = st.slider("Stock Price (S) for point", min_value=max(0.01, S_global * 0.7), max_value=S_global * 1.3, value=S_global, step=S_global * 0.01, format="%.2f", key="S_call_s_r_point")
        r_adj_for_point = st.slider("Risk-Free Rate (r) for point", min_value=max(0.0, r_global * 0.7), max_value=r_global * 1.3, value=r_global, step=0.001, format="%.3f", key="r_call_s_r_point")

    with col_plot:
        S_range_hp = S_global * 0.3
        r_range_hp = r_global * 0.5 if r_global > 0 else 0.02
        stock_prices_hp = np.linspace(max(0.01, S_global - S_range_hp), S_global + S_range_hp, num_points)
        r_rates_hp = np.linspace(max(0.0, r_global - r_range_hp), r_global + r_range_hp, num_points)

        call_matrix = np.zeros((num_points, num_points))
        for i, s_val in enumerate(stock_prices_hp):
            for j, r_val in enumerate(r_rates_hp):
                call_matrix[i, j] = black_scholes_call_price(s_val, k_fixed_for_s_r, t_fixed_for_s_r, r_val, sigma_fixed_for_s_r)

        plot_heatmap(call_matrix,
                     [f'{x:.3f}' for x in r_rates_hp], [f'{y:.2f}' for y in stock_prices_hp],
                     "Risk-Free Rate (r)", "Stock Price (S)",
                     "Call Price: S vs. r", 'RdYlGn', "Call Price")

    with col_sliders:
        current_call = black_scholes_call_price(s_adj_for_point, k_fixed_for_s_r, t_fixed_for_s_r, r_adj_for_point, sigma_fixed_for_s_r)
        st.metric(label="Current Call Price", value=f"${current_call:,.2f}")
        st.markdown(f"**Point Parameters:** S={s_adj_for_point:.2f}, r={r_adj_for_point:.3f}")
        st.markdown(f"**Fixed for Heatmap:** K={k_fixed_for_s_r:.2f}, T={t_fixed_for_s_r:.2f}, $\sigma$={sigma_fixed_for_s_r:.2f}")

st.markdown("---")

# --- Put Option Heatmaps ---
st.subheader("European Put Option Sensitivity")

# Put: K vs T
with st.expander("Put Price: Strike Price (K) vs. Time to Expiration (T)", expanded=True):
    col_plot, col_sliders = st.columns([2.5, 1])

    with col_sliders:
        st.markdown("#### Adjust Parameters")
        s_fixed_for_k_t = st.slider("Stock Price (S)", min_value=max(0.01, S_global * 0.7), max_value=S_global * 1.3, value=S_global, step=S_global * 0.01, format="%.2f", key="S_put_k_t_fix")
        r_fixed_for_k_t = st.slider("Risk-Free Rate (r)", min_value=max(0.0, r_global * 0.7), max_value=r_global * 1.3, value=r_global, step=0.001, format="%.3f", key="r_put_k_t_fix")
        sigma_fixed_for_k_t = st.slider("Volatility ($\sigma$)", min_value=max(0.01, sigma_global * 0.7), max_value=sigma_global * 1.3, value=sigma_global, step=0.01, format="%.2f", key="sigma_put_k_t_fix")

        k_adj_for_point = st.slider("Strike Price (K) for point", min_value=max(0.01, K_global * 0.7), max_value=K_global * 1.3, value=K_global, step=K_global * 0.01, format="%.2f", key="K_put_k_t_point")
        t_adj_for_point = st.slider("Time to Expiration (T) for point", min_value=max(0.01, T_global * 0.7), max_value=T_global * 1.3, value=T_global, step=T_global * 0.05, format="%.2f", key="T_put_k_t_point")

    with col_plot:
        K_range_hp = K_global * 0.3
        T_range_hp = T_global * 0.3
        strike_prices_hp = np.linspace(max(0.01, K_global - K_range_hp), K_global + K_range_hp, num_points)
        times_hp = np.linspace(max(0.01, T_global - T_range_hp), T_global + T_range_hp, num_points)

        put_matrix = np.zeros((num_points, num_points))
        for i, k_val in enumerate(strike_prices_hp):
            for j, t_val in enumerate(times_hp):
                put_matrix[i, j] = black_scholes_put_price(s_fixed_for_k_t, k_val, t_val, r_fixed_for_k_t, sigma_fixed_for_k_t)

        plot_heatmap(put_matrix,
                     [f'{x:.2f}' for x in times_hp], [f'{y:.2f}' for y in strike_prices_hp],
                     "Time to Expiration (T)", "Strike Price (K)",
                     "Put Price: K vs. T", 'RdYlGn_r', "Put Price") # _r reverses colormap

    with col_sliders:
        current_put = black_scholes_put_price(s_fixed_for_k_t, k_adj_for_point, t_adj_for_point, r_fixed_for_k_t, sigma_fixed_for_k_t)
        st.metric(label="Current Put Price", value=f"${current_put:,.2f}")
        st.markdown(f"**Point Parameters:** K={k_adj_for_point:.2f}, T={t_adj_for_point:.2f}")
        st.markdown(f"**Fixed for Heatmap:** S={s_fixed_for_k_t:.2f}, r={r_fixed_for_k_t:.3f}, $\sigma$={sigma_fixed_for_k_t:.2f}")

# Put: S vs Sigma
with st.expander("Put Price: Stock Price (S) vs. Volatility ($\sigma$)"):
    col_plot, col_sliders = st.columns([2.5, 1])

    with col_sliders:
        st.markdown("#### Adjust Parameters")
        k_fixed_for_s_sigma = st.slider("Strike Price (K)", min_value=max(0.01, K_global * 0.7), max_value=K_global * 1.3, value=K_global, step=K_global * 0.01, format="%.2f", key="K_put_s_sigma_fix")
        t_fixed_for_s_sigma = st.slider("Time to Expiration (T)", min_value=max(0.01, T_global * 0.7), max_value=T_global * 1.3, value=T_global, step=T_global * 0.05, format="%.2f", key="T_put_s_sigma_fix")
        r_fixed_for_s_sigma = st.slider("Risk-Free Rate (r)", min_value=max(0.0, r_global * 0.7), max_value=r_global * 1.3, value=r_global, step=0.001, format="%.3f", key="r_put_s_sigma_fix")

        s_adj_for_point = st.slider("Stock Price (S) for point", min_value=max(0.01, S_global * 0.7), max_value=S_global * 1.3, value=S_global, step=S_global * 0.01, format="%.2f", key="S_put_s_sigma_point")
        sigma_adj_for_point = st.slider("Volatility ($\sigma$) for point", min_value=max(0.01, sigma_global * 0.7), max_value=sigma_global * 1.3, value=sigma_global, step=0.01, format="%.2f", key="sigma_put_s_sigma_point")

    with col_plot:
        S_range_hp = S_global * 0.3
        sig_range_hp = sigma_global * 0.3
        stock_prices_hp = np.linspace(max(0.01, S_global - S_range_hp), S_global + S_range_hp, num_points)
        sigmas_hp = np.linspace(max(0.01, sigma_global - sig_range_hp), sigma_global + sig_range_hp, num_points)

        put_matrix = np.zeros((num_points, num_points))
        for i, s_val in enumerate(stock_prices_hp):
            for j, sig_val in enumerate(sigmas_hp):
                put_matrix[i, j] = black_scholes_put_price(s_val, k_fixed_for_s_sigma, t_fixed_for_s_sigma, r_fixed_for_s_sigma, sig_val)

        plot_heatmap(put_matrix,
                     [f'{x:.2f}' for x in sigmas_hp], [f'{y:.2f}' for y in stock_prices_hp],
                     "Volatility ($\sigma$)", "Stock Price (S)",
                     "Put Price: S vs. $\sigma$", 'RdYlGn_r', "Put Price")

    with col_sliders:
        current_put = black_scholes_put_price(s_adj_for_point, k_fixed_for_s_sigma, t_fixed_for_s_sigma, r_fixed_for_s_sigma, sigma_adj_for_point)
        st.metric(label="Current Put Price", value=f"${current_put:,.2f}")
        st.markdown(f"**Point Parameters:** S={s_adj_for_point:.2f}, $\sigma$={sigma_adj_for_point:.2f}")
        st.markdown(f"Fixed: K={k_fixed_for_s_sigma:.2f}, T={t_fixed_for_s_sigma:.2f}, r={r_fixed_for_s_sigma:.3f}")

# Put: K vs R
with st.expander("Put Price: Strike Price (K) vs. Risk-Free Rate (r)"):
    col_plot, col_sliders = st.columns([2.5, 1])

    with col_sliders:
        st.markdown("#### Adjust Parameters")
        s_fixed_for_k_r = st.slider("Stock Price (S)", min_value=max(0.01, S_global * 0.7), max_value=S_global * 1.3, value=S_global, step=S_global * 0.01, format="%.2f", key="S_put_k_r_fix")
        t_fixed_for_k_r = st.slider("Time to Expiration (T)", min_value=max(0.01, T_global * 0.7), max_value=T_global * 1.3, value=T_global, step=T_global * 0.05, format="%.2f", key="T_put_k_r_fix")
        sigma_fixed_for_k_r = st.slider("Volatility ($\sigma$)", min_value=max(0.01, sigma_global * 0.7), max_value=sigma_global * 1.3, value=sigma_global, step=0.01, format="%.2f", key="sigma_put_k_r_fix")

        k_adj_for_point = st.slider("Strike Price (K) for point", min_value=max(0.01, K_global * 0.7), max_value=K_global * 1.3, value=K_global, step=K_global * 0.01, format="%.2f", key="K_put_k_r_point")
        r_adj_for_point = st.slider("Risk-Free Rate (r) for point", min_value=max(0.0, r_global * 0.7), max_value=r_global * 1.3, value=r_global, step=0.001, format="%.3f", key="r_put_k_r_point")

    with col_plot:
        K_range_hp = K_global * 0.3
        r_range_hp = r_global * 0.5 if r_global > 0 else 0.02
        strike_prices_hp = np.linspace(max(0.01, K_global - K_range_hp), K_global + K_range_hp, num_points)
        r_rates_hp = np.linspace(max(0.0, r_global - r_range_hp), r_global + r_range_hp, num_points)

        put_matrix = np.zeros((num_points, num_points))
        for i, k_val in enumerate(strike_prices_hp):
            for j, r_val in enumerate(r_rates_hp):
                put_matrix[i, j] = black_scholes_put_price(s_fixed_for_k_r, k_val, t_fixed_for_k_r, r_val, sigma_fixed_for_k_r)

        plot_heatmap(put_matrix,
                     [f'{x:.3f}' for x in r_rates_hp], [f'{y:.2f}' for y in strike_prices_hp],
                     "Risk-Free Rate (r)", "Strike Price (K)",
                     "Put Price: K vs. r", 'RdYlGn_r', "Put Price")

    with col_sliders:
        current_put = black_scholes_put_price(s_fixed_for_k_r, k_adj_for_point, t_fixed_for_k_r, r_adj_for_point, sigma_fixed_for_k_r)
        st.metric(label="Current Put Price", value=f"${current_put:,.2f}")
        st.markdown(f"**Point Parameters:** K={k_adj_for_point:.2f}, r={r_adj_for_point:.3f}")
        st.markdown(f"Fixed: S={s_fixed_for_k_r:.2f}, T={t_fixed_for_k_r:.2f}, $\sigma$={sigma_fixed_for_k_r:.2f}")


st.markdown("---")
st.markdown("""
    **Note:** The Black-Scholes model assumes European options, no dividends,
    constant volatility and risk-free rate, and efficient markets.
    """)