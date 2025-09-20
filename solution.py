# %%

import numpy as np
import pandas as pd
import datetime

from scipy.optimize import minimize
import plotly.express as px

print('---> Python Script Start', t0 := datetime.datetime.now())

# %%

print('---> initial data set up')

# instrument data
df_bonds = pd.read_csv('data/data_bonds.csv')
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp']).apply(lambda d: d.date())

# albi data
df_albi = pd.read_csv('data/data_albi.csv')
df_albi['datestamp'] = pd.to_datetime(df_albi['datestamp']).apply(lambda d: d.date())

# macro data
df_macro = pd.read_csv('data/data_macro.csv')
df_macro['datestamp'] = pd.to_datetime(df_macro['datestamp']).apply(lambda d: d.date())

print('---> the parameters')

# training and test dates
start_train = datetime.date(2005, 1, 3)
start_test = datetime.date(2023, 1, 3)
end_test = df_bonds['datestamp'].max()

# %%

df_signals = pd.DataFrame(data={'datestamp':df_bonds.loc[(df_bonds['datestamp']>=start_test) & (df_bonds['datestamp']<=end_test), 'datestamp'].values})
df_signals.drop_duplicates(inplace=True)
df_signals.reset_index(drop=True, inplace=True)
df_signals.sort_values(by='datestamp', inplace=True)

weight_matrix = pd.DataFrame()

# %%

# Fire solution - simple but effective
prev_weights = [0.1]*10
p_active_md = 1.4
weight_bounds = (0.0, 0.2)

for i in range(len(df_signals)):
    
    current_date = df_signals.loc[i, 'datestamp']
    
    if i % 25 == 0:
        print(f'---> Processing {current_date} ({i+1}/{len(df_signals)})')
    
    # Get recent training data (faster)
    df_train_bonds = df_bonds[df_bonds['datestamp'] < current_date].tail(2000).copy()
    df_train_albi = df_albi[df_albi['datestamp'] < current_date].tail(500).copy()
    df_train_macro = df_macro[df_macro['datestamp'] < current_date].tail(500).copy()
    
    # Current day data
    df_current = df_bonds[df_bonds['datestamp'] == current_date].copy()
    
    if len(df_current) == 0:
        continue
        
    # Get ALBI duration
    p_albi_md = df_train_albi['modified_duration'].iloc[-1] if len(df_train_albi) > 0 else 6.0
    
    # === FIRE SIGNAL GENERATION ===
    
    # 1. Calculate rolling features for each bond (vectorized)
    df_train_bonds = df_train_bonds.sort_values(['bond_code', 'datestamp'])
    
    # Rolling returns and yields by bond
    df_train_bonds['return_5d'] = df_train_bonds.groupby('bond_code')['return'].rolling(5, min_periods=3).mean().reset_index(0, drop=True)
    df_train_bonds['return_20d'] = df_train_bonds.groupby('bond_code')['return'].rolling(20, min_periods=10).mean().reset_index(0, drop=True)
    df_train_bonds['yield_20d_avg'] = df_train_bonds.groupby('bond_code')['yield'].rolling(20, min_periods=10).mean().reset_index(0, drop=True)
    df_train_bonds['vol_20d'] = df_train_bonds.groupby('bond_code')['return'].rolling(20, min_periods=10).std().reset_index(0, drop=True)
    
    # Get latest features for each bond
    latest_features = df_train_bonds.groupby('bond_code').tail(1)[['bond_code', 'return_5d', 'return_20d', 'yield', 'yield_20d_avg', 'vol_20d', 'modified_duration', 'convexity']].reset_index(drop=True)
    
    # Merge with current day data
    df_current = df_current.merge(latest_features[['bond_code', 'return_5d', 'return_20d', 'yield_20d_avg', 'vol_20d']], on='bond_code', how='left')
    df_current = df_current.fillna(0)
    
    # 2. Generate signals
    
    # Signal 1: Mean reversion in yields (FIRE!)
    df_current['yield_reversion'] = (df_current['yield_20d_avg'] - df_current['yield']) / df_current['yield_20d_avg'].replace(0, 1)
    
    # Signal 2: Momentum with volatility adjustment
    df_current['risk_adj_momentum'] = df_current['return_20d'] / df_current['vol_20d'].replace(0, 1)
    
    # Signal 3: Convexity advantage
    df_current['convexity_signal'] = df_current['convexity'] / df_current['modified_duration'].replace(0, 1)
    
    # Signal 4: High yield preference (value)
    df_current['yield_rank'] = df_current['yield'].rank(pct=True)
    
    # Signal 5: Recent momentum
    df_current['recent_momentum'] = df_current['return_5d']
    
    # === COMBINE SIGNALS ===
    signal = (
        df_current['yield_reversion'] * 0.4 +           # Mean reversion (strongest)
        df_current['risk_adj_momentum'] * 0.25 +        # Risk-adjusted momentum  
        df_current['convexity_signal'] * 0.15 +         # Convexity advantage
        df_current['yield_rank'] * 0.15 +               # Value (high yield)
        np.clip(df_current['recent_momentum'] * 10, -0.5, 0.5) * 0.05  # Recent momentum
    )
    
    # Cap extreme signals
    signal = np.clip(signal, -3, 3)
    
    # === PORTFOLIO OPTIMIZATION ===
    
    def objective(weights, signal, prev_weights):
        expected_return = np.dot(weights, signal)
        turnover = np.sum(np.abs(weights - prev_weights))
        concentration = np.sum(weights**2)  # Penalize concentration
        return -(expected_return - 0.1 * turnover - 0.5 * concentration)
    
    def duration_constraint(weights, durations):
        port_duration = np.dot(weights, durations)
        lower_bound = port_duration - (p_albi_md - p_active_md)
        upper_bound = (p_albi_md + p_active_md) - port_duration
        return [lower_bound, upper_bound]
    
    # Smart initial weights based on signal
    if np.sum(np.abs(signal)) > 0:
        signal_weights = np.abs(signal) / np.sum(np.abs(signal))
        initial_weights = 0.7 * np.array(prev_weights) + 0.3 * signal_weights
    else:
        initial_weights = np.array(prev_weights)
    
    # Ensure bounds compliance
    initial_weights = np.clip(initial_weights, weight_bounds[0], weight_bounds[1])
    initial_weights = initial_weights / np.sum(initial_weights)
    
    # Optimization constraints
    bounds = [weight_bounds] * len(signal)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_current['modified_duration'])[0]},
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_current['modified_duration'])[1]}
    ]
    
    # Solve optimization
    try:
        result = minimize(
            objective, 
            initial_weights,
            args=(signal, prev_weights),
            bounds=bounds, 
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 100, 'ftol': 1e-4}
        )
        
        if result.success and 0.999 <= np.sum(result.x) <= 1.001:
            optimal_weights = result.x
        else:
            optimal_weights = initial_weights
            
    except Exception as e:
        optimal_weights = initial_weights
    
    # Store results
    weight_matrix_tmp = pd.DataFrame({
        'bond_code': df_current['bond_code'],
        'weight': optimal_weights,
        'datestamp': current_date
    })
    weight_matrix = pd.concat([weight_matrix, weight_matrix_tmp], ignore_index=True)
    
    prev_weights = optimal_weights.tolist()

print(f"---> Completed processing {len(weight_matrix['datestamp'].unique())} trading days")

# %%

def plot_payoff(weight_matrix):

    # check weights sum to one
    df_weight_sum = weight_matrix.groupby(['datestamp'])['weight'].sum()
    if df_weight_sum.min() < 0.9999 or df_weight_sum.max() > 1.0001:
        raise ValueError('The portfolio weights do not sum to one')
    
    # check weights between 0 and 0.2
    if weight_matrix['weight'].min() < 0 or weight_matrix['weight'].max() > 0.20001:
        raise ValueError(r'The instrument weights are not confined to 0%-20%')

    # plot weights through time
    fig_weights = px.area(weight_matrix, x="datestamp", y="weight", color="bond_code")
    fig_weights.show()

    port_data = weight_matrix.merge(df_bonds, on = ['bond_code', 'datestamp'], how = 'left')
    df_turnover = weight_matrix.copy()
    df_turnover['turnover'] = df_turnover.groupby(['bond_code'])['weight'].diff()

    port_data['port_return'] = port_data['return'] * port_data['weight']
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']

    port_data = port_data.groupby("datestamp")[['port_return', 'port_md']].sum().reset_index()
    port_data['turnover'] = df_turnover.groupby('datestamp').turnover.apply(lambda x: x.abs().sum()/2).to_list()
    port_data['penalty'] = 0.01*port_data['turnover']*port_data['port_md'].shift()
    port_data['net_return'] = port_data['port_return'].sub(port_data['penalty'], fill_value=0)
    port_data = port_data.merge(df_albi[['datestamp','return']], on = 'datestamp', how = 'left')
    port_data['portfolio_tri'] = (port_data['net_return']/100 +1).cumprod()
    port_data['albi_tri'] = (port_data['return']/100 +1).cumprod()

    #turnover chart
    fig_turnover = px.line(port_data, x='datestamp', y='turnover')
    fig_turnover.show()

    portfolio_return = (port_data['portfolio_tri'].values[-1]-1)*100
    albi_return = (port_data['albi_tri'].values[-1]-1)*100
    
    print(f"---> payoff for these buys between period {port_data['datestamp'].min()} and {port_data['datestamp'].max()} is {portfolio_return:.2f}%")
    print(f"---> payoff for the ALBI benchmark for this period is {albi_return:.2f}%")
    print(f"---> EXCESS RETURN: {portfolio_return - albi_return:.2f}%")

    port_data = pd.melt(port_data[['datestamp', 'portfolio_tri', 'albi_tri']], id_vars = 'datestamp')
    fig_payoff = px.line(port_data, x='datestamp', y='value', color = 'variable')
    fig_payoff.show()

def plot_md(weight_matrix):

    port_data = weight_matrix.merge(df_bonds, on = ['bond_code', 'datestamp'], how = 'left')
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']
    port_data = port_data.groupby("datestamp")[['port_md']].sum().reset_index()
    port_data = port_data.merge(df_albi[['datestamp','modified_duration']], on = 'datestamp', how = 'left')
    port_data['active_md'] = port_data['port_md'] - port_data['modified_duration']

    fig_payoff = px.line(port_data, x='datestamp', y='active_md')
    fig_payoff.show()

    if len(port_data[abs(port_data['active_md']) > 1.5]['datestamp']) == 0:
        print(f"---> The portfolio does not breach the modified duration constraint")
    else:
        raise ValueError('This buy matrix violates the modified duration constraint on the below dates: \n ' +  ", ".join(pd.to_datetime(port_data[abs(port_data['active_md']) > 1.5]['datestamp']).dt.strftime("%Y-%m-%d")))

plot_payoff(weight_matrix)
plot_md(weight_matrix)

# %%

print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)