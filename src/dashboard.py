"""
Bitcoin RL Trading Bot - Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import yaml
import json
from datetime import datetime
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import load_and_preprocess_data
from trading_env import BitcoinTradingEnv

# Page config
st.set_page_config(
    page_title="Bitcoin RL Trading Bot",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d5a87;
    }
    .profit {
        color: #00ff88;
    }
    .loss {
        color: #ff4444;
    }
    .stMetric {
        background-color: #0e1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)


def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def get_available_models():
    """Get list of available trained models."""
    models_path = Path(__file__).parent.parent / "models"
    if not models_path.exists():
        return []

    models = list(models_path.glob("*.zip"))
    return sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)


def get_trade_logs():
    """Get trade log files."""
    logs_path = Path(__file__).parent.parent / "logs" / "trades"
    if not logs_path.exists():
        return []

    logs = list(logs_path.glob("*.csv"))
    return sorted(logs, key=lambda x: x.stat().st_mtime, reverse=True)


def load_trade_history(log_path):
    """Load trade history from CSV."""
    if log_path and os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame()


def create_price_chart(df, trades_df=None):
    """Create interactive price chart with trades."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('BTC/USD Price', 'RSI', 'Volume')
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC/USD'
        ),
        row=1, col=1
    )

    # Add EMAs
    if 'ema_12' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ema_12'], name='EMA 12',
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    if 'ema_26' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ema_26'], name='EMA 26',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )

    # Add Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                      line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                      line=dict(color='gray', width=1, dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )

    # Add trade markers if available
    if trades_df is not None and not trades_df.empty:
        # Long entries
        long_trades = trades_df[trades_df['direction'] == 'long']
        if not long_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_trades['entry_step'],
                    y=long_trades['entry_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='Long Entry'
                ),
                row=1, col=1
            )

        # Short entries
        short_trades = trades_df[trades_df['direction'] == 'short']
        if not short_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_trades['entry_step'],
                    y=short_trades['entry_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Short Entry'
                ),
                row=1, col=1
            )

    # RSI
    if 'rsi_14' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi_14'], name='RSI 14',
                      line=dict(color='purple', width=1)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Volume
    colors = ['green' if row['close'] >= row['open'] else 'red'
              for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume',
               marker_color=colors, opacity=0.7),
        row=3, col=1
    )

    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_equity_curve(equity_data, title="Equity Curve"):
    """Create equity curve chart."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=equity_data,
            mode='lines',
            name='Equity',
            line=dict(color='#00ff88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        )
    )

    # Add drawdown visualization
    peak = np.maximum.accumulate(equity_data)
    drawdown = (peak - equity_data) / peak * 100

    fig.add_trace(
        go.Scatter(
            y=-drawdown,
            mode='lines',
            name='Drawdown %',
            line=dict(color='#ff4444', width=1),
            yaxis='y2'
        )
    )

    fig.update_layout(
        title=title,
        height=400,
        template='plotly_dark',
        yaxis=dict(title='Equity ($)', side='left'),
        yaxis2=dict(title='Drawdown (%)', side='right', overlaying='y', showgrid=False),
        hovermode='x unified'
    )

    return fig


def create_trade_analysis(trades_df):
    """Create trade analysis charts."""
    if trades_df.empty:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PnL Distribution', 'Win/Loss by Direction',
                       'Trade Duration', 'Cumulative PnL'),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )

    # PnL Distribution
    fig.add_trace(
        go.Histogram(x=trades_df['pnl'], nbinsx=30, name='PnL',
                    marker_color=trades_df['pnl'].apply(
                        lambda x: 'green' if x > 0 else 'red')),
        row=1, col=1
    )

    # Win/Loss by Direction
    win_loss = trades_df.groupby(['direction', trades_df['pnl'] > 0]).size().unstack(fill_value=0)
    if True in win_loss.columns and False in win_loss.columns:
        fig.add_trace(
            go.Bar(x=win_loss.index, y=win_loss[True], name='Wins', marker_color='green'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=win_loss.index, y=win_loss[False], name='Losses', marker_color='red'),
            row=1, col=2
        )

    # Trade Duration Distribution
    if 'duration' in trades_df.columns:
        fig.add_trace(
            go.Histogram(x=trades_df['duration'], nbinsx=20, name='Duration',
                        marker_color='blue'),
            row=2, col=1
        )

    # Cumulative PnL
    fig.add_trace(
        go.Scatter(y=trades_df['pnl'].cumsum(), mode='lines+markers',
                  name='Cumulative PnL', line=dict(color='#00ff88')),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=True,
        barmode='group'
    )

    return fig


def calculate_metrics(trades_df, initial_capital=10000):
    """Calculate trading performance metrics."""
    if trades_df.empty:
        return {}

    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] <= 0])

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0

    # Profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calculate equity curve for Sharpe/Sortino
    equity = initial_capital + trades_df['pnl'].cumsum()
    returns = equity.pct_change().dropna()

    # Sharpe Ratio (annualized, assuming hourly data)
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)
    else:
        sharpe = 0

    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_drawdown = drawdown.max()

    # Average trade duration
    avg_duration = trades_df['duration'].mean() if 'duration' in trades_df.columns else 0

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'avg_duration': avg_duration,
        'final_equity': equity.iloc[-1] if len(equity) > 0 else initial_capital
    }


# ============== PAGES ==============

def page_overview():
    """Overview/Dashboard page."""
    st.title("₿ Bitcoin RL Trading Bot")
    st.markdown("---")

    config = load_config()
    models = get_available_models()

    # Status cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Trained Models", len(models))

    with col2:
        data_path = Path(__file__).parent.parent / "data" / "btcusd_1-min_data.csv"
        st.metric("Data Status", "Ready" if data_path.exists() else "Missing")

    with col3:
        if config:
            st.metric("Timeframe", config['data']['timeframe'])
        else:
            st.metric("Timeframe", "N/A")

    with col4:
        if config:
            st.metric("Total Timesteps", f"{config['model']['total_timesteps']:,}")
        else:
            st.metric("Total Timesteps", "N/A")

    st.markdown("---")

    # Configuration display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Configuration")
        if config:
            st.json(config['model'])
        else:
            st.warning("Config file not found")

    with col2:
        st.subheader("Trading Parameters")
        if config:
            st.json(config['trading'])
        else:
            st.warning("Config file not found")

    # Recent models
    st.markdown("---")
    st.subheader("Available Models")

    if models:
        model_data = []
        for m in models[:10]:
            model_data.append({
                'Model': m.name,
                'Size': f"{m.stat().st_size / 1024 / 1024:.2f} MB",
                'Modified': datetime.fromtimestamp(m.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            })
        st.dataframe(pd.DataFrame(model_data), use_container_width=True)
    else:
        st.info("No trained models found. Go to Training page to train a model.")


def read_training_progress():
    """Read training progress from JSON file."""
    progress_file = Path(__file__).parent.parent / "logs" / "training_progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def page_training():
    """Training page."""
    st.title("Model Training")
    st.markdown("---")

    config = load_config()

    if not config:
        st.error("Configuration file not found!")
        return

    # Check if training is in progress
    progress_data = read_training_progress()
    is_training = False
    if progress_data and progress_data.get('status') == 'running':
        try:
            last_update = datetime.fromisoformat(progress_data.get('timestamp', '2000-01-01'))
            is_training = (datetime.now() - last_update).seconds < 10
        except Exception:
            is_training = False

    # Training parameters
    st.subheader("Training Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_timesteps = st.number_input(
            "Total Timesteps",
            min_value=10000,
            max_value=10000000,
            value=config['model']['total_timesteps'],
            step=10000,
            disabled=is_training
        )

        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.00001,
            max_value=0.01,
            value=config['model']['learning_rate'],
            format="%.5f",
            disabled=is_training
        )

    with col2:
        batch_size = st.selectbox(
            "Batch Size",
            options=[32, 64, 128, 256],
            index=[32, 64, 128, 256].index(config['model']['batch_size']),
            disabled=is_training
        )

        n_steps = st.selectbox(
            "N Steps",
            options=[512, 1024, 2048, 4096],
            index=[512, 1024, 2048, 4096].index(config['model']['n_steps']),
            disabled=is_training
        )

    with col3:
        gamma = st.slider(
            "Gamma (Discount Factor)",
            min_value=0.9,
            max_value=0.999,
            value=config['model']['gamma'],
            step=0.001,
            disabled=is_training
        )

        ent_coef = st.slider(
            "Entropy Coefficient",
            min_value=0.0,
            max_value=0.1,
            value=config['model']['ent_coef'],
            step=0.001,
            disabled=is_training
        )

    st.markdown("---")

    # Date range
    st.subheader("Data Range")
    col1, col2 = st.columns(2)

    with col1:
        train_start = st.date_input("Train Start", value=pd.to_datetime(config['data']['train_start']), disabled=is_training)
        train_end = st.date_input("Train End", value=pd.to_datetime(config['data']['train_end']), disabled=is_training)

    with col2:
        test_start = st.date_input("Test Start", value=pd.to_datetime(config['data']['test_start']), disabled=is_training)
        test_end = st.date_input("Test End", value=pd.to_datetime(config['data']['test_end']), disabled=is_training)

    st.markdown("---")

    # Show real-time progress if training is running
    if is_training:
        st.subheader("Training Progress")

        # Progress bar
        progress_percent = progress_data.get('percent', 0)
        st.progress(progress_percent / 100)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Progress",
                f"{progress_percent:.1f}%",
                f"{progress_data.get('current_step', 0):,} / {progress_data.get('total_steps', 0):,}"
            )

        with col2:
            st.metric(
                "Elapsed",
                progress_data.get('elapsed_formatted', '0s'),
                f"ETA: {progress_data.get('eta_formatted', 'N/A')}"
            )

        with col3:
            st.metric(
                "Episodes",
                progress_data.get('episodes', 0),
                f"{progress_data.get('steps_per_second', 0):.0f} steps/s"
            )

        with col4:
            avg_reward = progress_data.get('avg_reward', 0)
            st.metric(
                "Avg Reward (last 10)",
                f"{avg_reward:.1f}",
                f"Avg Length: {progress_data.get('avg_length', 0):.0f}"
            )

        # Auto-refresh
        st.info("Training in progress... Page will auto-refresh every 2 seconds.")
        time.sleep(2)
        st.rerun()

    else:
        # Training control
        if st.button("Start Training", type="primary", use_container_width=True):
            # Save updated config
            config['model']['total_timesteps'] = total_timesteps
            config['model']['learning_rate'] = learning_rate
            config['model']['batch_size'] = batch_size
            config['model']['n_steps'] = n_steps
            config['model']['gamma'] = gamma
            config['model']['ent_coef'] = ent_coef
            config['data']['train_start'] = str(train_start)
            config['data']['train_end'] = str(train_end)
            config['data']['test_start'] = str(test_start)
            config['data']['test_end'] = str(test_end)

            config_path = Path(__file__).parent.parent / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            # Clear old progress file
            progress_file = Path(__file__).parent.parent / "logs" / "training_progress.json"
            if progress_file.exists():
                progress_file.unlink()

            # Start training in a subprocess
            import subprocess
            train_script = Path(__file__).parent / "train.py"

            # Run training in background
            subprocess.Popen(
                [sys.executable, str(train_script)],
                cwd=str(Path(__file__).parent.parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )

            st.success("Training started! Progress will appear below...")
            time.sleep(2)
            st.rerun()

        # Show last training result if available
        if progress_data and progress_data.get('status') == 'completed':
            st.markdown("---")
            st.subheader("Last Training Result")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Status", "Completed")
            with col2:
                st.metric("Total Time", progress_data.get('elapsed_formatted', 'N/A'))
            with col3:
                st.metric("Episodes", progress_data.get('episodes', 0))
            with col4:
                st.metric("Final Avg Reward", f"{progress_data.get('avg_reward', 0):.1f}")

    # TensorBoard link
    st.markdown("---")
    st.subheader("TensorBoard")
    st.code("tensorboard --logdir logs/tensorboard/", language="bash")
    st.info("Run this command in terminal to view detailed training metrics")


def page_backtest():
    """Backtest page."""
    st.title("Backtest & Evaluation")
    st.markdown("---")

    models = get_available_models()
    config = load_config()

    if not models:
        st.warning("No trained models found. Train a model first.")
        return

    # Model selection
    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox(
            "Select Model",
            options=models,
            format_func=lambda x: x.name
        )

    with col2:
        test_start = st.date_input("Test Start", value=pd.to_datetime(config['data']['test_start']))
        test_end = st.date_input("Test End", value=pd.to_datetime(config['data']['test_end']))

    if st.button("Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            try:
                from stable_baselines3 import PPO

                # Load data
                data_path = Path(__file__).parent.parent / config['data']['train_path']
                test_df, test_features, scaler = load_and_preprocess_data(
                    str(data_path),
                    timeframe=config['data']['timeframe'],
                    start_date=str(test_start),
                    end_date=str(test_end),
                    normalize=config['features']['normalize'],
                    scaler_path=str(Path(__file__).parent.parent / "cache" / "scaler.pkl")
                )

                st.info(f"Loaded {len(test_df)} test samples")

                # Create environment
                env = BitcoinTradingEnv(
                    df=test_df,
                    feature_df=test_features,
                    window_size=config['environment']['window_size'],
                    initial_capital=config['environment']['initial_capital'],
                    max_position_duration=config['environment']['max_position_duration'],
                    max_position_pct=config['environment'].get('max_position_pct', 0.20),
                    simple_actions=config['environment'].get('simple_actions', True),
                    sl_options=config['trading']['sl_options'],
                    tp_options=config['trading']['tp_options'],
                    spread_pct=config['trading']['spread_pct'],
                    commission_pct=config['trading']['commission_pct']
                )

                # Load model
                model = PPO.load(str(selected_model))

                # Run backtest
                obs, info = env.reset()
                total_reward = 0
                done = False

                progress_bar = st.progress(0)

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    done = done or truncated

                    progress = min(env.current_step / len(test_df), 1.0)
                    progress_bar.progress(progress)

                # Get results
                trades_df = env.get_trade_history()
                equity_curve = env.get_equity_curve()

                # Store in session state
                st.session_state['backtest_trades'] = trades_df
                st.session_state['backtest_equity'] = equity_curve
                st.session_state['backtest_df'] = test_df
                st.session_state['backtest_info'] = info

                st.success("Backtest completed!")

            except Exception as e:
                st.error(f"Backtest error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display results if available
    if 'backtest_trades' in st.session_state:
        trades_df = st.session_state['backtest_trades']
        equity_curve = st.session_state['backtest_equity']
        info = st.session_state['backtest_info']

        st.markdown("---")
        st.subheader("Results")

        # Metrics
        metrics = calculate_metrics(trades_df, config['environment']['initial_capital'])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pnl_color = "normal" if metrics.get('total_pnl', 0) >= 0 else "inverse"
            st.metric("Total PnL", f"${metrics.get('total_pnl', 0):.2f}", delta_color=pnl_color)
            st.metric("Total Trades", metrics.get('total_trades', 0))

        with col2:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

        with col3:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")

        with col4:
            st.metric("Avg Win", f"${metrics.get('avg_win', 0):.2f}")
            st.metric("Avg Loss", f"${metrics.get('avg_loss', 0):.2f}")

        # Equity curve
        st.markdown("---")
        st.subheader("Equity Curve")
        fig = create_equity_curve(equity_curve)
        st.plotly_chart(fig, use_container_width=True)

        # Trade analysis
        st.markdown("---")
        st.subheader("Trade Analysis")
        fig = create_trade_analysis(trades_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Trade table
        st.markdown("---")
        st.subheader("Trade History")
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True)


def page_data_explorer():
    """Data exploration page."""
    st.title("Data Explorer")
    st.markdown("---")

    config = load_config()

    # Date range selection
    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))

    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-06-30"))

    with col3:
        timeframe = st.selectbox("Timeframe", options=['1h', '4h', '1d'], index=0)

    if st.button("Load Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                data_path = Path(__file__).parent.parent / config['data']['train_path']
                df, features, _ = load_and_preprocess_data(
                    str(data_path),
                    timeframe=timeframe,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    normalize=False
                )

                st.session_state['explorer_df'] = df
                st.session_state['explorer_features'] = features
                st.success(f"Loaded {len(df)} samples")

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Display data if available
    if 'explorer_df' in st.session_state:
        df = st.session_state['explorer_df']

        # Price chart
        st.markdown("---")
        st.subheader("Price Chart")
        fig = create_price_chart(df)
        st.plotly_chart(fig, use_container_width=True)

        # Data statistics
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Statistics")
            st.dataframe(df.describe(), use_container_width=True)

        with col2:
            st.subheader("Feature Correlations")
            features = st.session_state.get('explorer_features', df)
            corr_cols = ['close', 'rsi_14', 'macd', 'atr_14', 'adx_14']
            corr_cols = [c for c in corr_cols if c in features.columns]
            if corr_cols:
                fig = px.imshow(
                    features[corr_cols].corr(),
                    text_auto=True,
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)


def page_settings():
    """Settings page."""
    st.title("Settings")
    st.markdown("---")

    config = load_config()

    if not config:
        st.error("Configuration file not found!")
        return

    st.subheader("Environment Settings")

    col1, col2 = st.columns(2)

    with col1:
        window_size = st.number_input(
            "Window Size (hours)",
            min_value=12,
            max_value=168,
            value=config['environment']['window_size']
        )

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=config['environment']['initial_capital']
        )

    with col2:
        max_position_duration = st.number_input(
            "Max Position Duration (hours)",
            min_value=1,
            max_value=168,
            value=config['environment']['max_position_duration']
        )

    st.markdown("---")
    st.subheader("Trading Settings")

    col1, col2 = st.columns(2)

    with col1:
        spread_pct = st.number_input(
            "Spread (%)",
            min_value=0.0,
            max_value=1.0,
            value=config['trading']['spread_pct'] * 100,
            format="%.3f"
        ) / 100

        commission_pct = st.number_input(
            "Commission (%)",
            min_value=0.0,
            max_value=1.0,
            value=config['trading']['commission_pct'] * 100,
            format="%.3f"
        ) / 100

    with col2:
        st.write("Stop Loss Options (%)")
        sl_options = st.text_input(
            "SL Options",
            value=", ".join([str(x * 100) for x in config['trading']['sl_options']])
        )

        st.write("Take Profit Options (%)")
        tp_options = st.text_input(
            "TP Options",
            value=", ".join([str(x * 100) for x in config['trading']['tp_options']])
        )

    st.markdown("---")

    if st.button("Save Settings", type="primary"):
        try:
            config['environment']['window_size'] = window_size
            config['environment']['initial_capital'] = initial_capital
            config['environment']['max_position_duration'] = max_position_duration
            config['trading']['spread_pct'] = spread_pct
            config['trading']['commission_pct'] = commission_pct
            config['trading']['sl_options'] = [float(x.strip()) / 100 for x in sl_options.split(',')]
            config['trading']['tp_options'] = [float(x.strip()) / 100 for x in tp_options.split(',')]

            config_path = Path(__file__).parent.parent / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            st.success("Settings saved successfully!")

        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")

    # Display current config
    st.markdown("---")
    st.subheader("Current Configuration")
    st.json(config)


# ============== MAIN ==============

def main():
    """Main application."""

    # Sidebar navigation
    st.sidebar.title("Navigation")

    pages = {
        "Overview": page_overview,
        "Training": page_training,
        "Backtest": page_backtest,
        "Data Explorer": page_data_explorer,
        "Settings": page_settings
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Run selected page
    pages[selection]()

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Bitcoin RL Trading Bot\n\n"
        "Using PPO algorithm with\n"
        "Stable-Baselines3"
    )


if __name__ == "__main__":
    main()
