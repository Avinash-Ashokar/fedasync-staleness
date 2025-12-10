#!/usr/bin/env python3
"""
Comprehensive analysis of all experiments.
Compiles results from all CSV files and configs into a single table.
"""

import os
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_csv_results(csv_path):
    """Load and extract key metrics from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        
        return {
            'total_rounds': int(df['total_agg'].max()) if 'total_agg' in df.columns else len(df),
            'best_test_acc': float(df['test_acc'].max()) if 'test_acc' in df.columns else None,
            'final_test_acc': float(df['test_acc'].iloc[-1]) if 'test_acc' in df.columns else None,
            'best_train_acc': float(df['avg_train_acc'].max()) if 'avg_train_acc' in df.columns else None,
            'final_train_acc': float(df['avg_train_acc'].iloc[-1]) if 'avg_train_acc' in df.columns else None,
            'final_test_loss': float(df['test_loss'].iloc[-1]) if 'test_loss' in df.columns else None,
            'total_time_sec': float(df['time'].iloc[-1]) if 'time' in df.columns else None,
            'total_time_min': float(df['time'].iloc[-1] / 60.0) if 'time' in df.columns else None,
            'num_rows': len(df)
        }
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def load_config(config_path):
    """Load experiment configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return None

def extract_experiment_info(config, run_dir):
    """Extract key experiment parameters from config."""
    info = {
        'method': 'Unknown',
        'alpha': None,
        'stragglers': None,
        'clients': None,
        'local_epochs': None,
        'lr': None,
        'eta': None,
        'max_rounds': None,
        'batch_size': None,
        'seed': None,
    }
    
    # Determine method
    if 'trustweight' in config or 'TrustWeight' in str(run_dir):
        info['method'] = 'TrustWeight'
        if 'trustweight' in config:
            tw = config['trustweight']
            info['eta'] = tw.get('eta', None)
            info['freshness_alpha'] = tw.get('freshness_alpha', None)
            info['beta1'] = tw.get('beta1', None)
            info['beta2'] = tw.get('beta2', None)
    elif 'fedasync' in config or 'FedAsync' in str(run_dir):
        info['method'] = 'FedAsync'
        if 'async' in config:
            info['eta'] = config['async'].get('fedasync_mixing_alpha', None)
    elif 'fedbuff' in config or 'FedBuff' in str(run_dir):
        info['method'] = 'FedBuff'
        if 'async' in config:
            info['eta'] = config['async'].get('eta', None)
    
    # Extract common parameters
    if 'partition_alpha' in config:
        info['alpha'] = config['partition_alpha']
    elif 'data' in config and 'alpha' in config['data']:
        info['alpha'] = config['data']['alpha']
    
    if 'clients' in config:
        clients = config['clients']
        info['clients'] = clients.get('total', None)
        info['local_epochs'] = clients.get('local_epochs', None)
        info['lr'] = clients.get('lr', None)
        info['batch_size'] = clients.get('batch_size', None)
        info['stragglers'] = clients.get('struggle_percent', clients.get('straggler_fraction', None))
        if info['stragglers'] is None:
            info['stragglers'] = 0.0
    
    if 'train' in config:
        info['max_rounds'] = config['train'].get('max_rounds', None)
    
    info['seed'] = config.get('seed', None)
    
    return info

def scan_all_experiments():
    """Scan all experiment directories and compile results."""
    logs_dir = Path('logs')
    experiments = []
    
    # Scan avinash directory
    avinash_dir = logs_dir / 'avinash'
    if avinash_dir.exists():
        for run_dir in sorted(avinash_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            
            # Look for CSV files
            csv_files = {
                'FedBuff': run_dir / 'FedBuff.csv',
                'FedAsync': run_dir / 'FedAsync.csv',
                'TrustWeight': run_dir / 'TrustWeight.csv',
            }
            
            config_file = run_dir / 'CONFIG.yaml'
            if not config_file.exists():
                config_file = run_dir / 'CONFIG.yml'
            
            config = load_config(config_file) if config_file.exists() else None
            
            for method, csv_path in csv_files.items():
                if csv_path.exists():
                    results = load_csv_results(csv_path)
                    if results:
                        info = extract_experiment_info(config, run_dir) if config else {}
                        info['method'] = method
                        info['run_dir'] = str(run_dir)
                        info['run_date'] = run_dir.name.split('_')[1] if '_' in run_dir.name else None
                        info.update(results)
                        experiments.append(info)
    
    # Scan TrustWeight directory
    tw_dir = logs_dir / 'TrustWeight'
    if tw_dir.exists():
        for exp_dir in sorted(tw_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            for run_dir in sorted(exp_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                
                csv_path = run_dir / 'TrustWeight.csv'
                config_file = run_dir / 'CONFIG.yaml'
                
                if csv_path.exists():
                    results = load_csv_results(csv_path)
                    if results and results.get('num_rows', 0) > 0:  # Only add if has data
                        config = load_config(config_file) if config_file.exists() else None
                        info = extract_experiment_info(config, run_dir) if config else {}
                        info['method'] = 'TrustWeight'
                        info['exp_id'] = exp_dir.name
                        info['run_dir'] = str(run_dir)
                        info['run_date'] = run_dir.name.split('_')[1] if '_' in run_dir.name else None
                        info.update(results)
                        experiments.append(info)
    
    # Scan outside/ directory for additional experiments
    outside_dir = Path('outside')
    if outside_dir.exists():
        # Scan outside/logs
        outside_logs = outside_dir / 'logs'
        if outside_logs.exists():
            for method_dir in sorted(outside_logs.iterdir()):
                if not method_dir.is_dir():
                    continue
                
                method_name = method_dir.name
                if method_name not in ['FedAsync', 'FedBuff', 'TrustWeight']:
                    continue
                
                for exp_dir in sorted(method_dir.iterdir()):
                    if not exp_dir.is_dir():
                        continue
                    
                    for run_dir in sorted(exp_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue
                        
                        # Look for CSV files
                        csv_files = {
                            'FedBuff': run_dir / 'FedBuff.csv',
                            'FedAsync': run_dir / 'FedAsync.csv',
                            'TrustWeight': run_dir / 'TrustWeight.csv',
                        }
                        
                        config_file = run_dir / 'CONFIG.yaml'
                        if not config_file.exists():
                            config_file = run_dir / 'CONFIG.yml'
                        
                        csv_path = csv_files.get(method_name)
                        if csv_path and csv_path.exists():
                            results = load_csv_results(csv_path)
                            if results and results.get('num_rows', 0) > 0:
                                config = load_config(config_file) if config_file.exists() else None
                                info = extract_experiment_info(config, run_dir) if config else {}
                                info['method'] = method_name
                                info['exp_id'] = exp_dir.name
                                info['run_dir'] = str(run_dir)
                                info['run_date'] = run_dir.name.split('_')[1] if '_' in run_dir.name else None
                                info['location'] = 'outside/logs'
                                info.update(results)
                                experiments.append(info)
        
        # Scan outside/outside/logs (nested)
        outside_nested = outside_dir / 'outside' / 'logs'
        if outside_nested.exists():
            for method_dir in sorted(outside_nested.iterdir()):
                if not method_dir.is_dir():
                    continue
                
                method_name = method_dir.name
                if method_name not in ['FedAsync', 'FedBuff', 'TrustWeight']:
                    continue
                
                for exp_dir in sorted(method_dir.iterdir()):
                    if not exp_dir.is_dir():
                        continue
                    
                    for run_dir in sorted(exp_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue
                        
                        csv_files = {
                            'FedBuff': run_dir / 'FedBuff.csv',
                            'FedAsync': run_dir / 'FedAsync.csv',
                            'TrustWeight': run_dir / 'TrustWeight.csv',
                        }
                        
                        config_file = run_dir / 'CONFIG.yaml'
                        if not config_file.exists():
                            config_file = run_dir / 'CONFIG.yml'
                        
                        csv_path = csv_files.get(method_name)
                        if csv_path and csv_path.exists():
                            results = load_csv_results(csv_path)
                            if results and results.get('num_rows', 0) > 0:
                                config = load_config(config_file) if config_file.exists() else None
                                info = extract_experiment_info(config, run_dir) if config else {}
                                info['method'] = method_name
                                info['exp_id'] = exp_dir.name
                                info['run_dir'] = str(run_dir)
                                info['run_date'] = run_dir.name.split('_')[1] if '_' in run_dir.name else None
                                info['location'] = 'outside/outside/logs'
                                info.update(results)
                                experiments.append(info)
    
    return experiments

def create_summary_table(experiments):
    """Create a formatted summary table."""
    if not experiments:
        return "No experiments found."
    
    # Sort by method, then by date
    experiments.sort(key=lambda x: (x.get('method', ''), x.get('run_date', ''), x.get('alpha', 0)))
    
    # Create table
    print("="*150)
    print("COMPREHENSIVE EXPERIMENT RESULTS TABLE")
    print("="*150)
    print(f"\nTotal experiments found: {len(experiments)}\n")
    
    # Group by method
    by_method = defaultdict(list)
    for exp in experiments:
        by_method[exp.get('method', 'Unknown')].append(exp)
    
    # Print table for each method
    for method in sorted(by_method.keys()):
        exps = by_method[method]
        print(f"\n{'='*150}")
        print(f"{method.upper()} EXPERIMENTS ({len(exps)} runs)")
        print(f"{'='*150}")
        print(f"{'Run Date':<12} {'Exp ID':<8} {'Alpha':<8} {'Strag %':<8} {'Clients':<8} {'Rounds':<8} {'Best Acc':<10} {'Final Acc':<10} {'Time (min)':<12} {'Run Dir':<40}")
        print("-"*150)
        
        for exp in exps:
            run_date = exp.get('run_date', 'Unknown')[:10] if exp.get('run_date') else 'Unknown'
            exp_id = exp.get('exp_id', '-')
            alpha = f"{exp.get('alpha', 0):.1f}" if exp.get('alpha') is not None else '-'
            stragglers = f"{exp.get('stragglers', 0):.0f}" if exp.get('stragglers') is not None else '0'
            clients = f"{exp.get('clients', 0)}" if exp.get('clients') else '-'
            rounds = f"{exp.get('total_rounds', 0)}" if exp.get('total_rounds') else '-'
            best_acc = f"{exp.get('best_test_acc', 0):.4f}" if exp.get('best_test_acc') is not None else '-'
            final_acc = f"{exp.get('final_test_acc', 0):.4f}" if exp.get('final_test_acc') is not None else '-'
            time_min = f"{exp.get('total_time_min', 0):.2f}" if exp.get('total_time_min') else '-'
            run_dir_short = exp.get('run_dir', '').split('/')[-1] if exp.get('run_dir') else '-'
            
            print(f"{run_date:<12} {exp_id:<8} {alpha:<8} {stragglers:<8} {clients:<8} {rounds:<8} {best_acc:<10} {final_acc:<10} {time_min:<12} {run_dir_short:<40}")
    
    return experiments

def create_detailed_summary(experiments):
    """Create a more detailed summary with key statistics."""
    print("\n" + "="*150)
    print("DETAILED STATISTICS BY METHOD")
    print("="*150)
    
    by_method = defaultdict(list)
    for exp in experiments:
        by_method[exp.get('method', 'Unknown')].append(exp)
    
    for method in sorted(by_method.keys()):
        exps = by_method[method]
        if not exps:
            continue
        
        print(f"\n{method.upper()}:")
        print(f"  Total runs: {len(exps)}")
        
        # Filter valid results
        valid_exps = [e for e in exps if e.get('best_test_acc') is not None]
        if valid_exps:
            best_accs = [e['best_test_acc'] for e in valid_exps]
            final_accs = [e['final_test_acc'] for e in valid_exps if e.get('final_test_acc') is not None]
            
            print(f"  Valid results: {len(valid_exps)}")
            print(f"  Best accuracy range: {min(best_accs):.4f} - {max(best_accs):.4f} (avg: {sum(best_accs)/len(best_accs):.4f})")
            if final_accs:
                print(f"  Final accuracy range: {min(final_accs):.4f} - {max(final_accs):.4f} (avg: {sum(final_accs)/len(final_accs):.4f})")
            
            # Group by alpha
            by_alpha = defaultdict(list)
            for e in valid_exps:
                alpha = e.get('alpha', 'unknown')
                by_alpha[alpha].append(e)
            
            print(f"  Experiments by alpha:")
            # Sort alphas properly (handle mixed types)
            sorted_alphas = sorted(by_alpha.keys(), key=lambda x: (isinstance(x, str), x if isinstance(x, (int, float)) else 0))
            for alpha in sorted_alphas:
                alpha_exps = by_alpha[alpha]
                avg_best = sum(e['best_test_acc'] for e in alpha_exps) / len(alpha_exps)
                print(f"    α={alpha}: {len(alpha_exps)} runs, avg best acc: {avg_best:.4f}")

if __name__ == '__main__':
    print("Scanning all experiment results...")
    experiments = scan_all_experiments()
    
    if experiments:
        create_summary_table(experiments)
        create_detailed_summary(experiments)
        
        # Save to CSV
        df = pd.DataFrame(experiments)
        output_file = 'all_experiments_summary.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Detailed results saved to: {output_file}")
    else:
        print("No experiments found.")

