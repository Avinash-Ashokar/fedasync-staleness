#!/usr/bin/env python3
"""
Comprehensive analysis of all branches and their code differences.
"""

import subprocess
import os
from pathlib import Path
from collections import defaultdict
import json

def run_git_command(cmd):
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd()
        )
        return result.stdout.strip().split('\n') if result.returncode == 0 else []
    except Exception as e:
        print(f"Error running command: {cmd}\n{e}")
        return []

def get_all_branches():
    """Get all local and remote branches."""
    local_branches = run_git_command("git branch --format='%(refname:short)'")
    remote_branches = run_git_command("git branch -r --format='%(refname:short)'")
    
    # Filter out HEAD
    local_branches = [b for b in local_branches if b and 'HEAD' not in b]
    remote_branches = [b.replace('origin/', '') for b in remote_branches if b and 'HEAD' not in b]
    
    all_branches = list(set(local_branches + remote_branches))
    return sorted(all_branches)

def get_branch_files(branch):
    """Get list of files in a branch."""
    try:
        files = run_git_command(f"git ls-tree -r --name-only {branch}")
        return [f for f in files if f]
    except:
        return []

def get_file_diff(branch1, branch2, file_path):
    """Get diff for a specific file between two branches."""
    try:
        result = subprocess.run(
            f"git diff {branch1} {branch2} -- {file_path}",
            shell=True, capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else ""
    except:
        return ""

def get_branch_stats(branch):
    """Get statistics for a branch."""
    try:
        # Get commit count
        commits = run_git_command(f"git rev-list --count {branch}")
        commit_count = int(commits[0]) if commits and commits[0].isdigit() else 0
        
        # Get last commit
        last_commit = run_git_command(f"git log -1 --format='%h %s' {branch}")
        last_commit_msg = last_commit[0] if last_commit else "N/A"
        
        # Get file count
        files = get_branch_files(branch)
        file_count = len(files)
        
        # Get Python files
        python_files = [f for f in files if f.endswith('.py')]
        
        # Get notebook files
        notebook_files = [f for f in files if f.endswith('.ipynb')]
        
        # Get config files
        config_files = [f for f in files if f.endswith(('.yaml', '.yml', '.json'))]
        
        return {
            'commit_count': commit_count,
            'last_commit': last_commit_msg,
            'file_count': file_count,
            'python_files': len(python_files),
            'notebook_files': len(notebook_files),
            'config_files': len(config_files),
            'all_files': files
        }
    except Exception as e:
        print(f"Error getting stats for {branch}: {e}")
        return None

def compare_branches(branch1, branch2):
    """Compare two branches and return differences."""
    try:
        # Get files unique to each branch
        files1 = set(get_branch_files(branch1))
        files2 = set(get_branch_files(branch2))
        
        unique_to_1 = files1 - files2
        unique_to_2 = files2 - files1
        common = files1 & files2
        
        # Get modified files
        modified = []
        for file in common:
            diff = get_file_diff(branch1, branch2, file)
            if diff:
                # Count lines changed
                lines_added = len([l for l in diff.split('\n') if l.startswith('+') and not l.startswith('+++')])
                lines_removed = len([l for l in diff.split('\n') if l.startswith('-') and not l.startswith('---')])
                if lines_added > 0 or lines_removed > 0:
                    modified.append({
                        'file': file,
                        'lines_added': lines_added,
                        'lines_removed': lines_removed
                    })
        
        return {
            'unique_to_1': list(unique_to_1),
            'unique_to_2': list(unique_to_2),
            'common': list(common),
            'modified': modified
        }
    except Exception as e:
        print(f"Error comparing {branch1} and {branch2}: {e}")
        return None

def analyze_codebase():
    """Main analysis function."""
    print("="*80)
    print("COMPREHENSIVE BRANCH ANALYSIS")
    print("="*80)
    
    # Get all branches
    branches = get_all_branches()
    print(f"\nFound {len(branches)} branches:")
    for i, branch in enumerate(branches, 1):
        print(f"  {i}. {branch}")
    
    # Get current branch
    current = run_git_command("git branch --show-current")
    current_branch = current[0] if current else "unknown"
    print(f"\nCurrent branch: {current_branch}")
    
    # Analyze each branch
    print("\n" + "="*80)
    print("BRANCH STATISTICS")
    print("="*80)
    
    branch_stats = {}
    for branch in branches:
        print(f"\nAnalyzing {branch}...")
        stats = get_branch_stats(branch)
        if stats:
            branch_stats[branch] = stats
            print(f"  Commits: {stats['commit_count']}")
            print(f"  Files: {stats['file_count']} (Python: {stats['python_files']}, Notebooks: {stats['notebook_files']}, Configs: {stats['config_files']})")
            print(f"  Last commit: {stats['last_commit']}")
    
    # Compare main branches
    main_branches = ['main', 'avinash', 'fedbuff', 'staleness', 'TrustWeight']
    existing_main = [b for b in main_branches if b in branches]
    
    if len(existing_main) > 1:
        print("\n" + "="*80)
        print("BRANCH COMPARISONS")
        print("="*80)
        
        # Compare main with others
        base_branch = 'main' if 'main' in branches else existing_main[0]
        
        for branch in existing_main:
            if branch != base_branch:
                print(f"\n{base_branch} vs {branch}:")
                comparison = compare_branches(base_branch, branch)
                if comparison:
                    print(f"  Files unique to {base_branch}: {len(comparison['unique_to_1'])}")
                    print(f"  Files unique to {branch}: {len(comparison['unique_to_2'])}")
                    print(f"  Common files: {len(comparison['common'])}")
                    print(f"  Modified files: {len(comparison['modified'])}")
                    
                    if comparison['unique_to_2']:
                        print(f"\n  Files only in {branch}:")
                        for f in sorted(comparison['unique_to_2'])[:10]:
                            print(f"    - {f}")
                        if len(comparison['unique_to_2']) > 10:
                            print(f"    ... and {len(comparison['unique_to_2']) - 10} more")
                    
                    if comparison['modified']:
                        print(f"\n  Most modified files in {branch}:")
                        sorted_modified = sorted(comparison['modified'], key=lambda x: x['lines_added'] + x['lines_removed'], reverse=True)
                        for mod in sorted_modified[:10]:
                            print(f"    - {mod['file']}: +{mod['lines_added']} -{mod['lines_removed']}")
    
    # Key directories to analyze
    key_dirs = ['FedAsync', 'FedBuff', 'TrustWeight', 'utils', 'baseline']
    
    print("\n" + "="*80)
    print("KEY DIRECTORY ANALYSIS")
    print("="*80)
    
    for branch in existing_main[:3]:  # Analyze top 3 branches
        print(f"\n{branch}:")
        files = branch_stats.get(branch, {}).get('all_files', [])
        for dir_name in key_dirs:
            dir_files = [f for f in files if f.startswith(dir_name + '/')]
            if dir_files:
                print(f"  {dir_name}/: {len(dir_files)} files")
                # Show key files
                key_files = [f for f in dir_files if any(x in f for x in ['client.py', 'server.py', 'run.py', 'config', 'strategy.py'])]
                for kf in key_files[:5]:
                    print(f"    - {kf}")
    
    # Save results
    output = {
        'branches': branches,
        'current_branch': current_branch,
        'branch_stats': {k: {key: val for key, val in v.items() if key != 'all_files'} for k, v in branch_stats.items()},
        'comparisons': {}
    }
    
    # Add comparisons
    if len(existing_main) > 1:
        base = existing_main[0]
        for branch in existing_main[1:]:
            comp = compare_branches(base, branch)
            if comp:
                output['comparisons'][f"{base}_vs_{branch}"] = {
                    'unique_to_base': len(comp['unique_to_1']),
                    'unique_to_branch': len(comp['unique_to_2']),
                    'common': len(comp['common']),
                    'modified': len(comp['modified']),
                    'unique_files': comp['unique_to_2'][:20],
                    'top_modified': sorted(comp['modified'], key=lambda x: x['lines_added'] + x['lines_removed'], reverse=True)[:10]
                }
    
    with open('branch_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Analysis saved to branch_analysis.json")
    
    return output

if __name__ == '__main__':
    analyze_codebase()

