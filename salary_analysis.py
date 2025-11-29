#!/usr/bin/env python3
"""
Salary Analysis: AU vs NZ Cost Savings
Handles multi-line data automatically
"""

import pandas as pd
import numpy as np
import ast
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class IRDValidator:
    """Validates New Zealand IRD numbers."""
    
    @staticmethod
    def validate(ird_number: int) -> bool:
        """Validate NZ IRD number using checksum algorithm."""
        ird_str = str(ird_number)
        
        if len(ird_str) < 8 or len(ird_str) > 9:
            return False
        
        ird_str = ird_str.zfill(9)
        weights = [3, 2, 7, 6, 5, 4, 3, 2]
        
        try:
            digits = [int(d) for d in ird_str]
            weighted_sum = sum(d * w for d, w in zip(digits[:8], weights))
            remainder = weighted_sum % 11
            check_digit = 0 if remainder == 0 else 11 - remainder
            
            if check_digit == 10:
                return False
            
            return check_digit == digits[8]
        except (ValueError, IndexError):
            return False


class SalaryDataCleaner:
    """Clean and validate salary data."""
    
    def __init__(self, raw_data: str, exchange_rate: float):
        self.raw_data = raw_data
        self.exchange_rate = exchange_rate
        self.ird_validator = IRDValidator()
        
    def fix_line_breaks(self, text: str) -> str:
        """Merge lines that were broken up."""
        lines = text.strip().split('\n')
        fixed_lines = []
        current_line = ""
        
        print("Fixing line breaks in data...")
        
        for line in lines:
            line = line.strip()
            
            # Skip header
            if line.startswith('pay1'):
                continue
            
            if not line:
                continue
            
            # Start of a new record (starts with ")
            if line.startswith('"[{'):
                # Save previous record if exists
                if current_line:
                    fixed_lines.append(current_line)
                current_line = line
            else:
                # Continuation of previous line
                current_line += " " + line
        
        # Don't forget the last line
        if current_line:
            fixed_lines.append(current_line)
        
        print(f"Fixed {len(fixed_lines)} records")
        return '\n'.join(fixed_lines)
        
    def parse_data(self) -> pd.DataFrame:
        """Parse the messy string data."""
        # First, fix line breaks
        fixed_data = self.fix_line_breaks(self.raw_data)
        
        rows = []
        lines = fixed_data.strip().split('\n')
        
        print(f"Parsing {len(lines)} records...")
        
        for i, line in enumerate(lines):
            if i % 10 == 0 and i > 0:
                print(f"  Processed {i} records...")
                
            parts = self._split_pay_data(line)
            if len(parts) != 2:
                print(f"  Warning: Could not split line {i+1}")
                continue
            
            nz_data = self._parse_dict_list(parts[0])
            au_data = self._parse_dict_list(parts[1])
            
            rows.append({'nz_raw': nz_data, 'au_raw': au_data})
        
        return pd.DataFrame(rows)
    
    def _split_pay_data(self, line: str) -> List[str]:
        """Split pay data on separator."""
        line = line.strip()
        if not line:
            return []
        
        # Split by ","
        parts = line.split('","')
        if len(parts) == 2:
            return [parts[0].strip('"\''), parts[1].strip('"\'')]
        return []
    
    def _parse_dict_list(self, dict_str: str) -> Dict:
        """Parse string of list of dicts."""
        dict_str = dict_str.strip(' "\'')
        
        try:
            parsed = ast.literal_eval(dict_str)
            if isinstance(parsed, list):
                result = {}
                for item in parsed:
                    if isinstance(item, dict):
                        result.update(item)
                return result
        except (SyntaxError, ValueError) as e:
            pass
        
        result = {}
        pattern = r"['\"]?(\w+(?:\s+\w+)*)['\"]?\s*:\s*([0-9.]+)"
        matches = re.findall(pattern, dict_str)
        
        for key, value in matches:
            key = key.strip().replace(' ', '_').lower()
            try:
                result[key] = float(value) if '.' in value else int(value)
            except ValueError:
                result[key] = value
        
        return result
    
    def _normalize_key(self, data: Dict, possible_keys: List[str]) -> Optional[str]:
        """Find key from variations."""
        for key in possible_keys:
            if key in data:
                return key
        return None
    
    def extract_structured_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract structured data with typo handling."""
        records = []
        
        print("Extracting structured data...")
        
        for idx, row in df.iterrows():
            nz = row['nz_raw']
            au = row['au_raw']
            
            # Handle typos
            nz_base_key = self._normalize_key(nz, ['base_pay', 'bese_pay'])
            nz_shift_key = self._normalize_key(nz, ['shift_loading', 'shift_load'])
            au_shift_key = self._normalize_key(au, ['shift_loading', 'shift_load'])
            au_emp_key = self._normalize_key(au, ['emp_number', 'emp_no', 'emp_numbar'])
            nz_kiwi_key = self._normalize_key(nz, ['kiwi_saver_employee', 'kiwi_sever_employee'])
            
            record = {
                'row_index': idx,
                'nz_base_pay': nz.get(nz_base_key, 0) if nz_base_key else 0,
                'nz_shift_loading': nz.get(nz_shift_key, 0) if nz_shift_key else 0,
                'nz_net': nz.get('net', 0),
                'nz_kiwisaver': nz.get(nz_kiwi_key, 0) if nz_kiwi_key else 0,
                'nz_ird_number': nz.get('ird_number', nz.get('ird_no', 0)),
                'nz_future_positions': nz.get('future_total_positions', 0),
                'au_base_pay': au.get('base_pay', 0),
                'au_shift_loading': au.get(au_shift_key, 0) if au_shift_key else 0,
                'au_net': au.get('net', 0),
                'au_sg': au.get('sg', 0),
                'au_emp_number': au.get(au_emp_key, 0) if au_emp_key else 0,
                'au_future_positions': au.get('future_total_positions', au.get('future_total_postions', 0)),
            }
            
            record['nz_gross'] = record['nz_base_pay'] + record['nz_shift_loading']
            record['au_gross'] = record['au_base_pay'] + record['au_shift_loading']
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean data with validation."""
        df = df.copy()
        df['used'] = True
        df['rejection_reason'] = ''
        
        print("Validating data...")
        
        # 1. Invalid IRD numbers
        print("  Checking IRD numbers...")
        invalid_ird = ~df['nz_ird_number'].apply(self.ird_validator.validate)
        df.loc[invalid_ird, 'used'] = False
        df.loc[invalid_ird, 'rejection_reason'] = 'Invalid IRD number'
        
        # 2. Duplicates
        print("  Checking for duplicates...")
        dup_groups = df.groupby(['nz_ird_number', 'au_emp_number'])
        
        for name, group in dup_groups:
            if len(group) > 1:
                nz_diff = group['nz_gross'].max() - group['nz_gross'].min()
                au_diff = group['au_gross'].max() - group['au_gross'].min()
                
                if nz_diff < 1 and au_diff < 1:
                    keep_idx = group['nz_gross'].idxmin()
                    for idx in group.index:
                        if idx != keep_idx:
                            df.loc[idx, 'used'] = False
                            df.loc[idx, 'rejection_reason'] = 'Duplicate (kept lowest)'
                else:
                    for idx in group.index:
                        df.loc[idx, 'used'] = False
                        df.loc[idx, 'rejection_reason'] = 'Conflicting data'
        
        # 3. Net pay validation (lenient - government levies might not be included)
        print("  Validating net pay ratios...")
        for idx, row in df.iterrows():
            if not row['used']:
                continue
            
            nz_ratio = row['nz_net'] / row['nz_gross'] if row['nz_gross'] > 0 else 0
            au_ratio = row['au_net'] / row['au_gross'] if row['au_gross'] > 0 else 0
            
            if nz_ratio < 0.4 or nz_ratio > 0.95:
                df.loc[idx, 'used'] = False
                df.loc[idx, 'rejection_reason'] = 'Suspicious net/gross ratio'
            elif au_ratio < 0.4 or au_ratio > 0.95:
                df.loc[idx, 'used'] = False
                df.loc[idx, 'rejection_reason'] = 'Suspicious net/gross ratio'
        
        # Convert to AUD
        df['nz_gross_aud'] = df['nz_gross'] * self.exchange_rate
        df['nz_net_aud'] = df['nz_net'] * self.exchange_rate
        
        return df[df['used']].copy(), df[~df['used']].copy()


def perform_statistical_test(df: pd.DataFrame) -> Dict:
    """Paired t-test for AU vs NZ (manual calculation)."""
    au_gross = df['au_gross'].values
    nz_gross_aud = df['nz_gross_aud'].values
    
    differences = au_gross - nz_gross_aud
    n = len(differences)
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    t_stat = mean_diff / se_diff
    
    # Critical value at 0.05 for n=36 is about 2.03
    significant = abs(t_stat) > 2.03
    
    return {
        't_statistic': t_stat,
        'au_mean': au_gross.mean(),
        'nz_mean': nz_gross_aud.mean(),
        'significant': significant
    }


def calculate_cost_savings(df: pd.DataFrame) -> Dict:
    """Calculate cost savings scenarios."""
    future_au = df['au_future_positions'].sum()
    future_nz = df['nz_future_positions'].sum()
    avg_au = df['au_gross'].mean()
    avg_nz_aud = df['nz_gross_aud'].mean()
    
    # Scenario 1: AU → NZ
    au_to_nz_current = future_au * avg_au
    au_to_nz_new = future_au * avg_nz_aud
    au_to_nz_savings = au_to_nz_current - au_to_nz_new
    
    # Scenario 2: NZ → AU
    nz_to_au_current = future_nz * avg_nz_aud
    nz_to_au_new = future_nz * avg_au
    nz_to_au_savings = nz_to_au_current - nz_to_au_new
    
    return {
        'current_positions': len(df),
        'future_au': int(future_au),
        'future_nz': int(future_nz),
        'avg_au': avg_au,
        'avg_nz_aud': avg_nz_aud,
        'au_to_nz_savings': au_to_nz_savings,
        'au_to_nz_percent': (au_to_nz_savings / au_to_nz_current * 100) if au_to_nz_current > 0 else 0,
        'nz_to_au_savings': nz_to_au_savings,
        'nz_to_au_percent': (nz_to_au_savings / nz_to_au_current * 100) if nz_to_au_current > 0 else 0
    }


def main():
    print("=" * 60)
    print("SALARY ANALYSIS: AU vs NZ")
    print("=" * 60)
    print()
    
    input_file = 'salary_data.txt'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        raw_data = f.read()
    
    exchange_rate = 0.92  # NZD to AUD
    print(f"Exchange rate: 1 NZD = {exchange_rate} AUD")
    print()
    
    cleaner = SalaryDataCleaner(raw_data, exchange_rate)
    df_raw = cleaner.parse_data()
    print(f"✓ Parsed {len(df_raw)} rows")
    print()
    
    df_structured = cleaner.extract_structured_data(df_raw)
    df_clean, df_rejected = cleaner.clean_data(df_structured)
    
    print()
    print(f"✓ Clean rows: {len(df_clean)}")
    print(f"✗ Rejected: {len(df_rejected)}")
    print()
    
    if len(df_clean) == 0:
        print("ERROR: No valid data! Check rejection reasons in results.csv")
        # Still save results for debugging
        all_data = df_rejected
    else:
        # Statistical test
        print("=" * 60)
        print("STATISTICAL TEST")
        print("=" * 60)
        test = perform_statistical_test(df_clean)
        print(f"Sample size: {len(df_clean)} paired observations")
        print(f"AU mean: ${test['au_mean']:,.2f} AUD")
        print(f"NZ mean: ${test['nz_mean']:,.2f} AUD")
        print(f"Difference: ${test['nz_mean'] - test['au_mean']:,.2f}")
        print(f"t-statistic: {test['t_statistic']:.4f}")
        print(f"Significant: {'YES - NZ is higher' if test['significant'] else 'NO'}")
        print()
        
        # Cost savings
        print("=" * 60)
        print("COST SAVINGS ANALYSIS")
        print("=" * 60)
        savings = calculate_cost_savings(df_clean)
        print(f"Future AU positions: {savings['future_au']}")
        print(f"Future NZ positions: {savings['future_nz']}")
        print()
        print(f"Scenario 1 (AU→NZ): ${savings['au_to_nz_savings']:,.2f} ({savings['au_to_nz_percent']:.1f}%)")
        if savings['au_to_nz_savings'] < 0:
            print(f"  ⚠ COST INCREASE")
        print()
        print(f"Scenario 2 (NZ→AU): ${savings['nz_to_au_savings']:,.2f} ({savings['nz_to_au_percent']:.1f}%)")
        if savings['nz_to_au_savings'] > 0:
            print(f"  ✓ RECOMMENDED")
        print()
        
        all_data = pd.concat([df_clean, df_rejected], ignore_index=True)
    
    # Save results
    print("Saving results...")
    output = pd.DataFrame({
        'IRD_Number': all_data['nz_ird_number'],
        'Employee_Number': all_data['au_emp_number'],
        'AU_Gross': all_data['au_gross'].round(2),
        'AU_Net': all_data['au_net'].round(2),
        'NZ_Gross_NZD': all_data['nz_gross'].round(2),
        'NZ_Gross_AUD': all_data['nz_gross_aud'].round(2),
        'NZ_Net_NZD': all_data['nz_net'].round(2),
        'NZ_Net_AUD': all_data['nz_net_aud'].round(2),
        'Used': all_data['used'],
        'Rejection_Reason': all_data['rejection_reason']
    })
    
    output.to_csv('results.csv', index=False)
    print("✓ Saved results.csv")
    print()
    print("=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()