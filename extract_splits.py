"""Quick extraction of AEQ/IEQ active-passive and internal-external splits."""
import openpyxl
from collections import defaultdict

wb = openpyxl.load_workbook(
    'VFMC.Daily.20260430.CLIENT_DTF-Act3.AssetClass.xlsx',
    read_only=True, data_only=True
)
ws = wb['RiskReport']

# Header row is 21. Data starts at row 22.
# Key column positions (0-indexed):
# 1=Level, 2=Name, 3=RM PV AUD (Total), 11=*VFMC_Asset Class
# 133=*VFMC_Asset Class Category, 134=*VFMC_Sub Asset Class
# 135=*VFMC_Strategy, 140=*VFMC_Manager, 141=*VFMC_ManagerName, 142=*VFMC_Int_Ext

current_ac = None
aeq_mgrs = defaultdict(lambda: {'pv': 0.0, 'int_ext': '', 'strategy': '', 'mgr_code': ''})
ieq_mgrs = defaultdict(lambda: {'pv': 0.0, 'int_ext': '', 'strategy': '', 'mgr_code': ''})

row_count = 0
for row in ws.iter_rows(min_row=22, values_only=True):
    row_count += 1
    if row is None or len(row) < 143:
        continue

    level = row[1]
    name = str(row[2]).strip() if row[2] else ''
    pv = row[3]
    mgr_code = str(row[140]).strip() if row[140] else ''
    mgr_name = str(row[141]).strip() if row[141] else ''
    int_ext  = str(row[142]).strip() if row[142] else ''
    strategy = str(row[135]).strip() if row[135] else ''

    if not isinstance(pv, (int, float)):
        try:
            pv = float(str(pv).replace(',', '')) if pv else 0.0
        except (ValueError, TypeError):
            pv = 0.0

    if level == 0:
        nl = name.lower()
        if 'australian equit' in nl:
            current_ac = 'AEQ'
        elif any(k in nl for k in ['international equit', 'emerging market',
                                     'low volatility equit']):
            current_ac = 'IEQ'
        else:
            current_ac = None
        continue

    # Use leaf-level rows that have *VFMC_ManagerName populated
    if current_ac and mgr_name and mgr_name not in ('None', '*Unspecified', ''):
        if not isinstance(pv, (int, float)):
            continue
        if abs(pv) < 0.01:
            continue
        key = mgr_name
        tgt = aeq_mgrs if current_ac == 'AEQ' else ieq_mgrs
        tgt[key]['pv'] += pv
        if int_ext and int_ext not in ('None', '*Unspecified', ''):
            tgt[key]['int_ext'] = int_ext
        if mgr_code and mgr_code not in ('None', '*Unspecified', ''):
            tgt[key]['mgr_code'] = mgr_code
        if strategy and strategy not in ('None', '*Unspecified', ''):
            tgt[key]['strategy'] = strategy

wb.close()

# Debug: show what we found
print("AEQ managers found:")
for k, v in sorted(aeq_mgrs.items(), key=lambda x: x[1]['pv'], reverse=True):
    print(f"  {k}: PV={v['pv']/1e6:.1f}m, Int/Ext='{v['int_ext']}', MC='{v['mgr_code']}', Strat='{v['strategy']}'")
print(f"\nIEQ managers found:")
for k, v in sorted(ieq_mgrs.items(), key=lambda x: x[1]['pv'], reverse=True):
    print(f"  {k}: PV={v['pv']/1e6:.1f}m, Int/Ext='{v['int_ext']}', MC='{v['mgr_code']}', Strat='{v['strategy']}'")


# ── Classification functions ──────────────────────────────────────────
def classify_aeq(name, mc):
    e, m = name.lower(), mc.lower()
    for p in ['cooper investor', 'greencape', 'vinva', 'alphinity',
              'platypus', 'yarra', 'ifm', 'paradice']:
        if p in e or p in m: return 'Active'
    for p in ['obi', 'opportunistic', 'vader', 'imp']:
        if p in e or p in m: return 'Active'
    for p in ['state street', 'ssga', 'ssg']:
        if p in e or p in m: return 'Passive'
    for p in ['asx20', 'asx 20', 'plug']:
        if p in e or p in m: return 'Passive'
    return 'Active'

def classify_ieq(name, mc):
    e, m = name.lower(), mc.lower()
    for p in ['arrowstreet', 'wellington', 'sanders', 'c worldwide',
              'c worldw', 'orbis', 'artisan', 'wasatch', 'rwc',
              'jennison', 'gsam', 'goldman sachs', 'goldman']:
        if p in e or p in m: return 'Active'
    for p in ['elvis', 'quality plus', 'qualityplus', 'qlpl',
              'nvidia', 'nvda', 'plug']:
        if p in e or p in m: return 'Active'
    for p in ['state street', 'ssga', 'ssg', 'low carbon',
              'lchosg', 'lcwasg', 'optimised']:
        if p in e or p in m: return 'Passive'
    for p in ['total return swap', 'trs', 'swap', 'minvol', 'msvl',
              'low vol', 'lowvol', 'minimum volatility']:
        if p in e or p in m: return 'Passive'
    if 'ifm' in e or 'ifm' in m: return 'Active'
    return 'Active'

# ── Helpers ───────────────────────────────────────────────────────────
def fmt(v):
    sign = '-' if v < 0 else ''
    return f"${sign}{abs(v):,.1f}m"

def pct(v, t):
    return f"{v / t * 100:.1f}%" if t else "0.0%"

def cell(v, t):
    return f"{fmt(v / 1e6)} ({pct(v, t)})"

def is_int(ie):
    return ie and 'int' in str(ie).lower()

def is_ext(ie):
    return ie and 'ext' in str(ie).lower()

# ── Print table ───────────────────────────────────────────────────────
def print_table(label, managers, classify_fn):
    items = sorted(managers.items(), key=lambda x: x[1]['pv'], reverse=True)
    total = sum(v['pv'] for _, v in items)
    total_m = total / 1e6

    for k, v in items:
        v['ap'] = classify_fn(k, v['mgr_code'])

    print()
    print("=" * 90)
    print(f"  {label} as at 30-Apr-2026")
    print("=" * 90)
    print(f"{'Manager':<35} | {'PV ($m)':>12} | {'Active/Passive':<14} | {'Int/Ext':<10}")
    print("-" * 80)

    for name, v in items:
        pv_m = v['pv'] / 1e6
        ie = v['int_ext'] if v['int_ext'] and v['int_ext'] != 'None' else 'N/A'
        print(f"{name[:35]:<35} | {fmt(pv_m):>12} | {v['ap']:<14} | {ie:<10}")

    print("-" * 80)
    print(f"{'TOTAL':>35} | {fmt(total_m):>12}")

    active  = sum(v['pv'] for _, v in items if v['ap'] == 'Active')
    passive = sum(v['pv'] for _, v in items if v['ap'] == 'Passive')
    internal = sum(v['pv'] for _, v in items if is_int(v['int_ext']))
    external = sum(v['pv'] for _, v in items if is_ext(v['int_ext']))

    print(f"\n  SUMMARY - {label}:")
    print(f"    Active:    {fmt(active / 1e6):>14}  ({pct(active, total)})")
    print(f"    Passive:   {fmt(passive / 1e6):>14}  ({pct(passive, total)})")
    print(f"    Internal:  {fmt(internal / 1e6):>14}  ({pct(internal, total)})")
    print(f"    External:  {fmt(external / 1e6):>14}  ({pct(external, total)})")

    ai = sum(v['pv'] for _, v in items if v['ap'] == 'Active'  and is_int(v['int_ext']))
    ae = sum(v['pv'] for _, v in items if v['ap'] == 'Active'  and is_ext(v['int_ext']))
    pi = sum(v['pv'] for _, v in items if v['ap'] == 'Passive' and is_int(v['int_ext']))
    pe = sum(v['pv'] for _, v in items if v['ap'] == 'Passive' and is_ext(v['int_ext']))

    w = 22
    print(f"\n  2x2 Matrix:")
    print(f"  {'':<18} | {'Internal':>{w}} | {'External':>{w}} | {'Total':>{w}}")
    print(f"  {'-' * 72}")
    print(f"  {'Active':<18} | {cell(ai, total):>{w}} | {cell(ae, total):>{w}} | {cell(ai+ae, total):>{w}}")
    print(f"  {'Passive':<18} | {cell(pi, total):>{w}} | {cell(pe, total):>{w}} | {cell(pi+pe, total):>{w}}")

    total_str = fmt(total_m) + " (100.0%)"
    print(f"  {'Total':<18} | {cell(ai+pi, total):>{w}} | {cell(ae+pe, total):>{w}} | {total_str:>{w}}")


# ── Run ───────────────────────────────────────────────────────────────
print(f"Rows scanned: {row_count}")
print(f"AEQ managers found: {len(aeq_mgrs)}")
print(f"IEQ managers found: {len(ieq_mgrs)}")

if aeq_mgrs:
    print_table("AEQ (Australian Equities)", aeq_mgrs, classify_aeq)
if ieq_mgrs:
    print_table("IEQ (International Equities)", ieq_mgrs, classify_ieq)

if not aeq_mgrs and not ieq_mgrs:
    print("\n[!] No AEQ or IEQ data found. Check Level/AC hierarchy.")
