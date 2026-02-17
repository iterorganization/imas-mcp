# Facility Access Patterns

SSH access and data system patterns per facility.

## SSH Hosts

| Facility | SSH Alias | Data System | Reference Shot |
|----------|-----------|-------------|----------------|
| TCV | `tcv` | MDSplus + TDI | 84000 |
| JET | `jet` | MDSplus + PPF | 99000 |
| JT-60SA | `jt-60sa` | MDSplus | 30000 |
| ITER | `iter` | IMAS | N/A |

## Remote Tools

Always prefer these over standard Unix commands:
- `rg` over `grep -r` (pattern search)
- `fd` over `find` (file finder — **requires path argument on large FS**)
- `eza` over `ls -la` / `tree` (directory listing)
- `tokei` over `wc -l` (LOC by language)

Install on facility: `imas-codex tools install <facility>`

## TCV

### MDSplus Direct
```python
import MDSplus
tree = MDSplus.Tree('tcv_shot', 84000, 'readonly')
ip = tree.getNode('\\RESULTS::LIUQE:I_P').data()
```

### TDI Functions
```python
import MDSplus
conn = MDSplus.Connection('tcvdata.epfl.ch')
conn.openTree('tcv_shot', 84000)
ip = conn.get('tcv_eq("I_P")').data()
time = conn.get('dim_of(tcv_eq("I_P"))').data()
```

### Key Trees
- `tcv_shot`: Raw diagnostic data
- `results`: Analysis code outputs (LIUQE, ASTRA, etc.)
- `atlas`: Magnetics and basic plasma params

### TDI Function Directory
`/usr/local/mdsplus/tdi/tcv/` — contains `.FUN` files

## JET

### PPF Access
```python
import ppf
ppf.ppfgo(pulse=99000)
data, x, t, ier = ppf.ppfget(pulse=99000, dda='EFIT', dtype='Q95')
```

### MDSplus
```python
import MDSplus
conn = MDSplus.Connection('mdsjet.jet.uk')
conn.openTree('jet', 99000)
```

### Key DDAs
- `EFIT`: Equilibrium reconstruction
- `KK3`: ECE diagnostic
- `HRTS`: High-res Thomson scattering
- `BOLO`: Bolometry

## JT-60SA

### MDSplus
```python
import MDSplus
tree = MDSplus.Tree('jt-60sa', 30000, 'readonly')
```

## Safety Rules

1. **Read-only**: Never modify remote data
2. **Check excludes first**: `get_facility('tcv')['excludes']`
3. **Timeout awareness**: Commands on `/work` or `/home` can hang on NFS
4. **fd needs path**: `fd -e py /specific/path` — never bare `fd` on large FS
5. **Persist timeouts**: If a path times out, immediately call `update_infrastructure()`
