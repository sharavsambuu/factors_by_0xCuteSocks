# factors
```python
def inf_ratio(depth=None, trade=None, n=100):
    quasi = trade.p.diff().abs().rolling(n).sum().fillna(10)
    dif = trade.p.diff(n).abs().fillna(10)
    return quasi / (dif + quasi)
 
def depth_price_range(depth=None, trade=None, n=100):
    return (depth.ap1.rolling(n).max() / depth.ap1.rolling(n).min() - 1).fillna(0)

@nb.jit
def get_age(prices):
    last_value = prices[-1]
    age = 0
    for i in range(2, len(prices)):
        if prices[-i] != last_value:
            return age
        age += 1
    return age

def (ask/bid)_age(depth, trade, n=100):
        bp1 = depth['bp1']
        bp1_changes = bp1.rolling(n).apply(get_age, engine='numba', raw=True).fillna(0)
        return bp1_changes

def arrive_rate(depth, trade, n=300):
    res = trade['ts'].diff(n).fillna(0) / n
    return res

def cofi(depth, trade):
    a = depth['bv1']*np.where(depth['bp1'].diff()>=0,1,0)
    b = depth['bv1'].shift()*np.where(depth['bp1'].diff()<=0,1,0)
    c = depth['av1']*np.where(depth['ap1'].diff()<=0,1,0)
    d = depth['av1'].shift()*np.where(depth['ap1'].diff()>=0,1,0)
    return (a - b - c + d).fillna(0)

def bp_rank(depth, trade, n=100):
    return ((depth.bp1.rolling(n).rank())/n*2 - 1).fillna(0)

def ap_rank(depth, trade, n=100):
    return ((depth.ap1.rolling(n).rank())/n*2 - 1).fillna(0)

def price_impact(depth, trade, n=10):
    ask, bid, ask_v, bid_v = 0, 0, 0, 0
    for i in range(1, n+1):
        ask += depth[f'ap{i}'] * depth[f'av{i}']
        bid += depth[f'bp{i}'] * depth[f'bv{i}']
        ask_v += depth[f'av{i}']
        bid_v += depth[f'bv{i}']
    ask /= ask_v
    bid /= bid_v
    return pd.Series(-(depth['ap1'] - ask)/depth['ap1'] - (depth['bp1'] - bid)/depth['bp1'], name="price_impact")

def depth_price_skew(depth, trade):
    prices = ["bp5", "bp4", "bp3", "bp2", "bp1", "ap1", "ap2", "ap3", "ap4", "ap5"]
    return depth[prices].skew(axis=1)

def depth_price_kurt(depth, trade):
    prices = ["bp5", "bp4", "bp3", "bp2", "bp1", "ap1", "ap2", "ap3", "ap4", "ap5"]
    return depth[prices].kurt(axis=1)

def rolling_return(depth, trade, n=100):
    mp = ((depth.ap1 + depth.bp1)/2)
    return (mp.diff(n) / mp).fillna(0)

def buy_increasing(depth, trade, n=100):
    v = trade.v.copy()
    v[v<0] = 0
    return np.log1p(((v.rolling(2*n).sum() + 1) / (v.rolling(n).sum() + 1)).fillna(1))

def sell_increasing(depth, trade, n=100):
    v = trade.v.copy()
    v[v>0] = 0
    return np.log1p(((v.rolling(2*n).sum() - 1) / (v.rolling(n).sum() - 1)).fillna(1))

@nb.jit
def first_location_of_maximum(x):
    max_value=max(x)#一个for 循环
    for loc in range(len(x)):
        if x[loc]==max_value:
            return loc+1
        
def price_idxmax(depth, trade, n=20):
    return depth['ap1'].rolling(n).apply(first_location_of_maximum,engine='numba',raw=True).fillna(0)

@nb.jit
def mean_second_derivative_centra(x):
    sum_value=0
    for i in range(len(x)-5):
        sum_value+=(x[i+5]-2*x[i+3]+x[i])/2
    return sum_value/(2*(len(x)-5))

def center_deri_two(depth, trade, n=20):
    return depth['ap1'].rolling(n).apply(mean_second_derivative_centra,engine='numba',raw=True).fillna(0)

def quasi(depth, trade, n=100):
    return depth.ap1.diff(1).abs().rolling(n).sum().fillna(0)

@trade_to_depth
def last_range(depth, trade, n=100):
    return trade.p.diff(1).abs().rolling(n).sum().fillna(0)

@trade_to_depth
def arrive_rate(depth, trade, n=100):
    return (trade.ts.shift(n) - trade.ts).fillna(0)

@trade_to_depth
def avg_trade_volume(depth, trade, n=100):
    return (trade.v[::-1].abs().rolling(n).sum().shift(-n+1)).fillna(0)[::-1]

def avg_spread(depth, trade, n=200):
    return (depth.ap1 - depth.bp1).rolling(n).mean().fillna(0)

def avg_turnover(depth, trade, n=500):
    return depth[['av1', 'av2', 'av3', 'av4', ....., 'av10', 'bv1', 'bv2', 'bv3', 'bv4', ....., 'bv10']].sum(axis=1)

@trade_to_depth
def abs_volume_kurt(depth, trade, n=500):
    return trade.v.abs().rolling(n).kurt().fillna(0)

@trade_to_depth
def abs_volume_skew(depth, trade, n=500):
    return trade.v.abs().rolling(n).skew().fillna(0)

@trade_to_depth
def volume_kurt(depth, trade, n=500):
    return trade.v.rolling(n).kurt().fillna(0)

@trade_to_depth
def volume_skew(depth, trade, n=500):
    return trade.v.rolling(n).skew().fillna(0)

@trade_to_depth
def price_kurt(depth, trade, n=500):
    return trade.p.rolling(n).kurt().fillna(0)

@trade_to_depth
def price_skew(depth, trade, n=500):
    return trade.p.rolling(n).skew().abs().fillna(0)

def bv_divide_tn(depth, trade, n=10):
    bvs = depth.bv1 + depth.bv2 +  ... + depth.bv10
    @trade_to_depth
    def volume(depth, trade, n):
        return trade.v
    v = volume(depth=depth, trade=trade, n=n)
    v[v>0] = 0
    return (v.rolling(n).sum() / bvs).fillna(0)

def av_divide_tn(depth, trade, n=10):
    avs = depth.av1 + depth.av2 +  ... + depth.av10
    @trade_to_depth
    def volume(depth, trade, n):
        return trade.v
    v = volume(depth=depth, trade=trade, n=n)
    v[v<0] = 0
    return (v.rolling(n).sum() / avs).fillna(0)

def weighted_price_to_mid(depth, trade, levels=10, alpha=1):
    def get_columns(name, levels):
        return [name+str(i) for i in range(1, levels+1)]
    avs = depth[get_columns("av", levels)]
    bvs =  depth[get_columns("bv", levels)]
    aps = depth[get_columns("ap", levels)]
    bps =  depth[get_columns("bp", levels)]
    mp = (depth['ap1'] + depth['bp1'])/2
    return (avs.values * aps.values + bvs.values * bps.values).sum(axis=1) / (avs.values + bvs.values).sum(axis=1) - mp
   
@nb.njit
def _ask_withdraws_volume(l, n, levels=10):
    withdraws = 0
    for price_index in range(2,2+4*levels, 4):
        now_p = n[price_index]
        for price_last_index in range(2,2+4*levels,4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index+1] - l[price_last_index + 1], 0)
        
    return withdraws

@nb.njit
def _bid_withdraws_volume(l, n, levels=10):
    withdraws = 0
    for price_index in range(0,4*levels, 4):
        now_p = n[price_index]
        for price_last_index in range(0,4*levels,4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index+1] - l[price_last_index + 1], 0)
        
    return withdraws

def ask_withdraws(depth, trade):
    ob_values = depth.values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _ask_withdraws_volume(ob_values[i-1], ob_values[i])
    return pd.Series(flows)

def bid_withdraws(depth, trade):
    ob_values = depth.values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _bid_withdraws_volume(ob_values[i-1], ob_values[i])
    return pd.Series(flows)
```