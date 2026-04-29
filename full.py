import os
import json
import numpy as np
import pandas as pd
from brian2 import *
import warnings
from collections import defaultdict
from collections import deque
from brian2 import start_scope
start_scope()
# ========== USER SETTINGS ==========
CSV_NEURONS_PATH = 'unique_ids.csv'
PARQUET_CONN_PATH = 'connectivity_data (1).parquet'
RESULT_DIR = 'results'
os.makedirs(RESULT_DIR, exist_ok=True)

# Simulation parameters
t_run = 2000 * ms
stimulus_rate = 30 * Hz
stimulus_weight = 30 * pA
threshold_mV = -45.0
# ========== LOAD NEURON IDs ==========
print("Loading neuron IDs from CSV...")
neu_sugar = []
try:
    df_neurons = pd.read_csv(CSV_NEURONS_PATH, header=None)
    for val in df_neurons[0]:
        try:
            neu_sugar.append(int(str(val).strip()))
        except ValueError:
            print(f"⚠️ Skipping invalid row in CSV: {val}")
except Exception as e:
    print(f"⚠️ Error reading CSV: {e}")
    with open(CSV_NEURONS_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    neu_sugar.append(int(line))
                except ValueError:
                    print(f"⚠️ Skipping non-integer line: {line}")
neu_sugar = [x for x in neu_sugar if x]
N = len(neu_sugar)
fwid_to_idx = {fwid: idx for idx, fwid in enumerate(neu_sugar)}
print(f"✅ Loaded {N} neurons from CSV")
print([type(n) for n in neu_sugar[:5]])
# ========== LOAD CONNECTIVITY ==========
print("Loading connectivity from Parquet...")
df_conn = pd.read_parquet(PARQUET_CONN_PATH)
print(f"📊 Total connections in Parquet: {len(df_conn)}")
# ========== EXTRACT VALID CONNECTIONS ==========
pre_idx_list, post_idx_list, weight_list, conn_type_list = [], [], [], []

for _, row in df_conn.iterrows():
    pre_id = int(row["Presynaptic_ID"])
    post_id = int(row["Postsynaptic_ID"])

    if pre_id in fwid_to_idx and post_id in fwid_to_idx:
        pre_idx = fwid_to_idx[pre_id]
        post_idx = fwid_to_idx[post_id]

        if int(row.get("Connectivity", 0)) != 1:
            continue

        is_excitatory = int(row.get("Excitatory", 1))
        if is_excitatory:
            weight = 3000 * pA
            conn_type = "excitatory"
        else:
            weight = -1500 * pA
            conn_type = "inhibitory"

        pre_idx_list.append(pre_idx)
        post_idx_list.append(post_idx)
        weight_list.append(weight)
        conn_type_list.append(conn_type)
print(f"✅ Total valid connections: {len(pre_idx_list)}")
print(f"🔗 Unique presynaptic neurons: {len(set(pre_idx_list))}")
print(f"🔗 Unique postsynaptic neurons: {len(set(post_idx_list))}")
# ========== BRIAN2 SIMULATION SETUP ==========
start_scope()
prefs.codegen.target = 'numpy'
set_device('runtime')
eqs = '''
dv/dt = (gl*(El - v) + I_syn + I_ext)/Cm : volt
dI_syn/dt = -I_syn/tausyn : amp
I_ext : amp
v_at_spike : volt
spike_count : 1
'''

# Create neuron groups
all_neurons = np.unique(np.concatenate([df_conn['Presynaptic_ID'].values, df_conn['Postsynaptic_ID'].values]))
num_neurons = len(all_neurons)
neuron_index_map = {nid: i for i, nid in enumerate(all_neurons)}
reverse_index_map = {i: nid for nid, i in neuron_index_map.items()}

Cm, gl, El, tausyn, Vth = 100*pF, 5*nS, -70*mV, 10*ms, -62*mV
neurons = NeuronGroup(
    N,
    model=eqs,
    threshold='v > Vth',
    reset='''
        v_at_spike = v
        v = El
        spike_count += 1
    ''',
    method='euler'
)

# ==== MULTI-TARGET STIMULATION SETUP ====
# Example: Stimulate first 3 neurons with different rates
multi_target_ids = neu_sugar[:3]  # Or any list of neuron IDs you want to stimulate
multi_rates = [30*Hz, 50*Hz, 10*Hz]  # Match length to multi_target_ids
neurons.v = El
neurons.v_at_spike = El
neurons.spike_count = 0
neurons.I_ext = '(randn() * 100 + 200) * pA'

stim_indices = [fwid_to_idx[nid] for nid in multi_target_ids]
print(f"🎯 Stimulating neuron indices {stim_indices} with rates {multi_rates}")

# Stimulate all target neurons in stim_indices
poisson_input = PoissonInput(
    target=neurons[stim_indices],
    target_var='I_ext',
    N=1,
    rate=stimulus_rate,
    weight=stimulus_weight
)

# Create PoissonGroup with one neuron per target, each with its own rate
P = PoissonGroup(len(stim_indices), rates=multi_rates)

# Connect each Poisson neuron to its corresponding target neuron
S_stim = Synapses(P, neurons, on_pre='I_ext += stimulus_weight')
S_stim.connect(i=range(len(stim_indices)), j=stim_indices)

# ========== SYNAPTIC CONNECTIONS ==========
if len(pre_idx_list) > 0:
    synapses = Synapses(neurons, neurons, 'w : volt', on_pre="v_post += w")
    synapses.connect(i=pre_idx_list, j=post_idx_list)
    synapses.w = 15 * mV
    print(f"✅ Created {len(synapses)} synaptic connections")
else:
    synapses = None
    print("⚠️  No synapses created")

# ========== MONITORS ==========
spike_mon = SpikeMonitor(neurons)
state_mon = StateMonitor(neurons, ['v', 'spike_count'], record=True, dt=0.5*ms)
# ========== RUN SIMULATION ==========
print(f'🚀 Running simulation for {t_run}...')
net = Network(neurons, poisson_input, spike_mon, state_mon)
if synapses: net.add(synapses)
net.run(t_run)
print('✅ Simulation complete!')
import numpy as np
import pandas as pd
import os

# statemon: StateMonitor (must record 'v' and 'I' for all neurons at fine enough dt)
# spike_mon: SpikeMonitor (for spikes)
# reverse_index_map: index → neuron id mapping built earlier

# Ensure StateMonitor recorded all neurons you care about:
times = state_mon.t / ms
recorded_neurons = state_mon.record
statemon_idx_map = {n: i for i, n in enumerate(recorded_neurons)}

detailed_spikes = []

for spike_num, (neuron_idx, spike_time) in enumerate(zip(spike_mon.i, spike_mon.t)):
    t = spike_time / ms
    # Find closest sample in StateMonitor
    closest_idx = np.searchsorted(times, t)
    if closest_idx == len(times):
        closest_idx -= 1
    elif closest_idx > 0 and (abs(times[closest_idx] - t) > abs(times[closest_idx-1] - t)):
        closest_idx -= 1
    stat_idx = statemon_idx_map[neuron_idx]
    voltage = state_mon.v[stat_idx, closest_idx] / mV
    current_voltage = state_mon.spike_count[stat_idx, closest_idx] / mV  # Or /pA for pA
    neuron_id = reverse_index_map[neuron_idx]

    detailed_spikes.append({
        "spike_number": spike_num,
        "neuron_index": int(neuron_idx),
        "neuron_id": int(neuron_id),
        "spike_time_ms": t,
        "voltage_at_spike_mV": float(voltage),
        "current_voltage_mV": float(current_voltage)
    })
import os

# Name of the output file in the results directory
filename = "detailed_spikes_output.txt"
output_path = os.path.join(RESULT_DIR, filename)
# Create and write if file doesn't exist
if not os.path.exists(output_path):
    with open(output_path, "w") as f:
        for entry in detailed_spikes:
            f.write(str(entry) + "\n")
    print(f"File '{output_path}' created and spike data written.")
else:
    print(f"File '{output_path}' already exists.")
print(df_conn.columns)
# ========== LABEL CONNECTIVITY FOR ALL NEURONS ==========
# Build adjacency list (pre → post)
adjlist = defaultdict(set)
for pre_nid, post_nid in zip(df_conn['Presynaptic_ID'], df_conn['Postsynaptic_ID']):
    pre_idx, post_idx = neuron_index_map[pre_nid], neuron_index_map[post_nid]
    adjlist[pre_idx].add(post_idx)

labels = ['unconnected'] * num_neurons
connectivity_status = ["Not Connected"] * num_neurons
# Map your list of target neuron IDs (e.g. neu_sugar) to neuron simulation indices
stim_indices = [neuron_index_map[nid] for nid in neu_sugar if nid in neuron_index_map]

# Use the indexes of all target neurons you stimulated
# stim_indices must already be defined (list of neuron indices you stimulated)
for idx in stim_indices:
    labels[idx] = 'target'
    connectivity_status[idx] = 'Target'

visited = set(stim_indices)
queue = deque([(idx, 0) for idx in stim_indices])

while queue:
    current, depth = queue.popleft()
    if depth == 2:  # Only go up to 2 hops total (direct + indirect)
        continue
    for neighbor in adjlist[current]:
        if neighbor not in visited:
            if depth == 0:
                labels[neighbor] = 'direct'
                connectivity_status[neighbor] = 'Connected'
            elif depth == 1:
                labels[neighbor] = 'indirect'
                connectivity_status[neighbor] = 'Connected'
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))

# ========== FREQUENCY ANALYSIS ==========
print("📊 Analyzing frequencies...")
neuron_frequencies = {}
neuron_spike_counts = {}
neuron_spike_times = defaultdict(list)

# Group spikes by neuron
for i, t in zip(spike_mon.i, spike_mon.t):
    neuron_spike_times[int(i)].append(float(t/ms))

# Calculate frequencies
simulation_time_sec = float(t_run/ms) / 1000.0

for neuron_idx in range(N):
    spike_times = neuron_spike_times[neuron_idx]
    spike_count = len(spike_times)
    frequency = spike_count / simulation_time_sec
    neuron_frequencies[neu_sugar[neuron_idx]] = frequency
    neuron_spike_counts[neu_sugar[neuron_idx]] = spike_count

# ========== CONNECTIVITY CHECK ==========
def mark_connected_neurons(pre_idx_list, post_idx_list, fwid_to_idx, target_flywire_id, neu_sugar):
    target_idx = fwid_to_idx[target_flywire_id]
    connectivity_status = {nid: "Not Connected" for nid in neu_sugar}
    connectivity_status[target_flywire_id] = "Target"
    for pre, post in zip(pre_idx_list, post_idx_list):
        if pre == target_idx:
            connectivity_status[neu_sugar[post]] = "Connected"
        if post == target_idx:
            connectivity_status[neu_sugar[pre]] = "Connected"
    return connectivity_status

connectivity_status = mark_connected_neurons(pre_idx_list, post_idx_list, fwid_to_idx, target_flywire_id, neu_sugar)
# ========== DIRECT / INDIRECT CONNECTIONS (FIXED) ==========
target_idx = fwid_to_idx[target_flywire_id]
directly_connected_downstream = [post for pre, post in zip(pre_idx_list, post_idx_list) if pre == target_idx]
directly_connected_upstream = [pre for pre, post in zip(pre_idx_list, post_idx_list) if post == target_idx]
directly_connected = list(set(directly_connected_downstream + directly_connected_upstream))

indirectly_connected = []
for pre, post in zip(pre_idx_list, post_idx_list):
    if (pre in directly_connected and post != target_idx and post not in directly_connected):
        indirectly_connected.append(post)
    if (post in directly_connected and pre != target_idx and pre not in directly_connected):
        indirectly_connected.append(pre)
indirectly_connected = list(set(indirectly_connected))
print(f"Total neurons: {len(neu_sugar)}")
print("Example neuron IDs:", neu_sugar[:10])
print("fwid_to_idx mapping size:", len(fwid_to_idx))
print("Example mapping:", list(fwid_to_idx.items())[:5])
for pre, post in zip(pre_idx_list, post_idx_list):
    if pre == target_idx or post == target_idx:
        print(f"Connection: {neu_sugar[pre]} -> {neu_sugar[post]}")
downstream = [post for pre, post in zip(pre_idx_list, post_idx_list) if pre == target_idx]
upstream = [pre for pre, post in zip(pre_idx_list, post_idx_list) if post == target_idx]
print("Downstream:", len(downstream), "Upstream:", len(upstream))
# ========== PREPARE RESULTS ==========
frequency_results = []
for neuron_id in neu_sugar:
    neuron_idx = fwid_to_idx[neuron_id]
    freq = neuron_frequencies[neuron_id]
    scount = neuron_spike_counts[neuron_id]
    ctype = (
        "target" if neuron_id == target_flywire_id else
        "direct" if neuron_idx in directly_connected else
        "indirect" if neuron_idx in indirectly_connected else
        "unconnected"
    )
    frequency_results.append({
        "neuron_id": neuron_id,
        "neuron_index": neuron_idx,
        "frequency_hz": freq,
        "spike_count": scount,
        "connection_type": ctype,
        "connectivity_status": connectivity_status[neuron_id],
        "spike_times": neuron_spike_times[neuron_idx]
    })
# ========== SAVE RESULTS ==========
print("💾 Saving results...")
results_json = {
    "experiment_info": {
        "stimulus_frequency_hz": float(stimulus_rate/Hz),
        "target_neuron_id": target_flywire_id,
        "simulation_time_ms": float(t_run/ms),
        "total_neurons": N,
        "total_connections": len(pre_idx_list)
    },
    "frequency_analysis": frequency_results,
    "summary": {
        "neurons_with_activity": len([r for r in frequency_results if r['frequency_hz'] > 0]),
        "target_frequency": neuron_frequencies[target_flywire_id],
        "max_downstream_frequency": max(
            [r['frequency_hz'] for r in frequency_results if r['connection_type'] != 'target'], default=0),
        "directly_connected_count": len(directly_connected),
        "indirectly_connected_count": len(indirectly_connected)
    }
}

json_path = os.path.join(RESULT_DIR, f"frequency_analysis_{int(stimulus_rate/Hz)}Hz.json")
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)

csv_path = os.path.join(RESULT_DIR, f"neuron_frequencies_{int(stimulus_rate/Hz)}Hz.csv")
pd.DataFrame([{
    "neuron_id": r["neuron_id"],
    "frequency_hz": r["frequency_hz"],
    "spike_count": r["spike_count"],
    "connection_type": r["connection_type"],
    "connectivity_status": r["connectivity_status"]
} for r in frequency_results]).to_csv(csv_path, index=False)
# ========== SUMMARY ==========
print("\n📊 FREQUENCY ANALYSIS SUMMARY:")
print(f"🎯 Target neuron ({target_flywire_id}): {neuron_frequencies[target_flywire_id]:.2f} Hz")
print(f"📈 Active neurons: {len([r for r in frequency_results if r['frequency_hz'] > 0])}/{N}")
print(f"🔗 Direct responses: {len([r for r in frequency_results if r['connection_type']=='direct' and r['frequency_hz']>0])}")
print(f"🔗 Indirect responses: {len([r for r in frequency_results if r['connection_type']=='indirect' and r['frequency_hz']>0])}")

print(f"\n📄 Results saved to:")
print(f"  - JSON: {json_path}")
print(f"  - CSV: {csv_path}")
print("🎉 Analysis complete!")

