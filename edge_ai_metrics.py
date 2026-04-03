import numpy as np

class EdgeAIMetrics:
    def __init__(self):
        self.device_specs = {
            'ARM Cortex-M4': {
                'cpu_clock': 180,
                'ram_available': 256,
                'flash_storage': 1024,
                'base_power_mw': 120,
                'flops': 225
            },
            'ESP32': {
                'cpu_clock': 240,
                'ram_available': 520,
                'flash_storage': 4096,
                'base_power_mw': 180,
                'flops': 600
            },
            'Raspberry Pi': {
                'cpu_clock': 1500,
                'ram_available': 1024000,
                'flash_storage': 32768000,
                'base_power_mw': 2500,
                'flops': 24000
            },
            'Edge TPU': {
                'cpu_clock': 500,
                'ram_available': 8192,
                'flash_storage': 8192,
                'base_power_mw': 800,
                'flops': 4000000
            }
        }
        
        self.model_specs = {
            'Lightweight RF': {
                'size_kb': 45,
                'flops_per_inference': 150000,
                'quantization': 'INT8',
                'compression_ratio': 4
            },
            'Optimized SVM': {
                'size_kb': 32,
                'flops_per_inference': 80000,
                'quantization': 'INT8',
                'compression_ratio': 3
            },
            'Quantized NN': {
                'size_kb': 120,
                'flops_per_inference': 500000,
                'quantization': 'INT8',
                'compression_ratio': 8
            }
        }
    
    def compute_metrics(self, model_type, edge_device, signal_length):
        device = self.device_specs.get(edge_device, self.device_specs['ARM Cortex-M4'])
        model = self.model_specs.get(model_type, self.model_specs['Lightweight RF'])
        
        flops_required = model['flops_per_inference']
        device_flops = device['flops'] * 1000
        
        inference_time_ms = (flops_required / device_flops) * 1000
        
        inference_time_ms *= np.random.uniform(0.9, 1.1)
        
        model_size_kb = model['size_kb']
        
        signal_size_kb = (signal_length * 4) / 1024
        memory_usage_kb = model_size_kb + signal_size_kb + 20
        
        cpu_utilization = min(95, (flops_required / device_flops) * 100 * 1.2)
        
        memory_utilization = (memory_usage_kb / device['ram_available']) * 100
        
        power_mw = device['base_power_mw'] * (1 + cpu_utilization / 100)
        
        battery_capacity_mah = 2000
        battery_voltage = 3.7
        battery_wh = (battery_capacity_mah * battery_voltage) / 1000
        
        avg_power_w = power_mw / 1000
        battery_life_hours = battery_wh / avg_power_w if avg_power_w > 0 else 0
        
        return {
            'inference_time_ms': inference_time_ms,
            'model_size_kb': model_size_kb,
            'memory_usage_kb': memory_usage_kb,
            'power_mw': power_mw,
            'cpu_utilization': cpu_utilization,
            'memory_utilization': memory_utilization,
            'cpu_clock': device['cpu_clock'],
            'ram_available': device['ram_available'],
            'flash_storage': device['flash_storage'],
            'quantization': model['quantization'],
            'compression_ratio': model['compression_ratio'],
            'battery_life_hours': battery_life_hours,
            'throughput_fps': 1000 / inference_time_ms if inference_time_ms > 0 else 0
        }
