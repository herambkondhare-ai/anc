import numpy as np
import torch
import torchaudio
import onnxruntime as ort

class ANC3LayerPipeline:
    def __init__(self, onnx_model_path="models/denoiser_model.onnx", nlms_filter_len=32, nlms_mu=0.01, gate_threshold=0.015):
        """
        3-Layer Defense-Grade ANC Pipeline for Edge Execution (Raspberry Pi 5)
        Layer 1: Dynamic Impulsive Gate (Time-Domain Amplitude Gate)
        Layer 2: Optimized NLMS Adaptive Filter
        Layer 3: Quantized ONNX Neural Denoiser
        """
        self.gate_threshold = gate_threshold
        self.nlms_filter_len = nlms_filter_len
        self.nlms_mu = nlms_mu
        self.weights = np.zeros(nlms_filter_len)
        
        # Load ONNX Model Runtime for Layer 3
        try:
            self.ort_session = ort.InferenceSession(onnx_model_path)
            print(f"[+] Loaded ONNX Denoiser Model from {onnx_model_path}")
        except Exception as e:
            print(f"[!] Warning: Could not load ONNX model ({e}). Layer 3 will fallback to bypass.")
            self.ort_session = None

    def layer1_impulsive_gate(self, audio_signal):
        """Layer 1: Clamps high-amplitude impulsive peaks (e.g., blast/gunshot noise)"""
        return np.clip(audio_signal, -self.gate_threshold, self.gate_threshold)

    def layer2_nlms_filter(self, primary_signal, ref_noise):
        """Layer 2: Normalized Least Mean Squares Adaptive Filtering"""
        num_samples = len(primary_signal)
        output_signal = np.zeros(num_samples)
        buffer = np.zeros(self.nlms_filter_len)

        for i in range(num_samples):
            buffer[1:] = buffer[:-1]
            buffer[0] = ref_noise[i] if i < len(ref_noise) else 0.0

            prediction = np.dot(self.weights, buffer)
            error = primary_signal[i] - prediction
            output_signal[i] = error

            # Weight update with normalization factor
            norm_term = np.dot(buffer, buffer) + 1e-6
            self.weights += (self.nlms_mu / norm_term) * error * buffer

        return output_signal

    def layer3_onnx_denoiser(self, audio_signal, sample_rate=16000):
        """Layer 3: Neural Denoiser via ONNX Runtime"""
        if self.ort_session is None:
            return audio_signal

        # Prepare input tensor shape [1, 1, samples]
        input_tensor = np.expand_dims(np.expand_dims(audio_signal.astype(np.float32), axis=0), axis=0)
        input_name = self.ort_session.get_inputs()[0].name
        
        outputs = self.ort_session.run(None, {input_name: input_tensor})
        cleaned_signal = outputs[0].squeeze()
        return cleaned_signal

    def process_audio(self, noisy_signal, ref_noise=None, sample_rate=16000):
        """Executes the complete 3-layer pipeline in sequence"""
        if ref_noise is None:
            ref_noise = np.random.normal(0, 0.01, size=len(noisy_signal))

        # Layer 1
        l1_out = self.layer1_impulsive_gate(noisy_signal)
        # Layer 2
        l2_out = self.layer2_nlms_filter(l1_out, ref_noise)
        # Layer 3
        final_clean_audio = self.layer3_onnx_denoiser(l2_out, sample_rate)

        return final_clean_audio

if __name__ == "__main__":
    print("[+] Initializing 3-Layer ANC Pipeline...")
    pipeline = ANC3LayerPipeline()
    dummy_input = np.random.normal(0, 0.05, 16000)
    output = pipeline.process_audio(dummy_input)
    print(f"[+] Pipeline test complete. Input length: {len(dummy_input)}, Output length: {len(output)}")