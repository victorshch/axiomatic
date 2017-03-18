import numpy as np
from scipy.signal import resample

class Trajectory:

    def __init__(self, segments):
        self.segments = segments
    
    def generate(self, sigma=1, length_deformation_min=None, length_deformation_max=2, base=0, time=0, **kwargs):
        result = []

        for segment in self.segments:
            generated_segment = np.array(segment.generate(base))
            if generated_segment is None or len(generated_segment) == 0: continue
            base = generated_segment[-1] #latest value
            if length_deformation_min is None: length_deformation_min = 1
            if length_deformation_max is not None:
                new_length = int(len(generated_segment) * np.random.uniform(length_deformation_min, length_deformation_max))
                generated_segment = resample(generated_segment, new_length)
            if sigma is not None and sigma > 0:
                generated_segment = generated_segment + np.random.normal(0, sigma, generated_segment.shape)
            result.append(generated_segment[0: -1])

        return np.hstack(result)