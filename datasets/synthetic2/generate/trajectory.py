import numpy as np
from scipy.signal import resample
from sklearn.preprocessing import scale

class Trajectory:

    def __init__(self, segments):
        self.segments = segments
        self.n_dims = len(segments[0])
    
    def generate(self, amp_distortion, time_distortion, base=0, time=0, **kwargs):
        base = np.zeros((self.n_dims))
        result = [self.generate_multidimensional_segment(oneLetterSegments, amp_distortion, time_distortion, base) for oneLetterSegments in self.segments]

        return np.vstack(result)
    
    def generate_multidimensional_segment(self, oneLetterSegments, amp_distortion, time_distortion, base):
        multidimensional_segment = []
        new_length = None
        for dim, segment in enumerate(oneLetterSegments):
            generated_segment = np.array(segment.generate(0))
            old_length = len(generated_segment)
            if new_length is None:
                new_length = int(old_length * np.random.uniform(1.0 - time_distortion, 1.0 + time_distortion))
            generated_segment = old_length * scale(generated_segment, with_mean = False)
            generated_segment = resample(generated_segment, new_length)
            generated_segment = generated_segment + np.random.normal(0, amp_distortion * np.std(generated_segment), generated_segment.shape) + base[dim]
            base[dim] = generated_segment[-1]
            multidimensional_segment.append(generated_segment.reshape(-1, 1))
        return np.hstack(multidimensional_segment)