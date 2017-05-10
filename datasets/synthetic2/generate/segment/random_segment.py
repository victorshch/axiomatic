from segment import Segment
import numpy as np

class RandomSegment(Segment):
    def __init__(self, segment_list):
        self.segment_list = segment_list

    def generate(self, base, step = None):
        used_segment = np.random.choice(self.segment_list)
        return used_segment.generate(base, step)
