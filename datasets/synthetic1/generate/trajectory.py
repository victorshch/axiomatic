class Trajectory:

	def __init__(self, segments):
		self.segments = segments
	
	def generate(self, base=0, time=0):
		result = []

		for segment in self.segments:
			generated_segment, time = segment.generate(time, base)
			if generated_segment is not None and len(generated_segment) != 0:
				base = generated_segment[-1][1] #latest value			
			result += generated_segment[0: -1]

		return result