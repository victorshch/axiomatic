from segment import Segment

class CubeSegment(Segment):

	def getValue(self, x):
		return (x - self.length / 2) ** 3 + (self.length / 2) ** 3
