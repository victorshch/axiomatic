from generate.segment.segment import Segment

class DownLineSegment(Segment):

	def getValue(self, x):
		return -x
