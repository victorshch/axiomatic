from segment import Segment

class FullSquareSegment(Segment):

	def getValue(self, x):			
		return (x - self.length / 2) ** 2 - (self.length / 2) ** 2	
