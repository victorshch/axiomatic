from generate.segment.arc import ArcSegment
from generate.segment.cube import CubeSegment
from generate.segment.minuscube import MinusCubeSegment
from generate.segment.downline import DownLineSegment
from generate.segment.upline import UpLineSegment
from generate.segment.fullsquare import FullSquareSegment
from generate.segment.halfsquare import HalfSquareSegment

#result path
path = './generated_data'
csv_header = ['time', 'row_1']

#definition of creating dataset
dataset = {
	'gauss': {
		'mu': 0,
		'sigma': 2
	},

	'length_deformation': {
		'max': 2,
		'min': 1
	},

	'aliases': {
		'A': CubeSegment(10, ignore_base=True),
		'B': MinusCubeSegment(10, ignore_base=True),
		'C': ArcSegment(10, ignore_base=True),
		'D': DownLineSegment(10, ignore_base=True),
		'E': UpLineSegment(10, ignore_base=True),
		'F': FullSquareSegment(10, ignore_base=True),
		'G': HalfSquareSegment(10, ignore_base=True)
	},	

	'classes': {
		# first type of trajectories
		'normal': {							
			#count of creating trajectories
			'count': 10,
			#count of segments
			'length': 10,		
			'removeSequences': []
		},

		#another type of trajectories
		'bad1': {										
			#count of creating trajectories
			'count': 10,
			#count of segments
			'length': 10, 
			'abnormalSequence': '',
			'removeSequences': []
		}
	}	
}
