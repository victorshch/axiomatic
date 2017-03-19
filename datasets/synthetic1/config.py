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
datasetConf = {
    'noise': {
        'sigma': 2,
        'length_deformation_min': 1,
        'length_deformation_max': 2
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
        
    'abnormalSequenceLength_min': 3,
    'abnormalSequenceLength_max': 6,

    'classes': {
        # first type of trajectories
        'normal': {                            
            #count of creating trajectories
            'count': 30,
            #count of segments
            'length': 10,        
            'removeSequences': []
        },

        #another type of trajectories
        'bad1': {                                        
            #count of creating trajectories
            'count': 30,
            #count of segments
            'length': 10, 
            'abnormalSequence': '',
            'removeSequences': []
        },
                
        'bad2': {                                        
            #count of creating trajectories
            'count': 30,
            #count of segments
            'length': 10, 
            'abnormalSequence': '',
            'removeSequences': []
        }
    },
    
    'folds': {
        'train': { 'count_factor': 1.0 },
        'test': { 'count_factor': 1.0 },
        'validate': { 'count_factor': 1.0 }
    }
}
