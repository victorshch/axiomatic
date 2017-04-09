from generate.segment.arc import ArcSegment
from generate.segment.cube import CubeSegment
from generate.segment.minuscube import MinusCubeSegment
from generate.segment.downline import DownLineSegment
from generate.segment.upline import UpLineSegment
from generate.segment.fullsquare import FullSquareSegment
from generate.segment.halfsquare import HalfSquareSegment

dataset_config = {
    'segments': [
            CubeSegment(10),
            ArcSegment(10),
            MinusCubeSegment(10),
            DownLineSegment(10),
            UpLineSegment(10),
            FullSquareSegment(10),
            HalfSquareSegment(10)
            ],

    'dimension_count': 1,
    'alphabet_size': 7,
    
    #length in segments
    'abnormal_sequence_length_min': 3,
    'abnormal_sequence_length_max': 6,
    
    'noise': {
        'amp_distortion': 0.2,
        'time_distortion': 0.1
        },

    'classes': {
        # first type of trajectories
        'normal': {                            
            #count of creating trajectories
            'count': 30,
            #count of segments
            'length': 10
        },

        #another type of trajectories
        'bad1': {                                        
            #count of creating trajectories
            'count': 30,
            #count of segments
            'length': 10, 
            'abnormalSequence': ''
        },
                
        'bad2': {                                        
            #count of creating trajectories
            'count': 30,
            #count of segments
            'length': 10, 
            'abnormalSequence': ''
        }
    },
    
    'folds': {
        'train': { 'count_factor': 1.0 },
        'test': { 'count_factor': 1.0 },
        'validate': { 'count_factor': 1.0 }
    }

    }