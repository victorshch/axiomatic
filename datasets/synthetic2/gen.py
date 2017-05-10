import numpy as np
import pandas as pd
import copy
import random

from generate.trajectory import Trajectory
from generate.segment.random_segment import RandomSegment

def find_all(a_str, sub):
    start = 0
    if len(sub) > len(a_str):
        a_str, sub = sub, a_str
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += 1     

def getStringFromAlphabet(alphabet, length):
    result = ''
    for i in range(length):
        result += random.choice(alphabet)
    return result

def getOccurencesCount(string, substrings):
    count = 0
    for substr in substrings:
        count += len(list(find_all(string, substr)))
    return count

def generateAbnormalSequences(classCount, minLength, maxLength, alphabet):
    result = []
    for i in xrange(classCount):
        length = np.random.randint(minLength, maxLength + 1)
        while True:
            sequence = getStringFromAlphabet(alphabet, length)
            if getOccurencesCount(sequence, result) == 0:
                break
        result.append(sequence)
    return result

def inject(s, substr):
    pos = random.randint(0, len(s) - len(substr))
    return s[:pos] + substr + s[pos + len(substr):]

def generateSegmentsFromString(alphabet_to_segments, string):
    result = []
    for char in string:
        result.append(alphabet_to_segments[char])
    return result   

#very bad code (DO NOT USE IT!!!)
def generateSegments(data, aliases):
    length = data['length']
    alphabet = list(aliases.keys())
    result = getStringFromAlphabet(alphabet, length)
    do_again = False

    abnormal = data.get('abnormalSequence', None)
    injectSequences = data.get('injectSequences', [])
    removeSequences = data.get('removeSequences', [])

    while (True):
        if abnormal:    
            abnormal_length = len(abnormal)        
            occurences = list(find_all(result, abnormal))
            
            while (len(occurences) > 0):
                result = getStringFromAlphabet(alphabet, length)
                occurences = list(find_all(result, abnormal))

            result = inject(result, abnormal)
            occurences = list(find_all(result, abnormal))

            if (len(occurences) != 1):
                continue

        do_again = False

        for s in injectSequences:
            result = inject(result, s)

        while ('removeSequences' in data and getOccurencesCount(result, data['removeSequences']) != 0):
            result = getStringFromAlphabet(alphabet, length)

            if abnormal:
                do_again = True
                break

        if do_again:
            continue
        
        if abnormal:
            print(result.replace(abnormal, '_{0}_'.format(abnormal)))    
        elif injectSequences:
            s = result
            for injected in injectSequences:
                s = s.replace(injected, '<{0}>'.format(injected))
            print(s)
        else:
            print(result)
        
        break

    return (generateSegmentsFromString(aliases, result), result)

def generateNormalClass(normalClassConf, aliases, noiseConf, classes):
    
    def getSubstrings(str):
        result = []
        for i in range(len(str) - 1):
            result.append(str[i] + str[i+1])
        return result

    parts = []
    for name, data in classes.items():
        abnormalSequence = data.get('abnormalSequence', [])
        parts += getSubstrings(abnormalSequence)

    normalClassConf['injectSequences'] = list(set(parts))
    return generateClass('normal', normalClassConf, aliases, noiseConf)

def generateClass(name, classConf, aliases, noiseConf = dict(sigma=1,length_deformation_min=1, length_deformation_max=2), verbose=False):
    if verbose: 
        print('Generate segments...')
        if 'abnormalSequence' in classConf:    
            print('Abnormal: {0}'.format(classConf['abnormalSequence']))
        if 'injectSequences' in classConf:    
            print('Injected: {0}'.format(', '.join(classConf['injectSequences'])))
    result = []
    for i in xrange(classConf['count']):            
        segments, str_segments = generateSegments(classConf, aliases)
        trajectory = Trajectory(segments)
        #print "Segments: ", segments
        generatedTraj = np.array(trajectory.generate(**noiseConf))
        result.append(pd.DataFrame(generatedTraj))
    return result

def form_alphabet(dimension_count, alphabet_size, segments, **kwargs):
    used_segments = segments
    
    while len(used_segments) < alphabet_size: used_segments = used_segments + segments
    
    letter_segments = list(np.random.choice(used_segments, alphabet_size, replace=False))
    
    result = {}
    
    for letter_no in xrange(alphabet_size):
        current_multidim_segment = [RandomSegment(letter_segments) for i in xrange(dimension_count)]
        fixed_segment_dim = np.random.randint(dimension_count)
        current_multidim_segment[fixed_segment_dim] = letter_segments[letter_no]
        result[chr(ord('A') + letter_no)] = current_multidim_segment
    
    return result

def generate_dataset(dimension_count, alphabet_size, segments, verbose=True, **conf):
    alphabet_to_segments = form_alphabet(dimension_count, alphabet_size, segments)
    print "alphabet:", alphabet_to_segments
    alphabet = list(alphabet_to_segments.keys())
    
    result = {}
    conf = copy.deepcopy(conf)
    
    classList = conf['classes'].keys()
    classList.remove('normal')
    abnormalSequences = generateAbnormalSequences(len(classList), conf['abnormal_sequence_length_min'], conf['abnormal_sequence_length_max'], alphabet)
    abnormalSequencesForClasses = dict(zip(classList, abnormalSequences))
    
    for foldName, foldConf in conf['folds'].items():
        result[foldName] = {}
        for name, classConf in conf['classes'].items():
            if verbose: print('Generating trajectories for class: {0}'.format(name))
            if name == 'normal': continue
            classConf['abnormalSequence'] = abnormalSequencesForClasses[name]
            classConf['removeSequences'] = [abnormalSequencesForClasses[clName] for clName in classList if clName != name]
            oldCount = classConf['count']
            classConf['count'] = int(oldCount * foldConf['count_factor'])
            result[foldName][name] = generateClass(name, classConf, alphabet_to_segments, conf['noise'], verbose)
            classConf['count'] = oldCount
        if 'normal' in conf['classes']:
            oldCount = classConf['count']
            classConf['count'] = int(oldCount * foldConf['count_factor'])
            result[foldName]['normal'] = generateNormalClass(conf['classes']['normal'], alphabet_to_segments, conf['noise'], conf['classes'])
            classConf['count'] = oldCount
    return result

    # for each TS:
    # generate symbolic TS
    # convert symbolic TS to multidimensional numeric TS