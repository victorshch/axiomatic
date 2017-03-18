import os
import csv
import random
import sys
import numpy as np
import pandas as pd

from generate.trajectory import Trajectory

def find_all(a_str, sub):
    start = 0
    if len(sub) > len(a_str):
        a_str, sub = sub, a_str
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += 1

def generateSegmentsFromString(aliases, string):
    result = []
    for char in string:
        result.append(aliases[char])
    return result        

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

        while (getOccurencesCount(result, data['removeSequences']) != 0):
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
        generatedTraj = np.array(trajectory.generate(**noiseConf))
        result.append(pd.DataFrame(generatedTraj))
    return result

def generateFold(verbose=False, **conf):
    result = {}
    
    classList = conf['classes'].keys()
    classList.remove('normal')
    alphabet = conf['aliases'].keys()
    abnormalSequences = generateAbnormalSequences(len(classList), conf['abnormalSequenceLength_min'], conf['abnormalSequenceLength_max'], alphabet)
    abnormalSequencesForClasses = dict(zip(classList, abnormalSequences))
    
    for name, classConf in conf['classes'].items():
        if verbose: print('Generating trajectories for class: {0}'.format(name))
        if name == 'normal': continue
        classConf['abnormalSequence'] = abnormalSequencesForClasses[name]
        classConf['removeSequences'] = [abnormalSequencesForClasses[clName] for clName in classList if clName != name]
        result[name] = generateClass(name, classConf, conf['aliases'], conf['noise'], verbose)
    if 'normal' in conf['classes']:
        result['normal'] = generateNormalClass(conf['classes']['normal'], conf['aliases'], conf['noise'], conf['classes'])
    return result

#if 'length_deformation' in config.dataset:
    #deformation = config.dataset['length_deformation']
    #for segment in aliases.values():
        #value = random.uniform(deformation['min'], deformation['max'])
        #segment.setStep(1 if value == 0 else 1 / value)    

#if forceAbnormal:
    #print('Use abnormal: {0}'.format(forceAbnormal))
    #for name, data in config.dataset['classes'].items():
        #if name == 'normal':
            #config.dataset['classes']['normal']['removeSequences'] = [forceAbnormal]
        #else:
            #data['abnormalSequence'] = forceAbnormal

#for name, data in config.dataset['classes'].items():
    #print('Handle class: {0}'.format(name))
    #if name == 'normal':
        #handleNormalClass(data, config.dataset['classes'])
    #else:
        #handleClass(name, data)
    #print('')

#print('Finished')
    






