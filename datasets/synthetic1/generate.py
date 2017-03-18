import os
import csv
import random
import sys

from generate.trajectory import Trajectory

import config

path = sys.argv[1] if len(sys.argv) >= 2 else config.path
forceAbnormal = sys.argv[2] if len(sys.argv) >= 3 else None

if not os.path.exists(path):
	os.makedirs(path)	

aliases = config.dataset['aliases']
alphabet = list(aliases.keys())

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += 1

def generateSegmentsFromString(string):
	result = []
	for char in string:
		result.append(aliases[char])
	return result		

def getStringFromAlphabet(length):
	result = ''
	for i in range(length):
		result += random.choice(alphabet)
	return result

def getOccurencesCount(str, substrings):
	count = 0
	for substr in substrings:
		count += len(list(find_all(str, substr)))
	return count

def inject(s, substr):
	pos = random.randint(0, len(s) - len(substr))
	return s[:pos] + substr + s[pos + len(substr):]

#very bad code (DO NOT USE IT!!!)
def generateSegments(data):
	length = data['length']	
	result = getStringFromAlphabet(length)
	do_again = False

	abnormal = data.get('abnormalSequence', None)
	injectSequences = data.get('injectSequences', [])
	removeSequences = data.get('removeSequences', [])

	while (True):
		if abnormal:	
			abnormal_length = len(abnormal)		
			occurences = list(find_all(result, abnormal))
			
			while (len(occurences) > 0):
				result = getStringFromAlphabet(length)
				occurences = list(find_all(result, abnormal))

			result = inject(result, abnormal)
			occurences = list(find_all(result, abnormal))

			if (len(occurences) != 1):
				continue

		do_again = False

		for s in injectSequences:
			result = inject(result, s)

		while (getOccurencesCount(result, data['removeSequences']) != 0):
			result = getStringFromAlphabet(length)

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

	return (generateSegmentsFromString(result), result)

def handleNormalClass(normal_data, classes):
	
	def getSubstrings(str):
		result = []
		for i in range(len(str) - 1):
			result.append(str[i] + str[i+1])
		return result

	parts = []
	for name, data in classes.items():
		abnormalSequence = data.get('abnormalSequence', [])
		parts += getSubstrings(abnormalSequence)

	normal_data['injectSequences'] = list(set(parts))
	return handleClass('normal', normal_data)

def handleClass(name, data):
	dir = os.path.join(path, name).replace("\\","/")
	if not os.path.exists(dir):
		os.makedirs(dir)

	print('Generate segments...')
	if 'abnormalSequence' in data:	
		print('Abnormal: {0}'.format(data['abnormalSequence']))
	if 'injectSequences' in data:	
		print('Injected: {0}'.format(', '.join(data['injectSequences'])))
	for i in range(data['count']):			
		segments, str_segments = generateSegments(data)

		trajectory = Trajectory(segments)	
		result = trajectory.generate()

		file = os.path.join(path, name, '{0}_{1}_{2}.csv'.format(name, i, str_segments)).replace("\\","/")		
		with open(file, "w") as f:			
		    writer = csv.writer(f, delimiter=';', lineterminator=';\n')
		    writer.writerow(config.csv_header)
		    writer.writerows(result)


if 'gauss' in config.dataset:
	mu = config.dataset['gauss']['mu']
	sigma = config.dataset['gauss']['sigma']
	for segment in aliases.values():
		segment.setGaussMuAndSigma(mu, sigma)

if 'length_deformation' in config.dataset:
	deformation = config.dataset['length_deformation']
	for segment in aliases.values():
		value = random.uniform(deformation['min'], deformation['max'])
		segment.setStep(1 if value == 0 else 1 / value)	

if forceAbnormal:
	print('Use abnormal: {0}'.format(forceAbnormal))
	for name, data in config.dataset['classes'].items():
		if name == 'normal':
			config.dataset['classes']['normal']['removeSequences'] = [forceAbnormal]
		else:
			data['abnormalSequence'] = forceAbnormal

for name, data in config.dataset['classes'].items():
	print('Handle class: {0}'.format(name))
	if name == 'normal':
		handleNormalClass(data, config.dataset['classes'])
	else:
		handleClass(name, data)
	print('')

print('Finished')
	






