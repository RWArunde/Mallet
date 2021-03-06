# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# ------------------------------------------------------------------------
# Filename   : heatmap.py
# Date       : 2013-04-19
# Updated    : 2014-01-04
# Author     : @LotzJoe >> Joe Lotz
# Description: My attempt at reproducing the FlowingData graphic in Python
# Source     : http://flowingdata.com/2010/01/21/how-to-make-a-heatmap-a-quick-and-easy-solution/
#
# Other Links:
#     http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
#
# ------------------------------------------------------------------------

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import matplotlib.pyplot as plt
#import pandas as pd
from urllib2 import urlopen
import numpy as np
import json
import sys
#%pylab inline

page = urlopen("http://datasets.flowingdata.com/ppg2008.csv")
#nba = pd.read_csv(page, index_col=0)

#read in subreddit_topics data
subred = json.load(open("subreddit_topics.json"))
nba = []
ylabels = []

topsub = sys.argv[1]
nba.append(subred[topsub])
ylabels.append(topsub)
for k, v in subred.iteritems():
	if k == 'gg' or k == topsub:
		continue
	ylabels.append(k)
	nba.append(v)

#get xlabels
#xlabels = [line.strip().split('\t')[-1] for line in open('topic_words.txt').readlines()]
#hide them
#xlabels = ['' for l in xlabels]
xlabels = []

nba = np.array(nba)



# Plot it out
fig, ax = plt.subplots()
heatmap = ax.pcolor(nba, cmap=plt.cm.Blues, alpha=0.8)

# Format
fig = plt.gcf()

fig.set_size_inches(8,4)

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(nba.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(nba.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

# Set the labels
labels = xlabels

# note I could have used nba_sort.columns but made "labels" instead
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(ylabels, minor=False)

# rotate the x labels
# plt.xticks(rotation=90)

ax.grid(False)

# Turn off all the ticks
ax = plt.gca()

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

#fig.subplots_adjust(bottom = 0)
#fig.subplots_adjust(top = .5)
#fig.subplots_adjust(right = 1)
#fig.subplots_adjust(left = 0.1)

plt.savefig('LDA_football.png')
cpy = np.copy(nba)

for c in range(nba.shape[1]):
	minn = (nba[:,c]).min()
	maxx = (nba[:,c]).max()
	mean = (nba[:,c]).mean()
	nba[:,c] = (nba[:,c] - minn) / (maxx-minn)
	
ax.pcolor(nba, cmap=plt.cm.Blues, alpha=0.8)	
plt.savefig('LDA_football_normed.png')

nba = cpy
norms = []
for c in range(nba.shape[1]):
	norms.append([c,sum(nba[:,c])])
sort_cols = sorted(norms, key=lambda a: -a[1])
nba2 = np.copy(nba)
for c in range(nba.shape[1]):
	nba2[:,c] = nba[:,sort_cols[c][0]]


ax.pcolor(nba2, cmap=plt.cm.Blues, alpha=0.8)	
plt.savefig('LDA_'+sys.argv[2]+'_sorted.png')

for c in range(nba.shape[0]):	
	print ylabels[c] + "\t" + str(np.var(nba[c,:]))
	
