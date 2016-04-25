import sys, json, random


lines = [line.strip() for line in open(sys.argv[1],"r").readlines()]

data = [json.loads(line) for line in lines]

docsubbody = []

f = open('out.txt','w')
doc = None
body = ''
sub = ''
for d in data:
	if doc is None:
		doc = d['link_id']
	
	if doc != d['link_id']:
		if len(body) > 5:
			docsubbody.append( (doc, sub, body) )
			f.write(doc + '\t' + sub + '\t' + body + '\n')
		body = ''
	
	doc = d['link_id']
	try:
		if d['body'] != '[removed]' and d['body'] != '[deleted]':
			newbod = d['body'].replace('\n','')
			newbod = newbod.replace('\t','')
			newbod = newbod.replace('\r','')
			newbod = ''.join([i if ord(i) < 128 else ' ' for i in newbod])
			body += newbod + ' | '
	except Exception as e:
		print str(e)
		
	sub = d['subreddit']

docsubbody.append( (doc, sub, body) )	
f.write(doc + '\t' + sub + '\t' + body + '\n')
f.close()

g = open('out_shuffled.txt', 'w')
random.shuffle(docsubbody)
for (doc, sub, body) in docsubbody:
	g.write(doc + '\t' + sub + '\t' + body + '\n')
g.close()
