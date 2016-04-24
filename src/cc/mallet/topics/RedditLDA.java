/* Copyright (C) 2005 Univ. of Massachusetts Amherst, Computer Science Dept.
	 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
	 http://www.cs.umass.edu/~mccallum/mallet
	 This software is provided under the terms of the Common Public License,
	 version 1.0, as published by http://www.opensource.org.	For further
	 information, see the file `LICENSE' included with this distribution. */

package cc.mallet.topics;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.io.*;

import cc.mallet.types.*;
import cc.mallet.util.ArrayUtils;
import cc.mallet.util.Randoms;

import org.apache.commons.math3.special.Gamma;



/**
 * Latent Dirichlet Allocation for subreddit data
 * @author Andrew McCallum, Joe Runde
 */

// Think about support for incrementally adding more documents...
// (I think this means we might want to use FeatureSequence directly).
// We will also need to support a growing vocabulary!

public class RedditLDA {

	public int numTopics; // Number of topics to be fit
	double alphaScalar; //Scalar alpha value
	public double[][] alpha; //Matrix of alpha values, indexed by <subreddit, topic>
	double beta;	 // Prior on per-topic multinomial distribution over words
	double vBeta;
	double gamma;	// Bernoulli prior for each alpha to be (alphaScalar) rather than (alphaScalar * eta)
	double eta;		// "dropout" multiplier for each alpha
	
	InstanceList ilist;	// the data field of the instances is expected to hold a FeatureSequence
	int[][] topics; // indexed by <document index, sequence index>
	int numDocs;
	int numTypes;
	int numTokens;
	int numSubreddits; //number of subreddits
	int[] docsPerSub;
	int[] docSubreddit; //subreddit each document is from
	int[][] docTopicCounts; // indexed by <document index, topic index>
	int[][] typeTopicCounts; // indexed by <feature index, topic index>
	int[][][] subTypeTopicCounts; // indexed by <subreddit index, feature index, topic index>
	
	int[] tokensPerTopic; // indexed by <topic index>
	int[] tokensPerDoc; // number of tokens in each document
	int[][] tokensPerSubPerTopic; //indexed by <subreddit index, topic index>
	
	public HashMap<String, Integer> subredditMap;

	public RedditLDA (InstanceList documents, String topSub, int numberOfTopics, double alpha, double beta, double gamma, double eta)
	{
		this.numTopics = numberOfTopics;
		this.alphaScalar = alpha;
		this.beta = beta;
		this.ilist = documents;
		this.gamma = gamma;
		this.eta = eta;
		
		
		//get number of subreddits, map them to integers
		HashSet<String> targetset = new HashSet<String>();
		for(Instance inst : documents) {
			targetset.add(inst.getTarget().toString());
		}
		this.numSubreddits = targetset.size();
		System.out.print("Number of subreddits: ");
		System.out.println(this.numSubreddits);
		this.subredditMap = new HashMap<String, Integer>();
		this.subredditMap.put(topSub, 0);
		int num = 1;
		for(String sub: targetset) {
			if(!this.subredditMap.containsKey(sub)){
				this.subredditMap.put(sub, num);
				num++;
			}
		}
		
		//initialize alpha matrix
		this.alpha = new double[numSubreddits][numTopics];
		for(int c = 0; c < numSubreddits; c++)
			for(int i = 0; i < numTopics; i++)
				this.alpha[c][i] = alphaScalar;
				
				
		//initialize topic assignments randomly
		numTypes = ilist.getDataAlphabet().size();
		numDocs = ilist.size();
		topics = new int[numDocs][];
		docsPerSub = new int[numSubreddits];
		docSubreddit = new int[numDocs];
		tokensPerDoc = new int[numDocs];
		docTopicCounts = new int[numDocs][numTopics];
		typeTopicCounts = new int[numTypes][numTopics];
		tokensPerTopic = new int[numTopics];
		subTypeTopicCounts = new int[numSubreddits][numTypes][numTopics];
		tokensPerSubPerTopic = new int[numSubreddits][numTopics];
		vBeta = beta * numTypes;

		// Initialize with random assignments of tokens to topics
		// and finish allocating this.topics and this.tokens
		int topic, seqLen;
		FeatureSequence fs;
		Randoms r = new Randoms();
		for (int di = 0; di < numDocs; di++) {
			try {
				fs = (FeatureSequence) ilist.get(di).getData();
			} catch (ClassCastException e) {
				System.err.println ("LDA and other topic models expect FeatureSequence data, not FeatureVector data.	"
								+"With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
				throw e;
			}
			int sub = subredditMap.get(ilist.get(di).getTarget().toString());
			docSubreddit[di] = sub;
			docsPerSub[sub]++;
			seqLen = fs.getLength();
			numTokens += seqLen;
			tokensPerDoc[di] = seqLen;
			topics[di] = new int[seqLen];
			// Randomly assign tokens to topics
			for (int si = 0; si < seqLen; si++) {
				topic = r.nextInt(numTopics);
				topics[di][si] = topic;
				docTopicCounts[di][topic]++;
				typeTopicCounts[fs.getIndexAtPosition(si)][topic]++;
				subTypeTopicCounts[sub][fs.getIndexAtPosition(si)][topic]++;
				tokensPerTopic[topic]++;
				tokensPerSubPerTopic[sub][topic]++;
			}
		}
	}

	public void estimate (int numIterations, int showTopicsInterval,
						Randoms r)
	{
		this.estimate(0, numDocs, numIterations, showTopicsInterval, r);
	}
	
	/* Perform several rounds of Gibbs sampling on the documents in the given range. */ 
	public void estimate (int docIndexStart, int docIndexLength,
							int numIterations, int showTopicsInterval, Randoms r)
	{
		long startTime = System.currentTimeMillis();
		for (int iterations = 0; iterations < numIterations; iterations++) {
			if (iterations % 10 == 0) System.out.print (iterations);	else System.out.print (".");
			System.out.flush();
			if (showTopicsInterval != 0 && iterations % showTopicsInterval == 0 && iterations > 0) {
				System.out.println ();
				printTopWords (50, false);
			}
		
			sampleTopicsForDocs(docIndexStart, docIndexLength, r);
			sampleAlphas(r);
		}

		long seconds = Math.round((System.currentTimeMillis() - startTime)/1000.0);
		long minutes = seconds / 60;	seconds %= 60;
		long hours = minutes / 60;	minutes %= 60;
		long days = hours / 24;	hours %= 24;
		System.out.print ("\nTotal time: ");
		if (days != 0) { System.out.print(days); System.out.print(" days "); }
		if (hours != 0) { System.out.print(hours); System.out.print(" hours "); }
		if (minutes != 0) { System.out.print(minutes); System.out.print(" minutes "); }
		System.out.print(seconds); System.out.println(" seconds");
	}

	/* One iteration of Gibbs sampling, across all documents. */
	public void sampleTopicsForAllDocs (Randoms r)
	{
		sampleTopicsForDocs(0, numDocs, r);
	}

	/* One iteration of Gibbs sampling, across all documents. */
	public void sampleTopicsForDocs (int start, int length, Randoms r)
	{
		assert (start+length <= docTopicCounts.length);
		double[] topicWeights = new double[numTopics];
		// Loop over every word in the corpus
		for (int di = start; di < start+length; di++) {
			int sub = this.docSubreddit[di];//subredditMap.get(ilist.get(di).getTarget().toString());
			sampleTopicsForOneDoc ((FeatureSequence)ilist.get(di).getData(),
									 topics[di], docTopicCounts[di], topicWeights, sub, r);
		}
	}
	
	private double logsum(double x, double y) {
		if(x >= y) {
			return x + Math.exp(y - x);
		} else {
			return y + Math.exp(x - y);
		}
	}
	
	private double logAlphaNumerator(double x, double alphasMinusTopic, int doc, int topic) {
		return Gamma.logGamma(x + alphasMinusTopic) + Gamma.logGamma(x + this.docTopicCounts[doc][topic]);
	}
	
	private double logAlphaDenominator(double x, double alphasMinusTopic, int doc) {
		return Gamma.logGamma(x) + Gamma.logGamma(x + this.tokensPerDoc[doc] + alphasMinusTopic);
	}
	
	private void sampleAlphas(Randoms r) {
		
		// Don't sample alphas if this is regular LDA
		if(this.gamma >= 1) {
			return;
		}
		
		//don't resample for top subreddit
		for(int c = 1; c < numSubreddits; c++){
			double alphaSubSum = 0;
			int tokensInSub = 0;
			for(int i = 0; i < numTopics; i++){
				alphaSubSum += alpha[c][i];
				tokensInSub += tokensPerSubPerTopic[c][i];
			}
			
			
			for(int i = 0; i < numTopics; i++){
				double alphaSumMinusTopic = alphaSubSum - alpha[c][i];
				//System.out.println(alphaSumMinusTopic);
			
				double low = Math.log(1 - this.gamma) * this.docsPerSub[c];
				double high = Math.log(this.gamma) * this.docsPerSub[c];
				
				for(int d = 0; d < numDocs; d++) {
					if(this.docSubreddit[d] != c) {
						continue;
					}
					
					low += logAlphaNumerator(alphaScalar * eta, alphaSumMinusTopic, d, i);
					low -= logAlphaDenominator(alphaScalar * eta, alphaSumMinusTopic, d);
					high += logAlphaNumerator(alphaScalar, alphaSumMinusTopic, d, i);
					high -= logAlphaDenominator(alphaScalar, alphaSumMinusTopic, d);
				}
				
				/* Old calculations: aggregate counts in subreddit instead of product over documents
				double x = alphaScalar * eta;
				double top = Gamma.logGamma(x + alphaSumMinusTopic) + Gamma.logGamma(tokensPerSubPerTopic[c][i] + x);
				double bot = Gamma.logGamma(x) + Gamma.logGamma(x + alphaSumMinusTopic + (tokensInSub));
				double low = Math.log(1 - gamma) + top - bot;
				
				x = alphaScalar;
				top = Gamma.logGamma(x + alphaSumMinusTopic) + Gamma.logGamma(tokensPerSubPerTopic[c][i] + x);
				bot = Gamma.logGamma(x) + Gamma.logGamma(x + alphaSumMinusTopic + (tokensInSub));
				double high = Math.log(gamma) + top - bot; */

				//System.out.println("Low:  " + low);
				//System.out.println("High: " + high);
				
				
				double p = Math.exp(low - logsum(low, high));

				//System.out.println(p);
				//System.out.println();
				
				double a = r.nextFloat();
				if(a < p) {
					alpha[c][i] = alphaScalar * eta;
				} else {
					alpha[c][i] = alphaScalar;
				}
				
				
			}
		}
	}

	private void sampleTopicsForOneDoc (FeatureSequence oneDocTokens, int[] oneDocTopics, // indexed by seq position
										int[] oneDocTopicCounts, // indexed by topic index
										double[] topicWeights, int sub, Randoms r)
	{
		int[] currentTypeTopicCounts;
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLen = oneDocTokens.getLength();
		double tw;
		// Iterate over the positions (words) in the document
		for (int si = 0; si < docLen; si++) {
			type = oneDocTokens.getIndexAtPosition(si);
			oldTopic = oneDocTopics[si];
			// Remove this token from all counts
			oneDocTopicCounts[oldTopic]--;
			typeTopicCounts[type][oldTopic]--;
			subTypeTopicCounts[sub][type][oldTopic]--;
			tokensPerTopic[oldTopic]--;
			tokensPerSubPerTopic[sub][oldTopic]--;
			// Build a distribution over topics for this token
			Arrays.fill (topicWeights, 0.0);
			topicWeightsSum = 0;
			currentTypeTopicCounts = typeTopicCounts[type];
			for (int ti = 0; ti < numTopics; ti++) {
				tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
						* ((oneDocTopicCounts[ti] + alpha[sub][ti])); // (/docLen-1+tAlpha); is constant across all topics
				topicWeightsSum += tw;
				topicWeights[ti] = tw;
			}
			// Sample a topic assignment from this distribution
			newTopic = r.nextDiscrete (topicWeights, topicWeightsSum);

			// Put that new topic into the counts
			oneDocTopics[si] = newTopic;
			oneDocTopicCounts[newTopic]++;
			typeTopicCounts[type][newTopic]++;
			subTypeTopicCounts[sub][type][newTopic]++;
			tokensPerTopic[newTopic]++;
			tokensPerSubPerTopic[sub][newTopic]++;
		}
	}
	
	public double logLikeDocNum(double alphaSum, int doc) {
		double lognum = Gamma.logGamma(alphaSum);
		int sub = this.docSubreddit[doc];
		for(int top = 0; top < this.numTopics; top++) {
			lognum += Gamma.logGamma(this.docTopicCounts[doc][top] + this.alpha[sub][top]);
		}
		return lognum;
	}
	
	public double logLikeDocDen(double alphaSum, int doc) {
		double logden = Gamma.logGamma(alphaSum + this.tokensPerDoc[doc]);
		int sub = this.docSubreddit[doc];
		for(int top = 0; top < this.numTopics; top++) {
			logden += Gamma.logGamma(this.alpha[sub][top]);
		}
		return logden;
	}
	
	public double logLikeTopicNum(int top) {
		double lognum = Gamma.logGamma(this.vBeta);
		for(int tok = 0; tok < this.numTypes; tok++) {
			lognum += Gamma.logGamma(this.typeTopicCounts[tok][top] + this.beta);
		}
		return lognum;
	}
	
	public double logLikeTopicDen(int top) {
		double logden = Gamma.logGamma(this.tokensPerTopic[top] + this.vBeta);
		for(int tok = 0; tok < this.numTypes; tok++) {
			logden += Gamma.logGamma(this.beta);
		}
		return logden;
	}
	
	public double getLogLikelihood() {
		
		double logLike = 0;
		
		//first loop over subs / docs
		for(int sub = 0; sub < this.numSubreddits; sub++) {
			
			double alphaSubSum = 0;
			int tokensInSub = 0;
			for(int i = 0; i < numTopics; i++){
				alphaSubSum += alpha[sub][i];
				tokensInSub += tokensPerSubPerTopic[sub][i];
			}
			
			for(int doc = 0; doc < this.numDocs; doc++) {
				if(sub != this.docSubreddit[doc])
					continue;
				
				logLike += logLikeDocNum(alphaSubSum, doc);
				logLike -= logLikeDocDen(alphaSubSum, doc);
			}
		}
		
		//then loop over topics
		for(int top = 0; top < this.numTopics; top++) {
			logLike += logLikeTopicNum(top);
			logLike -= logLikeTopicDen(top);
		}
		
		return logLike;
		
	}
	
	public Set<String> getSubredditSet() {
		return this.subredditMap.keySet();
	}
	
	public HashMap<String, double[]> getSubredditTopicDistributions() {
		
		HashMap<String, double[]> targetTopics = new HashMap<String, double[]>();
        HashMap<String, Integer> targetTotals = new HashMap<String, Integer>();
        Set<String> targetset = this.subredditMap.keySet();
        for (String target : targetset) {
        	targetTopics.put(target, new double[numTopics]);
        	targetTotals.put(target,  0);
        }
        
        for(int c = 0; c < this.numDocs; c++) {
        	double[] topicDist = this.getTopicProbabilities(c);
        	String target = this.ilist.get(c).getTarget().toString();
        	double[] totalDist = targetTopics.get(target);
        	for(int i = 0; i < numTopics; i++){
        		totalDist[i] += topicDist[i];
        	}
        	targetTopics.put(target, totalDist);
        	targetTotals.put(target, targetTotals.get(target) + 1);
        }
        
        for(String target: targetset) {
        	double[] dist = targetTopics.get(target);      	
        	for(int c = 0; c < numTopics; c++) {
        		dist[c] = dist[c] / targetTotals.get(target);
        	}
        	targetTopics.put(target, dist);
        }
        
        return targetTopics;
	}
	
	
	public double[] getTopicProbabilities(int instanceID) {
		//get subreddit ID
		int sub = this.docSubreddit[instanceID];//this.subredditMap.get(this.ilist.get(instanceID).getTarget().toString());
		int[] topicCounts = docTopicCounts[instanceID];
		double[] a = this.alpha[sub];
		
		double[] topicProbs = new double[numTopics];
		double sum = 0;
		for(int i = 0; i < this.numTopics; i++){
			topicProbs[i] = topicCounts[i] + a[i];
			sum += topicProbs[i];
		}
		for(int i = 0; i < this.numTopics; i++) {
			topicProbs[i] /= sum;
		}
		return topicProbs;
	}
	
	public int[][] getDocTopicCounts(){
		return docTopicCounts;
	}
	
	public int[][] getTypeTopicCounts(){
		return typeTopicCounts;
	}

	public int[] getTokensPerTopic(){
		return tokensPerTopic;
	}

	public void printTopWords (int numWords, boolean useNewLines)
	{
		class WordProb implements Comparable {
			int wi;
			double p;
			public WordProb (int wi, double p) { this.wi = wi; this.p = p; }
			public final int compareTo (Object o2) {
				if (p > ((WordProb)o2).p)
					return -1;
				else if (p == ((WordProb)o2).p)
					return 0;
				else return 1;
			}
		}

		WordProb[] wp = new WordProb[numTypes];
		for (int ti = 0; ti < numTopics; ti++) {
			for (int wi = 0; wi < numTypes; wi++)
				wp[wi] = new WordProb (wi, ((double)typeTopicCounts[wi][ti]) / tokensPerTopic[ti]);
			Arrays.sort (wp);
			if (useNewLines) {
				System.out.println ("\nTopic "+ti);
				for (int i = 0; i < numWords; i++)
					System.out.println (ilist.getDataAlphabet().lookupObject(wp[i].wi).toString() + " " + wp[i].p);
			} else {
				System.out.print ("Topic "+ti+": ");
				for (int i = 0; i < numWords; i++)
					System.out.print (ilist.getDataAlphabet().lookupObject(wp[i].wi).toString() + " ");
				System.out.println();
			}
		}
		
		for (int sub = 0; sub < this.numSubreddits; sub++) {
			double alphaSubSum = 0;
			for(int i = 0; i < numTopics; i++){
				alphaSubSum += alpha[sub][i];
			}
			double numTops = alphaSubSum / this.alphaScalar;
			System.out.print("Subreddit " + sub);
			System.out.println(" Num topics: " + numTops);
		}
		System.out.println("Model Log Likelihood: " + this.getLogLikelihood());
	}
	
	
	public void saveTopWords (int numWords, boolean useNewLines, String fname) throws Exception
	{
		class WordProb implements Comparable {
			int wi;
			double p;
			public WordProb (int wi, double p) { this.wi = wi; this.p = p; }
			public final int compareTo (Object o2) {
				if (p > ((WordProb)o2).p)
					return -1;
				else if (p == ((WordProb)o2).p)
					return 0;
				else return 1;
			}
		}
		
		Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(fname), "utf-8"));

		WordProb[] wp = new WordProb[numTypes];
		for (int ti = 0; ti < numTopics; ti++) {
			for (int wi = 0; wi < numTypes; wi++)
				wp[wi] = new WordProb (wi, ((double)typeTopicCounts[wi][ti]) / tokensPerTopic[ti]);
			Arrays.sort (wp);
			if (useNewLines) {
				
				writer.write ("\nTopic "+ti+"\n");
				for (int i = 0; i < numWords; i++)
					writer.write (ilist.getDataAlphabet().lookupObject(wp[i].wi).toString() + " " + wp[i].p + "\n");
			} else {
				writer.write ("Topic "+ti+": ");
				for (int i = 0; i < numWords; i++)
					writer.write (ilist.getDataAlphabet().lookupObject(wp[i].wi).toString() + " ");
				writer.write("\n");
			}
		}
		writer.close();
	}

	
	

	public void printDocumentTopics (File f) throws IOException
	{
	printDocumentTopics (new PrintWriter (new FileWriter (f)));
	}

	public void printDocumentTopics (PrintWriter pw) {
	printDocumentTopics (pw, 0.0, -1);
	}

	public void printDocumentTopics (PrintWriter pw, double threshold, int max)
	{
	pw.println ("#doc source topic proportion ...");
	int docLen;
	double topicDist[] = new double[topics.length];
	for (int di = 0; di < topics.length; di++) {
		pw.print (di); pw.print (' ');
			if (ilist.get(di).getSource() != null){
				pw.print (ilist.get(di).getSource().toString()); 
			}
			else {
				pw.print("null-source");
			}
			pw.print (' ');
		docLen = topics[di].length;
		for (int ti = 0; ti < numTopics; ti++)
		topicDist[ti] = (((float)docTopicCounts[di][ti])/docLen);
		if (max < 0) max = numTopics;
		for (int tp = 0; tp < max; tp++) {
		double maxvalue = 0;
		int maxindex = -1;
		for (int ti = 0; ti < numTopics; ti++)
			if (topicDist[ti] > maxvalue) {
			maxvalue = topicDist[ti];
			maxindex = ti;
			}
		if (maxindex == -1 || topicDist[maxindex] < threshold)
			break;
		pw.print (maxindex+" "+topicDist[maxindex]+" ");
		topicDist[maxindex] = 0;
		}
		pw.println (' ');
	}
	}



	public void printState (File f) throws IOException
	{
		PrintWriter writer = new PrintWriter (new FileWriter(f));
		printState (writer);
		writer.close();
	}


	public void printState (PrintWriter pw)
	{
		Alphabet a = ilist.getDataAlphabet();
		pw.println ("#doc pos typeindex type topic");
		for (int di = 0; di < topics.length; di++) {
			FeatureSequence fs = (FeatureSequence) ilist.get(di).getData();
			for (int si = 0; si < topics[di].length; si++) {
				int type = fs.getIndexAtPosition(si);
				pw.print(di); pw.print(' ');
				pw.print(si); pw.print(' ');
				pw.print(type); pw.print(' ');
				pw.print(a.lookupObject(type)); pw.print(' ');
				pw.print(topics[di][si]); pw.println();
			}
		}
	}



	public InstanceList getInstanceList ()
	{
		return ilist;
	}

	// Recommended to use mallet/bin/vectors2topics instead.
	/*public static void main (String[] args) throws IOException
	{
		InstanceList ilist = InstanceList.load (new File(args[0]));
		int numIterations = args.length > 1 ? Integer.parseInt(args[1]) : 1000;
		int numTopWords = args.length > 2 ? Integer.parseInt(args[2]) : 20;
		System.out.println ("Data loaded.");
		LDA lda = new LDA (10);
		lda.estimate (ilist, numIterations, 50, 0, null, new Randoms());	// should be 1100
		lda.printTopWords (numTopWords, true);
		lda.printDocumentTopics (new File(args[0]+".lda"));
	}*/

}
