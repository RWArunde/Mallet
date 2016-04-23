package cc.mallet.examples;

import cc.mallet.util.*;
import cc.mallet.types.*;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;
import cc.mallet.topics.*;

import java.util.*;
import java.util.regex.*;
import java.io.*;

public class TestLDA {

	public static void main(String[] args) throws Exception {

		// Begin by importing documents from text to feature sequences
		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();

		// Pipes: lowercase, tokenize, remove stopwords, map to features
		pipeList.add( new CharSequenceLowercase() );
        pipeList.add( new CharSequenceReplace(Pattern.compile("[\\-'\"]"), "") );
        pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")) );
		pipeList.add( new TokenSequenceRemoveStopwords(new File("stoplists/en.txt"), "UTF-8", false, false, false) );
		pipeList.add( new TokenSequence2FeatureSequence() );

		InstanceList instances = new InstanceList (new SerialPipes(pipeList));

		Reader fileReader = new InputStreamReader(new FileInputStream(new File("out.txt")), "UTF-8");
		instances.addThruPipe(new CsvIterator (fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
											   3, 2, 1)); // data, label, name fields

		int numTopics = 20;
		//params are Documents, top subreddit, num topics, alpha, beta, gamma, eta
		
		//for sparsity
		//RedditLDA model = new RedditLDA(instances, "nhl", numTopics, .5, 0.01, 0.15, 0.01);

		//Regular LDA
		RedditLDA model = new RedditLDA(instances, "nhl", numTopics, 1, 0.01, 1, 0.01);
		
		//go through all documents, 100 iter, print every 10
		model.estimate(0, instances.size(), 200, 10, new Randoms());
		
		//System.out.println(instances.size());
		model.printTopWords(5, false);
		
		HashMap<String, double[]> targetTopics = model.getSubredditTopicDistributions();
		
		Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream("subreddit_topics.json"), "utf-8"));
        writer.write("{");
        
        for(String target : model.getSubredditSet()){
        	double[] dist = targetTopics.get(target);
        	writer.write("\"" + target + "\":[");
        	for(int c = 0; c < numTopics; c++) {
        		if(c > 0)
        			writer.write(",");
        		writer.write(String.valueOf(dist[c]));
        	}
        	writer.write("],");
        }
        writer.write("\"gg\":4}");
        writer.close();
	}

}