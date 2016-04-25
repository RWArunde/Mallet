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

	public static void LDA(String topsub, int iters, int topics, double alpha, double beta, InstanceList docs) throws Exception {
		System.out.println("Running LDA model");
		RedditLDA model = new RedditLDA(docs, topsub, topics, alpha, beta, 9001, 9001);
		//go through all documents, 100 iter, print every 10
		model.estimate(0, docs.size(), iters, 100, new Randoms());
		write_sub_dists(model, "subreddit_topics_LDA_"+topsub+"_"+alpha+"_"+beta+".json");
		write_top_words(model, "topic_words_LDA_"+topsub+"_"+alpha+"_"+beta+".txt");
		write_log_like(model, "model_log_like_LDA_"+topsub+"_"+alpha+"_"+beta+".txt");
	}
	
	public static void SparseLDA(String topsub, int iters, int topics, double alpha, double beta, double gamma, double eta, InstanceList docs) throws Exception {
		System.out.println("Running Sparse LDA model");
		RedditLDA model = new RedditLDA(docs, topsub, topics, alpha, beta, gamma, eta);
		//go through all documents, 100 iter, print every 10
		model.estimate(0, docs.size(), iters, 100, new Randoms());
		write_sub_dists(model, "subreddit_topics_sparse_"+topsub+"_"+alpha+"_"+beta+"_"+gamma+"_"+eta+".json");
		write_top_words(model, "topic_words_sparse_"+topsub+"_"+alpha+"_"+beta+"_"+gamma+"_"+eta+".txt");
		write_log_like(model, "model_log_like_sparse_"+topsub+"_"+alpha+"_"+beta+"_"+gamma+"_"+eta+".txt");
		write_alphas(model, "subreddit_alphas_sparse_"+topsub+"_"+alpha+"_"+beta+"_"+gamma+"_"+eta+".json");
	}
	
	public static void write_sub_dists(RedditLDA model, String fname) throws Exception {
		HashMap<String, double[]> targetTopics = model.getSubredditTopicDistributions(false);
		Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(fname), "utf-8"));
        writer.write("{");
        
        for(String target : model.getSubredditSet()){
        	double[] dist = targetTopics.get(target);
        	writer.write("\"" + target + "\":[");
        	for(int c = 0; c < model.numTopics; c++) {
        		if(c > 0)
        			writer.write(",");
        		writer.write(String.valueOf(dist[c]));
        	}
        	writer.write("],");
        }
        writer.write("\"gg\":4}");
        writer.close();
	}
	
	public static void write_alphas(RedditLDA model, String fname) throws Exception {
		double[][] alpha = model.alpha;
		Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(fname), "utf-8"));
        writer.write("{");
        
        for(String target : model.getSubredditSet()){
        	int sub = model.subredditMap.get(target);
        	
        	double[] dist = alpha[sub];
        	writer.write("\"" + target + "\":[");
        	for(int c = 0; c < model.numTopics; c++) {
        		if(c > 0)
        			writer.write(",");
        		writer.write(String.valueOf(dist[c]));
        	}
        	writer.write("],");
        }
        writer.write("\"gg\":4}");
        writer.close();
	}
	
	public static void write_top_words(RedditLDA model, String fname) throws Exception {
		model.saveTopWords(50, false, fname);
	}
	
	public static void write_log_like(RedditLDA model, String fname) throws Exception {
		double like = model.getLogLikelihood();
		Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(fname), "utf-8"));
		writer.write("" + like);
		writer.close();
	}
	
	
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

		//SparseLDA("nhl", 10, 20, 1, 0.01, 0.15, 0.01, instances);
		//LDA("nhl", 10, 20, 1, 0.01, instances);
		
		System.out.println("Stuff is working");
		
		//TestLDA.SparseLDA("nhl", 100, 20, 1, 0.01, 0.10, 0.005, instances);
		//TestLDA.LDA("nhl", 100, 20, 1, 0.01, instances);
		
		
		String topsub = "nfl";
		int iterations = 1000;
		int topics = 20;
		
		//Start workers for regular LDA
		for(double alpha : new double[]{0.05, 0.25, 0.5, 1, 5}) {
			LDAWorker worker = new LDAWorker(topsub, iterations, topics, alpha, 0.01, 9000, 9000, instances);
			worker.start();
			
			// Lose track of the worker. Assume it will finish by the time the SparseLDA workers do.
			// good enough for grad school.
			
			
			//LDA("nhl", 1000, 20, alpha, 0.01, instances);
		}
		
		for(double alpha: new double[]{0.25, 0.5, 1, 5}) {
			//keep track of these workers- we'll need to check when they finish. 
			ArrayList<LDAWorker> workers = new ArrayList<LDAWorker>();
			
			for(double gamma: new double[]{0.25, 0.15, 0.05, 0.005}) {
				for(double eta: new double[]{0.01, 0.005, 0.001}) {
					
					LDAWorker worker = new LDAWorker(topsub, iterations, topics, alpha, 0.01, gamma, eta, instances);
					worker.start();
					workers.add(worker);
					
					//SparseLDA("nhl", 1000, 20, alpha, 0.01, gamma, eta, instances);
				}
			}
			
			//wait for workers to finish
			for(LDAWorker e : workers) {
				e.join();
			}
		}
		
		
		
		
		
		/*int numTopics = 20;
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
        writer.close();*/
	}

}