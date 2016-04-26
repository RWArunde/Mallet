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
	
	public static void classifySparseLDA(String topsub, int iters, int topics, double alpha, double beta, double gamma, double eta, InstanceList docs) throws Exception {
		System.out.println("Running Sparse LDA classification");
		RedditLDA model = new RedditLDA(docs, topsub, topics, alpha, beta, gamma, eta);
		int end = (int) Math.floor(docs.size() * 0.8);
		model.estimate(0, end, iters, 100, new Randoms());
		int[][] confusion = new int[model.numSubreddits][model.numSubreddits];
		double accuracy = model.classifyDocuments(25, confusion);
		System.out.println("Accuracy: " + accuracy + "\tF-score (macro): " + model.FMacro);
		write_confusions(model, confusion, "confusions_sparse_"+topsub+"_"+alpha+"_"+beta+"_"+gamma+"_"+eta+".json");
		write_accuracy(model, accuracy, "accuracy_sparse_"+topsub+"_"+alpha+"_"+beta+"_"+gamma+"_"+eta+".txt");
	}
	
	public static void classifyLDA(String topsub, int iters, int topics, double alpha, double beta, InstanceList docs) throws Exception {
		System.out.println("Running LDA classification");
		RedditLDA model = new RedditLDA(docs, topsub, topics, alpha, beta, 9001, 9001);
		int end = (int) Math.floor(docs.size() * 0.8);
		model.estimate(0, end, iters, 100, new Randoms());
		int[][] confusion = new int[model.numSubreddits][model.numSubreddits];
		double accuracy = model.classifyDocuments(25, confusion);
		System.out.println("Accuracy: " + accuracy + "\tF-score (macro): " + model.FMacro);
		write_confusions(model, confusion, "confusions_LDA_"+topsub+"_"+alpha+"_"+beta+".json");
		write_accuracy(model, accuracy, "accuracy_LDA_"+topsub+"_"+alpha+"_"+beta+".txt");
	}

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
	
	public static void write_confusions(RedditLDA model, int[][] confusions, String fname) throws Exception {
		Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(fname), "utf-8"));
        writer.write("{");
        
        for(String target : model.getSubredditSet()){
        	int sub = model.subredditMap.get(target);
        	
        	int[] dist = confusions[sub];
        	writer.write("\"" + target + "\":[");
        	for(int c = 0; c < model.numSubreddits; c++) {
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
	
	public static void write_accuracy(RedditLDA model, double accuracy, String fname) throws Exception {
		Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(fname), "utf-8"));
		writer.write("" + accuracy + "\n" + model.FMacro);
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

		/*InstanceList instances = new InstanceList (new SerialPipes(pipeList));
		Reader fileReader = new InputStreamReader(new FileInputStream(new File("out.txt")), "UTF-8");
		instances.addThruPipe(new CsvIterator (fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
											   3, 2, 1)); // data, label, name fields*/
		
		InstanceList nflinstances = new InstanceList (new SerialPipes(pipeList));
		Reader fileReader2 = new InputStreamReader(new FileInputStream(new File("nfl.txt")), "UTF-8");
		nflinstances.addThruPipe(new CsvIterator (fileReader2, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
											   3, 2, 1)); // data, label, name fields
		
		InstanceList nhlinstances = new InstanceList (new SerialPipes(pipeList));
		Reader fileReader3 = new InputStreamReader(new FileInputStream(new File("nhl.txt")), "UTF-8");
		nhlinstances.addThruPipe(new CsvIterator (fileReader3, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
											   3, 2, 1)); // data, label, name fields

		
		//Run a couple models...
		//SparseLDA("nhl", 10, 20, 1, 0.01, 0.15, 0.01, instances);
		//LDA("nhl", 10, 20, 1, 0.01, instances);
		
		
		//Do a classification test...
		//classifySparseLDA("nhl", 40, 20, 1, 0.01, 0.15, 0.01, instances);
		//classifyLDA("nhl", 40, 20, 1, 0.01, instances);
		
		
		//Parallel classification tasks
		LDAClassifyWorker w1 = new LDAClassifyWorker("nhl", 1000, 30, 0.5, 0.01, 0.15, 0.01, nhlinstances);
		LDAClassifyWorker w2 = new LDAClassifyWorker("nhl", 1000, 30, 0.5, 0.01, 9000, 9000, nhlinstances);
		w1.start();
		w2.start();
		
		LDAClassifyWorker w3 = new LDAClassifyWorker("nfl", 1000, 30, 0.5, 0.01, 0.15, 0.01, nflinstances);
		LDAClassifyWorker w4 = new LDAClassifyWorker("nfl", 1000, 30, 0.5, 0.01, 9000, 9000, nflinstances);
		w3.start();
		w4.start();
		
		
		//Parallel whole corpus fitting tasks
		LDAWorker w5 = new LDAWorker("nhl", 1000, 30, 0.25, 0.01, 9000, 9000, nhlinstances);
		LDAWorker w6 = new LDAWorker("nhl", 1000, 30, 0.5,  0.01, 0.15, 0.01, nhlinstances);
		w5.start();
		w6.start();
		
		LDAWorker w7 = new LDAWorker("nfl", 1000, 30, 0.25, 0.01, 9000, 9000, nflinstances);
		LDAWorker w8 = new LDAWorker("nfl", 1000, 30, 0.5,  0.01, 0.15, 0.01, nflinstances);
		w7.start();
		w8.start();
		
		System.out.println("Stuff is working");
		
		w1.join();
		w2.join();
		w3.join();
		w4.join();
		w5.join();
		w6.join();
		w7.join();
		w8.join();
		
		
		//TestLDA.SparseLDA("nhl", 100, 20, 1, 0.01, 0.10, 0.005, instances);
		//TestLDA.LDA("nhl", 100, 20, 1, 0.01, instances);
		
		/*
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
			
			for(double gamma: new double[]{0.25, 0.15, 0.05, 0.005}) {
				//keep track of these workers- we'll need to check when they finish. 
				ArrayList<LDAWorker> workers = new ArrayList<LDAWorker>();
				
				for(double eta: new double[]{0.01, 0.005, 0.001, 0.0005}) {
					
					LDAWorker worker = new LDAWorker(topsub, iterations, topics, alpha, 0.01, gamma, eta, instances);
					worker.start();
					workers.add(worker);
					
					//SparseLDA("nhl", 1000, 20, alpha, 0.01, gamma, eta, instances);
				}
				
				//wait for workers to finish
				for(LDAWorker e : workers) {
					e.join();
				}
				
			}
			
		}*/
		
		
		
		
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