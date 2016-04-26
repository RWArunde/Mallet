package cc.mallet.examples;

import cc.mallet.types.InstanceList;

public class LDAClassifyWorker extends Thread {

	boolean sparse;
	String topsub;
	int iters;
	int topics;
	double alpha;
	double beta;
	double gamma;
	double eta;
	InstanceList docs;
	
	public LDAClassifyWorker(String topsub, int iters, int topics, double alpha, double beta, double gamma, double eta, InstanceList docs) {
		if(gamma >= 1) {
			this.sparse = true;
		}
		
		this.topsub = topsub;
		this.iters = iters;
		this.topics = topics;
		this.alpha = alpha;
		this.beta = beta;
		this.gamma = gamma;
		this.eta = eta;
		this.docs = docs;
		
	}
	
	public void run() {
		if(this.sparse) {
			try {
				TestLDA.classifyLDA(topsub, iters, topics, alpha, beta, docs);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			try {
				TestLDA.classifySparseLDA(topsub, iters, topics, alpha, beta, gamma, eta, docs);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	
}
