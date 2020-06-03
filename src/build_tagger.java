import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;
import java.util.Set;


public class build_tagger {
	public static void main(String[] args) throws IOException {
		build_tagger builder = new build_tagger();
		if(args.length == 3) {
			// default to interpolation (2) for smoothing,
			// and hapax legomena (2) for handling unknown words
			builder.train(args[0], args[1], args[2], 2, 2);
		} else if(args.length == 5) {
			int smoothType = Integer.parseInt(args[3]);
			int unkHandlingType = Integer.parseInt(args[4]);
			builder.train(args[0], args[1], args[2], smoothType, unkHandlingType);
		}
	}
	
	private HashMap<String,Integer> tagToHash;
	private double transProb[][]; // P(tag_j|tag_i) is stored in transProb[i][j]
	private HashMap<String, double[]> emProb; // P(w|tag_i) is stored in emProb.get(w)[i]
	
	private double tagProb[]; // P(tag_i) is stored in tagProb[i]
	
	public build_tagger() {
		tagToHash = new HashMap<String, Integer>();
		for(int i = 0; i < tagset.length; i++) {
			tagToHash.put(tagset[i], i);
		}
		transProb = new double[tagset.length][tagset.length];
		emProb = new HashMap<String, double[]>();
		tagProb = new double[tagset.length];
	}
	
	public void train(String trainFilename, String tuneFilename, String modelFilename,
					  int smoothType, int unkHandlingType) throws IOException {
		Scanner sc = new Scanner(new File(trainFilename));
		while(sc.hasNextLine()) { // collect bigram and unigram counts
			String sent = sc.nextLine();
			String words[] = sent.split(" ");
			String word, tag; int sep, tagIdx, prevTagIdx;
			int startIdx = tagToHash.get("<s>");
			int endIdx = tagToHash.get("</s>");
			
			sep = words[0].lastIndexOf('/');
			word = words[0].substring(0, sep);
			tag = words[0].substring(sep+1,words[0].length());
			tagIdx = tagToHash.get(tag);
			
			transProb[startIdx][tagIdx]++;
			if(!emProb.containsKey(word)) {
				emProb.put(word, new double[tagset.length]);
			}
			emProb.get(word)[tagIdx]++;
			tagProb[startIdx]++;
			
			prevTagIdx = tagIdx;
			for(int i = 1; i < words.length; i++) {
				sep = words[i].lastIndexOf('/');
				word = words[i].substring(0, sep);
				tag = words[i].substring(sep+1,words[i].length());
				tagIdx = tagToHash.get(tag);
				transProb[prevTagIdx][tagIdx]++;
				if(!emProb.containsKey(word)) {
					emProb.put(word, new double[tagset.length]);
				}
				emProb.get(word)[tagIdx]++;
				tagProb[tagIdx]++;
				prevTagIdx = tagIdx;
			}
			
			transProb[prevTagIdx][endIdx]++;
			tagProb[endIdx]++;
		}
		smoothing(smoothType, transProb, tagProb, tuneFilename);
		modelUnknownWords(unkHandlingType, emProb);
		printModel(modelFilename);
		sc.close();
	}
	
	private void smoothing(int type, double bProbTable[][], double uProbTable[], String tuneFilename) throws IOException {
		switch(type) {
		case 1: // add-one
			addOneSmoothing(bProbTable);
			countToProb(bProbTable);
			logProb(bProbTable);
			break;
		case 2: // linear interpolation
			countToProb(bProbTable);
			countToProb(uProbTable);
			double lambdas[] = tune(tuneFilename);
			applyWeights(bProbTable, uProbTable, lambdas);
			logProb(bProbTable);
			break;
		}
	}
	
	private void addOneSmoothing(double probTable[][]) {
		for(int i = 0; i < probTable.length; i++) {
			for(int j = 0; j < probTable.length; j++) {
				probTable[i][j]++;
			}
		}
	}
	
	private double[] tune(String tuneFilename) throws IOException {
		Scanner sc = new Scanner(new File(tuneFilename));
		double bCnt[][] = new double[tagset.length][tagset.length];
		double uCnt[] = new double [tagset.length];
		while(sc.hasNextLine()) { // collect bigram and unigram counts
			String sent = sc.nextLine();
			String words[] = sent.split(" ");
			String tag; int sep, tagIdx, prevTagIdx;
			int startIdx = tagToHash.get("<s>");
			int endIdx = tagToHash.get("</s>");
			
			sep = words[0].lastIndexOf('/');
			tag = words[0].substring(sep+1,words[0].length());
			tagIdx = tagToHash.get(tag);
			
			bCnt[startIdx][tagIdx]++;
			uCnt[startIdx]++;
			
			prevTagIdx = tagIdx;
			for(int i = 1; i < words.length; i++) {
				sep = words[i].lastIndexOf('/');
				tag = words[i].substring(sep+1,words[i].length());
				tagIdx = tagToHash.get(tag);
				bCnt[prevTagIdx][tagIdx]++;
				uCnt[tagIdx]++;
				prevTagIdx = tagIdx;
			}
			bCnt[prevTagIdx][endIdx]++;
			uCnt[endIdx]++;
		}
		double lambda1 = 0.0, lambda2 = 0.0;
		double uDenum = rowSum(uCnt);
		for(int i = 0; i < tagset.length; i++) { // deleted interpolation
			double bDenum = rowSum(bCnt[i]);
			for(int j = 0; j < tagset.length; j++) {
				double b = (bDenum-1 > 0) ? (bCnt[i][j]-1)/(bDenum-1) : 0;
				double u = (uDenum-1 > 0) ? (uCnt[j]-1)/(uDenum-1) : 0;
				if(b >= u) {
					lambda2++;
				} else {
					lambda1++;
				}
			}
		}
		double denum = lambda1 + lambda2;
		return new double[] { lambda1/denum, lambda2/denum };
	}
	
	private void applyWeights(double[][] bProbTable, double[] uProbTable, double[] lambdas) {
		for(int i = 0; i < bProbTable.length; i++) {
			for(int j = 0; j < bProbTable.length; j++) {
				bProbTable[i][j] = lambdas[0] * bProbTable[i][j] + lambdas[1] * uProbTable[j];
			}
		}
	}
	
	private void modelUnknownWords(int type, HashMap<String, double[]> probTable) {
		switch(type) {
		case 1:
			mostProbableTag(probTable);
			break;
		case 2:
			hapaxLegomena(probTable);
			break;
		}
		countToProb(probTable);
		logProb(probTable);
	}
	
	private void mostProbableTag(HashMap<String, double[]> probTable) {
		int startIdx = tagToHash.get("<s>");
		int endIdx = tagToHash.get("</s>");
		double max = Double.NEGATIVE_INFINITY;
		int tag = 0;
		for(int i = 0; i < tagset.length; i++) { // find the most probable tag
			if(i != startIdx && i != endIdx) {
				double cnt = rowSum(transProb[i]);
				if(cnt > max) {
					max = cnt; tag = i;
				}
			}
		}
		double unkCnt[] = new double[tagset.length];
		unkCnt[tag]++;
		probTable.put("<unk>", unkCnt);
	}
	
	private void hapaxLegomena(HashMap<String, double[]> probTable) {
		Set<String> words = probTable.keySet();
		double unkCnt[] = new double[tagset.length];
		for(String word : words) {
			double row[] = probTable.get(word); 
			if(rowSum(row) == 1) {
				for(int i = 0; i < tagset.length; i++) {
					unkCnt[i] += row[i];
				}
			}
		}
		probTable.put("<unk>", unkCnt);
	}
	
	private void countToProb(double[][] probTable) {
		for(int i = 0; i < probTable.length; i++) {
			double denum = rowSum(probTable[i]);  
			for(int j = 0; j < probTable.length; j++) {
				probTable[i][j] = probTable[i][j]/denum;
			}
		}
	}
	
	private void logProb(double[][] probTable) {
		for(int i = 0; i < probTable.length; i++) {
			for(int j = 0; j < probTable.length; j++) {
				probTable[i][j] = Math.log(probTable[i][j]);
			}
		}
	}
	
	private void countToProb(HashMap<String, double[]> probTable) {
		int startIdx = tagToHash.get("<s>");
		int endIdx = tagToHash.get("</s>");
		double colSum[] = new double[tagset.length];
		Set<String> words = probTable.keySet();
		for(String word : words) {
			for(int i = 0; i < tagset.length; i++) {
				if(i != startIdx && i != endIdx) {
					colSum[i] += probTable.get(word)[i];
				}
			}
		}
		for(String word : words) {
			for(int i = 0; i < tagset.length; i++) {
				if(i != startIdx && i != endIdx) {
					probTable.get(word)[i] = probTable.get(word)[i]/colSum[i];
				}
			}
		}
	}
	
	private void logProb(HashMap<String, double[]> probTable) {
		int startIdx = tagToHash.get("<s>");
		int endIdx = tagToHash.get("</s>");
		Set<String> words = probTable.keySet();
		for(String word : words) {
			for(int i = 0; i < tagset.length; i++) {
				if(i != startIdx && i != endIdx) {
					probTable.get(word)[i] = Math.log(probTable.get(word)[i]);
				}
			}
		}
	}
	
	private void countToProb(double[] probTable) {
		double total = rowSum(probTable);
		for(int i = 0; i < probTable.length; i++) {
			probTable[i] = probTable[i]/total;
		}
	}
	
	private double rowSum(double[] aRow) {
		double ret = 0.0;
		for(int i = 0; i < aRow.length; i++) {
			ret += aRow[i];
		}
		return ret;
	}
	
	private void printModel(String modelFilename) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(modelFilename)));
		for(int i = 0; i < transProb.length; i++) {
			for(int j = 0; j < transProb.length; j++) {
				if(j > 0) writer.write(" ");
				writer.write(""+transProb[i][j]);
			}
			writer.newLine();
		}
		for(String word : emProb.keySet()) {
			writer.write(word);
			for(int i = 0; i < emProb.get(word).length; i++) {
				writer.write(" "+emProb.get(word)[i]);
			}
			writer.newLine();
		}
		writer.close();
	}
	
	private final String tagset[] = {
			"<s>",
			"#",
			"$",
			"''",
			",",
			"-LRB-",
			"-RRB-",
			".",
			":",
			"CC",
			"CD",
			"DT",
			"EX",
			"FW",
			"IN",
			"JJ",
			"JJR",
			"JJS",
			"LS",
			"MD",
			"NN",
			"NNP",
			"NNPS",
			"NNS",
			"PDT",
			"POS",
			"PRP",
			"PRP$",
			"RB",
			"RBR",
			"RBS",
			"RP",
			"SYM",
			"TO",
			"UH",
			"VB",
			"VBD",
			"VBG",
			"VBN",
			"VBP",
			"VBZ",
			"WDT",
			"WP",
			"WP$",
			"WRB",
			"``",
			"</s>"
	};
}
