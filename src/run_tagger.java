import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;


public class run_tagger {
	public static void main(String[] args) throws IOException {
		run_tagger runner = new run_tagger(args[1]);
		if(args.length == 3) {
			runner.tag(args[0],args[2]);
			System.out.println("Accuracy for " + args[0] + ": " + runner.eval(args[2], "sents.devt"));
		} else if(args.length == 4) {
			runner.tag(args[0],args[2]);
			System.out.println("Accuracy for " + args[0] + ": " + runner.eval(args[2], args[3]));
		}
	}
	
	private HashMap<String,Integer> tagToHash;
	private double transProb[][];
	private HashMap<String, double[]> emProb;
	
	public run_tagger(String modelFilename) throws IOException {
		tagToHash = new HashMap<String, Integer>();
		for(int i = 0; i < tagset.length; i++) {
			tagToHash.put(tagset[i], i);
		}
		transProb = new double[tagset.length][tagset.length];
		emProb = new HashMap<String, double[]>();
		
		BufferedReader reader = new BufferedReader(new FileReader(modelFilename));
		String line = "";
		for(int i = 0; i < tagset.length; i++) {
			line = reader.readLine();
			String probs[] = line.split(" ");
			for(int j = 0; j < tagset.length; j++) {
				transProb[i][j] = Double.parseDouble(probs[j]);
			}
		}
		
		while((line = reader.readLine()) != null) {
			String tokens[] = line.split(" ");
			emProb.put(tokens[0], new double[tagset.length]);
			for(int i = 0; i < tagset.length; i++) {
				emProb.get(tokens[0])[i] = Double.parseDouble(tokens[i+1]);
			}
		}
	}
	
	public void tag(String testFilename, String resultFilename) throws IOException {
		Scanner sc = new Scanner(new File(testFilename));
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(resultFilename)));
		while(sc.hasNextLine()) {
			String sent = sc.nextLine();
			String words[] = sent.split(" ");
			String tags[] = new String[words.length];
			decode(Arrays.copyOf(words, words.length), tags);
			
			for(int i = 0; i < tags.length; i++) { // simple number handling
				if(isNumber(words[i])) {
					tags[i] = "CD";
				}
			}
			
			for(int i = 0; i < words.length; i++) {
				if(i > 0) writer.write(" ");
				writer.write(words[i]+"/"+tags[i]);
			}
			writer.newLine();
		}
		sc.close();
		writer.close();
	}
	
	private void decode(String words[], String tags[]) {
		int N = tagset.length;
		int T = words.length;
		double viterbi[][] = new double[N][T];
		int backptr[][] = new int[N][T];
		int startIdx = tagToHash.get("<s>");
		int endIdx = tagToHash.get("</s>");
		for(int i = 0; i < T; i++) {
			if(!emProb.containsKey(words[i])) {
				words[i] = "<unk>";
			}
		}
		for(int i = 0; i < N; i++) {
			if(i != startIdx && i != endIdx) {
				viterbi[i][0] = transProb[startIdx][i] + emProb.get(words[0])[i] ;
			}
		}
		for(int j = 1; j < T; j++) {
			for(int i = 0; i < N; i++) {
				if(i == startIdx || i == endIdx) continue;
				double maxProb = Double.NEGATIVE_INFINITY;
				for(int k = 0; k < N; k++) {
					if(k == startIdx || k == endIdx) continue;
					double newProb = viterbi[k][j-1] + transProb[k][i];
					if(newProb > maxProb) {
						maxProb = newProb;
						backptr[i][j] = k;
					}
				}
				viterbi[i][j] = maxProb + emProb.get(words[j])[i];
			}
		}
		int lastPtr = 0;
		double maxProb = Double.NEGATIVE_INFINITY;
		for(int k = 0; k < N; k++) {
			if(k == startIdx || k == endIdx) continue;
			double newProb = viterbi[k][T-1] + transProb[k][endIdx];
			if(newProb > maxProb) {
				maxProb = newProb;
				lastPtr = k;
			}
		}
		tags[T-1] = tagset[lastPtr];
		for(int i = T-1; i > 0; i--) {
			lastPtr = backptr[lastPtr][i];
			tags[i-1] = tagset[lastPtr];
		}
	}
	
	public double eval(String hypFilename, String goldFilename) throws IOException {
		double numCorrect = 0.0, total = 0.0;
		Scanner scH = new Scanner(new File(hypFilename));
		Scanner scG = new Scanner(new File(goldFilename));
		while(scH.hasNextLine()) {
			String hyp = scH.nextLine();
			String gold = scG.nextLine();
			String wordsH[] = hyp.split(" ");
			String wordsG[] = gold.split(" ");
			for(int i = 0; i < wordsH.length; i++) {
				if(wordsH[i].equals(wordsG[i])) numCorrect++;
				total++;
			}
		}
		scH.close(); scG.close();
		return numCorrect/total;
	}
	
	private static boolean isNumber(String guess) {
		guess = guess.replace(",","").replace("/", "").replace("-", "");
		if(guess.startsWith("'")) guess = guess.substring(1, guess.length());
		if(guess.endsWith("s")) guess = guess.substring(0, guess.length()-1);
		try {
			Double.parseDouble(guess);
			return true;
		} catch (NumberFormatException ex) {
			return false;
		}
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
