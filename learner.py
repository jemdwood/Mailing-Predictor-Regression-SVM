from __future__ import print_function
import collections, math, csv, operator, os, datetime, sys, random, string
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVR
import numpy as np

def mainMethod():

  ZIPCODES_PATH = "./zipdata.csv"
  DATASET_PATH = "./dataset.csv"
  AVG_LABEL = "Average"

  data_labels = ['prospectid',  'packageid', 'zip_income_percentile']
  external_data_labels = ['zip_income_percentile',]
  lines_to_read = 10000

  US_median_income = 51939.0
  
  Laplace_Smoothing = int(True) #For setting the priors
  LS = Laplace_Smoothing

  ID_Tag = 'MailCodeId' #The main ID that is sorted by
  Algorithm = "Regression" #Regression or svm
  REGRESSION_TYPE = "Linear" #Linear or Log, this is the default one
  LOSS_FN = "Abs Dev" #Least SQUARES or Abs Dev

  def sign(x):
      if x> 0:
        return 1.0
      else:
        return -1.0

  def getDate(datetimestring):
                datestring = datetimestring.split()  
                dates = datestring[0].split('-')
                date = datetime.date(*map(int,dates))
                return date 

  def dotProduct(d1, d2):
      """
      @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
      @param dict d2: same as d1
      @return float: the dot product between d1 and d2
      """
      if len(d1) < len(d2):
          return dotProduct(d2, d1)
      else:
          return sum(d1.get(f, 0) * v for f, v in d2.items())

  def increment(d1, scale, d2):
      """
      Implements d1 += scale * d2 for sparse vectors.
      @param dict d1: the feature vector which is mutated.
      @param float scale
      @param dict d2: a feature vector.
      """
      for f, v in d2.items():
          d1[f] = d1.get(f, 0) + v * scale

  def write_csv(name, list_to_write):
    name = " ".join([Algorithm, str(lines_to_read), name])
    with open(name, 'w') as cfile:
      writer = csv.writer(cfile, dialect = 'excel', delimiter = ',')
      
      writer.writerows(list_to_write)

  def packageLabel(primaryLabel, secondaryLabel):
      return " : ".join([primaryLabel, secondaryLabel])

  def unpackageLabel(label):
      return label.split(" : ")

  class Predictor:
        def __init__(self):
            self.leaderboard = list()
            self.data = None
            self.zipdata = None
            self.userData = dict()
            self.feature_keys = set()
            self.zipcode_incomes = dict()
            self.zipcodes = set()
            self.svmVecs = dict()
            self.priorAmountsAverages = dict().fromkeys(data_labels, 0)
            self.priorRatesAverages = dict().fromkeys(data_labels, 0)

        #Returns the corresponding rounded income
        def income_level(self,income):
          
          income_percentile = round(income, -2) #not the actual percentile, just a rounded income
          
          return income_percentile

        #Returns the income at a given zipcode. Usually used in conjunction with income_level
        def income_at(self,zipcode):
          if zipcode not in self.zipcode_incomes: income = US_median_income
          else: 
            digits = self.zipcode_incomes[zipcode].partition(',')
            income = "".join([digits[i] for i in range(len(digits))[::2]]) #removes any commas
          return float(income)


        #Opens and reads the zipcode data into the program
        def readZipData(self, filepath):
            with open(filepath, 'rb') as csvfile:
                 reader = csv.DictReader(csvfile)
                 for singleResponse in reader:
                     zipcode = singleResponse['zip']
                     median = singleResponse['median']
                     self.zipcode_incomes[zipcode] = median


        
        #Fn: readData
        #Reads the data into two self contained files, trainingSet and testSet, where testSet is the most recent 15%
        # of recorded interactions/entries. These datasets are then structured such that each is a dictionary 
        # mapping a ID_Tag label (e.g. prospect id) to a list of all interactions with that ID. Each interaction is a dictionary collection
        # that maps to datapoints.
        def readData(self, data_file):
            def processDataset(dataset): #processes a data set, mapping each entry to a dictionary of users/chosen ID_Tag and adding it to that users list of interactions
                userData = dict({entry[ID_Tag] :list() for entry in dataset}) #Creates a dict mapping all id tags to an empty list
                for entry in dataset:
                    idtag = entry[ID_Tag]
                    userData[idtag].append(entry) #adds entry to that id's list of interactions  
                print("Done processing...")
                return userData
                #--------------#
            data = list()           
            with open(data_file, 'rb') as csvfile:
                 reader = csv.DictReader(csvfile)
                 for singleResponse in reader:
                     if reader.line_num >lines_to_read: break #Limits amount read, couldn't operate on entire dataset
                     data.append(singleResponse)
                 print("Total entries used:", len(data))
                 data = sorted(data, key = lambda x: x['datemailed'], reverse = True)
                 self.data = processDataset(data)
                 IDs = self.data.keys()
                 trainingSetLen = int(0.85 * len(IDs)) #HARD CODED! Percent for Training = 0.85
                 self.trainingSetIDs = IDs[0:trainingSetLen]
                 self.testSetIDs = IDs[trainingSetLen: len(IDs)]

        #Sets the prior values for each feature by going entry by entry and adding it to the list of a given data labels. 
        #Then the program averages across each list and creates a new dictionary for both amounts and rates with the average.
        #Also, each priors dict has a special value "Average" for each data label that connotes the average value across all priors with
        #That data label (e.g. "Average : prospectid"))
        def setPriors(self, IDs):
            def update_priors(amounts, rates, entry):
                for label in data_labels:
                  if(label not in external_data_labels):  identifier = entry[label]
                  elif label == 'zip_income_percentile': identifier =   self.income_level(self.income_at(entry['zip']))
                  else: assert(False) #should never happen
                  tag = packageLabel(label, str(identifier))
                  if tag not in amounts: amounts[tag] = list()
                  if tag not in rates: rates[tag] = list()
                  amounts[tag].append(float(entry['amount']))
                  rates[tag].append(float(entry['donated']))
                  self.priorAmountsAverages[label] += float(entry['amount'])/len(IDs)
                  self.priorRatesAverages[label] += float(entry['donated'])/len(IDs)
            
            def average_priors(amounts, rates):
              avg_amounts = dict.fromkeys(data_labels, Counter())
              avg_rates = dict.fromkeys(data_labels, Counter())
              for tag in amounts: ####
                    avg_amt = 1.0*(sum((amounts[tag]))+1.0*LS)/(len(amounts[tag]) + 1.0*LS)####
                    avg_rate = 1.0*(sum(rates[tag])+.5*LS)/(len(rates[tag])+.5*LS) ####
                    avg_amounts[tag] = float(avg_amt)
                    avg_rates[tag] = float(avg_rate)
              return (avg_amounts, avg_rates)

            amounts = dict()
            rates = dict()
            for uid in IDs:
                udata = self.data[uid]
                for i in range(len(udata)):
                    update_priors(amounts, rates, udata[i])
            avg_amounts, avg_rates = average_priors(amounts, rates) 

            for dlabel in data_labels:
                avg_amounts[packageLabel(AVG_LABEL, dlabel)] = self.priorAmountsAverages[dlabel]
                avg_rates[packageLabel(AVG_LABEL, dlabel)] = self.priorRatesAverages[dlabel]
            self.prior_amounts = avg_amounts
            self.prior_rates = avg_rates
        #Takes a set of identifying IDs and outputs the set of examples, with extracted features, in form (amount examples, rate examples)
        # Where each set of examples consists of a list of tuples of input features, phi(x), and output, y. 
        def outputExamples(self, IDs):
            rate_examples = list()  
            amount_examples = list()  
              
            for uid in IDs:
                udata = (self.data[uid]) #took out list()
                for i in range(len(udata)):
                  x_amt = self.featureExtractorAmounts(udata[i])
                  x_rate = self.featureExtractorRates(udata[i])
                  y_amt = float(udata[i]['amount'])
                  y_rate = float(udata[i]['donated'])
                  rate_examples.append((x_rate, y_rate))
                  amount_examples.append((x_amt, y_amt))
            return (amount_examples, rate_examples)
                

        
        #Must be called after setPriors
        #Note: ignores last entry in userData (need to take care of case where only one entry)
        def getFeatureExtractor(self, priors):
          def featureExtractor(userDataP): 
          #investigating 71195  
            #'''
            featureV = Counter()
            entry = userDataP
            for label in data_labels:
                if(label not in external_data_labels):  value = entry[label]
                elif label == 'zip_income_percentile': value = self.income_level(self.income_at(entry['zip']))
                else: assert(False) #should never happen

                if(label == 'zip'): self.zipcodes.add(value)
                tag = packageLabel(label,str(value))

                if tag in priors: 
                  featureV[tag] = float(priors[tag]) 
                else:
                  featureV[tag] = float(priors[packageLabel(AVG_LABEL,unpackageLabel(tag)[0])])/2.0 #MAYBE THIS SHOULD BE WEIGHTED DOWN, SO NEVER ENCOUNTERED IS A BAD THING
            return featureV
          return featureExtractor

        def print_top_scores(self, scores, test_data):
            write_n = lines_to_read
            to_write = list()
            for ID, score in scores[0:write_n]:
              to_write.append([test_data[ID]['amount'], str(score)])
            
            
            if str("Regression_Type") not in locals(): Regression_Type = REGRESSION_TYPE
            print("\nAlgorithm: {alg} -- Regression Type: {reg} -- Loss Function: {LF}".format(alg=Algorithm, reg = Regression_Type, LF = LOSS_FN )) 
            print("--Final Stats--")
            top_avg =  sum([float(test_data[ID]['amount']) for ID, score in scores[0:int((0.75*len(scores)))]])/(0.75*len(scores))
            print ("Avg donated from top 75%:", top_avg)
            
            random_avg = 0
            sample_width = 100
            for i in range(sample_width):
              random_avg_sample = math.fsum([float(test_data[ID]['amount']) for ID, score in random.sample(scores, int(0.75*len(scores)))])/(0.75*len(scores))
              random_avg += random_avg_sample/sample_width
            print("Avg donated from random 75%:", random_avg)
            print("Success margin: {margin}%".format(margin = round((top_avg-random_avg)/random_avg*100, 2)))
            print("---------------")

        #Returns a sorted list of tuples consisting of (ID, score) for whatever chosen ID is. ID is not necessarily unique, though it probably should be.
        def score_and_rank(self, amounts_eval, rates_eval, test_data, evaluatorFn = lambda x, y, z: dotProduct(x, y)):
            user_scores = Counter()
            for ID, entry in test_data.items():
              amt_ft = self.featureExtractorAmounts(entry)
              amt_score = evaluatorFn(amounts_eval, amt_ft, 'amounts')
              rate_ft = self.featureExtractorRates(entry)
              rate_score = evaluatorFn(rates_eval, rate_ft, 'rates')  
              score = (amt_score**1.15) * rate_score
              user_scores[ID] += score # NOTE: ID is packaged as ID_Tag type and prospectid
            user_scores = sorted(user_scores.items(), key = operator.itemgetter(1), reverse=True)
            self.print_top_scores(user_scores, test_data)
            return user_scores



        #Writes the zipcodes to a csv. Each row has the zipcode and the income at that zipcode. Used for initial determination of
        #percentiles. The actual percentile calculation was just done in excel.
        def write_zipcodes(self):
          with open('zipcodesUsed.csv', 'w') as zfile:
                writer = csv.writer(zfile)
                for zipcode in list(self.zipcodes):
                    writer.writerow([str(zipcode),str(self.income_at(zipcode))])
            

        def predict_svm_val(self, svm, ft, title):
            x = self.svmVecs[title].transform([ft,])
            score = svm.predict(x)
            return score[0]


        def learnSVM(self, train_examples, test_examples, eps, c, title):
            
            X, Y = zip(*train_examples) 
            vec = DictVectorizer()
            X = vec.fit_transform(X)

            svm = SVR(epsilon = eps, C= c)
            svm.fit(X, Y)

            Xtest, Ytest = zip(*test_examples)
            Xtest = vec.transform(Xtest)

            print(title," score:", svm.score(Xtest, Ytest))
            self.svmVecs[title] = vec
            return svm

        
        #returns a basic leaderboard
        def solve(self):
            if self.data == None: quit();
            
            self.setPriors(self.trainingSetIDs)

            self.featureExtractorAmounts = self.getFeatureExtractor(self.prior_amounts)
            self.featureExtractorRates = self.getFeatureExtractor(self.prior_rates)
            
            train_amount_examples, train_rate_examples = self.outputExamples(self.trainingSetIDs) 
            test_amount_examples, test_rate_examples = self.outputExamples(self.testSetIDs)

            test_data = dict()
            for ID in self.testSetIDs: 
              entries = [self.data[ID][i] for i in range(len(self.data[ID]))]
              
              for entry in entries: 
                test_data[packageLabel(ID, entry['prospectid'])] = entry # ASSUMES YOU CANNOT HAVE multiple interactions with single user in the same mailing
              
            if(Algorithm == "Regression"):
              rates_weights = learnPredictor(train_rate_examples, test_rate_examples, self.featureExtractorRates, self.prior_rates, iterations = 2)
              amounts_weights = learnPredictor(train_amount_examples, test_amount_examples, self.featureExtractorAmounts, self.prior_amounts, weights = Counter(self.prior_amounts.keys()))
              scores = self.score_and_rank(amounts_weights, rates_weights, test_data)
            elif(Algorithm == "SVM"):    
              amounts_svm = self.learnSVM(train_amount_examples, test_amount_examples, eps = 0.1, c=1.0, title = 'amounts' )
              rates_svm = self.learnSVM(train_rate_examples, test_rate_examples, eps = 0.1, c = 1.0, title = 'rates')
              user_scores = self.score_and_rank(amounts_svm, rates_svm, test_data, self.predict_svm_val)
            else:
              print("Choose an appropriate algorithm")
            
            
            
            
  #A VERY loose approximation of how well we are doing. Ultimately, did not place much stake in this except while building. Thus it no longer prints anything
  def evaluatePredictor(examples, predictor):
      '''
      predictor: a function that takes an x and returns a predicted y.
      Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
      of mis  classiied examples.
      '''
      error_dif_list = list()
      error = 0
      total_above_zero = 0
      for x, y in examples:
          if(y > 0 ): 
               
               total_above_zero += 1.0
               if predictor(x) < 0.5: error+=1.0
          else:
              if (predictor(x) > 0.7): error+=1.0

          error_dif_list.append((predictor(x) - y)**2)
      return 1.0 * (1.0+error) / (len(examples))

  def normalize_weights(weights): #Not only normalizes, also reverses sign of
      mean = 1.0*sum(weights.values())/len(weights)
      d2 = list()
      for w in weights:
          d2.append((weights[w] - mean)**2)
      stdvar = math.sqrt(sum(d2) *1.0)
      newWeights = ({w: 1.0*((weights[w]-mean)+1.0)/(stdvar+1.0) for w in weights}) #added -1.0
      
      lower_bound = min(newWeights.values(), key = float)
      if(lower_bound > 0): lower_bound = 0
      for w in weights.keys():
          weights[w] = float(newWeights[w]) + abs(float(lower_bound))
      return stdvar



  def learnPredictor(trainExamples, testExamples, featureExtractor, priors, Regression_Type = REGRESSION_TYPE, weights = Counter(), iterations = 1):
      lambda_val = -0.5
      step_size = 0.05
      def loss_for_example(x, y, weights):
          features = Counter(x)
          if(Regression_Type == "Linear"):
            full_squared_loss = 1.0*(dotProduct(features, weights) - y) 
            if(LOSS_FN == "Least Squares"):
              loss_mult = 1.0*(dotProduct(features, weights) - y) #LEAST SQUARES LOSS
            elif(LOSS_FN == "Abs Dev"):
              if(dotProduct(features, weights) > y): #ABSOLUTE DEVIATION LOSS
                loss_mult = 1.0
              else:
                loss_mult = -1.0 
            else:
              print("Choose a proper loss function, not ", LOSS_FN)
              quit()  
            loss_gradient = {feature: features[feature] * loss_mult for feature in features}

          elif(Regression_Type == "Log"):
            loss_gradient = Counter()
            exp = y * dotProduct(features, weights)
            if(exp > 5): exp = 5 # ARBITRARY CAP TO PREVENT OVERFLOW
            try:
              mult =  2.0*y/(1.0+math.e**(exp))
              increment(loss_gradient,mult, features) #numerator
            except:
              print("Overflow on loss calculation")
              
          else:
            print("Requires an implemented regression type, not ", Regression_Type)
            quit()
          return loss_gradient
          
      def printEvaluations(weights):
          train_stat = evaluatePredictor(trainExamples, regress)
          test_stat = evaluatePredictor(testExamples, regress)
          testExamples.sort(lambda a, b: int((1.0*regress(a[0]) > regress(b[0]))), reverse = False)
          #print("Training error: {error}".format(error=train_stat))
          #print("Testing error: {error}".format(error=test_stat))
          n_top_samples = int(len(testExamples)*0.75)
          assert(n_top_samples != len(testExamples))
          print ("In-Process Training Data: \nAvg donated from top 75%:", sum([ex[1] for ex in testExamples[0:int(n_top_samples)]])/float(n_top_samples))
          return train_stat        

      def regress(x):
          return float((dotProduct(x, weights)))
      for itera in range(iterations):
          for x, y in trainExamples:
              loss_gradient = loss_for_example(x, y, weights)
              increment(weights, -step_size/(itera+1.0), loss_gradient)
          increment(weights, lambda_val/(itera+1.0), weights) #An attempt to generalize, NB lambda might be set to zero
          train_stat = printEvaluations(weights)
          if(train_stat == 0): break; #no point in going further, won't usually learn anything more
          stdvar = normalize_weights(weights)
      return weights
    

  Predictor = Predictor()
  Predictor.readData(DATASET_PATH)
  Predictor.readZipData(ZIPCODES_PATH)
  Predictor.solve() 

if __name__ == '__main__':
  mainMethod()
