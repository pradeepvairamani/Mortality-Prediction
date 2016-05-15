/**
* @author Pradeep Vairamani <pradeeprv@gatech.edu>.
*/
package edu.gatech.cse8803.main

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.SVMWithSGD
import java.text.SimpleDateFormat
import scala.collection.mutable
import edu.gatech.cse8803.ioutils.CSVUtils
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans}
import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vectors, Vector}
import org.apache.spark.mllib.classification.SVMWithSGD

// import org.apache.spark.ml.feature.VectorAssembler
import scala.io.Source
object Main {
    def main(args: Array[String]) {
        import org.apache.log4j.Logger
        import org.apache.log4j.Level
        
        Logger.getLogger("org").setLevel(Level.WARN)
        Logger.getLogger("akka").setLevel(Level.WARN)
        
        val sc = createContext
        val sqlContext = new SQLContext(sc)
        
        val auroc = baselineonly(sqlContext, sc)
        val lda_models = getLdaDistribution(sqlContext, sc)
        val auroc_combined = joinAllFeatures(sqlContext, sc, lda_models)        
        sc.stop
    }
    
    def createContext(appName: String, masterUrl: String): SparkContext = {
        val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
        new SparkContext(conf)
    }
    
    def createContext: SparkContext = createContext("CSE 8803 Project", "local")

    def getLdaDistribution(sqlContext:SQLContext, sc:SparkContext): RDD[(Long, Vector)] = {
        val corpus = CSVUtils.loadCSVAsTable(sqlContext, "data/nostopwords.csv", "lda12")
        val docs = corpus.map(_(2).toString)
        val hadmids = corpus.map(_(0).toString.toLong)
        // println(docs)
        // Split each document into a sequence of terms (words)
        val tokenized: RDD[Seq[String]] =
          docs.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 2).filter(_.forall(java.lang.Character.isLetter)))
        // Choose the vocabulary.
        //   termCounts: Sorted list of (term, termCount) pairs
        val termCounts: Array[(String, Long)] =
          tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
        //   vocabArray: Chosen vocab (removing common terms)
        // val numStopwords = 20
        // val vocabArray: Array[String] =
        //   termCounts.takeRight(termCounts.size - numStopwords).map(_._1)
          val vocabArray: Array[String] =
          termCounts.map(_._1)
        //   vocab: Map term -> term index
        val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap
        // println(vocab.size)
        val documents: RDD[(Long, Vector)] =
            tokenized.zip(hadmids).map { case (tokens, id) =>
            val counts = new mutable.HashMap[Int, Double]()
            tokens.foreach { term =>
              if (vocab.contains(term)) {
                val idx = vocab(term)
                counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
              }
            }
            (id, Vectors.sparse(vocab.size, counts.toSeq))
          }
        // Set LDA parameters
        val numTopics = 50
        val lda = new LDA().setK(numTopics).setMaxIterations(100)
        val ldaModel = lda.run(documents)
        val avgLogLikelihood = ldaModel.logLikelihood / documents.count()
        // Print topics, showing top-weighted 10 terms for each topic.
        val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
        topicIndices.foreach { case (terms, termWeights) =>
          println("--------- TOPIC -----------")
          terms.zip(termWeights).foreach { case (term, weight) =>
            println(s"${vocabArray(term.toInt)}\t$weight")
          }
      println()
    }
     ldaModel.topicDistributions
    }
    
    def getBaselineData(sqlContext:SQLContext, sc:SparkContext): Double = {
        val data_csv = CSVUtils.loadCSVAsTable(sqlContext, "data/baselinefeatures.csv", "baseline")
        val num_data = data_csv.map(line => if(line(2) == "M")  LabeledPoint(line(4).toString.toDouble, Vectors.dense(line(1).toString.toDouble, 0.0, line(3).toString.toDouble)) else LabeledPoint(line(4).toString.toDouble,Vectors.dense(line(1).toString.toDouble, 1.0, line(3).toString.toDouble)))
        // num_data.take(5).foreach(println)
        MLUtils.saveAsLibSVMFile(num_data, "SVMfile")
        val data = MLUtils.loadLibSVMFile(sc, "SVMfile")
        
        val splits = data.randomSplit(Array(0.7, 0.3), seed = 15L)
        val train = splits(0).cache()
        val test = splits(1).cache()
        val numIterations = 10
        val model = SVMWithSGD.train(train, numIterations)
        
        // import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
        //     val model = new LogisticRegressionWithLBFGS()
        //   .setNumClasses(10)
        //   .run(train)
        
        model.clearThreshold()
        
        // Compute raw scores on the test set.
        val scoreAndLabels = test.map { point =>
            val score = model.predict(point.features)
            (score, point.label)
        }
        
        // Get evaluation metrics.
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        val auROC = metrics.areaUnderROC()
        auROC
    }
    def joinAllFeatures(sqlContext:SQLContext, sc:SparkContext, topicdist:RDD[(Long, Vector)]){
        println("came to combine")
       val data_csv = CSVUtils.loadCSVAsTable(sqlContext, "data/allbaselinefeatures.csv", "baseline")
        val baselinekeyvalue = data_csv.map(line => (line(1).toString.toLong, line))
        // baselinekeyvalue.take(5).foreach(println)
        val topicdistmap = topicdist.collect().toMap
        val topicdistkeys = topicdistmap.keys
        // topicdistkeys.take(5).foreach(println)
        val joineddata = topicdist.join(baselinekeyvalue)
        val combinedFeatures = joineddata.map(line => line._2)
        // combinedFeatures.take(5).foreach(println)
        val combinedFeatures1 = combinedFeatures.map{
            case(y,x) => LabeledPoint(x(7).toString.toDouble, Vectors.dense(Array(x(2).toString.toDouble, x(3).toString.toDouble, x(4).toString.toDouble, x(6).toString.toDouble)
                ++ y.toArray))
        }

        MLUtils.saveAsLibSVMFile(combinedFeatures1, "SVMcombined")
        val data = MLUtils.loadLibSVMFile(sc, "SVMcombined")
        val splits = data.randomSplit(Array(0.7, 0.3), seed = 15L)
        val train = splits(0).cache()
        val test = splits(1).cache()
        import org.apache.spark.mllib.classification.SVMWithSGD
        var numIterations = 0
        var bestauROC = -1000.00
        for(numIterations <- 50 to 100){
          val model = SVMWithSGD.train(train, numIterations)
          /*val scModel = sc.broadcast(model)
          val predictionAndLabel = test.map(x => (scModel.value.predict(x.features), x.label))
          val accuracy = predictionAndLabel.filter(x => x._1 == x._2).count / test.count.toFloat
          println("testing Accuracy  = " + accuracy)*/
          // Clear the default threshold.
          model.clearThreshold()
          // Compute raw scores on the test set.
          val scoreAndLabels = test.map { point =>
            val score = model.predict(point.features)
            (score, point.label)
          }
        // Get evaluation metrics.
          val metrics = new BinaryClassificationMetrics(scoreAndLabels)
          val auROC = metrics.areaUnderROC()
          if(auROC > bestauROC){
              bestauROC = auROC      
          }
        
        // println(numIterations)          
     }
     println("Best Combined ROC " + bestauROC)
        (bestauROC)
    }

    def baselineonly(sqlContext:SQLContext, sc:SparkContext){
        println("came to baseline only")
       val data_csv = CSVUtils.loadCSVAsTable(sqlContext, "data/allbaselinefeatures.csv", "baseline")
       val num_data = data_csv.map(line => LabeledPoint(line(7).toString.toDouble, Vectors.dense(line(2).toString.toDouble, line(3).toString.toDouble, line(4).toString.toDouble, line(6).toString.toDouble)))
        // num_data.take(5).foreach(println)
        MLUtils.saveAsLibSVMFile(num_data, "SVMfile")
        val data = MLUtils.loadLibSVMFile(sc, "SVMfile")
        
        val splits = data.randomSplit(Array(0.7, 0.3), seed = 15L)
        val train = splits(0).cache()
        val test = splits(1).cache()
        var numIterations = 0
        var bestauROC = -1000.00
        for(numIterations <- 10 to 30){
          val model = SVMWithSGD.train(train, numIterations)
          /*val scModel = sc.broadcast(model)
          val predictionAndLabel = test.map(x => (scModel.value.predict(x.features), x.label))
          val accuracy = predictionAndLabel.filter(x => x._1 == x._2).count / test.count.toFloat
          println("testing Accuracy  = " + accuracy)*/
          // Clear the default threshold.
          model.clearThreshold()
          // Compute raw scores on the test set.
          val scoreAndLabels = test.map { point =>
            val score = model.predict(point.features)
            (score, point.label)
          }
        // Get evaluation metrics.
          val metrics = new BinaryClassificationMetrics(scoreAndLabels)
          val auROC = metrics.areaUnderROC()
          if(auROC > bestauROC){
              bestauROC = auROC      
          }
        
        // println(numIterations)          
     }
     println("Best Baseline ROC " + bestauROC)
        (bestauROC)
    }
    
}