using System;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningBaseBallHOF
{
    class Program
    {
		static void Main(string[] args)
		{
			Console.WriteLine("Starting Baseball HOF Training");
            
			var pipeline = new LearningPipeline();

			var trainingDataPath = "HOFTraining.txt";
			var validationDataPath = "HOFValidation.txt";

			pipeline.Add(new TextLoader<BaseballData>(trainingDataPath, separator: ","));

			pipeline.Add(new ColumnConcatenator("Features", "YearsPlayed", "AB", "H", "AllStarAppearances", "MVPs"));

			//var classifier = new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 };
			//var classifier = new BinaryLogisticRegressor();
			var classifier = new FastForestBinaryClassifier();

			pipeline.Add(classifier);

			var model = pipeline.Train<BaseballData, BaseballDataPrediction>();
            

			var testData = new TextLoader<BaseballData>(validationDataPath, useHeader: false, separator: ",");

			var samplePrediction = new BaseballData
			{
				AB = 100,
				AllStarAppearances = 3,
				FullPlayerName = "dfd",
				H = 400,
				ID = 3343,
				Label = false,
				MVPs = 1,
				YearsPlayed = 10
			};

			var result = model.Predict(samplePrediction);

			var evaluator = new BinaryClassificationEvaluator() { ProbabilityColumn = "Probability" };
			var metrics = evaluator.Evaluate(model, testData);

			Console.WriteLine("*****************");
			Console.WriteLine("AUC Score: " + metrics.Auc.ToString());
			Console.WriteLine("Precision: " + metrics.PositivePrecision.ToString());
			Console.WriteLine("Recall:    " + metrics.PositiveRecall.ToString());
			Console.WriteLine("Accuracy:  " + metrics.Accuracy.ToString());
			Console.WriteLine("*****************");

			Console.ReadLine();
		}
    }
}
