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
            var trainingDataPath = "HOFTraining.txt";
            var validationDataPath = "HOFValidation.txt";

            Console.WriteLine("Starting Baseball HOF Training");

            // 1) Create a new learning pipeline
            var pipeline = new LearningPipeline();

            // 2) Add a Text Loader
            pipeline.Add(new TextLoader<BaseballData>(trainingDataPath, separator: ","));

            // 3) Create Features
            pipeline.Add(new ColumnConcatenator("Features", "YearsPlayed", "AB", "H", "AllStarAppearances", "MVPs"));

            // 4) Create new classifier
            //var classifier = new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 };
            //var classifier = new BinaryLogisticRegressor();
            var classifier = new FastForestBinaryClassifier();

            pipeline.Add(classifier);

            // 5) Train Model
            var model = pipeline.Train<BaseballData, BaseballDataPrediction>();

            // 6) Sample Prediction
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

            // 7) Load Evaluation Data
            var testData = new TextLoader<BaseballData>(validationDataPath, useHeader: false, separator: ",");

            // 8) Evaluate trained model with test data
            var evaluator = new BinaryClassificationEvaluator() { ProbabilityColumn = "Probability" };
            var metrics = evaluator.Evaluate(model, testData);

            // 9) Print out Metrics (rounded to 4 decimals)
            Console.WriteLine("******************");
            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("******************");
            Console.WriteLine("AUC Score:  " + Math.Round(metrics.Auc, 4).ToString());
            Console.WriteLine("Precision:  " + Math.Round(metrics.PositivePrecision, 4).ToString());
            Console.WriteLine("Recall:     " + Math.Round(metrics.PositiveRecall, 4).ToString());
            Console.WriteLine("Accuracy:   " + Math.Round(metrics.Accuracy, 4).ToString());
            Console.WriteLine("******************");

            //10) Persist trained model
            model.WriteAsync("baseballhof-model.mlnet").Wait();
        }
    }
}
