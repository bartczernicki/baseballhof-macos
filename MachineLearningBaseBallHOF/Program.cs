using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MachineLearningBaseBallHOF;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

namespace MachineLearningBaseBallHOF
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // Training & Validation/Dev text CSV files
            var trainingDataPath = "HOFTraining.txt";
            var validationDataPath = "HOFValidation.txt";

            Utils.PrintConsoleMessage("Starting Baseball HOF Prediction", true);

            // 1) Create a new learning pipeline
            var pipeline = new LearningPipeline();

            // 2) Add a Text Loader
            pipeline.Add(new TextLoader<BaseballData>(trainingDataPath, separator: ",", allowQuotedStrings: false, trimWhitespace: true));

            // 3) Create Features
            pipeline.Add(new ColumnConcatenator("Features", "YearsPlayed",
            "AB", "R", "H", "Doubles", "Triples", "HR", "RBI", "SB",
            "AllStarAppearances", "MVPs", "TripleCrowns", "GoldGloves", "MajorLeaguePlayerOfTheYearAwards"));

            // pipeline.Add(new ColumnConcatenator("Features", "YearsPlayed"));

            // 4) Create new binary classifier (predict yes/no into Baseball HOF)
            var classifier = new FastTreeBinaryClassifier()
            {
                NumLeaves = 10,
                NumTrees = 60,
                MinDocumentsInLeafs = 2,
                BaggingSize = 5,
                AllowEmptyTrees = true,
                Caching = Microsoft.ML.Models.CachingOptions.Memory,
                OptimizationAlgorithm = BoostedTreeArgsOptimizationAlgorithmType.GradientDescent
            };
            // var classifier = new BinaryLogisticRegressor();
            // var classifier = new FastForestBinaryClassifier();
            //var classifier = new GeneralizedAdditiveModelBinaryClassifier();

            // Add the classifier to the pipeline
            pipeline.Add(classifier);

            // 5) Train Model
            var model = pipeline.Train<BaseballData, BaseballDataPrediction>();

            // 6) Sample Predictions

            // Bad Player with poor historical numbers
            var samplePredictionBadPlayer = new BaseballData
            {
                AB = 1000,
                AllStarAppearances = 0,
                FullPlayerName = "Bad Player",
                R = 30,
                H = 100,
                Doubles = 10,
                Triples = 10,
                HR = 10,
                RBI = 20,
                SB = 5,
                PlayerID = 10101,
                Label = false,
                MVPs = 0,
                GoldGloves = 0,
                MajorLeaguePlayerOfTheYearAwards = 0,
                TripleCrowns = 0,
                YearsPlayed = 2
            };
            var result = model.Predict(samplePredictionBadPlayer);

            Console.WriteLine("Bad Baseball Player Prediction");
            Console.WriteLine("******************************");
            Console.WriteLine("HOF Prediction: " + result.PredictedLabel.ToString() + " | " + "Probability: " + result.ProbabilityLabel);
            Console.WriteLine();


            // Great Player with great historical numbers worthy of HOF
            var samplePredictionGreatPlayer = new BaseballData
            {
                AB = 10000,
                AllStarAppearances = 12,
                FullPlayerName = "Great Player",
                R = 1100,
                H = 3200,
                Doubles = 450,
                Triples = 150,
                HR = 600,
                RBI = 1200,
                SB = 400,
                PlayerID = 20202,
                Label = true,
                MVPs = 3,
                GoldGloves = 8,
                MajorLeaguePlayerOfTheYearAwards = 4,
                TripleCrowns = 2,
                //SB = 200,
                YearsPlayed = 22
            };
            var greatPlayerPrediction = model.Predict(samplePredictionGreatPlayer);

            Console.WriteLine("Great Baseball Player Prediction");
            Console.WriteLine("******************************");
            Console.WriteLine("HOF Prediction: " + greatPlayerPrediction.PredictedLabel.ToString() + " | " + "Probability: " + greatPlayerPrediction.ProbabilityLabel);
            Console.WriteLine();
            Console.WriteLine();


            // 7) Load Evaluation Data
            var testData = new TextLoader<BaseballData>(validationDataPath, useHeader: false, separator: ",");

            // 8) Evaluate trained model with test data
            var evaluator = new BinaryClassificationEvaluator() { ProbabilityColumn = "Probability" };
            var metrics = evaluator.Evaluate(model, testData);

            // build a list of False Positives - Players not in the HOF, predicted by classifier to be in HOF
            var falsePostivePlayers = new List<Tuple<BaseballData, BaseballDataPrediction>>();
            // build a list of False Negatives - Players IN THE HOF, predicted by classifier not to be in HOF
            var falseNegativePlayers = new List<Tuple<BaseballData, BaseballDataPrediction>>();
            // true Pos
            var truePositivePlayers = new List<Tuple<BaseballData, BaseballDataPrediction>>();

            using (var environment = new TlcEnvironment())
            {
                var customSchema = "col=Label:BL:0 col=FullPlayerName:TX:1 col=YearsPlayed:R4:2 col=AB:R4:3 col=R:R4:4 col=H:R4:5 col=Doubles:R4:6 col=Triples:R4:7 col=HR:R4:8 col=RBI:R4:9 col=SB:R4:10 col=AllStarAppearances:R4:11 col=MVPs:R4:12 col=TripleCrowns:R4:13 col=GoldGloves:R4:14 col=MajorLeaguePlayerOfTheYearAwards:R4:15 col=PlayerID:R4:16 Separator=,";
                var inputFile = new SimpleFileHandle(environment, validationDataPath, false, false);
                var dataView = ImportTextData.ImportText(environment, new ImportTextData.Input { InputFile = inputFile, CustomSchema = customSchema }).Data;

                using (var cursor = dataView.GetRowCursor(col => true))
                {
                    cursor.Schema.TryGetColumnIndex("Label", out int labelCol);
                    cursor.Schema.TryGetColumnIndex("FullPlayerName", out int fullPlayerNameCol);
                    cursor.Schema.TryGetColumnIndex("YearsPlayed", out int yearsPlayedCol);
                    cursor.Schema.TryGetColumnIndex("AB", out int abCol);
                    cursor.Schema.TryGetColumnIndex("R", out int rCol);
                    cursor.Schema.TryGetColumnIndex("H", out int hCol);
                    cursor.Schema.TryGetColumnIndex("Doubles", out int doublesCol);
                    cursor.Schema.TryGetColumnIndex("Triples", out int triplesCol);
                    cursor.Schema.TryGetColumnIndex("HR", out int hrCol);
                    cursor.Schema.TryGetColumnIndex("RBI", out int rbiCol);
                    cursor.Schema.TryGetColumnIndex("SB", out int sbCol);
                    cursor.Schema.TryGetColumnIndex("AllStarAppearances", out int allStarAppearancesCol);
                    cursor.Schema.TryGetColumnIndex("MVPs", out int mvpsCol);
                    cursor.Schema.TryGetColumnIndex("TripleCrowns", out int tripleCrownsCol);
                    cursor.Schema.TryGetColumnIndex("GoldGloves", out int goldGlovesCol);
                    cursor.Schema.TryGetColumnIndex("MajorLeaguePlayerOfTheYearAwards", out int majorLeaguePlayerOfTheYearAwardsCol);
                    cursor.Schema.TryGetColumnIndex("PlayerID", out int idCol);

                    while (cursor.MoveNext())
                    {
                        // Label
                        var labelGetter = cursor.GetGetter<DvBool>(labelCol);
                        var label = default(DvBool);
                        labelGetter(ref label);
                        // Full Player Name
                        var fullPlayerNameGetter = cursor.GetGetter<DvText>(fullPlayerNameCol);
                        var fullPlayerName = default(DvText);
                        fullPlayerNameGetter(ref fullPlayerName);
                        // Years Played
                        var yearsPlayedGetter = cursor.GetGetter<float>(yearsPlayedCol);
                        var yearsPlayed = 0f;
                        yearsPlayedGetter(ref yearsPlayed);
                        // AB
                        var abGetter = cursor.GetGetter<float>(abCol);
                        var ab = 0f;
                        abGetter(ref ab);
                        // R
                        var rGetter = cursor.GetGetter<float>(rCol);
                        var r = 0f;
                        rGetter(ref r);
                        // H
                        var hGetter = cursor.GetGetter<float>(hCol);
                        var h = 0f;
                        hGetter(ref h);
                        // Doubles
                        var doublesGetter = cursor.GetGetter<float>(doublesCol);
                        float doubles = 0f;
                        doublesGetter(ref doubles);
                        // Triples
                        var triplesGetter = cursor.GetGetter<float>(triplesCol);
                        float triples = 0f;
                        triplesGetter(ref triples);
                        // HR
                        var hrGetter = cursor.GetGetter<float>(hrCol);
                        float hr = 0f;
                        hrGetter(ref hr);
                        // RBI
                        var rbiGetter = cursor.GetGetter<float>(rbiCol);
                        float rbi = 0f;
                        rbiGetter(ref rbi);
                        // SB
                        var sbGetter = cursor.GetGetter<float>(sbCol);
                        float sb = 0f;
                        sbGetter(ref sb);
                        // AllStarAppearances
                        var allStarAppearancesGetter = cursor.GetGetter<float>(allStarAppearancesCol);
                        float allStarAppearances = 0f;
                        allStarAppearancesGetter(ref allStarAppearances);
                        // MVPs
                        var mvpsGetter = cursor.GetGetter<float>(mvpsCol);
                        float mvps = 0f;
                        mvpsGetter(ref mvps);
                        // Tiple Crowns
                        var tripleCrownsGetter = cursor.GetGetter<float>(tripleCrownsCol);
                        float tripleCrowns = 0f;
                        tripleCrownsGetter(ref tripleCrowns);
                        // Gold Gloves
                        var goldGlovesGetter = cursor.GetGetter<float>(goldGlovesCol);
                        float goldGloves = 0f;
                        goldGlovesGetter(ref goldGloves);
                        // MajorLeaguePlayerOfTheYearAwards
                        var majorLeaguePlayerOfTheYearAwardsGetter = cursor.GetGetter<float>(majorLeaguePlayerOfTheYearAwardsCol);
                        float majorLeaguePlayerOfTheYearAwards = 0f;
                        majorLeaguePlayerOfTheYearAwardsGetter(ref majorLeaguePlayerOfTheYearAwards);
                        // PlayerID column
                        var idGetter = cursor.GetGetter<float>(idCol);
                        float id = 0f;
                        idGetter(ref id);

                        var baseBallData = new BaseballData()
                        {
                            AB = ab,
                            AllStarAppearances = allStarAppearances,
                            Doubles = doubles,
                            FullPlayerName = fullPlayerName.ToString(),
                            H = h,
                            HR = hr,
                            GoldGloves = goldGloves,
                            PlayerID = id,
                            Label = label.IsTrue ? true : false,
                            MajorLeaguePlayerOfTheYearAwards = majorLeaguePlayerOfTheYearAwards,
                            MVPs = mvps,
                            R = r,
                            RBI = rbi,
                            SB = sb,
                            TripleCrowns = tripleCrowns,
                            Triples = triples,
                            YearsPlayed = yearsPlayed
                        };

                        var prediction = model.Predict(baseBallData);

                        // True Positives
                        if ((prediction.PredictedLabel == true) && (baseBallData.Label == true))
                        {
                            truePositivePlayers.Add(new Tuple<BaseballData, BaseballDataPrediction>(baseBallData, prediction));
                        }
                        // False Positive Prediction
                        if ((prediction.PredictedLabel == true) && (baseBallData.Label == false))
                        {
                            falsePostivePlayers.Add(new Tuple<BaseballData, BaseballDataPrediction>(baseBallData, prediction));
                        }
                        else
                        // False Negative Prediction
                        if ((prediction.PredictedLabel == false) && (baseBallData.Label == true))
                        {
                            falseNegativePlayers.Add(new Tuple<BaseballData, BaseballDataPrediction>(baseBallData, prediction));
                        }
                    }
                }
            }

            // 9) Print out Metrics (rounded to 4 decimals)
            Console.WriteLine("******************");
            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("******************");
            Console.WriteLine("AUC Score:  " + Math.Round(metrics.Auc, 4).ToString());
            Console.WriteLine("Precision:  " + Math.Round(metrics.PositivePrecision, 4).ToString());
            Console.WriteLine("Recall:     " + Math.Round(metrics.PositiveRecall, 4).ToString());
            Console.WriteLine("Accuracy:   " + Math.Round(metrics.Accuracy, 4).ToString());
            Console.WriteLine("******************");

            Console.WriteLine();
            Console.WriteLine("******************");
            Console.WriteLine("True Positives");
            Console.WriteLine("******************");

            for (int i = 0; i != truePositivePlayers.Count; i++)
            {
                var player = truePositivePlayers[i].Item1;
                var playerPrediction = truePositivePlayers[i].Item2;
                Console.WriteLine(player.ToString() + "Prob: " + playerPrediction.ProbabilityLabel.ToString());
            }

            Console.WriteLine();
            Console.WriteLine("******************");
            Console.WriteLine("False Positives");
            Console.WriteLine("******************");

            for (int i = 0; i != falsePostivePlayers.Count; i++)
            {
                var player = falsePostivePlayers[i].Item1;
                var playerPrediction = falsePostivePlayers[i].Item2;
                Console.WriteLine(player.ToString() + "Prob: " + playerPrediction.ProbabilityLabel.ToString());
            }

            Console.WriteLine();
            Console.WriteLine("******************");
            Console.WriteLine("False Negatives");
            Console.WriteLine("******************");

            for (int i = 0; i != falseNegativePlayers.Count; i++)
            {
                var player = falseNegativePlayers[i].Item1;
                var playerPrediction = falseNegativePlayers[i].Item2;
                Console.WriteLine(player.ToString() + "Prob: " + playerPrediction.ProbabilityLabel.ToString());
            }

            //10) Persist trained model
            model.WriteAsync("baseballhof-model.mlnet").Wait();
        }
    }
}
