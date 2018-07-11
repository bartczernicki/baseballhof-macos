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
using Microsoft.ML.Data;

namespace MachineLearningBaseBallHOF
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Utils.PrintConsoleMessage("Starting Baseball HOF Prediction", true);

            // MCC Evaluation metric
            double mcc = 0.0;
            double mccNumerator = 0.0, mccDenominator = 0.0;

            // Training & Validation/Dev text CSV files
            var trainingDataPath = "HOFTraining.txt";
            var validationDataPath = "HOFValidation.txt";

            // 1) Create a new learning pipeline
            var pipeline = new LearningPipeline();

            // 2) Add a Text Loader
            var trainingLoader = new Microsoft.ML.Data.TextLoader(trainingDataPath).CreateFrom<BaseballData>(allowQuotedStrings: false, separator: ',');
            pipeline.Add(trainingLoader);

            // 3) Create Features
            pipeline.Add(new ColumnConcatenator("Features", 
                "YearsPlayed", "AB", 
                "R", "H", "Doubles", "Triples", "HR", "RBI", "SB",
                "AllStarAppearances", "MVPs", "TripleCrowns", "GoldGloves", "MajorLeaguePlayerOfTheYearAwards", "TB"));
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
            // var classifier = new LinearSvmBinaryClassifier();
            // var classifier = new FastForestBinaryClassifier();
            // var classifier = new GeneralizedAdditiveModelBinaryClassifier();
            // var classifier = new LightGbmBinaryClassifier { LearningRate = 0.02, MinDataPerLeaf = 4, VerboseEval = true };

            // Add the classifier to the pipeline
            pipeline.Add(classifier);

            // 5) Train Model
            var model = pipeline.Train<BaseballData, BaseballDataPrediction>();

            // 6) Sample Predictions

            // Bad Player with poor historical numbers
            var samplePredictionBadPlayer = new BaseballData
            {
                AB = 3000,
                AllStarAppearances = 0,
                FullPlayerName = "Bad Player",
                R = 90,
                H = 300,
                Doubles = 30,
                Triples = 30,
                HR = 30,
                RBI = 60,
                SB = 15,
                BattingAverage = 0.1f,
                SluggingPct = 0.25f,
                PlayerID = 10101,
                Label = false,
                MVPs = 0,
                GoldGloves = 0,
                MajorLeaguePlayerOfTheYearAwards = 0,
                TripleCrowns = 0,
                YearsPlayed = 3,
                TB = 570
            };
            var result = model.Predict(samplePredictionBadPlayer);

            Console.WriteLine("Bad Baseball Player Prediction");
            Console.WriteLine("******************************");
            Console.WriteLine("HOF Prediction: " + result.PredictedLabel.ToString() + " | " + "Probability: " + Math.Round(result.ProbabilityLabel, 7));
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
                BattingAverage = 0.32f,
                SluggingPct = 0.55f,
                PlayerID = 20202,
                Label = true,
                MVPs = 3,
                GoldGloves = 8,
                MajorLeaguePlayerOfTheYearAwards = 4,
                TripleCrowns = 2,
                YearsPlayed = 22,
                TB = 6700
            };
            var greatPlayerPrediction = model.Predict(samplePredictionGreatPlayer);

            Console.WriteLine("Great Baseball Player Prediction");
            Console.WriteLine("******************************");
            Console.WriteLine("HOF Prediction: " + greatPlayerPrediction.PredictedLabel.ToString() + " | " + "Probability: " + greatPlayerPrediction.ProbabilityLabel);
            Console.WriteLine();
            Console.WriteLine();


            // 7) Load Evaluation Data
            var testData = new Microsoft.ML.Data.TextLoader(validationDataPath).CreateFrom<BaseballData>(allowQuotedStrings: false, separator: ',');

            // 8) Evaluate trained model with test data
            var evaluator = new BinaryClassificationEvaluator() { ProbabilityColumn = "Probability" };
            var metrics = evaluator.Evaluate(model, testData);

            // build a list of False Positives - Players not in the HOF, predicted by classifier to be in HOF
            var falsePostivePlayers = new List<Tuple<BaseballData, BaseballDataPrediction>>();
            // build a list of False Negatives - Players IN THE HOF, predicted by classifier not to be in HOF
            var falseNegativePlayers = new List<Tuple<BaseballData, BaseballDataPrediction>>();
            // build a list of True Positives - Players IN THE HOF, predicted by classifier to be in HOF
            var truePositivePlayers = new List<Tuple<BaseballData, BaseballDataPrediction>>();
            // build a list of True Negataives - Players not in the HOF, predicted by classifier not to be in HOF
            var trueNegativePlayers = new List<Tuple<BaseballData, BaseballDataPrediction>>();


            using (var environment = new TlcEnvironment())
            {
                // note: custom schema not needed anymore
                // var customSchema = "col=Label:BL:0 col=FullPlayerName:TX:1 col=YearsPlayed:R4:2 col=AB:R4:3 col=R:R4:4 col=H:R4:5 col=Doubles:R4:6 col=Triples:R4:7 col=HR:R4:8 col=RBI:R4:9 col=SB:R4:10 col=BattingAverage:R4:11 col=SluggingPct:R4:12 col=AllStarAppearances:R4:13 col=MVPs:R4:14 col=TripleCrowns:R4:15 col=GoldGloves:R4:16 col=MajorLeaguePlayerOfTheYearAwards:R4:17 col=TB:R4:18 col=LastYearPlayed:R4:19 col=PlayerID:R4:20 Separator=,";

                var loader = new Microsoft.ML.Data.TextLoader(validationDataPath).CreateFrom<BaseballData>(allowQuotedStrings: false, separator: ',');

                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = loader.ApplyStep(null, experiment) as ILearningPipelineDataStep;

                experiment.Compile();
                loader.SetInput(environment, experiment);
                experiment.Run();

                IDataView data = experiment.GetOutput(output.Data);

                using (var cursor = data.GetRowCursor(col => true))
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
                    cursor.Schema.TryGetColumnIndex("TB", out int tbCol);
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
                        // TB
                        var tbGetter = cursor.GetGetter<float>(tbCol);
                        float tb = 0f;
                        tbGetter(ref tb);
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
                            TB = tb,
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

                        // True Negatives
                        if ((prediction.PredictedLabel == false) && (baseBallData.Label == false))
                        {
                            trueNegativePlayers.Add(new Tuple<BaseballData, BaseballDataPrediction>(baseBallData, prediction));
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
            mccNumerator = truePositivePlayers.Count * trueNegativePlayers.Count - falsePostivePlayers.Count * falseNegativePlayers.Count;
            mccDenominator = Math.Sqrt(
                1.0 * (truePositivePlayers.Count + falsePostivePlayers.Count) * (truePositivePlayers.Count + falseNegativePlayers.Count) * (trueNegativePlayers.Count + falsePostivePlayers.Count) * (trueNegativePlayers.Count + falseNegativePlayers.Count)
                                 );
            mcc = mccNumerator / mccDenominator;
            //Console.WriteLine(mccNumerator);
            //Console.WriteLine(mccDenominator);


            Console.WriteLine("******************");
            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("******************");
            Console.WriteLine("AUC Score:  " + Math.Round(metrics.Auc, 4).ToString());
            Console.WriteLine("Precision:  " + Math.Round(metrics.PositivePrecision, 4).ToString());
            Console.WriteLine("Recall:     " + Math.Round(metrics.PositiveRecall, 4).ToString());
            Console.WriteLine("Accuracy:   " + Math.Round(metrics.Accuracy, 4).ToString());
            Console.WriteLine("MCC:        " + Math.Round(mcc, 4).ToString());
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

            //11) Convert to ONNX & persist
            // will only work for FastTree, LightGBM, Logistic Regression
            var onnxPath = "baseballhof-model.onnx";
            var onnxAsJsonPath = "baseballhof-model.json";

            OnnxConverter converter = new OnnxConverter()
            {
                InputsToDrop = new[] { "Label" },
                OutputsToDrop = new[] { "Label", "Features" },
                Onnx = onnxPath,
                Json = onnxAsJsonPath,
                Domain = "com.baseballsample"
            };

            converter.Convert(model);
        }
    }
}
