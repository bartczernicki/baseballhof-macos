using System;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;

namespace MachineLearningBaseBallHOF
{
    public class BaseballData
    {
        [Column(ordinal: "0", name: "Label")]
        public bool Label;

        [Column(ordinal: "1")]
        public string FullPlayerName;

        [Column(ordinal: "2")]
        public float YearsPlayed;

        [Column(ordinal: "3")]
        public float AB;

        [Column(ordinal: "4")]
        public float R;

        [Column(ordinal: "5")]
        public float H;

        [Column(ordinal: "6")]
        public float Doubles;

        [Column(ordinal: "7")]
        public float Triples;

        [Column(ordinal: "8")]
        public float HR;

        [Column(ordinal: "9")]
        public float RBI;

        [Column(ordinal: "10")]
        public float SB;

        [Column(ordinal: "11")]
        public float BattingAverage;

        [Column(ordinal: "12")]
        public float SluggingPct;

        [Column(ordinal: "13")]
        public float AllStarAppearances;

        [Column(ordinal: "14")]
        public float MVPs;

        [Column(ordinal: "15")]
        public float TripleCrowns;

        [Column(ordinal: "16")]
        public float GoldGloves;

        [Column(ordinal: "17")]
        public float MajorLeaguePlayerOfTheYearAwards;

        [Column(ordinal: "18")]
        public float TB;

        [Column(ordinal: "19")]
        public float LastYearPlayed;

        [Column(ordinal: "20")]
        public float PlayerID;

        public override string ToString()
        {
            var test = string.Empty;
            return FullPlayerName + " | ID: " + PlayerID + " | Years: " + YearsPlayed + " | AllStarAppearances: " + AllStarAppearances + " | H: " + H + " | MVPs: " + MVPs + " | TB: " + TB + " ";
        }
    }


    public class BaseballDataPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel;

        [ColumnName("Probability")]
        public float ProbabilityLabel;

        [ColumnName("Score")]
        public float ScoreLabel;
    }
}