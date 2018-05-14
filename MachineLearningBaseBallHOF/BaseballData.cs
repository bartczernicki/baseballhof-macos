using System;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;

namespace MachineLearningBaseBallHOF
{
    public class BaseballData
    {
        [Column("0")]
        [ColumnName("Label")]
        public bool Label;

        [Column("1")]
        public string FullPlayerName;

        [Column("2")]
        public float YearsPlayed;

        [Column("3")]
        public float AB;

        [Column("4")]
        public float R;

        [Column("5")]
        public float H;

        [Column("6")]
        public float Doubles;

        [Column("7")]
        public float Triples;

        [Column("8")]
        public float HR;

        [Column("9")]
        public float RBI;

        [Column("10")]
        public float SB;

        [Column("11")]
        public float AllStarAppearances;

        [Column("12")]
        public float MVPs;

        [Column("13")]
        public float TripleCrowns;

        [Column("14")]
        public float GoldGloves;

        [Column("15")]
        public float MajorLeaguePlayerOfTheYearAwards;

        [Column("16")]
        public float PlayerID;

        public override string ToString()
        {
            var test = string.Empty;
            return FullPlayerName + " | ID: " + PlayerID + " | Years: " + YearsPlayed + " | AllStarAppearances: " + AllStarAppearances + " | H: " + H;
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