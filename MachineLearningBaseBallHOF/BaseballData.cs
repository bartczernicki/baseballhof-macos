using System;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;

namespace MachineLearningBaseBallHOF
{
	public class BaseballData
    {
        [Column("0")]
        public float ID;
        
        [Column("1")]
        public string FullPlayerName;
        
        [Column("2")]
		public float YearsPlayed;
        
        [Column("3")]
		public float AB;

        [Column("4")]
		public float H;

		[Column("5")]
		public float AllStarAppearances;

		[Column("6")]
		public float MVPs;

		[Column("7")]
		[ColumnName("Label")]
        public bool Label;
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
